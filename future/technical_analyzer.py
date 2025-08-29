# technical_analyzer.py
import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime
from collections import namedtuple

# ==========================
# ANSI color codes
# ==========================
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

QuickSig = namedtuple('QuickSig', ['signal', 'debug'])

# ==========================
# Futures-aware, cached fetcher
# ==========================
class FuturesDataFetcher:
    """
    Minimal futures OHLCV fetcher with a tiny TTL cache to avoid hammering the API.
    """
    def __init__(self, symbol='TAO/USDT', exchange_id='binanceusdm', api_key='', api_secret=''):
        self.symbol = symbol
        self.ttl = 2.0  # seconds
        self._cache = {}  # timeframe -> (ts, df)

        ctor = getattr(ccxt, exchange_id)
        self.exchange = ctor({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},
            'apiKey': api_key,
            'secret': api_secret,
            'timeout': 30000
        })
        self.exchange.load_markets()
        self.symbol = self._normalize_symbol(self.exchange, symbol)
        print(f"{Colors.CYAN}Futures fetcher using {self.exchange.id} symbol={self.symbol}{Colors.END}")

    @staticmethod
    def _normalize_symbol(exchange, symbol: str) -> str:
        if symbol in exchange.markets:
            return symbol
        flat = symbol.replace('/', '')
        for s in exchange.symbols:
            if s.replace('/', '') == flat:
                return s
        # fallbacks
        if 'BTC/USDT' in exchange.markets:
            print(f"{Colors.YELLOW}WARNING: {symbol} not found on {exchange.id}. Falling back to BTC/USDT{Colors.END}")
            return 'BTC/USDT'
        return exchange.symbols[0]

    def fetch_data(self, timeframe: str, limit: int = 200) -> pd.DataFrame:
        now = time.time()
        cached = self._cache.get(timeframe)
        if cached and (now - cached[0]) < self.ttl:
            return cached[1]
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        self._cache[timeframe] = (now, df)
        return df

# ==========================
# Fast 1m/5m scalping analyzer
# ==========================
class QuickScalpAnalyzer:
    """
    Ultra-fast 1m/5m decision engine:
      - EMA9 / EMA21 trend filter
      - MACD histogram direction
      - RSI(14) midline (50) confirm
      - 5m ADX(14) trend gate if 1m/5m disagree

    Returns: QuickSig(signal in {'Buy','Sell','Neutral'}, debug dict)
    """
    def __init__(self, symbol='TAO/USDT', exchange_id='binanceusdm', api_key='', api_secret=''):
        self.fetcher = FuturesDataFetcher(symbol, exchange_id, api_key, api_secret)
        self.last_snapshot = {}  # {'1m': {...}, '5m': {...}, 'consensus': {...}}

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.ewm(alpha=1/window, adjust=False).mean()
        roll_dn = down.ewm(alpha=1/window, adjust=False).mean()
        rs = roll_up / (roll_dn + 1e-12)
        return 100 - (100 / (1 + rs))

    def _adx(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        up_move = high.diff()
        dn_move = -low.diff()
        plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
        prev_close = close.shift(1)
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/window, adjust=False).mean()
        plus_di  = 100 * pd.Series(plus_dm,  index=high.index).ewm(alpha=1/window, adjust=False).mean() / (atr + 1e-12)
        minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1/window, adjust=False).mean() / (atr + 1e-12)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
        return dx.ewm(alpha=1/window, adjust=False).mean()

    def _macd_hist(self, close: pd.Series):
        ema12 = self._ema(close, 12)
        ema26 = self._ema(close, 26)
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return hist

    def _one_tf_signal(self, df: pd.DataFrame) -> QuickSig:
        if df is None or len(df) < 60:
            return QuickSig('Neutral', {'reason': 'insufficient_bars'})

        c = df['close']
        ema9 = self._ema(c, 9)
        ema21 = self._ema(c, 21)
        hist = self._macd_hist(c)
        rsi = self._rsi(c, 14)
        adx = self._adx(df['high'], df['low'], c, 14)

        cond_long  = (c.iloc[-1] > ema21.iloc[-1]) and (ema9.iloc[-1] > ema21.iloc[-1]) and (hist.iloc[-1] > 0) and (rsi.iloc[-1] > 50)
        cond_short = (c.iloc[-1] < ema21.iloc[-1]) and (ema9.iloc[-1] < ema21.iloc[-1]) and (hist.iloc[-1] < 0) and (rsi.iloc[-1] < 50)
        sig = 'Buy' if cond_long else ('Sell' if cond_short else 'Neutral')
        dbg = {
            'close': float(c.iloc[-1]),
            'ema9': float(ema9.iloc[-1]),
            'ema21': float(ema21.iloc[-1]),
            'macd_hist': float(hist.iloc[-1]),
            'rsi': float(rsi.iloc[-1]),
            'adx': float(adx.iloc[-1]),
        }
        return QuickSig(sig, dbg)

    def quick_decision(self) -> QuickSig:
        df1 = self.fetcher.fetch_data('1m', limit=200)
        df5 = self.fetcher.fetch_data('5m', limit=200)

        s1 = self._one_tf_signal(df1)
        s5 = self._one_tf_signal(df5)

        # Consensus
        if s1.signal == s5.signal:
            self.last_snapshot = {'1m': s1.debug, '5m': s5.debug, 'consensus': {'source': 'agreement', 'signal': s1.signal}}
            return s1

        # 5m trend gate
        adx5 = s5.debug.get('adx', 0.0)
        if adx5 >= 18.0:
            self.last_snapshot = {'1m': s1.debug, '5m': s5.debug, 'consensus': {'source': '5m_trend', 'signal': s5.signal, 'adx5': adx5}}
            return s5

        # Neutral if disagreement w/o trend
        self.last_snapshot = {'1m': s1.debug, '5m': s5.debug, 'consensus': {'source': 'disagree_no_trend', 'signal': 'Neutral', 'adx5': adx5}}
        return QuickSig('Neutral', self.last_snapshot['consensus'])

# Standalone quick check
if __name__ == "__main__":
    print(f"{Colors.BOLD}{Colors.CYAN}QuickScalpAnalyzer smoke test{Colors.END}")
    qa = QuickScalpAnalyzer('TAO/USDT', 'binanceusdm')
    sig = qa.quick_decision()
    print("Signal:", sig.signal)
    print("Snapshot:", qa.last_snapshot)
