# # technical_analyzer.py
# import argparse
# import ccxt
# import pandas as pd
# import numpy as np
# from datetime import datetime
# from pprint import pprint
# import warnings
# import os  # <--- ADD THIS LINE ---
# warnings.filterwarnings('ignore')
# # ==========================
# # ANSI color codes
# # ==========================
# class Colors:
#     RED = '\033[91m'
#     GREEN = '\033[92m'
#     YELLOW = '\033[93m'
#     BLUE = '\033[94m'
#     MAGENTA = '\033[95m'
#     CYAN = '\033[96m'
#     WHITE = '\033[97m'
#     BOLD = '\033[1m'
#     UNDERLINE = '\033[4m'
#     END = '\033[0m'

# def color_signal(signal):
#     if signal == 'Buy':
#         return f"{Colors.GREEN}{signal}{Colors.END}"
#     elif signal == 'Sell':
#         return f"{Colors.RED}{signal}{Colors.END}"
#     return f"{Colors.YELLOW}{signal}{Colors.END}"

# def color_recommendation(recommendation):
#     if 'Buy' in recommendation:
#         return f"{Colors.GREEN}{recommendation}{Colors.END}"
#     elif 'Sell' in recommendation:
#         return f"{Colors.RED}{recommendation}{Colors.END}"
#     return f"{Colors.YELLOW}{recommendation}{Colors.END}"

# # ==========================
# # Helpers
# # ==========================
# def timeframe_to_freq(timeframe):
#     mapping = {'1m': '1T', '5m': '5T', '15m': '15T', '1h': '1H', '1d': '1D', '1w': '1W', '1M': '1M'}
#     return mapping.get(timeframe, timeframe)

# # ==========================
# # Fast Data fetcher for Futures
# # ==========================
# class FastFuturesDataFetcher:
#     def __init__(self, symbol='TAO/USDT:USDT'):
#         self.symbol = symbol
#         # Use a single exchange instance, potentially configured for futures
#         self.exchange = ccxt.binance({
#             'apiKey': os.getenv('BINANCE_FUTURES_API_KEY', ''),
#             'secret': os.getenv('BINANCE_FUTURES_API_SECRET', ''),
#             'enableRateLimit': True,
#             'options': {'defaultType': 'future'} # Important for futures
#         })
#         try:
#             self.exchange.load_markets()
#             # Check if symbol exists for futures
#             if self.symbol not in self.exchange.symbols:
#                 print(f"Warning: Symbol {self.symbol} not found on {self.exchange.id} futures market.")
#         except Exception as e:
#             print(f"Error initializing futures exchange: {e}")

#     def fetch_fast_data(self, timeframe, limit=100): # Optimized fetch
#         try:
#             ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=limit)
#             if not ohlcv:
#                 raise Exception("No data returned")
#             df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
#             df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
#             df.set_index('timestamp', inplace=True)
#             return df
#         except Exception as e:
#             print(f"Error fetching {timeframe} data: {e}")
#             return pd.DataFrame() # Return empty DataFrame on error

# # ==========================
# # Oscillators (keeping only necessary ones)
# # ==========================
# class Oscillators:
#     @staticmethod
#     def rsi_wilder(data, window=9): # Shorter window for scalping
#         delta = data.diff()
#         up = delta.clip(lower=0)
#         down = -delta.clip(upper=0)
#         roll_up = up.ewm(alpha=1/window, adjust=False).mean()
#         roll_down = down.ewm(alpha=1/window, adjust=False).mean()
#         rs = roll_up / (roll_down + 1e-12)
#         return 100 - (100 / (1 + rs))

#     @staticmethod
#     def macd(data, fast=6, slow=13, signal=4): # Ultra-fast settings
#         ema_fast = data.ewm(span=fast, adjust=False).mean()
#         ema_slow = data.ewm(span=slow, adjust=False).mean()
#         macd_line = ema_fast - ema_slow
#         signal_line = macd_line.ewm(span=signal, adjust=False).mean()
#         return macd_line, signal_line, macd_line - signal_line

#     @staticmethod
#     def atr_wilder(high, low, close, window=14):
#         prev_close = close.shift(1)
#         tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
#         return tr.ewm(alpha=1/window, adjust=False).mean()

# # ==========================
# # Fast Scalping Analyzer
# # ==========================
# class FastScalpingAnalyzer:
#     def __init__(self, symbol='TAO/USDT:USDT'): # Futures symbol format
#         self.symbol = symbol
#         self.data_fetcher = FastFuturesDataFetcher(symbol)

#     def get_scalping_signal(self):
#         """
#         Get a quick Buy/Sell/Neutral signal based on 1m data.
#         Uses minimal, fast indicators.
#         """
#         df_1m = self.data_fetcher.fetch_fast_data('1m', limit=100)
#         if df_1m.empty or len(df_1m) < 20: # Need enough data
#             return 'Neutral', {}

#         current_price = df_1m['close'].iloc[-1]
#         # Example fast indicators
#         ema_fast = df_1m['close'].ewm(span=5, adjust=False).mean().iloc[-1]
#         ema_slow = df_1m['close'].ewm(span=10, adjust=False).mean().iloc[-1]
#         rsi = Oscillators.rsi_wilder(df_1m['close'], window=9).iloc[-1] # Using existing Oscillators class
#         macd_line, signal_line, histogram = Oscillators.macd(df_1m['close'], fast=6, slow=13, signal=4)
#         macd_hist_current = histogram.iloc[-1]
#         macd_hist_prev = histogram.iloc[-2] if len(histogram) > 1 else 0

#         signal_details = {
#             'price': current_price,
#             'ema_fast': ema_fast,
#             'ema_slow': ema_slow,
#             'rsi': rsi,
#             'macd_hist': macd_hist_current
#         }

#         # Simple Buy Logic: Price above fast EMA, fast EMA above slow EMA, RSI > 50, MACD histogram turning positive
#         if (current_price > ema_fast > ema_slow and
#             rsi > 50 and
#             macd_hist_current > 0 and macd_hist_prev <= 0):
#             return 'Buy', signal_details

#         # Simple Sell Logic: Price below fast EMA, fast EMA below slow EMA, RSI < 50, MACD histogram turning negative
#         elif (current_price < ema_fast < ema_slow and
#               rsi < 50 and
#               macd_hist_current < 0 and macd_hist_prev >= 0):
#             return 'Sell', signal_details

#         else:
#             return 'Neutral', signal_details

#     def compute_scalping_plan(self):
#         """
#         Compute a simple scalping plan using 5m for context and 1m for triggers.
#         """
#         df_5m = self.data_fetcher.fetch_fast_data('5m', limit=100)
#         df_1m = self.data_fetcher.fetch_fast_data('1m', limit=100)

#         if df_5m.empty or df_1m.empty:
#             return None

#         # Simple trend context from 5m
#         ema_20_5m = df_5m['close'].ewm(span=20, adjust=False).mean().iloc[-1]
#         last_price_5m = df_5m['close'].iloc[-1]
#         trend_long = last_price_5m > ema_20_5m
#         trend_short = last_price_5m < ema_20_5m

#         # Simple trigger levels from recent 1m highs/lows
#         recent_high = df_1m['high'].tail(10).max() # Last 10 1m candles high
#         recent_low = df_1m['low'].tail(10).min()   # Last 10 1m candles low

#         plan = {
#             'trend_long': trend_long,
#             'trend_short': trend_short,
#             'long_trigger': float(recent_high * 1.0005), # Slight buffer above recent high
#             'short_trigger': float(recent_low * 0.9995), # Slight buffer below recent low
#             # SL will be handled fixed at 2% in the bot logic
#             # TP can be fixed RR (e.g., 1:1.5) or also dynamic
#         }
#         return plan

# # ==========================
# # Execution Plan Logic (Updated for Futures Scalping Context)
# # ==========================
# def atr_wilder_series(high, low, close, window=14):
#     prev_close = close.shift(1)
#     tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
#     return tr.ewm(alpha=1/window, adjust=False).mean()

# def compute_scalping_plan_1m_5m(df_1m: pd.DataFrame, df_5m: pd.DataFrame, sentiment: float = 0.0):
#     """
#     Use last CLOSED 5m candle as regime; arm 1m breakout/pullback levels.
#     """
#     if len(df_5m) < 2 or len(df_1m) < 2:
#         return None
        
#     last_5m = df_5m.iloc[-2]  # last closed 5m bar
#     atr5m = atr_wilder_series(df_5m['high'], df_5m['low'], df_5m['close'], window=14).iloc[-2]
#     ema20_5m = df_5m['close'].ewm(span=20, adjust=False).mean().iloc[-2]

#     trend_long  = last_5m['close'] > ema20_5m
#     trend_short = last_5m['close'] < ema20_5m

#     prev_high_5m = last_5m['high']
#     prev_low_5m  = last_5m['low']

#     base_sl_mult = 1.2
#     base_tp_mult = 2.0

#     # Use recent 1m data for more precise triggers
#     recent_high_1m = df_1m['high'].tail(10).max()
#     recent_low_1m = df_1m['low'].tail(10).min()

#     plan = {
#         'trend_long': trend_long,
#         'trend_short': trend_short,
#         'long_trigger': float(max(prev_high_5m, recent_high_1m) * 1.0005), # Slightly above recent high
#         'short_trigger': float(min(prev_low_5m, recent_low_1m) * 0.9995),   # Slightly below recent low
#         'pullback_long': float(prev_high_5m - 0.25 * atr5m),  # if breakout missed
#         'pullback_short': float(prev_low_5m  + 0.25 * atr5m),
#         'atr5m': float(atr5m),
#         'tp_mult': base_tp_mult,
#         'sl_mult': base_sl_mult,
#         'ema20_5m': float(ema20_5m),
#         'prev_high_5m': float(prev_high_5m),
#         'prev_low_5m': float(prev_low_5m),
#     }

#     # Apply sentiment bias if needed (simplified)
#     s = max(min(sentiment, 0.8), -0.8)
#     plan['long_trigger']  *= (1 - 0.0005 * max(s, 0))   # bullish -> trigger slightly easier
#     plan['short_trigger'] *= (1 + 0.0005 * max(-s, 0))  # bearish -> easier for shorts
#     plan['tp_mult']       *= (1 + 0.3 * s)              # widen TP when bullish
#     plan['sl_mult']       *= (1 - 0.15 * s)             # slightly tighter SL when bullish

#     return plan

# # ==========================
# # Main (for testing the analyzer)
# # ==========================
# if __name__ == "__main__":
#     # This part can be used for testing the analyzer independently
#     class Args:
#       use_sentiment = True  # or False to disable
#     args = Args()

#     print(f"{Colors.BOLD}{Colors.CYAN}FUTURES SCALPING ANALYSIS{Colors.END}")
#     analyzer = FastScalpingAnalyzer('TAO/USDT:USDT')
    
#     # Get fast signal
#     signal, details = analyzer.get_scalping_signal()
#     print(f"Fast Scalping Signal: {signal}")
#     print(f"Signal Details: {details}")
    
#     # Build an execution plan using 1m (timing) + 5m (trend)
#     df_1m = analyzer.data_fetcher.fetch_fast_data('1m', limit=100)
#     df_5m = analyzer.data_fetcher.fetch_fast_data('5m', limit=100)
#     sentiment = 0.0 # Example sentiment
#     plan = compute_scalping_plan_1m_5m(df_1m, df_5m, sentiment=sentiment)
#     last_price_1m = float(df_1m['close'].iloc[-1]) if not df_1m.empty else 0.0
    
#     print(f"\n{Colors.BOLD}{Colors.BLUE}SCALPING PLAN{Colors.END}")
#     pprint(plan)
#     print(f"\nLast 1m price: {last_price_1m:.4f}")

#     print(f"\n{Colors.BOLD}{Colors.MAGENTA}Analysis Complete!{Colors.END}")




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
