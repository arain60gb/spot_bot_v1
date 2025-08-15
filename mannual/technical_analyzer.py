import argparse
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
from pprint import pprint
import warnings
warnings.filterwarnings('ignore')

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

def color_signal(signal):
    if signal == 'Buy':
        return f"{Colors.GREEN}{signal}{Colors.END}"
    elif signal == 'Sell':
        return f"{Colors.RED}{signal}{Colors.END}"
    return f"{Colors.YELLOW}{signal}{Colors.END}"

def color_recommendation(recommendation):
    if 'Buy' in recommendation:
        return f"{Colors.GREEN}{recommendation}{Colors.END}"
    elif 'Sell' in recommendation:
        return f"{Colors.RED}{recommendation}{Colors.END}"
    return f"{Colors.YELLOW}{recommendation}{Colors.END}"

# ==========================
# Helpers
# ==========================
def timeframe_to_freq(timeframe):
    mapping = {'5m': '5T', '15m': '15T', '1h': '1H', '1d': '1D', '1w': '1W', '1M': '1M'}
    return mapping.get(timeframe, timeframe)

# ==========================
# Data fetcher
# ==========================
class DataFetcher:
    def __init__(self, symbol='TAO/USDT'):
        self.symbol = symbol
        self.exchanges = [
            ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'spot'}}),
            ccxt.kraken({'enableRateLimit': True}),
            ccxt.kucoin({'enableRateLimit': True}),
            ccxt.gateio({'enableRateLimit': True})
        ]
        self.exchange = None
        self._find_exchange_with_symbol()

    def _find_exchange_with_symbol(self):
        for exchange in self.exchanges:
            try:
                exchange.load_markets()
                if self.symbol in exchange.symbols:
                    self.exchange = exchange
                    print(f"Using exchange: {exchange.id} for {self.symbol}")
                    return
            except Exception:
                continue
        print("TAO/USDT not found, searching for alternatives...")
        for exchange in self.exchanges:
            try:
                exchange.load_markets()
                tao_symbols = [s for s in exchange.symbols if 'TAO' in s]
                if tao_symbols:
                    self.symbol = tao_symbols[0]
                    self.exchange = exchange
                    print(f"Found alternative: {self.symbol} on {exchange.id}")
                    return
            except Exception:
                continue
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.symbol = 'BTC/USDT'
        print(f"{Colors.YELLOW}WARNING: Using BTC/USDT as fallback — TAO not found!{Colors.END}")

    def fetch_data(self, timeframe, limit=500):
        try:
            if not self.exchange:
                raise Exception("No exchange available")
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=limit)
            if not ohlcv:
                raise Exception("No data returned")
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception:
            return self._get_sample_data(timeframe, limit)

    def _get_sample_data(self, timeframe, limit):
        np.random.seed(42)
        base_price = 100 + np.random.random() * 50
        dates = pd.date_range(end=datetime.now(), periods=limit, freq=timeframe_to_freq(timeframe))
        prices = [base_price]
        for _ in range(1, limit):
            change = np.random.normal(0, 0.02)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 0.01))
        df = pd.DataFrame(index=dates)
        df['close'] = prices
        df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.01, limit)))
        df['low']  = df[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.01, limit)))
        df['volume'] = np.random.uniform(1000, 10000, limit)
        return df

# ==========================
# Pivot points
# ==========================
class PivotPoints:
    @staticmethod
    def standard(high, low, close):
        pp = (high + low + close) / 3
        r1 = 2 * pp - low
        s1 = 2 * pp - high
        r2 = pp + (high - low)
        s2 = pp - (high - low)
        r3 = high + 2 * (pp - low)
        s3 = low - 2 * (high - pp)
        return {'S3': s3, 'S2': s2, 'S1': s1, 'PP': pp, 'R1': r1, 'R2': r2, 'R3': r3}

    @staticmethod
    def fibonacci(high, low, close):
        pp = (high + low + close) / 3
        rng = high - low
        return {
            'S3': pp - (rng * 1.000), 'S2': pp - (rng * 0.618), 'S1': pp - (rng * 0.382),
            'PP': pp,
            'R1': pp + (rng * 0.382), 'R2': pp + (rng * 0.618), 'R3': pp + (rng * 1.000)
        }

    @staticmethod
    def camarilla(high, low, close):
        h_l = high - low
        pp = (high + low + close) / 3
        r1 = close + (h_l * 1.1 / 12)
        r2 = close + (h_l * 1.1 / 6)
        r3 = close + (h_l * 1.1 / 4)
        s1 = close - (h_l * 1.1 / 12)
        s2 = close - (h_l * 1.1 / 6)
        s3 = close - (h_l * 1.1 / 4)
        return {'S3': s3, 'S2': s2, 'S1': s1, 'PP': pp, 'R1': r1, 'R2': r2, 'R3': r3}

    @staticmethod
    def woodies(high, low, close, open_price):
        pp = (high + low + 2 * close) / 4
        r1 = (2 * pp) - low
        s1 = (2 * pp) - high
        r2 = pp + (high - low)
        s2 = pp - (high - low)
        r3 = high + 2 * (pp - low)
        s3 = low - 2 * (high - pp)
        return {'S3': s3, 'S2': s2, 'S1': s1, 'PP': pp, 'R1': r1, 'R2': r2, 'R3': r3}

    @staticmethod
    def demarks(high, low, close, open_price):
        if close < open_price:
            x = high + (2 * low) + close
        elif close > open_price:
            x = (2 * high) + low + close
        else:
            x = high + low + (2 * close)
        pp = x / 4
        r1 = x / 2 - low
        s1 = x / 2 - high
        return {'S1': s1, 'PP': pp, 'R1': r1}

# ==========================
# MAs
# ==========================
class MovingAverages:
    @staticmethod
    def sma(data, window):
        return data.rolling(window=window).mean()

    @staticmethod
    def ema(data, window):
        return data.ewm(span=window, adjust=False).mean()

# ==========================
# Oscillators (with Wilder + wrappers)
# ==========================
class Oscillators:
    @staticmethod
    def rsi_wilder(data, window=14):
        delta = data.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.ewm(alpha=1/window, adjust=False).mean()
        roll_down = down.ewm(alpha=1/window, adjust=False).mean()
        rs = roll_up / (roll_down + 1e-12)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def rsi(data, window=14):  # backward-compatible
        return Oscillators.rsi_wilder(data, window)

    @staticmethod
    def stochastic(high, low, close, k_window=14, d_window=3):
        low_min = low.rolling(window=k_window).min()
        high_max = high.rolling(window=k_window).max()
        k = 100 * ((close - low_min) / (high_max - low_min + 1e-12))
        d = k.rolling(window=d_window).mean()
        return k, d

    @staticmethod
    def stoch_rsi(data, window=14, eps=1e-12):
        rsi = Oscillators.rsi_wilder(data, window)
        rsi_min = rsi.rolling(window).min()
        rsi_max = rsi.rolling(window).max()
        return ((rsi - rsi_min) / (rsi_max - rsi_min + eps)) * 100

    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line, macd_line - signal_line

    @staticmethod
    def adx_wilder(high, low, close, window=14):
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        prev_close = close.shift(1)
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/window, adjust=False).mean()
        plus_di  = 100 * pd.Series(plus_dm,  index=high.index).ewm(alpha=1/window, adjust=False).mean() / (atr + 1e-12)
        minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1/window, adjust=False).mean() / (atr + 1e-12)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
        adx = dx.ewm(alpha=1/window, adjust=False).mean()
        return adx, plus_di, minus_di

    @staticmethod
    def adx(high, low, close, window=14):  # backward-compatible
        return Oscillators.adx_wilder(high, low, close, window)

    @staticmethod
    def cci(high, low, close, window=14):
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=window).mean()
        def mean_abs_dev(x):
            return np.mean(np.abs(x - np.mean(x)))
        mean_dev = tp.rolling(window=window).apply(mean_abs_dev, raw=True)
        return (tp - sma_tp) / (0.015 * (mean_dev + 1e-12))

    @staticmethod
    def williams_r(high, low, close, window=14):
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low + 1e-12)

    @staticmethod
    def roc(data, window=10):
        return (data - data.shift(window)) / (data.shift(window) + 1e-12) * 100

    @staticmethod
    def awesome_oscillator(high, low):
        median_price = (high + low) / 2
        return median_price.rolling(5).mean() - median_price.rolling(34).mean()

    @staticmethod
    def atr_wilder(high, low, close, window=14):
        prev_close = close.shift(1)
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1/window, adjust=False).mean()

    @staticmethod
    def atr(high, low, close, window=14):  # backward-compatible
        return Oscillators.atr_wilder(high, low, close, window)

# ==========================
# Signals
# ==========================
class SignalGenerator:
    @staticmethod
    def ma_signal(price, ma_value):
        if price > ma_value:
            return 'Buy'
        elif price < ma_value:
            return 'Sell'
        return 'Neutral'

    @staticmethod
    def rsi_signal(rsi_value):
        if rsi_value < 30:
            return 'Buy'
        elif rsi_value > 70:
            return 'Sell'
        return 'Neutral'

    @staticmethod
    def stochastic_signal(k_value, d_value):
        if k_value < 20 and d_value < 20:
            return 'Buy'
        elif k_value > 80 and d_value > 80:
            return 'Sell'
        return 'Neutral'

    @staticmethod
    def stoch_rsi_signal(stoch_rsi_value):
        if stoch_rsi_value < 20:
            return 'Buy'
        elif stoch_rsi_value > 80:
            return 'Sell'
        return 'Neutral'

    @staticmethod
    def macd_signal(macd_line, signal_line):
        if macd_line > signal_line:
            return 'Buy'
        elif macd_line < signal_line:
            return 'Sell'
        return 'Neutral'

    @staticmethod
    def adx_signal(adx_value, plus_di, minus_di):
        if adx_value > 25 and plus_di > minus_di:
            return 'Buy'
        elif adx_value > 25 and minus_di > plus_di:
            return 'Sell'
        return 'Neutral'

    @staticmethod
    def cci_signal(cci_value):
        if cci_value < -100:
            return 'Buy'
        elif cci_value > 100:
            return 'Sell'
        return 'Neutral'

    @staticmethod
    def williams_r_signal(wr_value):
        if wr_value < -80:
            return 'Buy'
        elif wr_value > -20:
            return 'Sell'
        return 'Neutral'

    @staticmethod
    def roc_signal(roc_value):
        if roc_value > 0:
            return 'Buy'
        elif roc_value < 0:
            return 'Sell'
        return 'Neutral'

    @staticmethod
    def awesome_oscillator_signal(ao_value, prev_ao):
        if ao_value > 0 and prev_ao < 0:
            return 'Buy'
        elif ao_value < 0 and prev_ao > 0:
            return 'Sell'
        return 'Neutral'

# ==========================
# Summary aggregator
# ==========================
class SummaryEngine:
    @staticmethod
    def generate_summary(ma_signals, osc_signals):
        total_buy = ma_signals.count('Buy') + osc_signals.count('Buy')
        total_sell = ma_signals.count('Sell') + osc_signals.count('Sell')
        if total_buy >= 2 * total_sell and total_buy >= 8:
            return 'Strong Buy'
        elif total_buy >= 1.5 * total_sell and total_buy >= 5:
            return 'Buy'
        elif total_sell >= 2 * total_buy and total_sell >= 8:
            return 'Strong Sell'
        elif total_sell >= 1.5 * total_buy and total_sell >= 5:
            return 'Sell'
        else:
            return 'Neutral'

# ==========================
# Printing helpers
# ==========================
def print_table(data, headers, title=""):
    if title:
        print(f"\n{Colors.BOLD}{Colors.CYAN}{title}{Colors.END}")
        print("=" * 100)
    header_str = " | ".join(f"{h:^22}" for h in headers)
    print(f"{Colors.BOLD}{header_str}{Colors.END}")
    print("-" * 100)
    for row in data:
        cells = []
        for i, cell in enumerate(row):
            if i == 2:
                cells.append(f"{color_signal(str(cell)):^22}")
            else:
                cells.append(f"{str(cell):^22}")
        print(" | ".join(cells))

def calculate_signal_percentage(signal_list):
    if not signal_list:
        return {'Buy': 0, 'Sell': 0, 'Neutral': 0}
    total = len(signal_list)
    return {
        'Buy': round(signal_list.count('Buy') / total * 100, 2),
        'Sell': round(signal_list.count('Sell') / total * 100, 2),
        'Neutral': round(signal_list.count('Neutral') / total * 100, 2),
    }

def format_results(results):
    if not results:
        print("No data available")
        return

    for timeframe, data in results.items():
        if data is None:
            print(f"\n{Colors.BOLD}{Colors.RED}{timeframe.upper()} Timeframe: Error in analysis{Colors.END}")
            continue

        print(f"\n{Colors.BOLD}{Colors.MAGENTA}{'='*120}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.MAGENTA}{timeframe.upper()} TIMEFRAME ANALYSIS{Colors.END}")
        print(f"{Colors.BOLD}{Colors.MAGENTA}{'='*120}{Colors.END}")

        print(f"\n{Colors.BOLD}Current Price: {Colors.CYAN}${data['current_price']:.4f}{Colors.END}")
        print(f"{Colors.BOLD}Overall Recommendation: {color_recommendation(data['overall_recommendation'])}")

        if 'Pivot_Points' in data['indicators'] and data['indicators']['Pivot_Points']:
            print(f"\n{Colors.BOLD}{Colors.BLUE}PIVOT POINTS (Previous Period){Colors.END}")
            print("=" * 100)
            pivot_methods = ['Classic', 'Fibonacci', 'Camarilla', 'Woodie', 'DeMark']
            pivot_header = "Method     |     S3     |     S2     |     S1     |     PP     |     R1     |     R2     |     R3     "
            print(f"{Colors.BOLD}{pivot_header}{Colors.END}")
            print("-" * 100)
            for method in pivot_methods:
                pivots = data['indicators']['Pivot_Points'].get(method)
                if not pivots: 
                    continue
                if method == 'DeMark':
                    line = f"{method:<10} |            |            | {pivots.get('S1', np.nan):>10.4f} | {pivots.get('PP', np.nan):>10.4f} | {pivots.get('R1', np.nan):>10.4f} |            |            "
                else:
                    line = f"{method:<10} | {pivots.get('S3', np.nan):>10.4f} | {pivots.get('S2', np.nan):>10.4f} | {pivots.get('S1', np.nan):>10.4f} | {pivots.get('PP', np.nan):>10.4f} | {pivots.get('R1', np.nan):>10.4f} | {pivots.get('R2', np.nan):>10.4f} | {pivots.get('R3', np.nan):>10.4f} "
                print(line)
            print("-" * 100)

        indicator_data = []
        all_signals = data['ma_signals'] + data['osc_signals']
        overall_signal_pct = calculate_signal_percentage(all_signals)

        for name, v in data['indicators'].items():
            if name == 'Pivot_Points':
                continue
            try:
                value_str = f"{v['value']:.4f}"
            except Exception:
                value_str = "N/A"
            signal = v['signal']
            if signal == 'Buy':
                signal_pct_str = f"{overall_signal_pct['Buy']:.1f}% Buy"
            elif signal == 'Sell':
                signal_pct_str = f"{overall_signal_pct['Sell']:.1f}% Sell"
            else:
                signal_pct_str = f"{overall_signal_pct['Neutral']:.1f}% Neutral"
            indicator_data.append([name, value_str, signal, signal_pct_str])

        print_table(indicator_data, ["Indicator", "Value", "Signal", "Signal %"], "TECHNICAL INDICATORS")

        buy_ma = data['ma_signals'].count('Buy'); sell_ma = data['ma_signals'].count('Sell'); neutral_ma = data['ma_signals'].count('Neutral')
        buy_osc = data['osc_signals'].count('Buy'); sell_osc = data['osc_signals'].count('Sell'); neutral_osc = data['osc_signals'].count('Neutral')

        print(f"\n{Colors.BOLD}{Colors.BLUE}SIGNAL COUNTS{Colors.END}")
        print("-" * 60)
        print(f"MA Signals    - {Colors.GREEN}Buy: {buy_ma}{Colors.END}, {Colors.RED}Sell: {sell_ma}{Colors.END}, {Colors.YELLOW}Neutral: {neutral_ma}{Colors.END}")
        print(f"Osc Signals   - {Colors.GREEN}Buy: {buy_osc}{Colors.END}, {Colors.RED}Sell: {sell_osc}{Colors.END}, {Colors.YELLOW}Neutral: {neutral_osc}{Colors.END}")
        print(f"Total         - {Colors.GREEN}Buy: {buy_ma+buy_osc}{Colors.END}, {Colors.RED}Sell: {sell_ma+sell_osc}{Colors.END}, {Colors.YELLOW}Neutral: {neutral_ma+neutral_osc}{Colors.END}")

        print(f"\n{Colors.BOLD}{Colors.BLUE}OVERALL SIGNAL PERCENTAGE{Colors.END}")
        print("-" * 60)
        print(f"Buy: {Colors.GREEN}{overall_signal_pct['Buy']:.1f}%{Colors.END}, "
              f"Sell: {Colors.RED}{overall_signal_pct['Sell']:.1f}%{Colors.END}, "
              f"Neutral: {Colors.YELLOW}{overall_signal_pct['Neutral']:.1f}%{Colors.END}")

# ==========================
# Analyzer
# ==========================
class TechnicalAnalyzer:
    def __init__(self, symbol='TAO/USDT'):
        self.symbol = symbol
        self.data_fetcher = DataFetcher(symbol)
        self.pivot_points = PivotPoints()
        self.ma = MovingAverages()
        self.osc = Oscillators()
        self.signal_gen = SignalGenerator()
        self.summary_engine = SummaryEngine()
        print(f"Initialized analyzer for {self.data_fetcher.symbol}")

    def calculate_pivot_points(self, df):
        if len(df) < 2:
            return {}
        prev_row = df.iloc[-2]
        return {
            'Classic':   self.pivot_points.standard(prev_row['high'], prev_row['low'], prev_row['close']),
            'Fibonacci': self.pivot_points.fibonacci(prev_row['high'], prev_row['low'], prev_row['close']),
            'Camarilla': self.pivot_points.camarilla(prev_row['high'], prev_row['low'], prev_row['close']),
            'Woodie':    self.pivot_points.woodies(prev_row['high'], prev_row['low'], prev_row['close'], prev_row['open']),
            'DeMark':    self.pivot_points.demarks(prev_row['high'], prev_row['low'], prev_row['close'], prev_row['open']),
        }

    def calculate_indicators(self, df):
        if len(df) < 34:
            print("Insufficient data for analysis")
            return None

        results = {}
        current_price = df['close'].iloc[-1]

        pivot_data = self.calculate_pivot_points(df)
        results['Pivot_Points'] = pivot_data

        ma_windows = [5, 10, 20, 50, 100, 200]
        ma_signals = []
        for window in ma_windows:
            sma_val = self.ma.sma(df['close'], window).iloc[-1]
            ema_val = self.ma.ema(df['close'], window).iloc[-1]
            results[f'SMA({window})'] = {'value': float(sma_val), 'signal': self.signal_gen.ma_signal(current_price, sma_val)}
            results[f'EMA({window})'] = {'value': float(ema_val), 'signal': self.signal_gen.ma_signal(current_price, ema_val)}
            ma_signals.append(self.signal_gen.ma_signal(current_price, sma_val))
            ma_signals.append(self.signal_gen.ma_signal(current_price, ema_val))

        osc_signals = []
        try:
            rsi_val = self.osc.rsi_wilder(df['close']).iloc[-1]
            results['RSI(14)'] = {'value': float(rsi_val), 'signal': self.signal_gen.rsi_signal(rsi_val)}
            osc_signals.append(self.signal_gen.rsi_signal(rsi_val))
        except Exception as e:
            print(f"Error calculating RSI: {e}")

        try:
            k_val, d_val = self.osc.stochastic(df['high'], df['low'], df['close'])
            k_val, d_val = k_val.iloc[-1], d_val.iloc[-1]
            results['Stoch %K'] = {'value': float(k_val), 'signal': self.signal_gen.stochastic_signal(k_val, d_val)}
            results['Stoch %D'] = {'value': float(d_val), 'signal': self.signal_gen.stochastic_signal(k_val, d_val)}
            osc_signals.append(self.signal_gen.stochastic_signal(k_val, d_val))
        except Exception as e:
            print(f"Error calculating Stochastic: {e}")

        try:
            stoch_rsi_val = self.osc.stoch_rsi(df['close']).iloc[-1]
            results['StochRSI'] = {'value': float(stoch_rsi_val), 'signal': self.signal_gen.stoch_rsi_signal(stoch_rsi_val)}
            osc_signals.append(self.signal_gen.stoch_rsi_signal(stoch_rsi_val))
        except Exception as e:
            print(f"Error calculating StochRSI: {e}")

        try:
            macd_line, signal_line, _ = self.osc.macd(df['close'])
            macd_val, signal_val = macd_line.iloc[-1], signal_line.iloc[-1]
            results['MACD']   = {'value': float(macd_val),   'signal': self.signal_gen.macd_signal(macd_val, signal_val)}
            results['Signal'] = {'value': float(signal_val), 'signal': self.signal_gen.macd_signal(macd_val, signal_val)}
            osc_signals.append(self.signal_gen.macd_signal(macd_val, signal_val))
        except Exception as e:
            print(f"Error calculating MACD: {e}")

        try:
            adx_val, plus_di, minus_di = self.osc.adx_wilder(df['high'], df['low'], df['close'])
            adx_v, pdi, mdi = adx_val.iloc[-1], plus_di.iloc[-1], minus_di.iloc[-1]
            results['ADX(14)'] = {'value': float(adx_v), 'signal': self.signal_gen.adx_signal(adx_v, pdi, mdi)}
            osc_signals.append(self.signal_gen.adx_signal(adx_v, pdi, mdi))
        except Exception as e:
            print(f"Error calculating ADX: {e}")

        try:
            cci_val = self.osc.cci(df['high'], df['low'], df['close']).iloc[-1]
            results['CCI(14)'] = {'value': float(cci_val), 'signal': self.signal_gen.cci_signal(cci_val)}
            osc_signals.append(self.signal_gen.cci_signal(cci_val))
        except Exception as e:
            print(f"Error calculating CCI: {e}")

        try:
            wr_val = self.osc.williams_r(df['high'], df['low'], df['close']).iloc[-1]
            results['Williams %R(14)'] = {'value': float(wr_val), 'signal': self.signal_gen.williams_r_signal(wr_val)}
            osc_signals.append(self.signal_gen.williams_r_signal(wr_val))
        except Exception as e:
            print(f"Error calculating Williams %R: {e}")

        try:
            roc_val = self.osc.roc(df['close']).iloc[-1]
            results['ROC'] = {'value': float(roc_val), 'signal': self.signal_gen.roc_signal(roc_val)}
            osc_signals.append(self.signal_gen.roc_signal(roc_val))
        except Exception as e:
            print(f"Error calculating ROC: {e}")

        try:
            ao_series = self.osc.awesome_oscillator(df['high'], df['low'])
            ao_val, prev_ao = ao_series.iloc[-1], ao_series.iloc[-2]
            results['AO'] = {'value': float(ao_val), 'signal': self.signal_gen.awesome_oscillator_signal(ao_val, prev_ao)}
            osc_signals.append(self.signal_gen.awesome_oscillator_signal(ao_val, prev_ao))
        except Exception as e:
            print(f"Error calculating AO: {e}")

        try:
            atr_val = self.osc.atr_wilder(df['high'], df['low'], df['close']).iloc[-1]
            results['ATR(14)'] = {'value': float(atr_val), 'signal': 'Neutral'}
        except Exception as e:
            print(f"Error calculating ATR: {e}")

        try:
            overall_recommendation = self.summary_engine.generate_summary(ma_signals, osc_signals)
        except Exception:
            overall_recommendation = 'Neutral'

        return {
            'current_price': float(current_price),
            'indicators': results,
            'ma_signals': ma_signals,
            'osc_signals': osc_signals,
            'overall_recommendation': overall_recommendation
        }

    def analyze_all_timeframes(self):
        timeframes = ['5m', '15m', '1h', '1d', '1w']
        results = {}
        for tf in timeframes:
            try:
                print(f"Analyzing {tf} timeframe...")
                df = self.data_fetcher.fetch_data(tf)
                results[tf] = self.calculate_indicators(df) if (df is not None and len(df) > 0) else None
            except Exception as e:
                print(f"Error analyzing {tf}: {e}")
                results[tf] = None
        return results

# ==========================
# Sentiment + Execution Plan
# ==========================
def sentiment_score_from_cli(use_sentiment: bool) -> float:
    """
    Returns sentiment in [-1, 1]. If --use-sentiment not provided, returns 0.0.
    Replace this stub with loading your real score (e.g., from JSON).
    """
    if not use_sentiment:
        return 0.0
    # Example from earlier DLNews analysis (bullish):
    return 0.71

def apply_sentiment_bias(levels: dict, news_score: float):
    s = max(min(news_score, 0.8), -0.8)
    levels['long_trigger']  *= (1 - 0.0005 * max(s, 0))   # bullish -> trigger slightly easier
    levels['short_trigger'] *= (1 + 0.0005 * max(-s, 0))  # bearish -> easier for shorts
    levels['tp_mult']       *= (1 + 0.3 * s)              # widen TP when bullish
    levels['sl_mult']       *= (1 - 0.15 * s)             # slightly tighter SL when bullish
    return levels

def atr_wilder_series(high, low, close, window=14):
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/window, adjust=False).mean()

def compute_trade_plan_1h_5m(df_1h: pd.DataFrame, df_5m: pd.DataFrame, sentiment: float = 0.0):
    """
    Use last CLOSED 1h candle as regime; arm 5m breakout/pullback levels.
    """
    last = df_1h.iloc[-2]  # last closed 1h bar
    atr1h = atr_wilder_series(df_1h['high'], df_1h['low'], df_1h['close'], window=14).iloc[-2]
    ema20 = df_1h['close'].ewm(span=20, adjust=False).mean().iloc[-2]

    trend_long  = last['close'] > ema20
    trend_short = last['close'] < ema20

    prev_high = last['high']
    prev_low  = last['low']

    base_sl_mult = 1.2
    base_tp_mult = 2.0

    plan = {
        'trend_long': trend_long,
        'trend_short': trend_short,
        'long_trigger': float(prev_high * 1.001),         # ~+0.1% above 1h high
        'short_trigger': float(prev_low  * 0.999),         # ~-0.1% below 1h low
        'pullback_long': float(prev_high - 0.25 * atr1h),  # if breakout missed
        'pullback_short': float(prev_low  + 0.25 * atr1h),
        'atr1h': float(atr1h),
        'tp_mult': base_tp_mult,
        'sl_mult': base_sl_mult,
        'ema20_1h': float(ema20),
        'prev_high': float(prev_high),
        'prev_low': float(prev_low),
    }

    plan = apply_sentiment_bias(plan, sentiment)
    return plan

def decide_orders(price_5m: float, plan: dict):
    """
    Decide entries/exits based on current 5m price and precomputed plan.
    Returns order intents; wire these to your ccxt order placement.
    """
    atr = plan['atr1h']
    sl_mult = plan['sl_mult']
    tp_mult = plan['tp_mult']
    orders = {'enter_long': None, 'enter_short': None, 'limit_long': None, 'limit_short': None,
              'sl': None, 'tp1': None, 'trail_at': None}

    if plan['trend_long']:
        if price_5m >= plan['long_trigger']:
            entry = price_5m
            orders['enter_long'] = entry
            orders['sl'] = entry - sl_mult * atr
            R = entry - orders['sl']
            orders['tp1'] = entry + tp_mult * R
            orders['trail_at'] = entry + 1.0 * R
        else:
            orders['limit_long'] = plan['pullback_long']

    if plan['trend_short']:
        if price_5m <= plan['short_trigger']:
            entry = price_5m
            orders['enter_short'] = entry
            orders['sl'] = entry + sl_mult * atr
            R = orders['sl'] - entry
            orders['tp1'] = entry - tp_mult * R
            orders['trail_at'] = entry - 1.0 * R
        else:
            orders['limit_short'] = plan['pullback_short']

    return orders

# ==========================
# Main
# ==========================
if __name__ == "__main__":
    class Args:
      use_sentiment = True  # or False to disable
    args = Args()

    print(f"{Colors.BOLD}{Colors.CYAN}BITTENSOR (TAO) TECHNICAL ANALYSIS{Colors.END}")
    analyzer = TechnicalAnalyzer('TAO/USDT')
    results = analyzer.analyze_all_timeframes()
    format_results(results)

    # Build an execution plan using 1h (trend) + 5m (timing)
    df_1h = analyzer.data_fetcher.fetch_data('1h', limit=300)
    df_5m = analyzer.data_fetcher.fetch_data('5m', limit=300)
    sentiment = sentiment_score_from_cli(args.use_sentiment)
    plan = compute_trade_plan_1h_5m(df_1h, df_5m, sentiment=sentiment)
    last_price_5m = float(df_5m['close'].iloc[-1])
    orders = decide_orders(last_price_5m, plan)

    print(f"\n{Colors.BOLD}{Colors.BLUE}EXECUTION PLAN (with {'sentiment' if args.use_sentiment else 'no sentiment'}){Colors.END}")
    pprint(plan)
    print(f"\nLast 5m price: {last_price_5m:.4f}")
    pprint(orders)

    print(f"\n{Colors.BOLD}{Colors.MAGENTA}Analysis Complete!{Colors.END}")
