import asyncio
import websockets
import json
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from collections import namedtuple, deque
import threading
import queue
import os
import requests

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
# Binance WebSocket Data Handler
# ==========================
class BinanceWebSocket:
    """
    Real-time Binance Futures WebSocket client for streaming market data.
    Handles 5m timeframe and maintains rolling data windows.
    """
    def __init__(self, symbol='TAOUSDT'):
        self.symbol = symbol.lower()
        # Only subscribe to 5m timeframe
        self.uri = f"wss://fstream.binance.com/stream?streams={self.symbol}@kline_5m"
        self.data_queue = queue.Queue()
        self.running = False
        self.ws = None
        self.data = {
            '5m': deque(maxlen=200)
        }
        self.current_price = 0.0  # Initialize with a default value
        self.last_update = time.time()
        self.connected = False
        self.message_count = 0  # For debugging
        
    async def connect(self):
        """Connect to Binance WebSocket and start streaming data"""
        self.running = True
        try:
            if not hasattr(self, 'silent_mode') or not self.silent_mode:
                print(f"{Colors.YELLOW}Connecting to: {self.uri}{Colors.END}")
            async with websockets.connect(self.uri) as websocket:
                self.ws = websocket
                self.connected = True
                if not hasattr(self, 'silent_mode') or not self.silent_mode:
                    print(f"{Colors.GREEN}Connected to Binance WebSocket for {self.symbol.upper()}{Colors.END}")
                
                # Listen for messages
                async for message in websocket:
                    if not self.running:
                        break
                    await self._handle_message(json.loads(message))
                    
        except Exception as e:
            if not hasattr(self, 'silent_mode') or not self.silent_mode:
                print(f"{Colors.RED}WebSocket error: {e}{Colors.END}")
            self.connected = False
            self.running = False
            
    async def _handle_message(self, data):
        """Process incoming WebSocket messages"""
        self.message_count += 1
        
        if 'stream' in data and 'data' in data:
            stream = data['stream']
            kline = data['data']['k']
            
            # Extract timeframe from stream name
            timeframe_parts = stream.split('@')
            if len(timeframe_parts) < 2:
                return
                
            timeframe = timeframe_parts[1].split('_')[1]
            
            # Parse kline data
            candle = {
                'timestamp': pd.to_datetime(kline['t'], unit='ms'),
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v']),
                'is_closed': kline['x']  # True if candle is closed
            }
            
            # Update current price
            self.current_price = candle['close']
            self.last_update = time.time()
            
            # Add to data queue for processing
            self.data_queue.put((timeframe, candle))
            
            # Update data store
            self._update_data(timeframe, candle)
            
        else:
            if not hasattr(self, 'silent_mode') or not self.silent_mode:
                print(f"{Colors.RED}Unexpected message format: {data}{Colors.END}")
            
    def _update_data(self, timeframe, candle):
        """Update the data store with new candle information"""
        if candle['is_closed']:
            # Add closed candle to history
            self.data[timeframe].append(candle)
        else:
            # Update current candle if it exists
            if self.data[timeframe] and not self.data[timeframe][-1]['is_closed']:
                self.data[timeframe][-1] = candle
            else:
                # Add new current candle
                self.data[timeframe].append(candle)
                
    def get_data(self, timeframe):
        """Get the latest data for a specific timeframe as a DataFrame"""
        if not self.data[timeframe]:
            return None
            
        # Convert deque to DataFrame
        df = pd.DataFrame(self.data[timeframe])
        df.set_index('timestamp', inplace=True)
        return df
        
    def stop(self):
        """Stop the WebSocket connection"""
        self.running = False
        self.connected = False
        if self.ws:
            asyncio.create_task(self.ws.close())

# ==========================
# Historical Data Fetcher
# ==========================
def fetch_historical_data(symbol='TAOUSDT', interval='5m', limit=200):
    """Fetch historical kline data from Binance REST API"""
    try:
        url = f"https://fapi.binance.com/fapi/v1/klines"
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': limit
        }
        response = requests.get(url, params=params)
        data = response.json()
        
        if not data:
            return []
            
        # Convert to our candle format
        candles = []
        for k in data:
            candle = {
                'timestamp': pd.to_datetime(k[0], unit='ms'),
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5]),
                'is_closed': True  # Historical candles are always closed
            }
            candles.append(candle)
            
        return candles
        
    except Exception as e:
        if not hasattr(BinanceWebSocket, 'silent_mode') or not BinanceWebSocket.silent_mode:
            print(f"{Colors.RED}Error fetching historical data: {e}{Colors.END}")
        return []

# ==========================
# Fast Real-time Technical Analyzer
# ==========================
class QuickScalpAnalyzer:
    """
    Real-time technical analyzer using 5m timeframe with robust confirmation:
      - Hull Moving Average (HMA) for trend detection
      - Optimized MACD (5,13,6) for signals
      - Stochastic Oscillator for momentum
      - Volume analysis for confirmation
      - Trend strength filtering
      - Multi-condition signal generation
      - Extended confirmation mechanism
    """
    def __init__(self, symbol='TAOUSDT', confirmation_candles=3, background_mode=False):
        self.symbol = symbol
        self.ws = BinanceWebSocket(symbol)
        self.last_snapshot = {
            '5m': {},
            'consensus': {}
        }
        self.signal_callback = None
        self.running = False
        self.update_interval = 0.5  # seconds between signal updates
        
        # Signal confirmation mechanism
        self.confirmation_candles = confirmation_candles
        self.current_signal = 'Neutral'
        self.confirmation_count = 0
        self.confirmed_signal = 'Neutral'
        self.signal_history = deque(maxlen=10)
        
        # Signal quality scoring
        self.signal_quality_threshold = 75
        
        # Additional filters
        self.min_trend_strength = 20.0
        self.min_volume_multiplier = 1.3
        
        # Background mode flag
        self.background_mode = background_mode
        # Set silent mode for WebSocket when in background
        self.ws.silent_mode = background_mode
        
    @staticmethod
    def _wma(series: pd.Series, window: int) -> pd.Series:
        """Weighted Moving Average"""
        weights = np.arange(1, window + 1)
        weights = weights / weights.sum()
        return series.rolling(window=window).apply(lambda x: np.dot(x, weights), raw=True)
    
    def _hma(self, series: pd.Series, window: int) -> pd.Series:
        """Hull Moving Average - significantly reduces lag"""
        wma_half = self._wma(series, window // 2)
        wma_full = self._wma(series, window)
        raw_hma = (2 * wma_half - wma_full).rolling(window=int(np.sqrt(window))).mean()
        return raw_hma
    
    @staticmethod
    def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.ewm(alpha=1/window, adjust=False).mean()
        roll_dn = down.ewm(alpha=1/window, adjust=False).mean()
        rs = roll_up / (roll_dn + 1e-12)
        return 100 - (100 / (1 + rs))
    
    def _macd_hist(self, close: pd.Series):
        """Optimized MACD with faster parameters"""
        ema_fast = close.ewm(span=5, adjust=False).mean()
        ema_slow = close.ewm(span=13, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=6, adjust=False).mean()
        return macd - signal
    
    def _stoch(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period=9, d_period=3) -> tuple:
        """Stochastic Oscillator - leading indicator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-12))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def _volume_sma(self, volume: pd.Series, window: int = 20) -> pd.Series:
        """Volume Simple Moving Average"""
        return volume.rolling(window=window).mean()
    
    def _volume_trend(self, volume: pd.Series, window: int = 5) -> str:
        """Determine volume trend"""
        vol_sma = self._volume_sma(volume, window)
        if volume.iloc[-1] > vol_sma.iloc[-1] * 1.5:
            return 'very_high'
        elif volume.iloc[-1] > vol_sma.iloc[-1] * 1.3:
            return 'high'
        elif volume.iloc[-1] > vol_sma.iloc[-1]:
            return 'above_avg'
        elif volume.iloc[-1] < vol_sma.iloc[-1] * 0.7:
            return 'low'
        else:
            return 'below_avg'
    
    def _trend_strength(self, close: pd.Series, window: int = 14) -> float:
        """Calculate trend strength using ADX-like method"""
        high = close.copy()
        low = close.copy()
        
        # Create synthetic high/low for ADX calculation
        for i in range(1, len(close)):
            high.iloc[i] = max(close.iloc[i], close.iloc[i-1])
            low.iloc[i] = min(close.iloc[i], close.iloc[i-1])
        
        up_move = high.diff()
        dn_move = -low.diff()
        
        plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
        
        tr = pd.concat([
            (high - low),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.ewm(alpha=1/window, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm, index=close.index).ewm(alpha=1/window, adjust=False).mean() / (atr + 1e-12)
        minus_di = 100 * pd.Series(minus_dm, index=close.index).ewm(alpha=1/window, adjust=False).mean() / (atr + 1e-12)
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
        adx = dx.ewm(alpha=1/window, adjust=False).mean()
        
        return adx.iloc[-1] if not np.isnan(adx.iloc[-1]) else 0
    
    def _price_momentum(self, close: pd.Series, window: int = 5) -> float:
        """Calculate price momentum"""
        if len(close) < window + 1:
            return 0.0
        
        # Calculate percentage change over window periods
        return ((close.iloc[-1] - close.iloc[-window-1]) / close.iloc[-window-1]) * 100
    
    def _candle_pattern(self, df: pd.DataFrame) -> str:
        """Identify reversal candle patterns"""
        if len(df) < 2:
            return 'none'
            
        body = abs(df['close'] - df['open'])
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        
        # Hammer pattern (bullish reversal)
        hammer = (lower_shadow.iloc[-1] > 2 * body.iloc[-1]) & (upper_shadow.iloc[-1] < body.iloc[-1] * 0.1)
        
        # Shooting star (bearish reversal)
        shooting_star = (upper_shadow.iloc[-1] > 2 * body.iloc[-1]) & (lower_shadow.iloc[-1] < body.iloc[-1] * 0.1)
        
        # Engulfing patterns
        prev_body = abs(df['close'].iloc[-2] - df['open'].iloc[-2])
        bullish_engulfing = (df['open'].iloc[-1] < df['close'].iloc[-2]) & (df['close'].iloc[-1] > df['open'].iloc[-2]) & (body.iloc[-1] > prev_body)
        bearish_engulfing = (df['open'].iloc[-1] > df['close'].iloc[-2]) & (df['close'].iloc[-1] < df['open'].iloc[-2]) & (body.iloc[-1] > prev_body)
        
        # Doji pattern (indecision)
        doji = body.iloc[-1] < (df['high'].iloc[-1] - df['low'].iloc[-1]) * 0.1
        
        if hammer:
            return 'hammer'
        elif shooting_star:
            return 'shooting_star'
        elif bullish_engulfing:
            return 'bullish_engulfing'
        elif bearish_engulfing:
            return 'bearish_engulfing'
        elif doji:
            return 'doji'
        return 'none'
    
    def _atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range for volatility measurement"""
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1/window, adjust=False).mean()
    
    def _early_signal(self, df: pd.DataFrame) -> str:
        """Detect early crossover signals before completion"""
        if len(df) < 3:
            return 'none'
        
        close = df['close']
        hma9 = self._hma(close, 9)
        hma21 = self._hma(close, 21)
        
        # Detect crossover before it completes
        prev_diff = hma9.iloc[-2] - hma21.iloc[-2]
        curr_diff = hma9.iloc[-1] - hma21.iloc[-1]
        
        if prev_diff < 0 and curr_diff > 0:
            return 'early_buy'
        elif prev_diff > 0 and curr_diff < 0:
            return 'early_sell'
        return 'none'
    
    def _calculate_signal_quality(self, signal_type, conditions_met, trend_strength, volume_trend, price_momentum) -> int:
        """Calculate signal quality score (0-100)"""
        # Base score from conditions met
        base_score = len(conditions_met) * 15  # Each condition is worth 15 points
        
        # Trend strength bonus (0-25 points)
        if trend_strength > 30:
            trend_bonus = 25
        elif trend_strength > 20:
            trend_bonus = 20
        elif trend_strength > 10:
            trend_bonus = 10
        else:
            trend_bonus = 0
        
        # Volume trend bonus (0-25 points)
        if volume_trend == 'very_high':
            volume_bonus = 25
        elif volume_trend == 'high':
            volume_bonus = 20
        elif volume_trend == 'above_avg':
            volume_bonus = 10
        else:
            volume_bonus = 0
        
        # Price momentum bonus (0-25 points)
        if signal_type == 'Buy' and price_momentum > 1.0:
            momentum_bonus = min(25, price_momentum * 5)
        elif signal_type == 'Sell' and price_momentum < -1.0:
            momentum_bonus = min(25, abs(price_momentum) * 5)
        else:
            momentum_bonus = 0
        
        # Pattern bonus (0-25 points)
        pattern_bonus = 0
        if signal_type == 'Buy' and conditions_met.get('pattern', '') in ['hammer', 'bullish_engulfing']:
            pattern_bonus = 25
        elif signal_type == 'Sell' and conditions_met.get('pattern', '') in ['shooting_star', 'bearish_engulfing']:
            pattern_bonus = 25
        
        # Calculate total quality score
        quality_score = min(100, base_score + trend_bonus + volume_bonus + momentum_bonus + pattern_bonus)
        
        return quality_score
    
    def _one_tf_signal(self, df: pd.DataFrame) -> QuickSig:
        if df is None or len(df) < 30:
            return QuickSig('Neutral', {'reason': 'insufficient_bars'})
        
        c = df['close']
        h = df['high']
        l = df['low']
        v = df['volume']
        
        # Calculate indicators
        hma9 = self._hma(c, 9)
        hma21 = self._hma(c, 21)
        macd_hist = self._macd_hist(c)
        k_percent, d_percent = self._stoch(h, l, c)
        atr = self._atr(h, l, c)
        pattern = self._candle_pattern(df)
        early_sig = self._early_signal(df)
        rsi = self._rsi(c, 14)
        
        # Calculate additional metrics
        trend_strength = self._trend_strength(c)
        volume_trend = self._volume_trend(v)
        price_momentum = self._price_momentum(c)
        volatility = atr.iloc[-1] / c.iloc[-1] * 100
        
        # Primary signal conditions
        hma_bull = hma9.iloc[-1] > hma21.iloc[-1]
        hma_bear = hma9.iloc[-1] < hma21.iloc[-1]
        macd_bull = macd_hist.iloc[-1] > 0
        macd_bear = macd_hist.iloc[-1] < 0
        stoch_bull = (k_percent.iloc[-1] > 20) and (d_percent.iloc[-1] > 20) and (k_percent.iloc[-1] > d_percent.iloc[-1])
        stoch_bear = (k_percent.iloc[-1] < 80) and (d_percent.iloc[-1] < 80) and (k_percent.iloc[-1] < d_percent.iloc[-1])
        rsi_bull = rsi.iloc[-1] > 50
        rsi_bear = rsi.iloc[-1] < 50
        
        # Volume confirmation
        volume_confirms_bull = volume_trend in ['high', 'very_high']
        volume_confirms_bear = volume_trend in ['high', 'very_high']
        
        # Trend strength filter
        strong_trend = trend_strength > self.min_trend_strength
        
        # Price momentum filter
        positive_momentum = price_momentum > 0.5
        negative_momentum = price_momentum < -0.5
        
        # Early signal with pattern
        early_buy_with_pattern = (early_sig == 'early_buy') and (pattern in ['hammer', 'bullish_engulfing'])
        early_sell_with_pattern = (early_sig == 'early_sell') and (pattern in ['shooting_star', 'bearish_engulfing'])
        
        # Track conditions met for quality scoring
        buy_conditions = {}
        sell_conditions = {}
        
        # Buy signal conditions
        buy_conditions['hma'] = hma_bull
        buy_conditions['macd'] = macd_bull
        buy_conditions['stoch'] = stoch_bull
        buy_conditions['rsi'] = rsi_bull
        buy_conditions['volume'] = volume_confirms_bull
        buy_conditions['trend'] = strong_trend
        buy_conditions['momentum'] = positive_momentum
        buy_conditions['pattern'] = early_buy_with_pattern
        
        # Count buy conditions met
        buy_conditions_met = {k: v for k, v in buy_conditions.items() if v}
        buy_score = len(buy_conditions_met)
        
        # Sell signal conditions
        sell_conditions['hma'] = hma_bear
        sell_conditions['macd'] = macd_bear
        sell_conditions['stoch'] = stoch_bear
        sell_conditions['rsi'] = rsi_bear
        sell_conditions['volume'] = volume_confirms_bear
        sell_conditions['trend'] = strong_trend
        sell_conditions['momentum'] = negative_momentum
        sell_conditions['pattern'] = early_sell_with_pattern
        
        # Count sell conditions met
        sell_conditions_met = {k: v for k, v in sell_conditions.items() if v}
        sell_score = len(sell_conditions_met)
        
        # Determine signal based on conditions met
        min_conditions = 5
        
        if buy_score >= min_conditions:
            signal = 'Buy'
            conditions_met = buy_conditions_met
        elif sell_score >= min_conditions:
            signal = 'Sell'
            conditions_met = sell_conditions_met
        else:
            signal = 'Neutral'
            conditions_met = {}
        
        # Calculate signal quality
        quality_score = 0
        if signal != 'Neutral':
            quality_score = self._calculate_signal_quality(signal, conditions_met, trend_strength, volume_trend, price_momentum)
        
        # Only generate signal if quality meets threshold
        if signal != 'Neutral' and quality_score < self.signal_quality_threshold:
            signal = 'Neutral'
        
        # Prepare debug info
        dbg = {
            'close': float(c.iloc[-1]),
            'hma9': float(hma9.iloc[-1]),
            'hma21': float(hma21.iloc[-1]),
            'macd_hist': float(macd_hist.iloc[-1]),
            'stoch_k': float(k_percent.iloc[-1]),
            'stoch_d': float(d_percent.iloc[-1]),
            'rsi': float(rsi.iloc[-1]),
            'atr': float(atr.iloc[-1]),
            'volatility': float(volatility),
            'pattern': pattern,
            'early_signal': early_sig,
            'trend_strength': float(trend_strength),
            'volume_trend': volume_trend,
            'price_momentum': float(price_momentum),
            'quality_score': quality_score,
            'conditions_met': conditions_met
        }
        
        return QuickSig(signal, dbg)
    
    def _update_signal_confirmation(self, new_signal):
        """Update signal confirmation mechanism"""
        # Add new signal to history
        self.signal_history.append(new_signal)
        
        # If signal is the same as current, increment confirmation count
        if new_signal == self.current_signal:
            self.confirmation_count += 1
        else:
            # Signal changed, reset confirmation
            self.current_signal = new_signal
            self.confirmation_count = 1
        
        # Check if we have enough confirmation
        if self.confirmation_count >= self.confirmation_candles:
            # Only update confirmed signal if it's different
            if new_signal != self.confirmed_signal:
                self.confirmed_signal = new_signal
                if not self.background_mode:
                    print(f"{Colors.CYAN}Signal confirmed: {new_signal} after {self.confirmation_count} candles ({self.confirmation_candles * 5} minutes){Colors.END}")
        elif self.confirmation_count == 1:
            if not self.background_mode:
                print(f"{Colors.YELLOW}New signal detected: {new_signal} (confirmation: 1/{self.confirmation_candles}){Colors.END}")
        else:
            if not self.background_mode:
                print(f"{Colors.BLUE}Signal continuing: {new_signal} (confirmation: {self.confirmation_count}/{self.confirmation_candles}){Colors.END}")
    
    def quick_decision(self) -> QuickSig:
        # Get data for 5m timeframe
        df5 = self.ws.get_data('5m')
        
        # If we don't have enough data, return neutral
        if df5 is None or len(df5) < 30:
            return QuickSig('Neutral', {'reason': 'insufficient_data'})
        
        # Get signal for 5m timeframe
        sig5 = self._one_tf_signal(df5)
        
        # Update signal confirmation mechanism
        self._update_signal_confirmation(sig5.signal)
        
        # Create consensus object
        consensus = {
            'signal': self.confirmed_signal,
            'current_signal': self.current_signal,
            'confirmation_count': self.confirmation_count,
            'confirmation_needed': self.confirmation_candles,
            '5m_signal': sig5.signal,
            'current_price': self.ws.current_price
        }
        
        # Update last snapshot
        self.last_snapshot = {
            '5m': sig5.debug,
            'consensus': consensus
        }
        
        return QuickSig(self.confirmed_signal, consensus)
    
    def start(self):
        """Start the real-time analysis"""
        self.running = True
        
        # Fetch historical data to bootstrap
        if not self.background_mode:
            print(f"{Colors.YELLOW}Fetching historical data...{Colors.END}")
        
        hist_5m = fetch_historical_data(self.symbol, '5m', 200)
        
        # Populate the data with historical data
        for candle in hist_5m:
            self.ws.data['5m'].append(candle)
            
        # Set current price to the latest close price
        if hist_5m:
            self.ws.current_price = hist_5m[-1]['close']
            if not self.background_mode:
                print(f"{Colors.GREEN}Set initial price to {self.ws.current_price}{Colors.END}")
        
        # Start WebSocket in a separate thread
        ws_thread = threading.Thread(target=self._run_websocket)
        ws_thread.daemon = True
        ws_thread.start()
        
        # Wait for WebSocket to connect
        if not self.background_mode:
            print(f"{Colors.YELLOW}Connecting to Binance WebSocket...{Colors.END}")
        
        timeout = 10  # seconds
        start_time = time.time()
        
        while not self.ws.connected and time.time() - start_time < timeout:
            time.sleep(0.5)
        
        if not self.ws.connected and not self.background_mode:
            print(f"{Colors.RED}Failed to connect to WebSocket. Using historical data only.{Colors.END}")
        
        # Only start signal processing loop if not in background mode
        if not self.background_mode:
            self._process_signals()
        
    def _run_websocket(self):
        """Run the WebSocket connection in a separate thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.ws.connect())
        
    def _process_signals(self):
        """Process signals at regular intervals"""
        last_signal_time = 0
        
        while self.running:
            current_time = time.time()
            
            # Update signals at specified interval
            if current_time - last_signal_time >= self.update_interval:
                sig = self.quick_decision()
                
                # Only display if not in background mode
                if not self.background_mode:
                    self._display_signal(sig)
                
                # Call callback if registered
                if self.signal_callback:
                    self.signal_callback(sig)
                
                last_signal_time = current_time
            
            time.sleep(0.1)  # Small sleep to prevent high CPU usage
            
    def _display_signal(self, sig):
        """Display the current signal and market status"""
        if self.background_mode:
            return
            
        os.system('clear' if os.name == 'posix' else 'cls')  # Clear screen
        
        # Get current price safely
        current_price = self.ws.current_price if self.ws.current_price is not None else 0.0
        
        # Get consensus info
        consensus = self.last_snapshot.get('consensus', {})
        current_signal = consensus.get('current_signal', 'Neutral')
        confirmation_count = consensus.get('confirmation_count', 0)
        confirmation_needed = consensus.get('confirmation_needed', 3)
        signal_5m = consensus.get('5m_signal', 'Neutral')
        
        # Get debug info
        debug5 = self.last_snapshot.get('5m', {})
        quality_score = debug5.get('quality_score', 0)
        conditions_met = debug5.get('conditions_met', {})
        trend_strength = debug5.get('trend_strength', 0)
        volume_trend = debug5.get('volume_trend', 'unknown')
        price_momentum = debug5.get('price_momentum', 0)
        
        # Display current price prominently
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.WHITE}5-MINUTE TECHNICAL ANALYZER - {self.symbol.upper()}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}Current Price: {Colors.GREEN if sig.signal == 'Buy' else Colors.RED if sig.signal == 'Sell' else Colors.YELLOW}{current_price:.2f}{Colors.END}")
        
        # Display signal with prominent formatting
        signal_color = Colors.GREEN if sig.signal == 'Buy' else Colors.RED if sig.signal == 'Sell' else Colors.YELLOW
        current_color = Colors.GREEN if current_signal == 'Buy' else Colors.RED if current_signal == 'Sell' else Colors.YELLOW
        
        # Quality score color
        if quality_score >= 80:
            quality_color = Colors.GREEN
        elif quality_score >= 60:
            quality_color = Colors.YELLOW
        else:
            quality_color = Colors.RED
        
        # ===== FINAL SIGNAL BOX =====
        print(f"\n{Colors.BOLD}{signal_color}╔{'═' * 58}╗{Colors.END}")
        print(f"{Colors.BOLD}{signal_color}║{' ' * 58}║{Colors.END}")
        print(f"{Colors.BOLD}{signal_color}║        CONFIRMED 5m SIGNAL        ║{Colors.END}")
        print(f"{Colors.BOLD}{signal_color}║{' ' * 58}║{Colors.END}")
        print(f"{Colors.BOLD}{signal_color}║            {sig.signal.upper():^34}            ║{Colors.END}")
        print(f"{Colors.BOLD}{signal_color}║{' ' * 58}║{Colors.END}")
        print(f"{Colors.BOLD}{signal_color}║     Quality: {quality_color}{quality_score:>3.0f}/100{signal_color}     ║{Colors.END}")
        print(f"{Colors.BOLD}{signal_color}╚{'═' * 58}╝{Colors.END}")
        
        # ===== SIGNAL CONFIRMATION STATUS =====
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.WHITE}SIGNAL CONFIRMATION STATUS{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
        
        # Confirmation progress bar
        progress = int((confirmation_count / confirmation_needed) * 20)
        progress_bar = '[' + '█' * progress + ' ' * (20 - progress) + ']'
        
        print(f"{Colors.BOLD}Current Signal: {current_color}{current_signal}{Colors.END}")
        print(f"{Colors.BOLD}Confirmation: {Colors.CYAN}{progress_bar} {confirmation_count}/{confirmation_needed}{Colors.END}")
        print(f"{Colors.BOLD}Confirmation Time: {Colors.CYAN}{confirmation_count * 5}/{confirmation_needed * 5} minutes{Colors.END}")
        
        if confirmation_count >= confirmation_needed:
            print(f"{Colors.BOLD}Status: {Colors.GREEN}CONFIRMED{Colors.END}")
        else:
            print(f"{Colors.BOLD}Status: {Colors.YELLOW}PENDING CONFIRMATION{Colors.END}")
        
        # ===== TECHNICAL INDICATORS =====
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.WHITE}TECHNICAL INDICATORS{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
        
        # 5m indicators
        print(f"{Colors.BOLD}5m Timeframe ({len(self.ws.data['5m'])} candles):{Colors.END}")
        print(f"  HMA9: {debug5.get('hma9', 0):.2f} | HMA21: {debug5.get('hma21', 0):.2f}")
        print(f"  MACD Hist: {debug5.get('macd_hist', 0):.4f}")
        print(f"  Stoch K: {debug5.get('stoch_k', 0):.1f} | Stoch D: {debug5.get('stoch_d', 0):.1f}")
        print(f"  RSI: {debug5.get('rsi', 0):.1f}")
        print(f"  Pattern: {debug5.get('pattern', 'none')}")
        
        # ===== MARKET CONDITIONS =====
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.WHITE}MARKET CONDITIONS{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}Volatility: {Colors.YELLOW}{debug5.get('volatility', 0):.2f}%{Colors.END}")
        print(f"{Colors.BOLD}Trend Strength: {Colors.YELLOW}{trend_strength:.1f} (Min: {self.min_trend_strength}){Colors.END}")
        print(f"{Colors.BOLD}Volume Trend: {Colors.YELLOW}{volume_trend}{Colors.END}")
        print(f"{Colors.BOLD}Price Momentum: {Colors.YELLOW}{price_momentum:.2f}%{Colors.END}")
        print(f"{Colors.BOLD}ATR: {Colors.YELLOW}{debug5.get('atr', 0):.2f}{Colors.END}")
        
        # ===== CONDITIONS MET =====
        if conditions_met:
            print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
            print(f"{Colors.BOLD}{Colors.WHITE}CONDITIONS MET{Colors.END}")
            print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
            for condition, met in conditions_met.items():
                status_color = Colors.GREEN if met else Colors.RED
                status = "✓" if met else "✗"
                print(f"{Colors.BOLD}{status} {condition}: {status_color}{met}{Colors.END}")
        
        # ===== SIGNAL EXPLANATION =====
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.WHITE}SIGNAL EXPLANATION{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
        
        if sig.signal == 'Buy':
            print(f"{Colors.GREEN}BUY SIGNAL CONFIRMED:{Colors.END}")
            print("- Multiple bullish conditions confirmed across indicators")
            print("- High volume supports the upward movement")
            print("- Strong trend strength and positive price momentum")
            print(f"- Signal quality score: {quality_score}/100")
            print("- Signal confirmed over {confirmation_needed * 5} minutes")
            print("- Consider entering long position with confidence")
                
        elif sig.signal == 'Sell':
            print(f"{Colors.RED}SELL SIGNAL CONFIRMED:{Colors.END}")
            print("- Multiple bearish conditions confirmed across indicators")
            print("- High volume supports the downward movement")
            print("- Strong trend strength and negative price momentum")
            print(f"- Signal quality score: {quality_score}/100")
            print("- Signal confirmed over {confirmation_needed * 5} minutes")
            print("- Consider entering short position with confidence")
                
        else:  # Neutral
            if current_signal == 'Neutral':
                print(f"{Colors.YELLOW}NEUTRAL SIGNAL CONFIRMED:{Colors.END}")
                print("- No clear trend direction detected")
                print("- Market may be in consolidation")
                print("- Insufficient conditions met for a reliable signal")
            else:
                print(f"{Colors.YELLOW}NEUTRAL SIGNAL (PENDING):{Colors.END}")
                print(f"- {current_signal.upper()} signal detected but not yet confirmed")
                print(f"- Awaiting {confirmation_needed - confirmation_count} more candle(s) for confirmation")
                print(f"- Current signal may change before confirmation")
                print(f"- Total confirmation time needed: {confirmation_needed * 5} minutes")
    
    def stop(self):
        """Stop the analyzer"""
        self.running = False
        self.ws.stop()

# Standalone execution
if __name__ == "__main__":
    print(f"{Colors.BOLD}{Colors.CYAN}Starting 5-Minute Technical Analyzer with Extended Confirmation...{Colors.END}")
    
    # Create and start analyzer with 3-candle confirmation (15 minutes)
    analyzer = QuickScalpAnalyzer('TAOUSDT', confirmation_candles=3)
    
    try:
        analyzer.start()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Stopping analyzer...{Colors.END}")
        analyzer.stop()