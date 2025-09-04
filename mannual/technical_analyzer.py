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
    Handles 1h timeframe and maintains rolling data windows.
    """
    def __init__(self, symbol='TAOUSDT'):
        self.symbol = symbol.lower()
        # Only subscribe to 1h timeframe
        self.uri = f"wss://fstream.binance.com/stream?streams={self.symbol}@kline_1h"
        self.data_queue = queue.Queue()
        self.running = False
        self.ws = None
        self.data = {
            '1h': deque(maxlen=100)  # Smaller deque for 1h as it has fewer candles
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
            
            # Only process 1h timeframe
            if timeframe != '1h':
                return
            
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
        if timeframe not in self.data or not self.data[timeframe]:
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
            try:
                # Check if we're in an asyncio event loop
                try:
                    loop = asyncio.get_running_loop()
                    # We're in a running loop, create a task
                    loop.create_task(self.ws.close())
                except RuntimeError:
                    # No running event loop, create a new one
                    asyncio.run(self.ws.close())
            except Exception as e:
                print(f"Error closing WebSocket: {e}")
# ==========================
# Historical Data Fetcher
# ==========================
def fetch_historical_data(symbol='TAOUSDT', interval='1h', limit=100):
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
    Real-time technical analyzer using 1h timeframe with robust confirmation:
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
            '1h': {},
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
        
    def get_status(self):
        """Return the current status of the analyzer"""
        status = {
            'running': self.running,
            'ws_connected': self.ws.connected if hasattr(self, 'ws') else False,
            'current_price': self.ws.current_price if hasattr(self, 'ws') else None,
            'last_update': self.ws.last_update if hasattr(self, 'ws') else None,
            'confirmed_signal': self.confirmed_signal,
            'current_signal': self.current_signal,
            'confirmation_count': self.confirmation_count,
            'data_points': len(self.ws.data.get('1h', [])) if hasattr(self, 'ws') else 0
        }
        return status
        
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
                    print(f"{Colors.CYAN}Signal confirmed: {new_signal} after {self.confirmation_count} candles ({self.confirmation_candles} hours){Colors.END}")
        elif self.confirmation_count == 1:
            if not self.background_mode:
                print(f"{Colors.YELLOW}New signal detected: {new_signal} (confirmation: 1/{self.confirmation_candles}){Colors.END}")
        else:
            if not self.background_mode:
                print(f"{Colors.BLUE}Signal continuing: {new_signal} (confirmation: {self.confirmation_count}/{self.confirmation_candles}){Colors.END}")
    
    def quick_decision(self) -> QuickSig:
        # Get data for 1h timeframe
        df1h = self.ws.get_data('1h')
        
        # If we don't have enough data, return neutral
        if df1h is None or len(df1h) < 30:
            return QuickSig('Neutral', {'reason': 'insufficient_data'})
        
        # Get signal for 1h timeframe
        sig1h = self._one_tf_signal(df1h)
        
        # Update signal confirmation mechanism
        self._update_signal_confirmation(sig1h.signal)
        
        # Create consensus object
        consensus = {
            'signal': self.confirmed_signal,
            'current_signal': self.current_signal,
            'confirmation_count': self.confirmation_count,
            'confirmation_needed': self.confirmation_candles,
            '1h_signal': sig1h.signal,
            'current_price': self.ws.current_price
        }
        
        # Update last snapshot
        self.last_snapshot = {
            '1h': sig1h.debug,
            'consensus': consensus
        }
        
        return QuickSig(self.confirmed_signal, consensus)
    
    def start(self):
        """Start the real-time analysis"""
        self.running = True
        
        # Fetch historical data to bootstrap
        if not self.background_mode:
            print(f"{Colors.YELLOW}Fetching historical data...{Colors.END}")
        
        # Fetch 1h historical data
        hist_1h = fetch_historical_data(self.symbol, '1h', 100)
        
        # Populate the data with historical data
        for candle in hist_1h:
            self.ws.data['1h'].append(candle)
            
        # Set current price to the latest close price
        if hist_1h:
            self.ws.current_price = hist_1h[-1]['close']
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
        signal_1h = consensus.get('1h_signal', 'Neutral')
        
        # Get debug info
        debug1h = self.last_snapshot.get('1h', {})
        quality_score = debug1h.get('quality_score', 0)
        conditions_met = debug1h.get('conditions_met', {})
        trend_strength = debug1h.get('trend_strength', 0)
        volume_trend = debug1h.get('volume_trend', 'unknown')
        price_momentum = debug1h.get('price_momentum', 0)
        
        # Display current price prominently
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.WHITE}1-HOUR TECHNICAL ANALYZER - {self.symbol.upper()}{Colors.END}")
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
        print(f"{Colors.BOLD}{signal_color}║        CONFIRMED 1h SIGNAL        ║{Colors.END}")
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
        print(f"{Colors.BOLD}Confirmation Time: {Colors.CYAN}{confirmation_count}/{confirmation_needed} hours{Colors.END}")
        
        if confirmation_count >= confirmation_needed:
            print(f"{Colors.BOLD}Status: {Colors.GREEN}CONFIRMED{Colors.END}")
        else:
            print(f"{Colors.BOLD}Status: {Colors.YELLOW}PENDING CONFIRMATION{Colors.END}")
        
        # ===== TECHNICAL INDICATORS =====
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.WHITE}TECHNICAL INDICATORS{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
        
        # 1h indicators
        print(f"{Colors.BOLD}1h Timeframe ({len(self.ws.data['1h'])} candles):{Colors.END}")
        print(f"  HMA9: {debug1h.get('hma9', 0):.2f} | HMA21: {debug1h.get('hma21', 0):.2f}")
        print(f"  MACD Hist: {debug1h.get('macd_hist', 0):.4f}")
        print(f"  Stoch K: {debug1h.get('stoch_k', 0):.1f} | Stoch D: {debug1h.get('stoch_d', 0):.1f}")
        print(f"  RSI: {debug1h.get('rsi', 0):.1f}")
        print(f"  Pattern: {debug1h.get('pattern', 'none')}")
        
        # ===== MARKET CONDITIONS =====
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.WHITE}MARKET CONDITIONS{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}Volatility: {Colors.YELLOW}{debug1h.get('volatility', 0):.2f}%{Colors.END}")
        print(f"{Colors.BOLD}Trend Strength: {Colors.YELLOW}{trend_strength:.1f} (Min: {self.min_trend_strength}){Colors.END}")
        print(f"{Colors.BOLD}Volume Trend: {Colors.YELLOW}{volume_trend}{Colors.END}")
        print(f"{Colors.BOLD}Price Momentum: {Colors.YELLOW}{price_momentum:.2f}%{Colors.END}")
        print(f"{Colors.BOLD}ATR: {Colors.YELLOW}{debug1h.get('atr', 0):.2f}{Colors.END}")
        
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
            print("- Signal confirmed over {confirmation_needed} hours")
            print("- Consider entering long position with confidence")
                
        elif sig.signal == 'Sell':
            print(f"{Colors.RED}SELL SIGNAL CONFIRMED:{Colors.END}")
            print("- Multiple bearish conditions confirmed across indicators")
            print("- High volume supports the downward movement")
            print("- Strong trend strength and negative price momentum")
            print(f"- Signal quality score: {quality_score}/100")
            print("- Signal confirmed over {confirmation_needed} hours")
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
                print(f"- Total confirmation time needed: {confirmation_needed} hours")
    
    def stop(self):
        """Stop the analyzer"""
        self.running = False
        self.ws.stop()
# Standalone execution
if __name__ == "__main__":
    print(f"{Colors.BOLD}{Colors.CYAN}Starting 1-Hour Technical Analyzer with Extended Confirmation...{Colors.END}")
    
    # Create and start analyzer with 3-candle confirmation (3 hours)
    analyzer = QuickScalpAnalyzer('TAOUSDT', confirmation_candles=3)
    
    try:
        analyzer.start()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Stopping analyzer...{Colors.END}")
        analyzer.stop()


# # technical_analyzer.py
# import argparse
# import ccxt
# import pandas as pd
# import numpy as np
# from datetime import datetime
# from pprint import pprint
# import warnings
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
#     mapping = {'5m': '5T', '15m': '15T', '1h': '1H', '1d': '1D', '1w': '1W', '1M': '1M'}
#     return mapping.get(timeframe, timeframe)

# # ==========================
# # Data fetcher
# # ==========================
# class DataFetcher:
#     def __init__(self, symbol='TAO/USDT'):
#         self.symbol = symbol
#         self.exchanges = [
#             ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'spot'}}),
#             ccxt.kraken({'enableRateLimit': True}),
#             ccxt.kucoin({'enableRateLimit': True}),
#             ccxt.gateio({'enableRateLimit': True})
#         ]
#         self.exchange = None
#         self._find_exchange_with_symbol()

#     def _find_exchange_with_symbol(self):
#         for exchange in self.exchanges:
#             try:
#                 exchange.load_markets()
#                 if self.symbol in exchange.symbols:
#                     self.exchange = exchange
#                     print(f"Using exchange: {exchange.id} for {self.symbol}")
#                     return
#             except Exception:
#                 continue
#         print("TAO/USDT not found, searching for alternatives...")
#         for exchange in self.exchanges:
#             try:
#                 exchange.load_markets()
#                 tao_symbols = [s for s in exchange.symbols if 'TAO' in s]
#                 if tao_symbols:
#                     self.symbol = tao_symbols[0]
#                     self.exchange = exchange
#                     print(f"Found alternative: {self.symbol} on {exchange.id}")
#                     return
#             except Exception:
#                 continue
#         self.exchange = ccxt.binance({'enableRateLimit': True})
#         self.symbol = 'BTC/USDT'
#         print(f"{Colors.YELLOW}WARNING: Using BTC/USDT as fallback — TAO not found!{Colors.END}")

#     def fetch_data(self, timeframe, limit=500):
#         try:
#             if not self.exchange:
#                 raise Exception("No exchange available")
#             ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=limit)
#             if not ohlcv:
#                 raise Exception("No data returned")
#             df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
#             df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
#             df.set_index('timestamp', inplace=True)
#             return df
#         except Exception:
#             return self._get_sample_data(timeframe, limit)

#     def _get_sample_data(self, timeframe, limit):
#         np.random.seed(42)
#         base_price = 100 + np.random.random() * 50
#         dates = pd.date_range(end=datetime.now(), periods=limit, freq=timeframe_to_freq(timeframe))
#         prices = [base_price]
#         for _ in range(1, limit):
#             change = np.random.normal(0, 0.02)
#             new_price = prices[-1] * (1 + change)
#             prices.append(max(new_price, 0.01))
#         df = pd.DataFrame(index=dates)
#         df['close'] = prices
#         df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
#         df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.01, limit)))
#         df['low']  = df[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.01, limit)))
#         df['volume'] = np.random.uniform(1000, 10000, limit)
#         return df

# # ==========================
# # Pivot points
# # ==========================
# class PivotPoints:
#     @staticmethod
#     def standard(high, low, close):
#         pp = (high + low + close) / 3
#         r1 = 2 * pp - low
#         s1 = 2 * pp - high
#         r2 = pp + (high - low)
#         s2 = pp - (high - low)
#         r3 = high + 2 * (pp - low)
#         s3 = low - 2 * (high - pp)
#         return {'S3': s3, 'S2': s2, 'S1': s1, 'PP': pp, 'R1': r1, 'R2': r2, 'R3': r3}

#     @staticmethod
#     def fibonacci(high, low, close):
#         pp = (high + low + close) / 3
#         rng = high - low
#         return {
#             'S3': pp - (rng * 1.000), 'S2': pp - (rng * 0.618), 'S1': pp - (rng * 0.382),
#             'PP': pp,
#             'R1': pp + (rng * 0.382), 'R2': pp + (rng * 0.618), 'R3': pp + (rng * 1.000)
#         }

#     @staticmethod
#     def camarilla(high, low, close):
#         h_l = high - low
#         pp = (high + low + close) / 3
#         r1 = close + (h_l * 1.1 / 12)
#         r2 = close + (h_l * 1.1 / 6)
#         r3 = close + (h_l * 1.1 / 4)
#         s1 = close - (h_l * 1.1 / 12)
#         s2 = close - (h_l * 1.1 / 6)
#         s3 = close - (h_l * 1.1 / 4)
#         return {'S3': s3, 'S2': s2, 'S1': s1, 'PP': pp, 'R1': r1, 'R2': r2, 'R3': r3}

#     @staticmethod
#     def woodies(high, low, close, open_price):
#         pp = (high + low + 2 * close) / 4
#         r1 = (2 * pp) - low
#         s1 = (2 * pp) - high
#         r2 = pp + (high - low)
#         s2 = pp - (high - low)
#         r3 = high + 2 * (pp - low)
#         s3 = low - 2 * (high - pp)
#         return {'S3': s3, 'S2': s2, 'S1': s1, 'PP': pp, 'R1': r1, 'R2': r2, 'R3': r3}

#     @staticmethod
#     def demarks(high, low, close, open_price):
#         if close < open_price:
#             x = high + (2 * low) + close
#         elif close > open_price:
#             x = (2 * high) + low + close
#         else:
#             x = high + low + (2 * close)
#         pp = x / 4
#         r1 = x / 2 - low
#         s1 = x / 2 - high
#         return {'S1': s1, 'PP': pp, 'R1': r1}

# # ==========================
# # MAs
# # ==========================
# class MovingAverages:
#     @staticmethod
#     def sma(data, window):
#         return data.rolling(window=window).mean()

#     @staticmethod
#     def ema(data, window):
#         return data.ewm(span=window, adjust=False).mean()

# # ==========================
# # Oscillators (with Wilder + wrappers)
# # ==========================
# class Oscillators:
#     @staticmethod
#     def rsi_wilder(data, window=14):
#         delta = data.diff()
#         up = delta.clip(lower=0)
#         down = -delta.clip(upper=0)
#         roll_up = up.ewm(alpha=1/window, adjust=False).mean()
#         roll_down = down.ewm(alpha=1/window, adjust=False).mean()
#         rs = roll_up / (roll_down + 1e-12)
#         return 100 - (100 / (1 + rs))

#     @staticmethod
#     def rsi(data, window=14):  # backward-compatible
#         return Oscillators.rsi_wilder(data, window)

#     @staticmethod
#     def stochastic(high, low, close, k_window=14, d_window=3):
#         low_min = low.rolling(window=k_window).min()
#         high_max = high.rolling(window=k_window).max()
#         k = 100 * ((close - low_min) / (high_max - low_min + 1e-12))
#         d = k.rolling(window=d_window).mean()
#         return k, d

#     @staticmethod
#     def stoch_rsi(data, window=14, eps=1e-12):
#         rsi = Oscillators.rsi_wilder(data, window)
#         rsi_min = rsi.rolling(window).min()
#         rsi_max = rsi.rolling(window).max()
#         return ((rsi - rsi_min) / (rsi_max - rsi_min + eps)) * 100

#     @staticmethod
#     def macd(data, fast=12, slow=26, signal=9):
#         ema_fast = data.ewm(span=fast, adjust=False).mean()
#         ema_slow = data.ewm(span=slow, adjust=False).mean()
#         macd_line = ema_fast - ema_slow
#         signal_line = macd_line.ewm(span=signal, adjust=False).mean()
#         return macd_line, signal_line, macd_line - signal_line

#     @staticmethod
#     def adx_wilder(high, low, close, window=14):
#         up_move = high.diff()
#         down_move = -low.diff()
#         plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
#         minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
#         prev_close = close.shift(1)
#         tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
#         atr = tr.ewm(alpha=1/window, adjust=False).mean()
#         plus_di  = 100 * pd.Series(plus_dm,  index=high.index).ewm(alpha=1/window, adjust=False).mean() / (atr + 1e-12)
#         minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1/window, adjust=False).mean() / (atr + 1e-12)
#         dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
#         adx = dx.ewm(alpha=1/window, adjust=False).mean()
#         return adx, plus_di, minus_di

#     @staticmethod
#     def adx(high, low, close, window=14):  # backward-compatible
#         return Oscillators.adx_wilder(high, low, close, window)

#     @staticmethod
#     def cci(high, low, close, window=14):
#         tp = (high + low + close) / 3
#         sma_tp = tp.rolling(window=window).mean()
#         def mean_abs_dev(x):
#             return np.mean(np.abs(x - np.mean(x)))
#         mean_dev = tp.rolling(window=window).apply(mean_abs_dev, raw=True)
#         return (tp - sma_tp) / (0.015 * (mean_dev + 1e-12))

#     @staticmethod
#     def williams_r(high, low, close, window=14):
#         highest_high = high.rolling(window=window).max()
#         lowest_low = low.rolling(window=window).min()
#         return -100 * (highest_high - close) / (highest_high - lowest_low + 1e-12)

#     @staticmethod
#     def roc(data, window=10):
#         return (data - data.shift(window)) / (data.shift(window) + 1e-12) * 100

#     @staticmethod
#     def awesome_oscillator(high, low):
#         median_price = (high + low) / 2
#         return median_price.rolling(5).mean() - median_price.rolling(34).mean()

#     @staticmethod
#     def atr_wilder(high, low, close, window=14):
#         prev_close = close.shift(1)
#         tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
#         return tr.ewm(alpha=1/window, adjust=False).mean()

#     @staticmethod
#     def atr(high, low, close, window=14):  # backward-compatible
#         return Oscillators.atr_wilder(high, low, close, window)

# # ==========================
# # Signals
# # ==========================
# class SignalGenerator:
#     @staticmethod
#     def ma_signal(price, ma_value):
#         if price > ma_value:
#             return 'Buy'
#         elif price < ma_value:
#             return 'Sell'
#         return 'Neutral'

#     @staticmethod
#     def rsi_signal(rsi_value):
#         if rsi_value < 30:
#             return 'Buy'
#         elif rsi_value > 70:
#             return 'Sell'
#         return 'Neutral'

#     @staticmethod
#     def stochastic_signal(k_value, d_value):
#         if k_value < 20 and d_value < 20:
#             return 'Buy'
#         elif k_value > 80 and d_value > 80:
#             return 'Sell'
#         return 'Neutral'

#     @staticmethod
#     def stoch_rsi_signal(stoch_rsi_value):
#         if stoch_rsi_value < 20:
#             return 'Buy'
#         elif stoch_rsi_value > 80:
#             return 'Sell'
#         return 'Neutral'

#     @staticmethod
#     def macd_signal(macd_line, signal_line):
#         if macd_line > signal_line:
#             return 'Buy'
#         elif macd_line < signal_line:
#             return 'Sell'
#         return 'Neutral'

#     @staticmethod
#     def adx_signal(adx_value, plus_di, minus_di):
#         if adx_value > 25 and plus_di > minus_di:
#             return 'Buy'
#         elif adx_value > 25 and minus_di > plus_di:
#             return 'Sell'
#         return 'Neutral'

#     @staticmethod
#     def cci_signal(cci_value):
#         if cci_value < -100:
#             return 'Buy'
#         elif cci_value > 100:
#             return 'Sell'
#         return 'Neutral'

#     @staticmethod
#     def williams_r_signal(wr_value):
#         if wr_value < -80:
#             return 'Buy'
#         elif wr_value > -20:
#             return 'Sell'
#         return 'Neutral'

#     @staticmethod
#     def roc_signal(roc_value):
#         if roc_value > 0:
#             return 'Buy'
#         elif roc_value < 0:
#             return 'Sell'
#         return 'Neutral'

#     @staticmethod
#     def awesome_oscillator_signal(ao_value, prev_ao):
#         if ao_value > 0 and prev_ao < 0:
#             return 'Buy'
#         elif ao_value < 0 and prev_ao > 0:
#             return 'Sell'
#         return 'Neutral'

# # ==========================
# # Summary aggregator
# # ==========================
# class SummaryEngine:
#     @staticmethod
#     def generate_summary(ma_signals, osc_signals):
#         total_buy = ma_signals.count('Buy') + osc_signals.count('Buy')
#         total_sell = ma_signals.count('Sell') + osc_signals.count('Sell')
#         if total_buy >= 2 * total_sell and total_buy >= 8:
#             return 'Strong Buy'
#         elif total_buy >= 1.5 * total_sell and total_buy >= 5:
#             return 'Buy'
#         elif total_sell >= 2 * total_buy and total_sell >= 8:
#             return 'Strong Sell'
#         elif total_sell >= 1.5 * total_buy and total_sell >= 5:
#             return 'Sell'
#         else:
#             return 'Neutral'

# # ==========================
# # Printing helpers
# # ==========================
# def print_table(data, headers, title=""):
#     if title:
#         print(f"\n{Colors.BOLD}{Colors.CYAN}{title}{Colors.END}")
#         print("=" * 100)
#     header_str = " | ".join(f"{h:^22}" for h in headers)
#     print(f"{Colors.BOLD}{header_str}{Colors.END}")
#     print("-" * 100)
#     for row in data:
#         cells = []
#         for i, cell in enumerate(row):
#             if i == 2:
#                 cells.append(f"{color_signal(str(cell)):^22}")
#             else:
#                 cells.append(f"{str(cell):^22}")
#         print(" | ".join(cells))

# def calculate_signal_percentage(signal_list):
#     if not signal_list:
#         return {'Buy': 0, 'Sell': 0, 'Neutral': 0}
#     total = len(signal_list)
#     return {
#         'Buy': round(signal_list.count('Buy') / total * 100, 2),
#         'Sell': round(signal_list.count('Sell') / total * 100, 2),
#         'Neutral': round(signal_list.count('Neutral') / total * 100, 2),
#     }

# def format_results(results):
#     if not results:
#         print("No data available")
#         return

#     for timeframe, data in results.items():
#         if data is None:
#             print(f"\n{Colors.BOLD}{Colors.RED}{timeframe.upper()} Timeframe: Error in analysis{Colors.END}")
#             continue

#         print(f"\n{Colors.BOLD}{Colors.MAGENTA}{'='*120}{Colors.END}")
#         print(f"{Colors.BOLD}{Colors.MAGENTA}{timeframe.upper()} TIMEFRAME ANALYSIS{Colors.END}")
#         print(f"{Colors.BOLD}{Colors.MAGENTA}{'='*120}{Colors.END}")

#         print(f"\n{Colors.BOLD}Current Price: {Colors.CYAN}${data['current_price']:.4f}{Colors.END}")
#         print(f"{Colors.BOLD}Overall Recommendation: {color_recommendation(data['overall_recommendation'])}")

#         if 'Pivot_Points' in data['indicators'] and data['indicators']['Pivot_Points']:
#             print(f"\n{Colors.BOLD}{Colors.BLUE}PIVOT POINTS (Previous Period){Colors.END}")
#             print("=" * 100)
#             pivot_methods = ['Classic', 'Fibonacci', 'Camarilla', 'Woodie', 'DeMark']
#             pivot_header = "Method     |     S3     |     S2     |     S1     |     PP     |     R1     |     R2     |     R3     "
#             print(f"{Colors.BOLD}{pivot_header}{Colors.END}")
#             print("-" * 100)
#             for method in pivot_methods:
#                 pivots = data['indicators']['Pivot_Points'].get(method)
#                 if not pivots: 
#                     continue
#                 if method == 'DeMark':
#                     line = f"{method:<10} |            |            | {pivots.get('S1', np.nan):>10.4f} | {pivots.get('PP', np.nan):>10.4f} | {pivots.get('R1', np.nan):>10.4f} |            |            "
#                 else:
#                     line = f"{method:<10} | {pivots.get('S3', np.nan):>10.4f} | {pivots.get('S2', np.nan):>10.4f} | {pivots.get('S1', np.nan):>10.4f} | {pivots.get('PP', np.nan):>10.4f} | {pivots.get('R1', np.nan):>10.4f} | {pivots.get('R2', np.nan):>10.4f} | {pivots.get('R3', np.nan):>10.4f} "
#                 print(line)
#             print("-" * 100)

#         indicator_data = []
#         all_signals = data['ma_signals'] + data['osc_signals']
#         overall_signal_pct = calculate_signal_percentage(all_signals)

#         for name, v in data['indicators'].items():
#             if name == 'Pivot_Points':
#                 continue
#             try:
#                 value_str = f"{v['value']:.4f}"
#             except Exception:
#                 value_str = "N/A"
#             signal = v['signal']
#             if signal == 'Buy':
#                 signal_pct_str = f"{overall_signal_pct['Buy']:.1f}% Buy"
#             elif signal == 'Sell':
#                 signal_pct_str = f"{overall_signal_pct['Sell']:.1f}% Sell"
#             else:
#                 signal_pct_str = f"{overall_signal_pct['Neutral']:.1f}% Neutral"
#             indicator_data.append([name, value_str, signal, signal_pct_str])

#         print_table(indicator_data, ["Indicator", "Value", "Signal", "Signal %"], "TECHNICAL INDICATORS")

#         buy_ma = data['ma_signals'].count('Buy'); sell_ma = data['ma_signals'].count('Sell'); neutral_ma = data['ma_signals'].count('Neutral')
#         buy_osc = data['osc_signals'].count('Buy'); sell_osc = data['osc_signals'].count('Sell'); neutral_osc = data['osc_signals'].count('Neutral')

#         print(f"\n{Colors.BOLD}{Colors.BLUE}SIGNAL COUNTS{Colors.END}")
#         print("-" * 60)
#         print(f"MA Signals    - {Colors.GREEN}Buy: {buy_ma}{Colors.END}, {Colors.RED}Sell: {sell_ma}{Colors.END}, {Colors.YELLOW}Neutral: {neutral_ma}{Colors.END}")
#         print(f"Osc Signals   - {Colors.GREEN}Buy: {buy_osc}{Colors.END}, {Colors.RED}Sell: {sell_osc}{Colors.END}, {Colors.YELLOW}Neutral: {neutral_osc}{Colors.END}")
#         print(f"Total         - {Colors.GREEN}Buy: {buy_ma+buy_osc}{Colors.END}, {Colors.RED}Sell: {sell_ma+sell_osc}{Colors.END}, {Colors.YELLOW}Neutral: {neutral_ma+neutral_osc}{Colors.END}")

#         print(f"\n{Colors.BOLD}{Colors.BLUE}OVERALL SIGNAL PERCENTAGE{Colors.END}")
#         print("-" * 60)
#         print(f"Buy: {Colors.GREEN}{overall_signal_pct['Buy']:.1f}%{Colors.END}, "
#               f"Sell: {Colors.RED}{overall_signal_pct['Sell']:.1f}%{Colors.END}, "
#               f"Neutral: {Colors.YELLOW}{overall_signal_pct['Neutral']:.1f}%{Colors.END}")

# # ==========================
# # Analyzer
# # ==========================
# class TechnicalAnalyzer:
#     def __init__(self, symbol='TAO/USDT'):
#         self.symbol = symbol
#         self.data_fetcher = DataFetcher(symbol)
#         self.pivot_points = PivotPoints()
#         self.ma = MovingAverages()
#         self.osc = Oscillators()
#         self.signal_gen = SignalGenerator()
#         self.summary_engine = SummaryEngine()
#         print(f"Initialized analyzer for {self.data_fetcher.symbol}")

#     def calculate_pivot_points(self, df):
#         if len(df) < 2:
#             return {}
#         prev_row = df.iloc[-2]
#         return {
#             'Classic':   self.pivot_points.standard(prev_row['high'], prev_row['low'], prev_row['close']),
#             'Fibonacci': self.pivot_points.fibonacci(prev_row['high'], prev_row['low'], prev_row['close']),
#             'Camarilla': self.pivot_points.camarilla(prev_row['high'], prev_row['low'], prev_row['close']),
#             'Woodie':    self.pivot_points.woodies(prev_row['high'], prev_row['low'], prev_row['close'], prev_row['open']),
#             'DeMark':    self.pivot_points.demarks(prev_row['high'], prev_row['low'], prev_row['close'], prev_row['open']),
#         }

#     def calculate_indicators(self, df):
#         if len(df) < 34:
#             print("Insufficient data for analysis")
#             return None

#         results = {}
#         current_price = df['close'].iloc[-1]

#         pivot_data = self.calculate_pivot_points(df)
#         results['Pivot_Points'] = pivot_data

#         ma_windows = [5, 10, 20, 50, 100, 200]
#         ma_signals = []
#         for window in ma_windows:
#             sma_val = self.ma.sma(df['close'], window).iloc[-1]
#             ema_val = self.ma.ema(df['close'], window).iloc[-1]
#             results[f'SMA({window})'] = {'value': float(sma_val), 'signal': self.signal_gen.ma_signal(current_price, sma_val)}
#             results[f'EMA({window})'] = {'value': float(ema_val), 'signal': self.signal_gen.ma_signal(current_price, ema_val)}
#             ma_signals.append(self.signal_gen.ma_signal(current_price, sma_val))
#             ma_signals.append(self.signal_gen.ma_signal(current_price, ema_val))

#         osc_signals = []
#         try:
#             rsi_val = self.osc.rsi_wilder(df['close']).iloc[-1]
#             results['RSI(14)'] = {'value': float(rsi_val), 'signal': self.signal_gen.rsi_signal(rsi_val)}
#             osc_signals.append(self.signal_gen.rsi_signal(rsi_val))
#         except Exception as e:
#             print(f"Error calculating RSI: {e}")

#         try:
#             k_val, d_val = self.osc.stochastic(df['high'], df['low'], df['close'])
#             k_val, d_val = k_val.iloc[-1], d_val.iloc[-1]
#             results['Stoch %K'] = {'value': float(k_val), 'signal': self.signal_gen.stochastic_signal(k_val, d_val)}
#             results['Stoch %D'] = {'value': float(d_val), 'signal': self.signal_gen.stochastic_signal(k_val, d_val)}
#             osc_signals.append(self.signal_gen.stochastic_signal(k_val, d_val))
#         except Exception as e:
#             print(f"Error calculating Stochastic: {e}")

#         try:
#             stoch_rsi_val = self.osc.stoch_rsi(df['close']).iloc[-1]
#             results['StochRSI'] = {'value': float(stoch_rsi_val), 'signal': self.signal_gen.stoch_rsi_signal(stoch_rsi_val)}
#             osc_signals.append(self.signal_gen.stoch_rsi_signal(stoch_rsi_val))
#         except Exception as e:
#             print(f"Error calculating StochRSI: {e}")

#         try:
#             macd_line, signal_line, _ = self.osc.macd(df['close'])
#             macd_val, signal_val = macd_line.iloc[-1], signal_line.iloc[-1]
#             results['MACD']   = {'value': float(macd_val),   'signal': self.signal_gen.macd_signal(macd_val, signal_val)}
#             results['Signal'] = {'value': float(signal_val), 'signal': self.signal_gen.macd_signal(macd_val, signal_val)}
#             osc_signals.append(self.signal_gen.macd_signal(macd_val, signal_val))
#         except Exception as e:
#             print(f"Error calculating MACD: {e}")

#         try:
#             adx_val, plus_di, minus_di = self.osc.adx_wilder(df['high'], df['low'], df['close'])
#             adx_v, pdi, mdi = adx_val.iloc[-1], plus_di.iloc[-1], minus_di.iloc[-1]
#             results['ADX(14)'] = {'value': float(adx_v), 'signal': self.signal_gen.adx_signal(adx_v, pdi, mdi)}
#             osc_signals.append(self.signal_gen.adx_signal(adx_v, pdi, mdi))
#         except Exception as e:
#             print(f"Error calculating ADX: {e}")

#         try:
#             cci_val = self.osc.cci(df['high'], df['low'], df['close']).iloc[-1]
#             results['CCI(14)'] = {'value': float(cci_val), 'signal': self.signal_gen.cci_signal(cci_val)}
#             osc_signals.append(self.signal_gen.cci_signal(cci_val))
#         except Exception as e:
#             print(f"Error calculating CCI: {e}")

#         try:
#             wr_val = self.osc.williams_r(df['high'], df['low'], df['close']).iloc[-1]
#             results['Williams %R(14)'] = {'value': float(wr_val), 'signal': self.signal_gen.williams_r_signal(wr_val)}
#             osc_signals.append(self.signal_gen.williams_r_signal(wr_val))
#         except Exception as e:
#             print(f"Error calculating Williams %R: {e}")

#         try:
#             roc_val = self.osc.roc(df['close']).iloc[-1]
#             results['ROC'] = {'value': float(roc_val), 'signal': self.signal_gen.roc_signal(roc_val)}
#             osc_signals.append(self.signal_gen.roc_signal(roc_val))
#         except Exception as e:
#             print(f"Error calculating ROC: {e}")

#         try:
#             ao_series = self.osc.awesome_oscillator(df['high'], df['low'])
#             ao_val, prev_ao = ao_series.iloc[-1], ao_series.iloc[-2]
#             results['AO'] = {'value': float(ao_val), 'signal': self.signal_gen.awesome_oscillator_signal(ao_val, prev_ao)}
#             osc_signals.append(self.signal_gen.awesome_oscillator_signal(ao_val, prev_ao))
#         except Exception as e:
#             print(f"Error calculating AO: {e}")

#         try:
#             atr_val = self.osc.atr_wilder(df['high'], df['low'], df['close']).iloc[-1]
#             results['ATR(14)'] = {'value': float(atr_val), 'signal': 'Neutral'}
#         except Exception as e:
#             print(f"Error calculating ATR: {e}")

#         try:
#             overall_recommendation = self.summary_engine.generate_summary(ma_signals, osc_signals)
#         except Exception:
#             overall_recommendation = 'Neutral'

#         return {
#             'current_price': float(current_price),
#             'indicators': results,
#             'ma_signals': ma_signals,
#             'osc_signals': osc_signals,
#             'overall_recommendation': overall_recommendation
#         }

#     def analyze_all_timeframes(self):
#         timeframes = ['5m', '15m', '1h', '1d', '1w']
#         results = {}
#         for tf in timeframes:
#             try:
#                 print(f"Analyzing {tf} timeframe...")
#                 df = self.data_fetcher.fetch_data(tf)
#                 results[tf] = self.calculate_indicators(df) if (df is not None and len(df) > 0) else None
#             except Exception as e:
#                 print(f"Error analyzing {tf}: {e}")
#                 results[tf] = None
#         return results

# # ==========================
# # Sentiment + Execution Plan
# # ==========================
# def sentiment_score_from_cli(use_sentiment: bool) -> float:
#     """
#     Returns sentiment in [-1, 1]. If --use-sentiment not provided, returns 0.0.
#     Replace this stub with loading your real score (e.g., from JSON).
#     """
#     if not use_sentiment:
#         return 0.0
#     # Example from earlier DLNews analysis (bullish):
#     return 0.71

# def apply_sentiment_bias(levels: dict, news_score: float):
#     s = max(min(news_score, 0.8), -0.8)
#     levels['long_trigger']  *= (1 - 0.0005 * max(s, 0))   # bullish -> trigger slightly easier
#     levels['short_trigger'] *= (1 + 0.0005 * max(-s, 0))  # bearish -> easier for shorts
#     levels['tp_mult']       *= (1 + 0.3 * s)              # widen TP when bullish
#     levels['sl_mult']       *= (1 - 0.15 * s)             # slightly tighter SL when bullish
#     return levels

# def atr_wilder_series(high, low, close, window=14):
#     prev_close = close.shift(1)
#     tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
#     return tr.ewm(alpha=1/window, adjust=False).mean()

# def compute_trade_plan_1h_5m(df_1h: pd.DataFrame, df_5m: pd.DataFrame, sentiment: float = 0.0):
#     """
#     Use last CLOSED 1h candle as regime; arm 5m breakout/pullback levels.
#     """
#     last = df_1h.iloc[-2]  # last closed 1h bar
#     atr1h = atr_wilder_series(df_1h['high'], df_1h['low'], df_1h['close'], window=14).iloc[-2]
#     ema20 = df_1h['close'].ewm(span=20, adjust=False).mean().iloc[-2]

#     trend_long  = last['close'] > ema20
#     trend_short = last['close'] < ema20

#     prev_high = last['high']
#     prev_low  = last['low']

#     base_sl_mult = 1.2
#     base_tp_mult = 2.0

#     plan = {
#         'trend_long': trend_long,
#         'trend_short': trend_short,
#         'long_trigger': float(prev_high * 1.001),         # ~+0.1% above 1h high
#         'short_trigger': float(prev_low  * 0.999),         # ~-0.1% below 1h low
#         'pullback_long': float(prev_high - 0.25 * atr1h),  # if breakout missed
#         'pullback_short': float(prev_low  + 0.25 * atr1h),
#         'atr1h': float(atr1h),
#         'tp_mult': base_tp_mult,
#         'sl_mult': base_sl_mult,
#         'ema20_1h': float(ema20),
#         'prev_high': float(prev_high),
#         'prev_low': float(prev_low),
#     }

#     plan = apply_sentiment_bias(plan, sentiment)
#     return plan

# def decide_orders(price_5m: float, plan: dict):
#     """
#     Decide entries/exits based on current 5m price and precomputed plan.
#     Returns order intents; wire these to your ccxt order placement.
#     """
#     atr = plan['atr1h']
#     sl_mult = plan['sl_mult']
#     tp_mult = plan['tp_mult']
#     orders = {'enter_long': None, 'enter_short': None, 'limit_long': None, 'limit_short': None,
#               'sl': None, 'tp1': None, 'trail_at': None}

#     if plan['trend_long']:
#         if price_5m >= plan['long_trigger']:
#             entry = price_5m
#             orders['enter_long'] = entry
#             orders['sl'] = entry - sl_mult * atr
#             R = entry - orders['sl']
#             orders['tp1'] = entry + tp_mult * R
#             orders['trail_at'] = entry + 1.0 * R
#         else:
#             orders['limit_long'] = plan['pullback_long']

#     if plan['trend_short']:
#         if price_5m <= plan['short_trigger']:
#             entry = price_5m
#             orders['enter_short'] = entry
#             orders['sl'] = entry + sl_mult * atr
#             R = orders['sl'] - entry
#             orders['tp1'] = entry - tp_mult * R
#             orders['trail_at'] = entry - 1.0 * R
#         else:
#             orders['limit_short'] = plan['pullback_short']

#     return orders

# # ==========================
# # Main
# # ==========================
# if __name__ == "__main__":
#     class Args:
#       use_sentiment = True  # or False to disable
#     args = Args()

#     print(f"{Colors.BOLD}{Colors.CYAN}BITTENSOR (TAO) TECHNICAL ANALYSIS{Colors.END}")
#     analyzer = TechnicalAnalyzer('TAO/USDT')
#     results = analyzer.analyze_all_timeframes()
#     format_results(results)

#     # Build an execution plan using 1h (trend) + 5m (timing)
#     df_1h = analyzer.data_fetcher.fetch_data('1h', limit=300)
#     df_5m = analyzer.data_fetcher.fetch_data('5m', limit=300)
#     sentiment = sentiment_score_from_cli(args.use_sentiment)
#     plan = compute_trade_plan_1h_5m(df_1h, df_5m, sentiment=sentiment)
#     last_price_5m = float(df_5m['close'].iloc[-1])
#     orders = decide_orders(last_price_5m, plan)

#     print(f"\n{Colors.BOLD}{Colors.BLUE}EXECUTION PLAN (with {'sentiment' if args.use_sentiment else 'no sentiment'}){Colors.END}")
#     pprint(plan)
#     print(f"\nLast 5m price: {last_price_5m:.4f}")
#     pprint(orders)

#     print(f"\n{Colors.BOLD}{Colors.MAGENTA}Analysis Complete!{Colors.END}")
