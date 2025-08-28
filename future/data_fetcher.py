# data_fetcher.py
"""
Module for fetching historical OHLCV data for backtesting.
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

class BacktestDataFetcher:
    """
    Fetches historical OHLCV data for backtesting, handling pagination efficiently.
    """

    def __init__(self, symbol='TAO/USDT', exchange_id='binance'):
        """
        Initializes the data fetcher.
        Args:
            symbol (str): Trading pair symbol (e.g., 'TAO/USDT').
            exchange_id (str): ID of the exchange to use (e.g., 'binance').
        """
        self.symbol = symbol
        try:
            self.exchange = getattr(ccxt, exchange_id)({
                'enableRateLimit': True,
                'timeout': 30000,  # 30 seconds timeout
                'options': {'adjustForTimeDifference': True}
            })
            markets = self.exchange.load_markets()
            if self.symbol not in markets:
                print(f"Warning: Symbol {self.symbol} not found on {exchange_id}. Please check availability.")
        except Exception as e:
            print(f"Error initializing exchange {exchange_id}: {e}")
            raise

    def _calculate_target_limit(self, timeframe, days):
        """Calculate the target number of candles based on timeframe and days."""
        timeframe_to_minutes = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
            '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
        }
        minutes_per_candle = timeframe_to_minutes.get(timeframe, 60)
        buffer_candles = 250  # Buffer for indicators like 200 SMA
        estimated_candles = int((days * 24 * 60) / minutes_per_candle) + buffer_candles
        return min(estimated_candles, 15000)  # Cap to avoid excessive requests

    def fetch_historical_data(self, timeframe='15m', days=30):
        """
        Fetches historical OHLCV data for a given period, attempting to get as many candles as possible.
        Args:
            timeframe (str): Time interval (e.g., '5m', '15m', '1h', '1d').
            days (int): Number of past days to fetch.
        Returns:
            pd.DataFrame: DataFrame with OHLCV data indexed by timestamp, sorted oldest to newest.
        """
        try:
            target_limit = self._calculate_target_limit(timeframe, days)
            print(f"Calculating data needs for {timeframe} ({days}d): ~{target_limit - 250} candles needed.")

            since_initial = self.exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)
            print(f"Fetching {self.symbol} ({timeframe}) data for last {days} days...")

            all_ohlcv = []
            current_since = since_initial
            candles_fetched_in_this_request = 1
            total_candles_fetched = 0
            request_count = 0
            exchange_limit_per_request = 1000  # Common limit for Binance

            while total_candles_fetched < target_limit and candles_fetched_in_this_request > 0:
                request_count += 1
                if request_count % 10 == 0:  # Progress indicator
                    print(f"  Request {request_count}: Fetching up to {exchange_limit_per_request} candles since {pd.to_datetime(current_since, unit='ms')}...")

                # --- CORRECTED: Removed params={'timeout': ...} to fix Binance API error ---
                ohlcv_chunk = self.exchange.fetch_ohlcv(
                    self.symbol, timeframe, since=current_since, limit=exchange_limit_per_request
                    # Timeout handled by exchange object initialization
                )
                # --- END CORRECTION ---

                candles_fetched_in_this_request = len(ohlcv_chunk)
                total_candles_fetched += candles_fetched_in_this_request
                # print(f"    Fetched {candles_fetched_in_this_request} candles in this request. Total so far: {total_candles_fetched}")

                if candles_fetched_in_this_request == 0:
                    # print("    No more data available from exchange.")
                    break

                all_ohlcv.extend(ohlcv_chunk)
                last_timestamp = ohlcv_chunk[-1][0]
                current_since = last_timestamp + 1

                if request_count > 50:  # Safety break
                    print("    Safety break: Too many requests. Stopping fetch.")
                    break

            if not all_ohlcv:
                print("Warning: No historical data returned after all requests.")
                return pd.DataFrame()

            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            df = df[~df.index.duplicated(keep='first')].sort_index()

            # Trim to the target_limit (latest candles)
            if len(df) > target_limit:
                df = df.tail(target_limit)
                print(f"    Trimmed data to the most recent {target_limit} candles.")

            print(f"Successfully fetched {len(df)} unique {timeframe} candles from {df.index[0]} to {df.index[-1]}.")
            return df

        except ccxt.RequestTimeout as e:
            print(f"Request timeout fetching {timeframe}  {e}")
        except ccxt.NetworkError as e:
            print(f"Network error fetching {timeframe}  {e}")
        except ccxt.ExchangeError as e:
            print(f"Exchange error fetching {timeframe}  {e}")
        except Exception as e:
            print(f"An unexpected error occurred while fetching {timeframe} data: {e}")

        return pd.DataFrame()


def fetch_multiple_timeframes_concurrently(fetcher_instance, timeframes_and_days):
    """
    Fetches data for multiple timeframes concurrently using ThreadPoolExecutor.

    Args:
        fetcher_instance (BacktestDataFetcher): An instance of the data fetcher.
        timeframes_and_days (dict): A dictionary like {'1h': 35, '15m': 32}.

    Returns:
        dict: A dictionary mapping timeframe strings to their respective DataFrames.
    """
    all_timeframe_data = {}
    print("Fetching data for all required timeframes...")

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_tf = {
            executor.submit(fetcher_instance.fetch_historical_data, tf, days): (tf, days)
            for tf, days in timeframes_and_days.items()
        }
        for future in as_completed(future_to_tf):
            tf, days = future_to_tf[future]
            try:
                tf_data = future.result()
                all_timeframe_data[tf] = tf_data
                print(f"  Fetched {len(tf_data)} candles for {tf}")
            except Exception as e:
                print(f"  Error fetching data for {tf} ({days} days): {e}")
                all_timeframe_data[tf] = pd.DataFrame()  # Empty DataFrame on error

    return all_timeframe_data
