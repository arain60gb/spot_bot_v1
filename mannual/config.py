# config.py
import os

# --- TRADING PARAMETERS ---
SYMBOL = 'TAO/USDT'  # Or TAO/USDT if available
TRADE_SYMBOL = 'TAO' # Base asset
QUOTE_CURRENCY = 'USDT' # Quote asset
INITIAL_USDT = 10000.0
INITIAL_TAO = 50.0
FIXED_ORDER_SIZE_USDT = 2000.0 # Amount of USDT to use per buy trade
STOP_LOSS_PERCENT = 1.5  # 5% Stop Loss
TAKE_PROFIT_PERCENT = 10.0 # 10% Take Profit

# --- MODE CONFIGURATION ---
# Set this to 'backtest', 'paper', or 'live'
# In a real scenario, you might load this from environment variables or command line args
MODE = os.getenv('TRADING_MODE', 'paper') # Default to paper if not set

# --- BINANCE API KEYS (NEVER COMMIT THESE!) ---
# Use environment variables for security
API_KEY = os.getenv('BINANCE_API_KEY', 'V6sDKtCZLEzgRygjQHF2kfRfRSir2KsDNjX9qB0K1cds97CHRH5W9gTmfS5usocr')
API_SECRET = os.getenv('BINANCE_API_SECRET', 'kBZ1eKhddgPIhawX2hvIisMgOSzCn2Qw4oKchBExQfsnPLz5TW2uvgOHcWneGeHX')

# --- LOGGING ---
LOG_FILE = 'trading_bot.log'
TRADE_HISTORY_FILE = 'trade_history.json'