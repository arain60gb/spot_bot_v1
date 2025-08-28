# # config.py
# import os

# # --- TRADING PARAMETERS (FUTURES) ---
# SYMBOL = 'TAO/USDT:USDT'  # Futures contract symbol format
# TRADE_SYMBOL = 'TAO'

# QUOTE_CURRENCY = 'USDT'
# INITIAL_USDT = 10000.0
# INITIAL_TAO = 0.0 # Likely not needed for futures unless you hold spot too

# FIXED_ORDER_SIZE_USDT = 100.0 # Amount of USDT margin to use per trade
# LEVERAGE = 10 # 10x Leverage
# STOP_LOSS_PERCENT = 2.0  # 2% Stop Loss on position value
# TAKE_PROFIT_PERCENT = 3.0 # Example TP, or use fixed RR like 1:1.5

# # --- MODE CONFIGURATION ---
# MODE = os.getenv('TRADING_MODE', 'paper') # Default to paper if not set

# # Use environment variables for security
# API_KEY = os.getenv('BINANCE_API_KEY', 'V6sDKtCZLEzgRygjQHF2kfRfRSir2KsDNjX9qB0K1cds97CHRH5W9gTmfS5usocr')
# API_SECRET = os.getenv('BINANCE_API_SECRET', 'kBZ1eKhddgPIhawX2hvIisMgOSzCn2Qw4oKchBExQfsnPLz5TW2uvgOHcWneGeHX')
# # --- LOGGING ---
# LOG_FILE = 'futures_trading_bot.log' # Differentiate logs
# TRADE_HISTORY_FILE = 'futures_trade_history.json'




# config.py
import os

# ── TRADING PARAMETERS (Futures Scalper) ──────────────────────────────────────
SYMBOL = 'TAO/USDT'                 # ccxt-normalized symbol; bot will normalize if needed
INITIAL_USDT = 10_000.0             # Paper account equity
FIXED_ORDER_SIZE_USDT = 100.0       # ← margin per trade ($100)
LEVERAGE = 50                       # ← 50x, ~ $5,000 notional per trade
STOP_LOSS_PERCENT = 2.0             # ← 2% SL from entry
TAKE_PROFIT_PERCENT = 5.0          # ← 10% TP from entry
# ── FUTURES / EXCHANGE CONFIG ────────────────────────────────────────────────
USE_FUTURES = True
FUTURES_EXCHANGE_ID = 'binanceusdm'  # ccxt futures exchange id
FUTURES_TAKER_FEE_BPS = 4            # 0.04% taker taker fee (per fill)

# ── MODE ─────────────────────────────────────────────────────────────────────
# backtest | paper | live
MODE = os.getenv('TRADING_MODE', 'paper')

# Use environment variables for security
API_KEY = os.getenv('BINANCE_API_KEY', 'V6sDKtCZLEzgRygjQHF2kfRfRSir2KsDNjX9qB0K1cds97CHRH5W9gTmfS5usocr')
API_SECRET = os.getenv('BINANCE_API_SECRET', 'kBZ1eKhddgPIhawX2hvIisMgOSzCn2Qw4oKchBExQfsnPLz5TW2uvgOHcWneGeHX')
# ── LOGGING & HISTORY ───────────────────────────────────────────────────────
LOG_FILE = 'trading_bot.log'
# Leave empty to let bot auto-name per symbol/mode to avoid collisions:
TRADE_HISTORY_FILE = os.getenv('TRADE_HISTORY_FILE', '')
