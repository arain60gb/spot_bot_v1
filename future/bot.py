# # bot.py  — updated for Futures Scalping
# import sys
# import os
# import json
# import csv
# import time
# import atexit
# import threading
# import argparse
# import logging
# import re
# import requests
# import pandas as pd
# import numpy as np
# import ccxt
# from datetime import datetime, timezone, timedelta
# from pathlib import Path
# from logging.handlers import RotatingFileHandler
# # ── Config ──────────────────────────────────────────────────────────────────────
# from config import (
#     SYMBOL,
#     INITIAL_USDT,
#     INITIAL_TAO,
#     FIXED_ORDER_SIZE_USDT,
#     MODE,
#     STOP_LOSS_PERCENT,
#     TAKE_PROFIT_PERCENT,
#     TRADE_HISTORY_FILE,
#     LOG_FILE,
#     LEVERAGE,
#     API_KEY,
#     API_SECRET
# )
# # ── Technical analysis utilities ───────────────────────────────────────────────
# # Import the new futures-specific analyzer
# from technical_analyzer import (
#     FastScalpingAnalyzer,
#     FastFuturesDataFetcher,
#     Colors,
#     color_signal,
#     compute_scalping_plan_1m_5m,   # <— updated plan builder
# )
# # ── Historical data fetcher (back-test) ────────────────────────────────────────
# # Note: Backtesting for futures might need adjustments, but we'll keep the structure
# from data_fetcher import BacktestDataFetcher, fetch_multiple_timeframes_concurrently
# # ── Dash & Plotly ─────────────────────────────────────────────────────────────
# import dash
# import dash_bootstrap_components as dbc
# from dash import dcc, html, ctx
# from dash.dependencies import Output, Input, State
# from dash.dash_table import DataTable
# import plotly.graph_objs as go
# # Light Bootstrap theme
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
# # Light palette
# PALETTE = {
#     "page_bg": "#f7f9fc",
#     "panel": "#ffffff",
#     "text": "#1f2937",
#     "muted": "#6b7280",
#     "grid": "#e5e7eb",
#     "accent": "#0d6efd",
#     "entry": "#22c55e",
#     "exit": "#ef4444",
#     "candle_up": "#16a34a",
#     "candle_down": "#dc2626",
#     "trade_win": "#15803d",
#     "trade_loss": "#b91c1c",
#     "halo": "rgba(0,0,0,0.08)",
#     "button_bg": "#dc3545",
#     "button_hover": "#bb2d3b",
#     "start_button_bg": "#198754",
#     "start_button_hover": "#157347",
# }
# # Marker density controls
# MAX_MARKERS_ON_CHART = 120
# SPARSE_MARKER_SIZE = 10
# DENSE_MARKER_SIZE = 6
# # ── Logging setup ------------------------------------------------------------
# def setup_logging(log_path: str | Path):
#     log_path = Path(log_path) if log_path else Path("futures_trading_bot.log")
#     log_path.parent.mkdir(parents=True, exist_ok=True)
#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)
#     fmt = logging.Formatter(
#         fmt="%(asctime)s | %(levelname)s | %(message)s",
#         datefmt="%Y-%m-%d %H:%M:%S",
#     )
#     class AnsiStrippingFileHandler(RotatingFileHandler):
#         ansi_re = re.compile(r"\x1b\[[0-9;]*m")
#         def emit(self, record):
#             if isinstance(record.msg, str):
#                 record.msg = self.ansi_re.sub("", record.msg)
#             super().emit(record)
#     fh = AnsiStrippingFileHandler(log_path, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
#     fh.setFormatter(fmt); fh.setLevel(logging.INFO)
#     ch = logging.StreamHandler()
#     ch.setFormatter(fmt); ch.setLevel(logging.INFO)
#     logger.handlers.clear()
#     logger.addHandler(fh); logger.addHandler(ch)
#     class _LoggerWriter:
#         def __init__(self, level_func): self.level_func = level_func
#         def write(self, message):
#             if message.strip():
#                 for line in message.rstrip().splitlines():
#                     self.level_func(line)
#         def flush(self): pass
#     sys.stdout = _LoggerWriter(logger.info)
#     sys.stderr = _LoggerWriter(logger.error)
# setup_logging(LOG_FILE)
# logger = logging.getLogger(__name__)
# logger.info("Logger initialised – MODE=%s – SYMBOL=%s", MODE, SYMBOL)
# # ── Trading strategy ---------------------------------------------------------
# class FuturesScalpingStrategy:
#     def __init__(self, symbol, primary_tf="1m"): # Changed primary timeframe to 1m for scalping
#         self.analyzer = FastScalpingAnalyzer(symbol)
#         self.symbol = symbol
#         self.primary_tf = primary_tf

#     def get_signals(self):
#         if MODE in ["paper", "live"]:
#             try:
#                 # Use the fast signal method
#                 signal, details = self.analyzer.get_scalping_signal()
#                 return {'1m': {'overall_recommendation': signal, 'details': details}}
#             except Exception:
#                 logger.exception("Error getting live signals")
#                 print("Error getting live signals")
#                 return {}
#         return {}
# # ── Core bot ---------------------------------------------------------------
# class FuturesScalpingBot:
#     def __init__(self, symbol, primary_tf="1m"): # Changed primary timeframe to 1m
#         self.symbol = symbol
#         self.base_asset, self.quote_asset = symbol.split("/")[0], symbol.split("/")[1].split(":")[0] # Handle TAO/USDT:USDT
#         self.primary_tf = primary_tf
#         # position state
#         self.position_state = "Flat"       # Flat | Long | Short
#         self.entry_price = None
#         self.amount_base = 0.0 # This will represent contract size in futures
#         self.sl_level = None               # dynamic SL for open position
#         self.tp_level = None               # dynamic TP for open position
#         # optional trade-logic state (kept for future rules/UI)
#         self.seen_strong_follow_through = False
#         self.open_signal = None
#         # trading state
#         self.trading_enabled = True
#         # price (for pill & chart)
#         self.last_price = None
#         self.simulated_balance = { self.quote_asset: INITIAL_USDT, self.base_asset: INITIAL_TAO }
#         self.strategy = FuturesScalpingStrategy(symbol, primary_tf)
#         self.trade_history = self._load_trade_history()
#         # dashboard data
#         self.completed_trades = []   # dicts include side, entry/exit, pnl, sl, tp, open_signal, close_reason
#         self.portfolio_history = []  # snapshots
#         self.candle_history = []     # OHLC for primary tf
#         self._open_trades_csv = {}
#         # current signal (for dashboard)
#         self.current_signal = "Neutral"
#         # last plan for cards
#         self.last_plan = None
#         # live mode data fetcher & candle aggregator
#         self.live_fetcher = None
#         if MODE in ["paper", "live"]:
#             try:
#                 self.live_fetcher = FastFuturesDataFetcher(symbol) # Use futures fetcher
#             except Exception:
#                 logger.exception("Could not init DataFetcher")
#         self._dash_lock = threading.Lock()
#         self._dash_server = None
#         self._dash_thread = None
#         atexit.register(self._shutdown_dash_server)
        
#         if MODE in ["paper", "live"]:
#             cfg = {
#                 "apiKey": API_KEY,
#                 "secret": API_SECRET,
#                 "enableRateLimit": True, 
#                 "timeout": 30_000,
#                 "options": {'defaultType': 'future'} # Crucial for futures
#             }
#             self.exchange = ccxt.binance(cfg)
#             self.exchange.load_markets()
#             # Check if symbol exists
#             if self.symbol not in self.exchange.markets:
#                  logger.error(f"Symbol {self.symbol} not available for futures trading on {self.exchange.id}")
#                  raise ValueError(f"Symbol {self.symbol} not found in futures markets")
                 
#             # Set leverage
#             try:
#                 self.exchange.set_leverage(LEVERAGE, self.symbol)
#                 logger.info(f"Leverage set to {LEVERAGE}x for {self.symbol}")
#             except Exception as e:
#                 logger.error(f"Failed to set leverage: {e}")
#         else:
#             self.exchange = None
            
#         if MODE == "backtest":
#             self.backtest_fetcher = BacktestDataFetcher(symbol)
#         print(f"Futures Bot initialised for {symbol} – mode={MODE} – TF={primary_tf}")
#         logger.info("Futures Bot init – %s – %s – %s", symbol, MODE, primary_tf)
#         self._setup_dashboard()
#     # ── Helpers for markers ---------------------------------------------------
#     def _thin_points(self, xs, ys, max_points):
#         """Evenly sample down to <= max_points, preserving endpoints."""
#         n = len(xs)
#         if n <= max_points:
#             return xs, ys
#         idx = np.linspace(0, n - 1, max_points, dtype=int)
#         return [xs[i] for i in idx], [ys[i] for i in idx]
#     # ── Small UI helpers -----------------------------------------------------
#     def _signal_pill(self, rec: str):
#         rec = rec or "Neutral"
#         cmap = {
#             "Strong Buy": ("#065f46", "#ecfdf5"),
#             "Buy":        ("#15803d", "#ecfdf5"),
#             "Neutral":    ("#374151", "#f3f4f6"),
#             "Sell":       ("#b91c1c", "#fee2e2"),
#             "Strong Sell":("#7f1d1d", "#fee2e2"),
#         }
#         fg, bg = cmap.get(rec, cmap["Neutral"])
#         return html.Span(
#             rec,
#             style={
#                 "display": "inline-block",
#                 "padding": "4px 10px",
#                 "borderRadius": "9999px",
#                 "fontWeight": 600,
#                 "color": fg,
#                 "backgroundColor": bg,
#                 "border": f"1px solid {PALETTE['grid']}",
#                 "marginLeft": "6px",
#             },
#         )
#     def _price_pill(self, price):
#         txt = f"{price:,.4f} {self.quote_asset}" if price is not None else "—"
#         return html.Span(
#             txt,
#             style={
#                 "display": "inline-block",
#                 "padding": "4px 10px",
#                 "borderRadius": "8px",
#                 "fontWeight": 600,
#                 "color": "#111827",
#                 "backgroundColor": "#eef2ff",
#                 "border": f"1px solid {PALETTE['grid']}",
#                 "marginLeft": "6px",
#             },
#         )
#     # ── NEW: Plan card (left) -------------------------------------------------
#     def _build_plan_card(self):
#         with self._dash_lock:
#             pricepill = self._price_pill(self.last_price)
#             pill = self._signal_pill(self.current_signal)
#             plan = self.last_plan
#         items = []
#         items.append(html.Div([html.Span("Current signal:"), pill], className="mb-2"))
#         items.append(html.Div([html.Span("Current price:"), pricepill], className="mb-2"))
#         if plan:
#             items += [
#                 html.P(f"Trend: Long={plan.get('trend_long', False)} | Short={plan.get('trend_short', False)}", className="mb-1"),
#                 html.P(f"5M EMA20: {plan.get('ema20_5m', float('nan')):.4f}", className="mb-1"),
#                 html.P(f"LongTrig: {plan.get('long_trigger', float('nan')):.4f} • PullLong: {plan.get('pullback_long', float('nan')):.4f}", className="mb-1"),
#                 html.P(f"ShortTrig: {plan.get('short_trigger', float('nan')):.4f} • PullShort: {plan.get('pullback_short', float('nan')):.4f}", className="mb-1"),
#                 html.P(f"ATR5m: {plan.get('atr5m', float('nan')):.4f} • SLx: {plan.get('sl_mult', float('nan')):.4f} • TPx: {plan.get('tp_mult', float('nan')):.4f}", className="mb-1"),
#             ]
#         else:
#             items.append(html.P("Plan not computed yet…", className="mb-1", style={"color": PALETTE["muted"]}))
#         return dbc.Card(
#             [
#                 dbc.CardHeader("Futures Scalping Plan"),
#                 dbc.CardBody(items),
#             ],
#             className="mt-2",
#             style={"backgroundColor": PALETTE["panel"], "color": PALETTE["text"],
#                    "border": f"1px solid {PALETTE['grid']}", "borderRadius": "10px"}
#         )
#     # ── NEW: Position card (center) ------------------------------------------
#     def _build_position_card(self):
#         with self._dash_lock:
#             pill = self._signal_pill(self.current_signal)
#             pricepill = self._price_pill(self.last_price)
#             pos = self.position_state
#             entry = self.entry_price
#             sl = self.sl_level
#             tp = self.tp_level
#             unreal = 0.0
#             if pos != "Flat" and entry:
#                 p = self.last_price if self.last_price is not None else entry
#                 # PnL for futures: (Current Price - Entry Price) * Contract Size * Leverage (for Long)
#                 # PnL for futures: (Entry Price - Current Price) * Contract Size * Leverage (for Short)
#                 # But we track contract size in amount_base
#                 unreal = ((p - entry) * self.amount_base) if pos == "Long" else ((entry - p) * self.amount_base)
#             seen_strong = self.seen_strong_follow_through
#             arm_neutral = True
#             post_neutral = False
#         items = [
#             html.Div([html.Span("Current signal:"), pill], className="mb-2"),
#             html.Div([html.Span("Current price:"), pricepill], className="mb-2"),
#             html.P(f"Timeframe: {self.primary_tf}", className="mb-1"),
#             html.P(f"Side: {pos}", className="mb-1"),
#             html.P(f"Leverage: {LEVERAGE}x", className="mb-1"), # Show leverage
#         ]
#         if pos != "Flat" and entry:
#             items += [
#                 html.P(f"Entry: {entry:.4f} {self.quote_asset}", className="mb-1"),
#                 html.P(f"SL: {sl:.4f}", className="mb-1"),
#                 html.P(f"TP: {tp:.4f}", className="mb-1"),
#                 html.P(f"Contracts: {self.amount_base:.6f}", className="mb-1"), # Show contract size
#                 html.P(f"Strong seen after entry: {seen_strong}", className="mb-1"),
#                 html.P(f"Arm exit on Neutral: {arm_neutral}", className="mb-1"),
#                 html.P(f"Post-Neutral State: {post_neutral}", className="mb-1"),
#                 html.P(f"Unrealised PnL: {unreal:.4f} {self.quote_asset}", className="mb-1"),
#             ]
#         else:
#             items.append(html.P("No open position.", className="mb-1", style={"color": PALETTE["muted"]}))
#         # Buttons row (now includes ShortTrig / LongTrig)
#         items.append(
#             html.Div(
#                 [
#                     dbc.Button(
#                         "Close Trade", id="close-button", color="danger",
#                         className="mt-2 me-2",
#                         style={"backgroundColor": PALETTE["button_bg"], "borderColor": PALETTE["button_bg"]}
#                     ),
#                     dbc.Button(
#                         "Start Trade", id="start-button", color="success",
#                         className="mt-2 me-2",
#                         style={"backgroundColor": PALETTE["start_button_bg"], "borderColor": PALETTE["start_button_bg"]}
#                     ),
#                     dbc.Button(
#                         "ShortTrig", id="shorttrig-button", color="warning",
#                         className="mt-2 me-2",
#                         title="Open a SELL (short) trade now, ignoring signals",
#                     ),
#                     dbc.Button(
#                         "LongTrig", id="longtrig-button", color="primary",
#                         className="mt-2",
#                         title="Open a BUY (long) trade now, ignoring signals",
#                     ),
#                 ]
#             )
#         )
#         return dbc.Card(
#             [dbc.CardHeader("Futures Position"), dbc.CardBody(items)],
#             className="mt-2",
#             style={"backgroundColor": PALETTE["panel"], "color": PALETTE["text"],
#                    "border": f"1px solid {PALETTE['grid']}", "borderRadius": "10px"}
#         )
#     # ── Manual Close Position Method ──────────────────────────────────────────
#     def close_position(self, ts, price, closed_by="Manual Close"):
#         if self.position_state == "Flat":
#             logger.info("Manual close requested, but no position is open.")
#             print("No open position to close.")
#             return
#         logger.info("Manual close initiated – %s @ %s", self.position_state, price)
#         print(f"{Colors.YELLOW}[{ts}] Manual close initiated – {self.position_state} @ {price:.4f}{Colors.END}")
#         if self.position_state == "Long":
#             self._execute_sell(price, ts, closed_by=closed_by)
#         elif self.position_state == "Short":
#             self._execute_buy(price, ts, closed_by=closed_by)
#         self.trading_enabled = False
#         logger.info("Trading disabled after manual close. Waiting for Start Trade button.")
#     def enable_trading(self):
#         if not self.trading_enabled:
#             self.trading_enabled = True
#             logger.info("Trading enabled via Start Trade button.")
#             print("Trading enabled. Bot will now respond to signals.")
#     # ── Price figure ---------------------------------------------------------
#     def _build_price_figure(self):
#         with self._dash_lock:
#             candles = pd.DataFrame(self.candle_history)
#             completed = list(self.completed_trades)
#             sig = self.current_signal
#             cur_price = self.last_price
#         candle_trace = go.Candlestick(
#             x=candles["timestamp"] if not candles.empty else [],
#             open=candles["open"] if not candles.empty else [],
#             high=candles["high"] if not candles.empty else [],
#             low=candles["low"] if not candles.empty else [],
#             close=candles["close"] if not candles.empty else [],
#             name="Price",
#             increasing=dict(line=dict(color=PALETTE["candle_up"]), fillcolor=PALETTE["candle_up"]),
#             decreasing=dict(line=dict(color=PALETTE["candle_down"]), fillcolor=PALETTE["candle_down"]),
#             whiskerwidth=0.4
#         )
#         entry_x = [t["entry_time"] for t in completed]
#         entry_y = [t["entry_price"] for t in completed]
#         exit_x  = [t["exit_time"] for t in completed]
#         exit_y  = [t["exit_price"] for t in completed]
#         total_markers = len(entry_x) + len(exit_x)
#         dense = total_markers > MAX_MARKERS_ON_CHART
#         if dense:
#             budget_each = max(1, MAX_MARKERS_ON_CHART // 2)
#             entry_x, entry_y = self._thin_points(entry_x, entry_y, budget_each)
#             exit_x,  exit_y  = self._thin_points(exit_x,  exit_y,  budget_each)
#         msize = DENSE_MARKER_SIZE if dense else SPARSE_MARKER_SIZE
#         outline = dict(width=1, color=PALETTE["panel"])
#         fig = go.Figure()
#         fig.add_trace(candle_trace)
#         if not dense:
#             fig.add_trace(
#                 go.Scatter(
#                     x=entry_x, y=entry_y, mode="markers", name="Entry halo",
#                     marker=dict(size=msize+8, color=PALETTE["halo"], line=dict(width=0)),
#                     hoverinfo="skip", showlegend=False
#                 )
#             )
#             fig.add_trace(
#                 go.Scatter(
#                     x=exit_x, y=exit_y, mode="markers", name="Exit halo",
#                     marker=dict(size=msize+8, color=PALETTE["halo"], line=dict(width=0)),
#                     hoverinfo="skip", showlegend=False
#                 )
#             )
#         fig.add_trace(
#             go.Scatter(
#                 x=entry_x, y=entry_y, mode="markers", name="Entry",
#                 marker=dict(size=msize, color=PALETTE["entry"], line=outline, symbol="circle"),
#                 hovertemplate="Entry @ %{y:.4f}<br>%{x|%Y-%m-%d %H:%M}<extra></extra>"
#             )
#         )
#         fig.add_trace(
#             go.Scatter(
#                 x=exit_x, y=exit_y, mode="markers", name="Exit",
#                 marker=dict(size=msize, color=PALETTE["exit"], line=outline, symbol="circle"),
#                 hovertemplate="Exit @ %{y:.4f}<br>%{x|%Y-%m-%d %H:%M}<extra></extra>"
#             )
#         )
#         for t in completed:
#             is_win = (t.get("pnl", 0) or 0) >= 0
#             col = PALETTE["trade_win"] if is_win else PALETTE["trade_loss"]
#             fig.add_trace(
#                 go.Scatter(
#                     x=[t["entry_time"], t["exit_time"]],
#                     y=[t["entry_price"], t["exit_price"]],
#                     mode="lines",
#                     line=dict(color=col, width=3),
#                     hoverinfo="skip",
#                     name="Trade",
#                     showlegend=False,
#                 )
#             )
#         total_pnl = sum((t.get("pnl") or 0) for t in completed)
#         pnl_color = PALETTE["trade_win"] if total_pnl >= 0 else PALETTE["trade_loss"]
#         price_txt = f" • Current: {cur_price:,.4f} {self.quote_asset}" if cur_price is not None else ""
#         density_note = " • Markers thinned" if dense else ""
#         status = "ENABLED" if self.trading_enabled else "DISABLED"
#         status_color = "#15803d" if self.trading_enabled else "#b91c1c"
#         fig.update_layout(
#             title=dict(
#                 text=(
#                     f"Total PnL: {total_pnl:,.2f} {self.quote_asset} • "
#                     f"Signal: {sig}{price_txt}{density_note}<br>"
#                     f"Trading: <span style='color:{status_color}'>{status}</span>"
#                 ),
#                 x=0.5,
#                 font=dict(color=pnl_color, size=20),
#             ),
#             template="plotly_white",
#             paper_bgcolor=PALETTE["panel"],
#             plot_bgcolor=PALETTE["panel"],
#             font=dict(color=PALETTE["text"]),
#             margin=dict(l=40, r=20, t=90, b=40),
#             legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
#             hovermode="x unified",
#         )
#         fig.update_xaxes(title_text="Time", gridcolor=PALETTE["grid"], showspikes=True)
#         fig.update_yaxes(title_text=f"Price ({self.quote_asset})", gridcolor=PALETTE["grid"], showspikes=True)
#         return fig
#     # ── Win/Loss pie ----------------------------------------------------------
#     def _build_winloss_pie(self):
#         with self._dash_lock:
#             completed = list(self.completed_trades)
#         win = sum(1 for t in completed if (t.get("pnl") or 0) > 0)
#         loss = sum(1 for t in completed if (t.get("pnl") or 0) < 0)
#         if win == 0 and loss == 0:
#             win, loss = 1, 1
#         vals = [win, loss]
#         labels = ["Winning", "Losing"]
#         total = sum(vals)
#         texts = [f"{lbl} {v/total:.0%}" if v > 0 else "" for lbl, v in zip(labels, vals)]
#         fig = go.Figure(
#             data=[go.Pie(
#                 labels=labels, values=vals, hole=0.55,
#                 text=texts, textinfo="text", textposition="inside",
#                 insidetextorientation="horizontal",
#                 marker=dict(colors=[PALETTE["trade_win"], PALETTE["trade_loss"]]),
#                 sort=False
#             )]
#         )
#         fig.update_layout(
#             template="plotly_white",
#             paper_bgcolor=PALETTE["panel"],
#             plot_bgcolor=PALETTE["panel"],
#             font=dict(color=PALETTE["text"]),
#             title=dict(text="Win / Loss", x=0.5, y=0.97),
#             margin=dict(l=10, r=10, t=60, b=10),
#         )
#         return fig
#     # ── Trades table ---------------------------------------------------------
#     def _build_trades_table(self):
#         base = self.base_asset
#         quote = self.quote_asset
#         with self._dash_lock:
#             rows = []
#             for t in self.completed_trades:
#                 amt_base = t.get("amount") # This is contract size now
#                 entry_price = t.get("entry_price")
#                 amt_quote = (entry_price or 0.0) * (amt_base or 0.0) / LEVERAGE # Adjusted for leverage context
#                 rows.append({
#                     "Timestamp (open)": t["entry_time"].strftime("%Y-%m-%d %H:%M"),
#                     "Side": t["side"],
#                     "Entry": round(entry_price, 4) if entry_price is not None else "-",
#                     "Exit": round(t.get("exit_price", np.nan), 4) if t.get("exit_price") is not None else "-",
#                     "PnL": round(t.get("pnl", 0.0), 4),
#                     f"Contracts {base}": round(amt_base, 6) if amt_base is not None else "-", # Show contracts
#                     f"Value {quote}": round(amt_quote * LEVERAGE, 2) if amt_base is not None else "-", # Show notional value
#                     "SL": round(t.get("sl"), 4) if t.get("sl") is not None else "-",
#                     "TP": round(t.get("tp"), 4) if t.get("tp") is not None else "-",
#                     "Open Signal": t.get("open_signal", "-"),
#                     "Close Reason": t.get("close_reason", "signal"),
#                 })
#         cols_order = [
#             "Timestamp (open)", "Side", "Entry", "Exit", "PnL",
#             f"Contracts {base}", f"Value {quote}", "SL", "TP", "Open Signal", "Close Reason"
#         ]
#         return DataTable(
#             id="trades-table",
#             columns=[{"name": c, "id": c} for c in cols_order],
#             data=rows,
#             style_header={
#                 "backgroundColor": PALETTE["panel"],
#                 "color": PALETTE["text"],
#                 "border": f"1px solid {PALETTE['grid']}",
#                 "fontWeight": "600",
#             },
#             style_cell={
#                 "backgroundColor": PALETTE["panel"],
#                 "color": PALETTE["text"],
#                 "border": f"1px solid {PALETTE['grid']}",
#                 "padding": "8px",
#                 "fontFamily": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
#                 "textAlign": "center",
#             },
#             style_table={"overflowX": "auto"},
#             style_data_conditional=[
#                 {"if": {"filter_query": "{PnL} >= 0", "column_id": "PnL"},
#                  "color": PALETTE["trade_win"], "fontWeight": "600"},
#                 {"if": {"filter_query": "{PnL} < 0", "column_id": "PnL"},
#                  "color": PALETTE["trade_loss"], "fontWeight": "600"},
#             ],
#             page_size=100,
#         )
#     # ── Dashboard layout & callbacks -----------------------------------------
#     def _setup_dashboard(self):
#         app.bot_instance = self
#         app.layout = dbc.Container(
#             [
#                 # Row 1: Price graph (full width)
#                 dbc.Row(
#                     dbc.Col(
#                         dcc.Graph(
#                             id="live-graph",
#                             config={"displayModeBar": True},
#                             style={"height": "62vh", "border": f"1px solid {PALETTE['grid']}",
#                                    "borderRadius": "10px", "background": PALETTE["panel"]},
#                         ),
#                         width=12,
#                     ),
#                     className="g-3",
#                     style={"marginTop": "8px"}
#                 ),
#                 # Row 2: Plan (left) + Position (center) + Win/Loss (right)
#                 dbc.Row(
#                     [
#                         dbc.Col(html.Div(id="plan-card"), width=4),
#                         dbc.Col(html.Div(id="position-card"), width=4),
#                         dbc.Col(
#                             dcc.Graph(
#                                 id="win-loss-pie",
#                                 config={"displayModeBar": False},
#                                 style={"height": "36vh", "border": f"1px solid {PALETTE['grid']}",
#                                        "borderRadius": "10px", "background": PALETTE["panel"]},
#                             ),
#                             width=4,
#                         ),
#                     ],
#                     className="g-3"
#                 ),
#                 # Row 3: Trades table
#                 dbc.Row(
#                     dbc.Col(
#                         html.Div(
#                             id="trades-table-wrap",
#                             style={
#                                 "border": f"1px solid {PALETTE['grid']}",
#                                 "borderRadius": "10px",
#                                 "padding": "8px",
#                                 "background": PALETTE["panel"],
#                             },
#                         ),
#                         width=12,
#                     ),
#                     className="g-3"
#                 ),
#                 dcc.Store(id='close-button-store', data=0),
#                 dcc.Store(id='start-button-store', data=0),
#                 dcc.Interval(id="graph-update", interval=2000, n_intervals=0), # Update every 2 seconds for scalping
#                 dbc.Row(
#                     dbc.Col(
#                         html.Div(
#                             "Close the browser or press Ctrl-C to stop the bot.",
#                             style={"textAlign": "center", "marginTop": "8px", "color": PALETTE["muted"]},
#                         ),
#                         width=12,
#                     )
#                 ),
#             ],
#             fluid=True,
#             className="p-3",
#             style={"backgroundColor": PALETTE["page_bg"]}
#         )
#         @app.callback(Output("live-graph", "figure"), Input("graph-update", "n_intervals"))
#         def _update_graph(_):
#             return self._build_price_figure()
#         # NEW: one callback for Plan + Position + Pie
#         @app.callback(
#             [Output("plan-card", "children"),
#              Output("position-card", "children"),
#              Output("win-loss-pie", "figure")],
#             Input("graph-update", "n_intervals"),
#         )
#         def _update_cards(_):
#             return self._build_plan_card(), self._build_position_card(), self._build_winloss_pie()
#         # UPDATED: handle new ShortTrig / LongTrig buttons
#         @app.callback(
#             Output("trades-table-wrap", "children"),
#             [
#                 Input("graph-update", "n_intervals"),
#                 Input("close-button", "n_clicks"),
#                 Input("start-button", "n_clicks"),
#                 Input("shorttrig-button", "n_clicks"),
#                 Input("longtrig-button", "n_clicks"),
#             ],
#         )
#         def _update_table_and_handle_buttons(_, close_clicks, start_clicks, shorttrig_clicks, longtrig_clicks):
#             trig = ctx.triggered_id
#             if trig == "close-button" and close_clicks:
#                 price = self._get_price()
#                 ts = datetime.utcnow().replace(tzinfo=timezone.utc)
#                 if price is not None:
#                     self.close_position(ts, price)
#                 else:
#                     logger.warning("Could not fetch price for manual close.")
#             if trig == "start-button" and start_clicks:
#                 self.enable_trading()
#             # Manual Short trigger — open SELL immediately, regardless of signals or trading_enabled
#             if trig == "shorttrig-button" and shorttrig_clicks:
#                 price = self._get_price()
#                 ts = datetime.utcnow().replace(tzinfo=timezone.utc)
#                 if price is not None:
#                     if self.position_state == "Flat":
#                         self._execute_sell(price, ts, closed_by="manual_shorttrig")
#                         logger.info("Manual ShortTrig executed at %.6f", price)
#                     else:
#                         logger.info("ShortTrig ignored: position not flat (%s)", self.position_state)
#                 else:
#                     logger.warning("ShortTrig: no price available")
#             # Manual Long trigger — open BUY immediately, regardless of signals or trading_enabled
#             if trig == "longtrig-button" and longtrig_clicks:
#                 price = self._get_price()
#                 ts = datetime.utcnow().replace(tzinfo=timezone.utc)
#                 if price is not None:
#                     if self.position_state == "Flat":
#                         self._execute_buy(price, ts, closed_by="manual_longtrig")
#                         logger.info("Manual LongTrig executed at %.6f", price)
#                     else:
#                         logger.info("LongTrig ignored: position not flat (%s)", self.position_state)
#                 else:
#                     logger.warning("LongTrig: no price available")
#             return self._build_trades_table()
#     # ── Trade-history helpers ------------------------------------------------
#     def _get_trade_history_file(self):
#         if TRADE_HISTORY_FILE:
#             p = Path(TRADE_HISTORY_FILE)
#         else:
#             p = Path(f"futures_trade_history_{self.symbol.replace('/', '_')}_{MODE}.json")
#         p.parent.mkdir(parents=True, exist_ok=True)
#         return str(p)
#     def _load_trade_history(self):
#         fn = self._get_trade_history_file()
#         if os.path.exists(fn):
#             try:
#                 with open(fn, "r", encoding="utf-8") as f:
#                     data = json.load(f)
#                     if isinstance(data, list):
#                         print(f"Loaded {len(data)} trade records")
#                         return data
#             except Exception:
#                 logger.exception("Error loading trade history")
#         print("No trade-history file – starting fresh.")
#         return []
#     def _save_trade_history(self):
#         fn = self._get_trade_history_file()
#         try:
#             def ser(o):
#                 if isinstance(o, (datetime, pd.Timestamp)):
#                     return o.isoformat()
#                 raise TypeError
#             with open(fn, "w", encoding="utf-8") as f:
#                 json.dump(self.trade_history, f, indent=4, default=ser)
#         except Exception:
#             logger.exception("Error saving trade history")
#             print(f"{Colors.RED}Error saving trade history{Colors.END}")
#     def _log_event(self, event_type, details):
#         ts = details.get("timestamp", datetime.utcnow().isoformat())
#         entry = {"timestamp": ts, "event": event_type, "details": details}
#         self.trade_history.append(entry)
#         print(f"[{ts}] {event_type.upper()}: {details}")
#         logger.info("Trade event – %s – %s", event_type, details)
#         self._save_trade_history()
#         if event_type in ["BUY_EXECUTED", "SELL_EXECUTED"]:
#             self._log_to_csv(details)
#         return entry
#     # ── CSV logger ------------------------------------------------------------
#     def _get_csv_file(self):
#         p = Path(f"futures_trade_cycle_{self.symbol.replace('/', '_')}_{MODE}.csv")
#         p.parent.mkdir(parents=True, exist_ok=True)
#         return str(p)
#     def _log_to_csv(self, details):
#         csv_fn = self._get_csv_file()
#         file_exists = os.path.isfile(csv_fn)
#         typ = details.get("type", "")
#         is_open = typ in ["BUY_OPEN_LONG", "SELL_OPEN_SHORT"]
#         is_close = typ in ["SELL_CLOSE_LONG", "BUY_CLOSE_SHORT"]
#         if not (is_open or is_close):
#             return
#         trade_id = f"{typ}_{details.get('amount_base', 0)}_{details.get('timestamp', '')}"
#         if is_open:
#             self._open_trades_csv[trade_id] = details.copy()
#             return
#         counterpart = "BUY_OPEN_LONG" if typ == "SELL_CLOSE_LONG" else "SELL_OPEN_SHORT"
#         match_key, open_detail = None, None
#         for k, v in list(self._open_trades_csv.items()):
#             if v.get("type") == counterpart and abs(v.get("amount_base", 0) - details.get("amount_base", 0)) < 1e-9:
#                 match_key, open_detail = k, v
#                 break
#         if not open_detail:
#             logger.warning("No matching open trade for %s", typ)
#             return
#         row = {
#             "timestamp": open_detail.get("timestamp", ""),
#             "type": f"{open_detail['type'].split('_')[2]}_{typ.split('_')[2]}",
#             "symbol": self.symbol,
#             "entry_price": open_detail.get("price", ""),
#             "exit_price": details.get("price", ""),
#             "contracts": open_detail.get("amount_base", ""), # Log contracts
#             "pnl": details.get("pnl", ""),
#             "fee": (open_detail.get("fee_quote", 0) or 0) + (details.get("fee_quote", 0) or 0),
#             "is_sl_tp": details.get("closed_by", "signal"),
#         }
#         fields = ["timestamp", "type", "symbol", "entry_price", "exit_price", "contracts", "pnl", "fee", "is_sl_tp"]
#         with open(csv_fn, "a", newline="", encoding="utf-8") as f:
#             writer = csv.DictWriter(f, fieldnames=fields)
#             if not file_exists:
#                 writer.writeheader()
#                 logger.info("Created CSV %s", csv_fn)
#             writer.writerow(row)
#         self._register_completed_trade(open_detail, details)
#         self._open_trades_csv.pop(match_key, None)
#     # ── Dashboard data registration -------------------------------------------
#     def _register_completed_trade(self, open_d, close_d):
#         side = "Long" if close_d["type"] == "SELL_CLOSE_LONG" else "Short"
#         entry_price = float(open_d.get("price", np.nan))
#         sl = open_d.get("sl")
#         tp = open_d.get("tp")
#         closed_by = close_d.get("closed_by", "signal")
#         close_reason = closed_by
#         if isinstance(closed_by, str) and closed_by.startswith("signal:"):
#             close_reason = closed_by.split("signal:", 1)[1]
#         with self._dash_lock:
#             self.completed_trades.append({
#                 "side": side,
#                 "entry_time": pd.to_datetime(open_d.get("timestamp")),
#                 "entry_price": entry_price,
#                 "exit_time": pd.to_datetime(close_d.get("timestamp")),
#                 "exit_price": float(close_d.get("price", np.nan)),
#                 "amount": float(open_d.get("amount_base", np.nan)), # Store contract size
#                 "pnl": float(close_d.get("pnl", np.nan)),
#                 "sl": float(sl) if sl is not None else None,
#                 "tp": float(tp) if tp is not None else None,
#                 "open_signal": open_d.get("trigger_signal", "-"),
#                 "close_reason": close_reason,
#             })
#     # ── Balance & price helpers ----------------------------------------------
#     def _get_balance(self):
#         if MODE == "live":
#             try:
#                 return self.exchange.fetch_balance()["total"]
#             except Exception:
#                 logger.exception("Live balance fetch error")
#         return self.simulated_balance
#     def _update_balance(self, asset, delta):
#         if MODE != "live":
#             self.simulated_balance[asset] = self.simulated_balance.get(asset, 0) + delta
#     def _get_price(self):
#         try:
#             if MODE in ("live", "paper") and self.exchange:
#                 return self.exchange.fetch_ticker(self.symbol)["last"]
#         except Exception:
#             logger.exception("Price fetch error")
#         if hasattr(self, "historical_data") and not self.historical_data.empty:
#             idx = getattr(self, "current_backtest_index", None)
#             if idx is not None and 0 <= idx < len(self.historical_data):
#                 return self.historical_data["close"].iloc[idx]
#         return None
#     # ── Signal processing (TABLE-DRIVEN) --------------------------------------
#     def _process_signal(self, rec, price, ts):
#         """
#         Apply trade actions per decision table:
#         Current Pos | Signal                 | Action            | Function
#         ------------|------------------------|-------------------|------------------------------
#         Flat        | Buy / Strong Buy       | Open Long         | _execute_buy()
#         Flat        | Sell / Strong Sell     | Open Short        | _execute_sell()
#         Long        | Sell / Strong Sell     | Close Long        | _execute_sell(closed_by=signal)
#         Short       | Buy / Strong Buy       | Close Short       | _execute_buy(closed_by=signal)
#         Long        | Neutral / Buy / SB     | Hold              | —
#         Short       | Neutral / Sell / SS    | Hold              | —
#         SL/TP exits are handled earlier and take priority.
#         """
#         self.current_signal = rec or "Neutral"
#         self.last_price = price
#         print(f"{Colors.BOLD}[{ts}] SIGNAL: {rec}{Colors.END}")
#         if not self.trading_enabled:
#             print(f"{Colors.YELLOW}[{ts}] Trading is disabled. Ignoring signal: {rec}{Colors.END}")
#             return
#         rec = rec or "Neutral"
#         # When Flat: open positions on Buy/SB or Sell/SS
#         if self.position_state == "Flat":
#             if rec in ("Buy", "Strong Buy"):
#                 self._execute_buy(price, ts)  # OPEN LONG
#                 self.open_signal = rec
#                 self.seen_strong_follow_through = False
#             elif rec in ("Sell", "Strong Sell"):
#                 self._execute_sell(price, ts)  # OPEN SHORT
#                 self.open_signal = rec
#                 self.seen_strong_follow_through = False
#             # Neutral -> Hold
#             return
#         # When Long: close on Sell/Strong Sell; else Hold
#         if self.position_state == "Long":
#             if rec in ("Sell", "Strong Sell"):
#                 self._execute_sell(price, ts, closed_by=f"signal:{rec}")  # CLOSE LONG
#                 self.open_signal = None
#                 self.seen_strong_follow_through = False
#             return
#         # When Short: close on Buy/Strong Buy; else Hold
#         if self.position_state == "Short":
#             if rec in ("Buy", "Strong Buy"):
#                 self._execute_buy(price, ts, closed_by=f"signal:{rec}")   # CLOSE SHORT
#                 self.open_signal = None
#                 self.seen_strong_follow_through = False
#             return
#     # ── Execution helpers (FUTURES-SPECIFIC) -----------------------------------------------------
#     def _calculate_contract_size(self, usdt_amount, price):
#         """Calculate the number of contracts based on USD amount and leverage."""
#         # In Binance Futures, contract size is often based on the notional value
#         # For TAO, it's typically 1 TAO per contract.
#         # So, number of contracts = (USD amount * Leverage) / (Price per contract)
#         notional_value = usdt_amount * LEVERAGE
#         contracts = notional_value / price
#         return contracts

#     def _execute_buy(self, price, ts, closed_by=None):
#         if self.position_state == "Flat":
#             # OPEN LONG
#             if MODE == "live":
#                 try:
#                     invested_usdt = FIXED_ORDER_SIZE_USDT
#                     contract_size = self._calculate_contract_size(invested_usdt, price)
#                     params = {
#                         'positionSide': 'LONG', # Important for futures
#                     }
#                     order = self.exchange.create_order(
#                         symbol=self.symbol,
#                         type='market',
#                         side='buy',
#                         amount=contract_size, # Number of contracts
#                         price=None,
#                         params=params
#                     )
#                     # Process order response (get actual filled price, fees)
#                     executed_price = order['average'] if 'average' in order else order['price']
#                     executed_contracts = order['filled'] if 'filled' in order else contract_size
#                     fee = order.get('fee', {}).get('cost', 0) or sum(f.get('cost', 0) for f in order.get('fees', []))
                    
#                     self.position_state = "Long"
#                     self.entry_price = executed_price
#                     self.amount_base = executed_contracts # Store contract size
#                     # Calculate SL and TP based on 2% risk
#                     risk_amount_usdt = invested_usdt * (STOP_LOSS_PERCENT / 100)
#                     sl_distance = (risk_amount_usdt * executed_price) / (invested_usdt * LEVERAGE)
#                     self.sl_level = executed_price - sl_distance
#                     tp_distance = sl_distance * (TAKE_PROFIT_PERCENT / STOP_LOSS_PERCENT)
#                     self.tp_level = executed_price + tp_distance
                    
#                     print(f"{Colors.GREEN}[{ts}] LIVE BUY LONG @ {executed_price:.4f} – {executed_contracts:.6f} contracts{Colors.END}")
#                     print(f"   Entry: {executed_price:.4f} | SL: {self.sl_level:.4f} | TP: {self.tp_level:.4f}")
#                     trade = {
#                         "type": "BUY_OPEN_LONG",
#                         "timestamp": ts.isoformat(),
#                         "symbol": self.symbol,
#                         "price": executed_price,
#                         "amount_quote": invested_usdt,
#                         "amount_base": executed_contracts, # Log contracts
#                         "fee_quote": fee,
#                         "sl": self.sl_level,
#                         "tp": self.tp_level,
#                         "trigger_signal": self.current_signal or "Buy",
#                     }
#                     self._log_event("BUY_EXECUTED", trade)
#                 except Exception as e:
#                     logger.error(f"Error placing live LONG order: {e}")
#                     print(f"{Colors.RED}Error placing live LONG order: {e}{Colors.END}")
            
#             elif MODE in ["paper", "backtest"]:
#                 usdt = self.simulated_balance.get(self.quote_asset, 0)
#                 spend = min(FIXED_ORDER_SIZE_USDT, usdt)
#                 if spend <= 0:
#                     print("Insufficient USDT to open long.")
#                     return
#                 contract_size = self._calculate_contract_size(spend, price)
#                 # Simulate fee (futures fees are usually lower)
#                 fee = spend * 0.0002 # Example 0.02% taker fee
#                 self._update_balance(self.quote_asset, -spend)
                
#                 self.position_state = "Long"
#                 self.entry_price = price
#                 self.amount_base = contract_size # Store contract size
                
#                 # Calculate SL and TP based on 2% risk
#                 risk_amount_usdt = spend * (STOP_LOSS_PERCENT / 100)
#                 sl_distance = (risk_amount_usdt * price) / (spend * LEVERAGE)
#                 self.sl_level = price - sl_distance
#                 tp_distance = sl_distance * (TAKE_PROFIT_PERCENT / STOP_LOSS_PERCENT)
#                 self.tp_level = price + tp_distance
                
#                 print(f"{Colors.GREEN}[{ts}] {'PAPER' if MODE == 'paper' else 'BACKTEST'} BUY LONG @ {price:.4f} – {contract_size:.6f} contracts{Colors.END}")
#                 print(f"   Entry: {price:.4f} | SL: {self.sl_level:.4f} | TP: {self.tp_level:.4f}")
#                 trade = {
#                     "type": "BUY_OPEN_LONG",
#                     "timestamp": ts.isoformat(),
#                     "symbol": self.symbol,
#                     "price": price,
#                     "amount_quote": spend,
#                     "amount_base": contract_size, # Log contracts
#                     "fee_quote": fee,
#                     "sl": self.sl_level,
#                     "tp": self.tp_level,
#                     "trigger_signal": self.current_signal or "Buy",
#                 }
#                 self._log_event("BUY_EXECUTED", trade)

#         elif self.position_state == "Short":
#             # CLOSE SHORT
#             amount = self.amount_base # This is the contract size
#             if MODE == "live":
#                 try:
#                     params = {
#                         'positionSide': 'SHORT', # Important for futures
#                     }
#                     order = self.exchange.create_order(
#                         symbol=self.symbol,
#                         type='market',
#                         side='buy', # Buy to close a short
#                         amount=amount, # Number of contracts to close
#                         price=None,
#                         params=params
#                     )
#                     # Process order response
#                     executed_price = order['average'] if 'average' in order else order['price']
#                     fee = order.get('fee', {}).get('cost', 0) or sum(f.get('cost', 0) for f in order.get('fees', []))
                    
#                     # PnL for closing a short: (Entry Price - Exit Price) * Contracts
#                     pnl = (self.entry_price - executed_price) * amount
                    
#                     self.position_state = "Flat"
#                     self.entry_price = None
#                     self.amount_base = 0.0
#                     self.sl_level = None
#                     self.tp_level = None
                    
#                     print(f"{Colors.GREEN}[{ts}] LIVE BUY CLOSE SHORT @ {executed_price:.4f}{Colors.END}")
#                     print(f"   PnL: {pnl:.4f} {self.quote_asset}")
#                     trade = {
#                         "type": "BUY_CLOSE_SHORT",
#                         "timestamp": ts.isoformat(),
#                         "symbol": self.symbol,
#                         "price": executed_price,
#                         "amount_base": amount, # Contracts closed
#                         "pnl": pnl,
#                         "trigger_signal": "Buy",
#                         "closed_by": closed_by or "signal",
#                         "fee_quote": fee,
#                     }
#                     self._log_event("BUY_EXECUTED", trade)
#                 except Exception as e:
#                     logger.error(f"Error placing live BUY to close SHORT order: {e}")
#                     print(f"{Colors.RED}Error placing live BUY to close SHORT order: {e}{Colors.END}")
            
#             elif MODE in ["paper", "backtest"]:
#                 # Simulate closing the short
#                 received_usdt = amount * price # Value of contracts closed
#                 # Simulate fee
#                 fee = received_usdt * 0.0002 # Example 0.02% taker fee
#                 received_usdt -= fee
                
#                 # PnL for closing a short: (Entry Price - Exit Price) * Contracts
#                 pnl = (self.entry_price - price) * amount
                
#                 # Update balance (add the USD value received)
#                 self._update_balance(self.quote_asset, received_usdt)
                
#                 self.position_state = "Flat"
#                 self.entry_price = None
#                 self.amount_base = 0.0
#                 self.sl_level = None
#                 self.tp_level = None
                
#                 print(f"{Colors.GREEN}[{ts}] {'PAPER' if MODE == 'paper' else 'BACKTEST'} BUY CLOSE SHORT @ {price:.4f}{Colors.END}")
#                 print(f"   PnL: {pnl:.4f} {self.quote_asset}")
#                 trade = {
#                     "type": "BUY_CLOSE_SHORT",
#                     "timestamp": ts.isoformat(),
#                     "symbol": self.symbol,
#                     "price": price,
#                     "amount_base": amount, # Contracts closed
#                     "pnl": pnl,
#                     "trigger_signal": "Buy",
#                     "closed_by": closed_by or "signal",
#                     "fee_quote": fee,
#                 }
#                 self._log_event("BUY_EXECUTED", trade)

#     def _execute_sell(self, price, ts, closed_by=None):
#         if self.position_state == "Flat":
#             # OPEN SHORT
#             if MODE == "live":
#                 try:
#                     invested_usdt = FIXED_ORDER_SIZE_USDT
#                     contract_size = self._calculate_contract_size(invested_usdt, price)
#                     params = {
#                         'positionSide': 'SHORT', # Important for futures
#                     }
#                     order = self.exchange.create_order(
#                         symbol=self.symbol,
#                         type='market',
#                         side='sell',
#                         amount=contract_size, # Number of contracts
#                         price=None,
#                         params=params
#                     )
#                     # Process order response
#                     executed_price = order['average'] if 'average' in order else order['price']
#                     executed_contracts = order['filled'] if 'filled' in order else contract_size
#                     fee = order.get('fee', {}).get('cost', 0) or sum(f.get('cost', 0) for f in order.get('fees', []))
                    
#                     self.position_state = "Short"
#                     self.entry_price = executed_price
#                     self.amount_base = executed_contracts # Store contract size
                    
#                     # Calculate SL and TP based on 2% risk
#                     risk_amount_usdt = invested_usdt * (STOP_LOSS_PERCENT / 100)
#                     sl_distance = (risk_amount_usdt * executed_price) / (invested_usdt * LEVERAGE)
#                     self.sl_level = executed_price + sl_distance # SL is higher for shorts
#                     tp_distance = sl_distance * (TAKE_PROFIT_PERCENT / STOP_LOSS_PERCENT)
#                     self.tp_level = executed_price - tp_distance # TP is lower for shorts
                    
#                     print(f"{Colors.RED}[{ts}] LIVE SELL SHORT @ {executed_price:.4f} – {executed_contracts:.6f} contracts{Colors.END}")
#                     print(f"   Entry: {executed_price:.4f} | SL: {self.sl_level:.4f} | TP: {self.tp_level:.4f}")
#                     trade = {
#                         "type": "SELL_OPEN_SHORT",
#                         "timestamp": ts.isoformat(),
#                         "symbol": self.symbol,
#                         "price": executed_price,
#                         "amount_base": executed_contracts, # Log contracts
#                         "fee_quote": fee,
#                         "sl": self.sl_level,
#                         "tp": self.tp_level,
#                         "trigger_signal": self.current_signal or "Sell",
#                     }
#                     self._log_event("SELL_EXECUTED", trade)
#                 except Exception as e:
#                     logger.error(f"Error placing live SHORT order: {e}")
#                     print(f"{Colors.RED}Error placing live SHORT order: {e}{Colors.END}")
            
#             elif MODE in ["paper", "backtest"]:
#                 base_bal = self.simulated_balance.get(self.base_asset, 0) # Not directly used for futures margin, but for tracking
#                 # In futures, we use USD margin, not base asset balance
#                 usdt_bal = self.simulated_balance.get(self.quote_asset, 0)
#                 spend_usdt = min(FIXED_ORDER_SIZE_USDT, usdt_bal)
#                 if spend_usdt <= 0:
#                     print("Insufficient USDT margin to open short.")
#                     return
#                 contract_size = self._calculate_contract_size(spend_usdt, price)
#                 # Simulate fee
#                 fee = spend_usdt * 0.0002 # Example 0.02% taker fee
#                 self._update_balance(self.quote_asset, -spend_usdt)
                
#                 self.position_state = "Short"
#                 self.entry_price = price
#                 self.amount_base = contract_size # Store contract size
                
#                 # Calculate SL and TP based on 2% risk
#                 risk_amount_usdt = spend_usdt * (STOP_LOSS_PERCENT / 100)
#                 sl_distance = (risk_amount_usdt * price) / (spend_usdt * LEVERAGE)
#                 self.sl_level = price + sl_distance # SL is higher for shorts
#                 tp_distance = sl_distance * (TAKE_PROFIT_PERCENT / STOP_LOSS_PERCENT)
#                 self.tp_level = price - tp_distance # TP is lower for shorts
                
#                 print(f"{Colors.RED}[{ts}] {'PAPER' if MODE == 'paper' else 'BACKTEST'} SELL SHORT @ {price:.4f} – {contract_size:.6f} contracts{Colors.END}")
#                 print(f"   Entry: {price:.4f} | SL: {self.sl_level:.4f} | TP: {self.tp_level:.4f}")
#                 trade = {
#                     "type": "SELL_OPEN_SHORT",
#                     "timestamp": ts.isoformat(),
#                     "symbol": self.symbol,
#                     "price": price,
#                     "amount_base": contract_size, # Log contracts
#                     "fee_quote": fee,
#                     "sl": self.sl_level,
#                     "tp": self.tp_level,
#                     "trigger_signal": self.current_signal or "Sell",
#                 }
#                 self._log_event("SELL_EXECUTED", trade)

#         elif self.position_state == "Long":
#             # CLOSE LONG
#             amount = self.amount_base # This is the contract size
#             if MODE == "live":
#                 try:
#                     params = {
#                         'positionSide': 'LONG', # Important for futures
#                     }
#                     order = self.exchange.create_order(
#                         symbol=self.symbol,
#                         type='market',
#                         side='sell', # Sell to close a long
#                         amount=amount, # Number of contracts to close
#                         price=None,
#                         params=params
#                     )
#                     # Process order response
#                     executed_price = order['average'] if 'average' in order else order['price']
#                     fee = order.get('fee', {}).get('cost', 0) or sum(f.get('cost', 0) for f in order.get('fees', []))
                    
#                     # PnL for closing a long: (Exit Price - Entry Price) * Contracts
#                     pnl = (executed_price - self.entry_price) * amount
                    
#                     self.position_state = "Flat"
#                     self.entry_price = None
#                     self.amount_base = 0.0
#                     self.sl_level = None
#                     self.tp_level = None
                    
#                     print(f"{Colors.RED}[{ts}] LIVE SELL CLOSE LONG @ {executed_price:.4f}{Colors.END}")
#                     print(f"   PnL: {pnl:.4f} {self.quote_asset}")
#                     trade = {
#                         "type": "SELL_CLOSE_LONG",
#                         "timestamp": ts.isoformat(),
#                         "symbol": self.symbol,
#                         "price": executed_price,
#                         "amount_base": amount, # Contracts closed
#                         "pnl": pnl,
#                         "trigger_signal": "Sell",
#                         "closed_by": closed_by or "signal",
#                         "fee_quote": fee,
#                     }
#                     self._log_event("SELL_EXECUTED", trade)
#                 except Exception as e:
#                     logger.error(f"Error placing live SELL to close LONG order: {e}")
#                     print(f"{Colors.RED}Error placing live SELL to close LONG order: {e}{Colors.END}")
            
#             elif MODE in ["paper", "backtest"]:
#                 # Simulate closing the long
#                 received_usdt = amount * price # Value of contracts closed
#                 # Simulate fee
#                 fee = received_usdt * 0.0002 # Example 0.02% taker fee
#                 received_usdt -= fee
                
#                 # PnL for closing a long: (Exit Price - Entry Price) * Contracts
#                 pnl = (price - self.entry_price) * amount
                
#                 # Update balance (add the USD value received)
#                 self._update_balance(self.quote_asset, received_usdt)
                
#                 self.position_state = "Flat"
#                 self.entry_price = None
#                 self.amount_base = 0.0
#                 self.sl_level = None
#                 self.tp_level = None
                
#                 print(f"{Colors.RED}[{ts}] {'PAPER' if MODE == 'paper' else 'BACKTEST'} SELL CLOSE LONG @ {price:.4f}{Colors.END}")
#                 print(f"   PnL: {pnl:.4f} {self.quote_asset}")
#                 trade = {
#                     "type": "SELL_CLOSE_LONG",
#                     "timestamp": ts.isoformat(),
#                     "symbol": self.symbol,
#                     "price": price,
#                     "amount_base": amount, # Contracts closed
#                     "pnl": pnl,
#                     "trigger_signal": "Sell",
#                     "closed_by": closed_by or "signal",
#                     "fee_quote": fee,
#                 }
#                 self._log_event("SELL_EXECUTED", trade)

#     # ── SL/TP checks (Remain Largely the Same, Logic is Correct for Futures) ----------------------------------------------------------
#     def _check_sl_tp_on_candle(self, candle_ts, high, low):
#         if self.position_state == "Flat" or self.entry_price is None:
#             return False
#         eps = 1e-9
#         if self.position_state == "Long":
#             if self.sl_level is not None and low <= self.sl_level + eps:
#                 self._execute_sell(self.sl_level, candle_ts, closed_by="SL")
#                 self.open_signal = None; self.seen_strong_follow_through = False
#                 return True
#             if self.tp_level is not None and high >= self.tp_level - eps:
#                 self._execute_sell(self.tp_level, candle_ts, closed_by="TP")
#                 self.open_signal = None; self.seen_strong_follow_through = False
#                 return True
#         elif self.position_state == "Short":
#             if self.sl_level is not None and high >= self.sl_level - eps:
#                 self._execute_buy(self.sl_level, candle_ts, closed_by="SL")
#                 self.open_signal = None; self.seen_strong_follow_through = False
#                 return True
#             if self.tp_level is not None and low <= self.tp_level + eps:
#                 self._execute_buy(self.tp_level, candle_ts, closed_by="TP")
#                 self.open_signal = None; self.seen_strong_follow_through = False
#                 return True
#         return False
#     def _check_sl_tp_on_tick(self, ts, price):
#         if self.position_state == "Flat" or self.entry_price is None:
#             return False
#         if self.position_state == "Long":
#             if self.sl_level is not None and price <= self.sl_level:
#                 self._execute_sell(self.sl_level, ts, closed_by="SL")
#                 self.open_signal = None; self.seen_strong_follow_through = False
#                 return True
#             if self.tp_level is not None and price >= self.tp_level:
#                 self._execute_sell(self.tp_level, ts, closed_by="TP")
#                 self.open_signal = None; self.seen_strong_follow_through = False
#                 return True
#         else: # Short position
#             if self.sl_level is not None and price >= self.sl_level:
#                 self._execute_buy(self.sl_level, ts, closed_by="SL")
#                 self.open_signal = None; self.seen_strong_follow_through = False
#                 return True
#             if self.tp_level is not None and price <= self.tp_level:
#                 self._execute_buy(self.tp_level, ts, closed_by="TP")
#                 self.open_signal = None; self.seen_strong_follow_through = False
#                 return True
#         return False
#     # ── Live candle utilities -------------------------------------------------
#     @staticmethod
#     def _tf_seconds(tf: str) -> int:
#         if tf.endswith("m"):
#             return int(tf[:-1]) * 60
#         if tf.endswith("h"):
#             return int(tf[:-1]) * 3600
#         if tf.endswith("d"):
#             return int(tf[:-1]) * 86400
#         if tf.endswith("w"):
#             return int(tf[:-1]) * 7 * 86400
#         return 60
#     @staticmethod
#     def _floor_time(ts: datetime, seconds: int) -> datetime:
#         epoch = int(ts.replace(tzinfo=timezone.utc).timestamp())
#         floored = epoch - (epoch % seconds)
#         return datetime.fromtimestamp(floored, tz=timezone.utc)
#     def _prime_live_candles(self, limit: int = 200):
#         if not self.live_fetcher:
#             return
#         try:
#             df = self.live_fetcher.fetch_fast_data(self.primary_tf, limit=limit)
#             if df is not None and not df.empty:
#                 with self._dash_lock:
#                     self.candle_history = [
#                         {"timestamp": ts.to_pydatetime().replace(tzinfo=timezone.utc),
#                          "open": float(r["open"]), "high": float(r["high"]),
#                          "low": float(r["low"]), "close": float(r["close"])}
#                         for ts, r in df.iterrows()
#                     ]
#         except Exception:
#             logger.exception("Prime live candles failed")
#     def _update_live_candle(self, now: datetime, price: float):
#         secs = self._tf_seconds(self.primary_tf)
#         bucket = self._floor_time(now, secs)
#         with self._dash_lock:
#             if not self.candle_history:
#                 self.candle_history.append(
#                     {"timestamp": bucket, "open": price, "high": price, "low": price, "close": price}
#                 )
#                 return
#             last = self.candle_history[-1]
#             last_bucket = last["timestamp"]
#             if isinstance(last_bucket, pd.Timestamp):
#                 last_bucket = last_bucket.to_pydatetime().replace(tzinfo=timezone.utc)
#             if bucket > last_bucket:
#                 self.candle_history.append(
#                     {"timestamp": bucket, "open": price, "high": price, "low": price, "close": price}
#                 )
#             else:
#                 last["close"] = price
#                 if price > last["high"]:
#                     last["high"] = price
#                 if price < last["low"]:
#                     last["low"] = price
#     # ── Back-test driver (Needs adjustments for futures logic, simplified here) -------------------------------------------------------
#     def run_backtest(self, timeframe=None, days=30):
#         tf = timeframe or self.primary_tf
#         print(f"{Colors.MAGENTA}Futures Back-test start – {self.symbol} – TF={tf} – {days}d{Colors.END}")
#         logger.info("Futures Back-test start – %s – %s – %d days", self.symbol, tf, days)
#         # Note: Backtesting futures with precise contract mechanics is complex.
#         # This is a simplified version using spot-like logic for demonstration.
#         # A real backtest would need historical futures data including funding rates, precise contract values, etc.
#         self.historical_data = self.backtest_fetcher.fetch_historical_data(tf, days)
#         if self.historical_data.empty:
#             print("No data – abort."); return
#         tf_map = {t: days + 20 for t in ["1m", "5m", "15m", "1h"]} # Focus on shorter timeframes for scalping
#         all_tf = fetch_multiple_timeframes_concurrently(self.backtest_fetcher, tf_map)
#         warmup = {"1m": 500, "5m": 200, "15m": 150, "1h": 100}.get(tf, 100)
#         if len(self.historical_data) <= warmup:
#             print("Not enough candles for warm-up."); return
#         start_idx = warmup
#         self._start_dash_server()
#         start_price = self.historical_data["close"].iloc[0]
#         peak_val = self.simulated_balance[self.quote_asset] # + self.simulated_balance[self.base_asset] * start_price (Not used in futures)
#         max_dd = 0.0
#         for i in range(start_idx, len(self.historical_data)):
#             self.current_backtest_index = i
#             candle = self.historical_data.iloc[i]
#             ts = candle.name
#             open_ = candle["open"]; high = candle["high"]; low = candle["low"]; close = candle["close"]
#             with self._dash_lock:
#                 self.candle_history.append(
#                     {"timestamp": ts, "open": open_, "high": high, "low": low, "close": close}
#                 )
#                 # Portfolio value in futures is just USDT balance for this simulation
#                 port_val = self.simulated_balance[self.quote_asset] # Simplified
#                 self.portfolio_history.append({"timestamp": ts, "price": close, "portfolio_value": port_val})
#                 self.last_price = close
#             # SL/TP first
#             if self._check_sl_tp_on_candle(ts, high, low):
#                 pass
#             # Build slices for strategy (simplified)
#             slices = {k: v[v.index <= ts] for k, v in all_tf.items()}
            
#             # Plan for cards: compute from slices (if available)
#             try:
#                 df1h = slices.get("1h", pd.DataFrame()).tail(100)
#                 df5m = slices.get("5m", pd.DataFrame()).tail(100)
#                 if not df1h.empty and not df5m.empty:
#                     news_score = 0.0 # Simplified
#                     self.last_plan = compute_scalping_plan_1m_5m(df5m, df1h, sentiment=news_score) # Order swapped for new function
#             except Exception:
#                 logger.exception("Plan compute failed (backtest)")
            
#             # Use the fast scalping analyzer for signal
#             temp_analyzer = FastScalpingAnalyzer(self.symbol)
#             # Mock the fetcher to use slices
#             temp_analyzer.data_fetcher = type(
#                 "SliceFetcher",
#                 (),
#                 {"fetch_fast_data": lambda self, tf, limit=100: slices.get(tf, pd.DataFrame()).tail(limit)},
#             )()
            
#             try:
#                 signal, details = temp_analyzer.get_scalping_signal()
#                 signals = {'1m': {'overall_recommendation': signal, 'details': details}}
#             except Exception:
#                 logger.exception("Signal error at %s", ts)
#                 signals = {}
                
#             rec = signals.get(tf, {}).get("overall_recommendation", "")
#             self._process_signal(rec, close, ts)
            
#             if port_val > peak_val:
#                 peak_val = port_val
#             else:
#                 dd = (peak_val - port_val) / peak_val * 100 if peak_val > 0 else 0
#                 max_dd = max(max_dd, dd)
                
#         final_price = self.historical_data["close"].iloc[-1]
#         final_ts = self.historical_data.index[-1]
#         final_port = self.simulated_balance[self.quote_asset] # Simplified
#         with self._dash_lock:
#             self.portfolio_history.append(
#                 {"timestamp": final_ts, "price": final_price, "portfolio_value": final_port}
#             )
#         net = final_port - peak_val # This needs careful consideration for futures PnL
#         net_pct = net / peak_val * 100 if peak_val > 0 else 0
#         print(f"\n{Colors.CYAN}Back-test completed – Net PnL: {net:.2f} ({net_pct:.2f}%) – Max DD: {max_dd:.2f}%{Colors.END}")
#         logger.info("Back-test finished – Net PnL %.2f – Max DD %.2f", net, max_dd)
#         self._save_trade_history()
#     # ── Live / paper loop -----------------------------------------------------
#     def run_live_paper(self):
#         print(f"{Colors.CYAN}Futures Live/Paper mode – TF={self.primary_tf}{Colors.END}")
#         self._start_dash_server()
#         self._prime_live_candles(limit=200)
#         iteration = 0
#         while True:
#             iteration += 1
#             print(f"\n--- Iteration {iteration} ---")
#             price = self._get_price()
#             if price is None:
#                 print("No price – wait.")
#                 time.sleep(3)
#                 continue
#             now = datetime.utcnow().replace(tzinfo=timezone.utc)
#             self.last_price = price
#             self._update_live_candle(now, price)
#             port_val = self.simulated_balance[self.quote_asset] # Simplified for paper
#             with self._dash_lock:
#                 self.portfolio_history.append(
#                     {"timestamp": now, "price": price, "portfolio_value": port_val}
#                 )
#             # update plan regularly for cards
#             try:
#                 df_1m = self.live_fetcher.fetch_fast_data("1m", limit=100)
#                 df_5m = self.live_fetcher.fetch_fast_data("5m", limit=100)
#                 news_score = 0.0 # Simplified
#                 self.last_plan = compute_scalping_plan_1m_5m(df_1m, df_5m, sentiment=news_score)
#             except Exception:
#                 logger.exception("Plan compute failed (live/paper)")
                
#             if self._check_sl_tp_on_tick(now, price):
#                 time.sleep(1) # Slightly faster check for scalping
#                 continue
                
#             signals = self.strategy.get_signals()
#             rec = signals.get(self.primary_tf, {}).get("overall_recommendation", "")
#             self._process_signal(rec, price, now)
#             time.sleep(1) # Slightly faster loop for scalping (was 2)
#     # ── Server helpers --------------------------------------------------------
#     def _start_dash_server(self):
#         if self._dash_server:
#             return
#         # Optional: graceful shutdown endpoint to match _shutdown_dash_server()
#         @app.server.route("/_shutdown")
#         def _shutdown():
#             try:
#                 from flask import request
#                 func = request.environ.get('werkzeug.server.shutdown')
#                 if func:
#                     func()
#             except Exception:
#                 pass
#             return "Shutting down..."
#         def run():
#             self._dash_server = app.server
#             logger.info("Dash server started – http://127.0.0.1:8050")
#             app.run(host="127.0.0.1", port=8050, debug=False, use_reloader=False)
#         self._dash_thread = threading.Thread(target=run, daemon=True)
#         self._dash_thread.start()
#         print("\nDashboard launched – open http://127.0.0.1:8050")
#         logger.info("Dashboard launched")
#     def _shutdown_dash_server(self):
#         if self._dash_server:
#             try:
#                 requests.get("http://127.0.0.1:8050/_shutdown", timeout=1)
#                 logger.info("Dashboard shutdown request sent")
#             except Exception:
#                 logger.debug("Dashboard shutdown endpoint not available")
#             finally:
#                 self._dash_server = None
#     # ── Runner ---------------------------------------------------------------
#     def run(self, backtest_tf=None, backtest_days=30):
#         if MODE == "backtest":
#             self.run_backtest(backtest_tf, backtest_days)
#         else:
#             self.run_live_paper()
#     def stop(self):
#         print("Stopping futures bot...")
#         self._save_trade_history()
#         self._shutdown_dash_server()
#         logger.info("Futures bot stopped")
#         print("Futures bot stopped.")
# # ── CLI ----------------------------------------------------------------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Futures Scalping Trading Bot")
#     parser.add_argument("--timeframe", default="1m", choices=["1m", "5m", "15m", "1h"]) # Default to 1m for scalping
#     parser.add_argument("--days", type=int, default=30)
#     parser.add_argument("--symbol", default=SYMBOL)
#     parser.add_argument("--no-sentiment", dest="no_sentiment", action="store_true",
#                         help="Disable sentiment bias in plan")
#     args = parser.parse_args()
#     bot = FuturesScalpingBot(symbol=args.symbol, primary_tf=args.timeframe)
#     bot.use_sentiment = not args.no_sentiment  # toggle sentiment used in plan
#     try:
#         bot.run(backtest_tf=args.timeframe, backtest_days=args.days)
#     except KeyboardInterrupt:
#         print(f"{Colors.YELLOW}Interrupted by user.{Colors.END}")
#         logger.info("Interrupted by user")
#     finally:
#         bot.stop()




# bot.py — Futures scalper with Dash UI (Plan • Position • Win/Loss • Trades)
import sys, os, json, csv, time, atexit, threading, argparse, logging, re, requests
import pandas as pd, numpy as np, ccxt
from datetime import datetime, timezone
from pathlib import Path
from logging.handlers import RotatingFileHandler

from config import (
    SYMBOL, INITIAL_USDT, FIXED_ORDER_SIZE_USDT, MODE,
    STOP_LOSS_PERCENT, TAKE_PROFIT_PERCENT, LOG_FILE, TRADE_HISTORY_FILE,
    FUTURES_EXCHANGE_ID, LEVERAGE, FUTURES_TAKER_FEE_BPS, USE_FUTURES,
    API_KEY, API_SECRET
)
from technical_analyzer import QuickScalpAnalyzer, Colors

import dash, dash_bootstrap_components as dbc
from dash import dcc, html, ctx
from dash.dependencies import Output, Input
from dash.dash_table import DataTable
import plotly.graph_objs as go

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
PALETTE = {
    "page_bg": "#f7f9fc","panel": "#ffffff","text": "#1f2937","muted": "#6b7280",
    "grid": "#e5e7eb","accent": "#0d6efd","entry": "#22c55e","exit": "#ef4444",
    "candle_up": "#16a34a","candle_down": "#dc2626","trade_win": "#15803d",
    "trade_loss": "#b91c1c","halo": "rgba(0,0,0,0.08)","button_bg": "#dc3545",
    "button_hover": "#bb2d3b","start_button_bg": "#198754","start_button_hover": "#157347",
}
MAX_MARKERS_ON_CHART = 120; SPARSE_MARKER_SIZE = 10; DENSE_MARKER_SIZE = 6

def setup_logging(log_path: str | Path):
    log_path = Path(log_path) if log_path else Path("trading_bot.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(); logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s","%Y-%m-%d %H:%M:%S")
    class AnsiStrippingFileHandler(RotatingFileHandler):
        ansi_re = re.compile(r"\x1b\[[0-9;]*m")
        def emit(self, record):
            if isinstance(record.msg, str): record.msg = self.ansi_re.sub("", record.msg)
            super().emit(record)
    fh = AnsiStrippingFileHandler(log_path, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt); fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(); ch.setFormatter(fmt); ch.setLevel(logging.INFO)
    logger.handlers.clear(); logger.addHandler(fh); logger.addHandler(ch)
    class _LoggerWriter:
        def __init__(self, level_func): self.level_func = level_func
        def write(self, message):
            if message.strip():
                for line in message.rstrip().splitlines(): self.level_func(line)
        def flush(self): pass
    sys.stdout = _LoggerWriter(logger.info); sys.stderr = _LoggerWriter(logger.error)
setup_logging(LOG_FILE); logger = logging.getLogger(__name__)
logger.info("Logger initialised – MODE=%s – SYMBOL=%s – Futures=%s", MODE, SYMBOL, USE_FUTURES)

class TradingStrategy:
    def __init__(self, symbol):
        self.quick = QuickScalpAnalyzer(symbol, exchange_id=FUTURES_EXCHANGE_ID, api_key=API_KEY, api_secret=API_SECRET)
    def quick_signal(self):
        try: return self.quick.quick_decision()
        except Exception:
            logger.exception("quick_decision failed")
            return type("S", (), {"signal": "Neutral", "debug": {"reason": "exception"}})()

class FuturesScalperBot:
    def __init__(self, symbol):
        self.symbol = symbol
        self.base_asset, self.quote_asset = symbol.split("/")
        self.leverage = LEVERAGE; self.taker_fee_bps = FUTURES_TAKER_FEE_BPS
        self.equity_usdt = INITIAL_USDT; self.margin_locked = 0.0
        self.position_side = "Flat"; self.entry_price = None; self.qty_base = 0.0
        self.sl_level = None; self.tp_level = None
        self.trading_enabled = True; self.last_price = None; self.current_signal = "Neutral"
        self.completed_trades = []; self.portfolio_history = []; self.candle_history = []
        self.strategy = TradingStrategy(symbol)
        self.exchange = None; self.symbol_ex = symbol
        if MODE in ["paper", "live"]:
            if USE_FUTURES:
                ctor = getattr(ccxt, FUTURES_EXCHANGE_ID)
                self.exchange = ctor({'enableRateLimit': True,'options': {'defaultType': 'future'},
                                      'apiKey': API_KEY,'secret': API_SECRET,'timeout': 30000})
            else:
                self.exchange = ccxt.binance({'enableRateLimit': True, 'timeout': 30000})
            try:
                self.exchange.load_markets()
                self.symbol_ex = self._normalize_symbol(self.exchange, self.symbol)
                if USE_FUTURES:
                    try: self.exchange.set_leverage(self.leverage, self.symbol_ex)
                    except Exception: logger.info("set_leverage not supported or already set")
            except Exception: logger.exception("Exchange init failed")
        self._dash_lock = threading.Lock(); self._dash_server = None; self._dash_thread = None
        atexit.register(self._shutdown_dash_server)
        print(f"Bot initialised for {self.symbol_ex} – mode={MODE} – lev={self.leverage}x")
        self._setup_dashboard()

    @staticmethod
    def _normalize_symbol(ex, symbol):
        if symbol in getattr(ex, 'markets', {}): return symbol
        flat = symbol.replace('/', '')
        for s in getattr(ex, 'symbols', []):
            if s.replace('/', '') == flat: return s
        return symbol

    def _notional_per_trade(self) -> float:
        return FIXED_ORDER_SIZE_USDT * self.leverage

    def _qty_for_price(self, price: float):
        notional = self._notional_per_trade()
        qty = notional / price
        return qty, notional

    def _taker_fee_quote(self, notional: float) -> float:
        return notional * (self.taker_fee_bps / 10_000.0)

    def _pnl_quote(self, entry, exit, qty, side):
        return (exit - entry) * qty if side == "Long" else (entry - exit) * qty

    def _get_price(self):
        try:
            if self.exchange: return self.exchange.fetch_ticker(self.symbol_ex)['last']
        except Exception: logger.exception("Price fetch error")
        return None

    def _open_long(self, price, ts, trigger="signal"):
        if self.position_side != "Flat": return
        qty, notional = self._qty_for_price(price); fee = self._taker_fee_quote(notional)
        margin = notional / self.leverage
        if MODE != "live":
            if self.equity_usdt < (margin + fee): print("Insufficient equity to open long."); return
            self.equity_usdt -= (margin + fee); self.margin_locked = margin
        self.position_side = "Long"; self.entry_price = price; self.qty_base = qty
        self.sl_level = price * (1 - STOP_LOSS_PERCENT / 100.0)
        self.tp_level = price * (1 + TAKE_PROFIT_PERCENT / 100.0) if TAKE_PROFIT_PERCENT > 0 else None
        logger.info("OPEN LONG @ %.4f qty=%.6f notional=%.2f fee=%.4f", price, qty, notional, fee)
        self._log_event("FUTURES_OPEN_LONG", {
            "timestamp": ts.isoformat(),"price": price,"qty": qty,"notional": notional,
            "fee_quote": fee,"sl": self.sl_level,"tp": self.tp_level,"trigger": trigger
        })

    def _open_short(self, price, ts, trigger="signal"):
        if self.position_side != "Flat": return
        qty, notional = self._qty_for_price(price); fee = self._taker_fee_quote(notional)
        margin = notional / self.leverage
        if MODE != "live":
            if self.equity_usdt < (margin + fee): print("Insufficient equity to open short."); return
            self.equity_usdt -= (margin + fee); self.margin_locked = margin
        self.position_side = "Short"; self.entry_price = price; self.qty_base = qty
        self.sl_level = price * (1 + STOP_LOSS_PERCENT / 100.0)
        self.tp_level = price * (1 - TAKE_PROFIT_PERCENT / 100.0) if TAKE_PROFIT_PERCENT > 0 else None
        logger.info("OPEN SHORT @ %.4f qty=%.6f notional=%.2f fee=%.4f", price, qty, notional, fee)
        self._log_event("FUTURES_OPEN_SHORT", {
            "timestamp": ts.isoformat(),"price": price,"qty": qty,"notional": notional,
            "fee_quote": fee,"sl": self.sl_level,"tp": self.tp_level,"trigger": trigger
        })

    def _close_position(self, price, ts, closed_by="signal"):
        if self.position_side == "Flat": return
        side = self.position_side; qty = self.qty_base; entry = self.entry_price
        notional_exit = price * qty; fee_exit = self._taker_fee_quote(notional_exit)
        pnl = self._pnl_quote(entry, price, qty, side)
        if MODE != "live":
            self.equity_usdt += self.margin_locked; self.equity_usdt += pnl; self.equity_usdt -= fee_exit
        self._register_completed_trade(
            {"timestamp": ts.isoformat(),"price": entry,"amount_base": qty,
             "sl": self.sl_level,"tp": self.tp_level,"trigger_signal": side},
            {"timestamp": ts.isoformat(),"price": price,"amount_base": qty,
             "pnl": pnl - fee_exit,"closed_by": closed_by,
             "type": "SELL_CLOSE_LONG" if side == "Long" else "BUY_CLOSE_SHORT"}
        )
        self._log_event("FUTURES_CLOSE", {
            "timestamp": ts.isoformat(),"side": side,"price": price,"qty": qty,
            "pnl": pnl,"fee_quote": fee_exit,"closed_by": closed_by
        })
        self.position_side = "Flat"; self.entry_price = None; self.qty_base = 0.0
        self.sl_level = None; self.tp_level = None; self.margin_locked = 0.0

    def _check_sl_tp_on_tick(self, ts, price):
        if self.position_side == "Flat" or self.entry_price is None: return False
        if self.position_side == "Long":
            if price <= self.sl_level: self._close_position(self.sl_level, ts, closed_by="SL"); return True
            if self.tp_level and price >= self.tp_level: self._close_position(self.tp_level, ts, closed_by="TP"); return True
        else:
            if price >= self.sl_level: self._close_position(self.sl_level, ts, closed_by="SL"); return True
            if self.tp_level and price <= self.tp_level: self._close_position(self.tp_level, ts, closed_by="TP"); return True
        return False

    def _process_quick_signal(self, quick_sig, price, ts):
        rec = quick_sig.signal or "Neutral"; self.current_signal = rec; self.last_price = price
        if not self.trading_enabled: return
        if self.position_side == "Flat":
            if rec == "Buy": self._open_long(price, ts, trigger="quick_1m_5m")
            if rec == "Sell": self._open_short(price, ts, trigger="quick_1m_5m")
            return
        if self.position_side == "Long" and rec == "Sell":
            self._close_position(price, ts, closed_by="signal:flip_to_short"); self._open_short(price, ts, trigger="flip"); return
        if self.position_side == "Short" and rec == "Buy":
            self._close_position(price, ts, closed_by="signal:flip_to_long"); self._open_long(price, ts, trigger="flip"); return

    def _get_trade_history_file(self):
        if TRADE_HISTORY_FILE: p = Path(TRADE_HISTORY_FILE)
        else: p = Path(f"trade_history_{self.symbol_ex.replace('/', '_')}_{MODE}.json")
        p.parent.mkdir(parents=True, exist_ok=True); return str(p)

    def _load_trade_history(self):
        fn = self._get_trade_history_file()
        if os.path.exists(fn):
            try:
                with open(fn, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list): print(f"Loaded {len(data)} trade records"); return data
            except Exception: logger.exception("Error loading trade history")
        print("No trade-history file – starting fresh."); return []

    def _save_trade_history(self):
        fn = self._get_trade_history_file()
        try:
            def ser(o):
                if isinstance(o, (datetime, pd.Timestamp)): return o.isoformat()
                raise TypeError
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(self.trade_history, f, indent=4, default=ser)
        except Exception:
            logger.exception("Error saving trade history"); print(f"{Colors.RED}Error saving trade history{Colors.END}")

    def _log_event(self, event_type, details):
        ts = details.get("timestamp", datetime.utcnow().isoformat())
        entry = {"timestamp": ts, "event": event_type, "details": details}
        self.trade_history.append(entry); print(f"[{ts}] {event_type.upper()}: {details}")
        logger.info("Trade event – %s – %s", event_type, details); self._save_trade_history(); return entry

    def _register_completed_trade(self, open_d, close_d):
        side = "Long" if close_d["type"] == "SELL_CLOSE_LONG" else "Short"
        entry_price = float(open_d.get("price"))
        with self._dash_lock:
            self.completed_trades.append({
                "side": side,"entry_time": pd.to_datetime(open_d.get("timestamp")),"entry_price": entry_price,
                "exit_time": pd.to_datetime(close_d.get("timestamp")),"exit_price": float(close_d.get("price")),
                "amount": float(open_d.get("amount_base")),"pnl": float(close_d.get("pnl")),
                "sl": float(open_d.get("sl")) if open_d.get("sl") is not None else None,
                "tp": float(open_d.get("tp")) if open_d.get("tp") is not None else None,
                "open_signal": open_d.get("trigger_signal", "-"),
                "close_reason": close_d.get("closed_by", "signal"),
            })

    def _signal_pill(self, rec: str):
        rec = rec or "Neutral"
        cmap = {"Strong Buy": ("#065f46","#ecfdf5"),"Buy":("#15803d","#ecfdf5"),"Neutral":("#374151","#f3f4f6"),
                "Sell":("#b91c1c","#fee2e2"),"Strong Sell":("#7f1d1d","#fee2e2")}
        fg, bg = cmap.get(rec, cmap["Neutral"])
        return html.Span(rec, style={"display":"inline-block","padding":"4px 10px","borderRadius":"9999px",
                                     "fontWeight":600,"color":fg,"backgroundColor":bg,
                                     "border":f"1px solid {PALETTE['grid']}","marginLeft":"6px"})

    def _price_pill(self, price):
        txt = f"{price:,.4f} {self.quote_asset}" if price is not None else "—"
        return html.Span(txt, style={"display":"inline-block","padding":"4px 10px","borderRadius":"8px",
                                     "fontWeight":600,"color":"#111827","backgroundColor":"#eef2ff",
                                     "border":f"1px solid {PALETTE['grid']}","marginLeft":"6px"})

    def _build_plan_card(self):
        with self._dash_lock:
            pricepill = self._price_pill(self.last_price); pill = self._signal_pill(self.current_signal)
        # FIX: read the correct last_snapshot (no .quick)
        snap = {}
        try: snap = self.strategy.quick.last_snapshot or {}
        except Exception: snap = {}

        def fmt(v, dec=4):
            try: return f"{float(v):.{dec}f}"
            except Exception: return "-"

        def row_for(tf):
            m = snap.get(tf, {})
            if not m: return html.P(f"{tf}: —", className="mb-1", style={"color": PALETTE["muted"]})
            return html.P(
                f"{tf}: close={fmt(m.get('close'))} • ema9={fmt(m.get('ema9'))} • ema21={fmt(m.get('ema21'))} • "
                f"rsi={fmt(m.get('rsi'),1)} • hist={fmt(m.get('macd_hist'),5)} • adx={fmt(m.get('adx'),1)}",
                className="mb-1"
            )

        cons = snap.get('consensus', {})
        items = [
            html.Div([html.Span("Signal:"), pill], className="mb-2"),
            html.Div([html.Span("Price:"), pricepill], className="mb-2"),
            row_for('1m'),
            row_for('5m'),
            html.P(f"Consensus: {cons.get('source','-')} → {cons.get('signal','-')}", className="mb-1"),
        ]
        return dbc.Card([dbc.CardHeader("Quick Plan (1m/5m)"), dbc.CardBody(items)],
                        className="mt-2",
                        style={"backgroundColor": PALETTE["panel"], "color": PALETTE["text"],
                               "border": f"1px solid {PALETTE['grid']}", "borderRadius": "10px"})

    def _build_position_card(self):
        with self._dash_lock:
            pill = self._signal_pill(self.current_signal); pricepill = self._price_pill(self.last_price)
            side = self.position_side; entry = self.entry_price; sl = self.sl_level; tp = self.tp_level
            unreal = 0.0
            if side != "Flat" and entry and self.last_price:
                p = self.last_price
                unreal = (p - entry) * self.qty_base if side == "Long" else (entry - p) * self.qty_base
        items = [
            html.Div([html.Span("Signal:"), pill], className="mb-2"),
            html.Div([html.Span("Price:"), pricepill], className="mb-2"),
            html.P(f"Leverage: {self.leverage}x", className="mb-1"),
            html.P(f"Side: {side}", className="mb-1"),
        ]
        if side != "Flat" and entry:
            notional = self.qty_base * entry
            items += [
                html.P(f"Entry: {entry:.4f} {self.quote_asset}", className="mb-1"),
                html.P(f"Qty (base): {self.qty_base:.6f}", className="mb-1"),
                html.P(f"Notional: {notional:.2f} {self.quote_asset}", className="mb-1"),
                html.P(f"Margin locked: {self.margin_locked:.2f} {self.quote_asset}", className="mb-1"),
                html.P(f"SL: {sl:.4f}" if sl else "SL: —", className="mb-1"),
                html.P(f"TP: {tp:.4f}" if tp else "TP: —", className="mb-1"),
                html.P(f"Unrealised PnL: {unreal:.4f} {self.quote_asset}", className="mb-1"),
            ]
        else:
            items.append(html.P("No open position.", className="mb-1", style={"color": PALETTE["muted"]}))
        items.append(html.Div([
            dbc.Button("Close Trade", id="close-button", color="danger",
                       className="mt-2 me-2",
                       style={"backgroundColor": PALETTE["button_bg"], "borderColor": PALETTE["button_bg"]}),
            dbc.Button("Start Trade", id="start-button", color="success",
                       className="mt-2 me-2",
                       style={"backgroundColor": PALETTE["start_button_bg"], "borderColor": PALETTE["start_button_bg"]}),
            dbc.Button("ShortTrig", id="shorttrig-button", color="warning", className="mt-2 me-2",
                       title="Open a SHORT immediately"),
            dbc.Button("LongTrig", id="longtrig-button", color="primary", className="mt-2",
                       title="Open a LONG immediately"),
        ]))
        return dbc.Card([dbc.CardHeader("Position"), dbc.CardBody(items)], className="mt-2",
                        style={"backgroundColor": PALETTE["panel"], "color": PALETTE["text"],
                               "border": f"1px solid {PALETTE['grid']}", "borderRadius": "10px"})

    def _build_winloss_pie(self):
        with self._dash_lock: completed = list(self.completed_trades)
        win = sum(1 for t in completed if (t.get("pnl") or 0) > 0); loss = sum(1 for t in completed if (t.get("pnl") or 0) < 0)
        if win == 0 and loss == 0: win, loss = 1, 1
        vals = [win, loss]; labels = ["Winning", "Losing"]; total = sum(vals)
        texts = [f"{lbl} {v/total:.0%}" if v > 0 else "" for lbl, v in zip(labels, vals)]
        fig = go.Figure(data=[go.Pie(labels=labels, values=vals, hole=0.55, text=texts, textinfo="text",
                                     textposition="inside", insidetextorientation="horizontal",
                                     marker=dict(colors=[PALETTE["trade_win"], PALETTE["trade_loss"]]), sort=False)])
        fig.update_layout(template="plotly_white", paper_bgcolor=PALETTE["panel"], plot_bgcolor=PALETTE["panel"],
                          font=dict(color=PALETTE["text"]), title=dict(text="Win / Loss", x=0.5, y=0.97),
                          margin=dict(l=10, r=10, t=60, b=10))
        return fig

    def _thin_points(self, xs, ys, max_points):
        n = len(xs)
        if n <= max_points: return xs, ys
        idx = np.linspace(0, n - 1, max_points, dtype=int)
        return [xs[i] for i in idx], [ys[i] for i in idx]

    def _build_price_figure(self):
        with self._dash_lock:
            candles = pd.DataFrame(self.candle_history)
            completed = list(self.completed_trades)
            sig = self.current_signal; cur_price = self.last_price; equity = self.equity_usdt
        candle_trace = go.Candlestick(
            x=candles["timestamp"] if not candles.empty else [],
            open=candles["open"] if not candles.empty else [],
            high=candles["high"] if not candles.empty else [],
            low=candles["low"] if not candles.empty else [],
            close=candles["close"] if not candles.empty else [],
            name="Price", increasing=dict(line=dict(color=PALETTE["candle_up"]), fillcolor=PALETTE["candle_up"]),
            decreasing=dict(line=dict(color=PALETTE["candle_down"]), fillcolor=PALETTE["candle_down"]), whiskerwidth=0.4
        )
        entry_x = [t["entry_time"] for t in completed]; entry_y = [t["entry_price"] for t in completed]
        exit_x  = [t["exit_time"] for t in completed];  exit_y  = [t["exit_price"] for t in completed]
        total_markers = len(entry_x) + len(exit_x); dense = total_markers > MAX_MARKERS_ON_CHART
        if dense:
            budget_each = max(1, MAX_MARKERS_ON_CHART // 2)
            entry_x, entry_y = self._thin_points(entry_x, entry_y, budget_each)
            exit_x,  exit_y  = self._thin_points(exit_x,  exit_y,  budget_each)
        msize = DENSE_MARKER_SIZE if dense else SPARSE_MARKER_SIZE; outline = dict(width=1, color=PALETTE["panel"])
        fig = go.Figure(); fig.add_trace(candle_trace)
        if not dense:
            fig.add_trace(go.Scatter(x=entry_x, y=entry_y, mode="markers", name="Entry halo",
                                     marker=dict(size=msize+8, color=PALETTE["halo"], line=dict(width=0)),
                                     hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=exit_x, y=exit_y, mode="markers", name="Exit halo",
                                     marker=dict(size=msize+8, color=PALETTE["halo"], line=dict(width=0)),
                                     hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=entry_x, y=entry_y, mode="markers", name="Entry",
                                 marker=dict(size=msize, color=PALETTE["entry"], line=outline, symbol="circle"),
                                 hovertemplate="Entry @ %{y:.4f}<br>%{x|%Y-%m-%d %H:%M}<extra></extra>"))
        fig.add_trace(go.Scatter(x=exit_x, y=exit_y, mode="markers", name="Exit",
                                 marker=dict(size=msize, color=PALETTE["exit"], line=outline, symbol="circle"),
                                 hovertemplate="Exit @ %{y:.4f}<br>%{x|%Y-%m-%d %H:%M}<extra></extra>"))
        for t in completed:
            is_win = (t.get("pnl", 0) or 0) >= 0
            col = PALETTE["trade_win"] if is_win else PALETTE["trade_loss"]
            fig.add_trace(go.Scatter(x=[t["entry_time"], t["exit_time"]], y=[t["entry_price"], t["exit_price"]],
                                     mode="lines", line=dict(color=col, width=3), hoverinfo="skip",
                                     name="Trade", showlegend=False))
        total_pnl = sum((t.get("pnl") or 0) for t in completed)
        pnl_color = PALETTE["trade_win"] if total_pnl >= 0 else PALETTE["trade_loss"]
        price_txt = f" • Current: {cur_price:,.4f} {self.quote_asset}" if cur_price is not None else ""
        status = "ENABLED" if self.trading_enabled else "DISABLED"; status_color = "#15803d" if self.trading_enabled else "#b91c1c"
        fig.update_layout(
            title=dict(text=(f"Equity: {equity:,.2f} {self.quote_asset} • Lev: {self.leverage}x • "
                             f"Signal: {sig}{price_txt}<br>"
                             f"Trading: <span style='color:{status_color}'>{status}</span> • "
                             f"Realized PnL: {total_pnl:,.2f} {self.quote_asset}"),
                       x=0.5, font=dict(color=pnl_color, size=20)),
            template="plotly_white", paper_bgcolor=PALETTE["panel"], plot_bgcolor=PALETTE["panel"],
            font=dict(color=PALETTE["text"]), margin=dict(l=40, r=20, t=90, b=40),
            legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"), hovermode="x unified",
        )
        fig.update_xaxes(title_text="Time", gridcolor=PALETTE["grid"], showspikes=True)
        fig.update_yaxes(title_text=f"Price ({self.quote_asset})", gridcolor=PALETTE["grid"], showspikes=True)
        return fig

    def _build_trades_table(self):
        with self._dash_lock:
            rows = []
            for t in self.completed_trades:
                amt_base = t.get("amount"); entry_price = t.get("entry_price")
                rows.append({
                    "Timestamp (open)": t["entry_time"].strftime("%Y-%m-%d %H:%M"),
                    "Side": t["side"],
                    "Entry": round(entry_price, 4) if entry_price is not None else "-",
                    "Exit": round(t.get("exit_price", np.nan), 4) if t.get("exit_price") is not None else "-",
                    "PnL": round(t.get("pnl", 0.0), 4),
                    "Qty (base)": round(amt_base, 6) if amt_base is not None else "-",
                    "Notional (quote)": round((entry_price or 0.0) * (amt_base or 0.0), 2) if amt_base is not None else "-",
                    "SL": round(t.get("sl"), 4) if t.get("sl") is not None else "-",
                    "TP": round(t.get("tp"), 4) if t.get("tp") is not None else "-",
                    "Open Signal": t.get("open_signal", "-"),
                    "Close Reason": t.get("close_reason", "signal"),
                })
        cols = ["Timestamp (open)", "Side", "Entry", "Exit", "PnL",
                "Qty (base)", "Notional (quote)", "SL", "TP", "Open Signal", "Close Reason"]
        return DataTable(
            id="trades-table", columns=[{"name": c, "id": c} for c in cols], data=rows,
            style_header={"backgroundColor": PALETTE["panel"], "color": PALETTE["text"],
                          "border": f"1px solid {PALETTE['grid']}", "fontWeight": "600"},
            style_cell={"backgroundColor": PALETTE["panel"], "color": PALETTE["text"],
                        "border": f"1px solid {PALETTE['grid']}", "padding": "8px",
                        "fontFamily": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
                        "textAlign": "center"},
            style_table={"overflowX": "auto"},
            style_data_conditional=[
                {"if": {"filter_query": "{PnL} >= 0", "column_id": "PnL"}, "color": PALETTE["trade_win"], "fontWeight": "600"},
                {"if": {"filter_query": "{PnL} < 0", "column_id": "PnL"}, "color": PALETTE["trade_loss"], "fontWeight": "600"},
            ],
            page_size=100,
        )

    def _setup_dashboard(self):
        self.trade_history = self._load_trade_history()
        app.bot_instance = self
        app.layout = dbc.Container(
            [
                dbc.Row(
                    dbc.Col(dcc.Graph(id="live-graph", config={"displayModeBar": True},
                                      style={"height": "62vh","border": f"1px solid {PALETTE['grid']}",
                                             "borderRadius": "10px","background": PALETTE["panel"]}),
                            width=12),
                    className="g-3", style={"marginTop": "8px"}
                ),
                dbc.Row(
                    [dbc.Col(html.Div(id="plan-card"), width=4),
                     dbc.Col(html.Div(id="position-card"), width=4),
                     dbc.Col(dcc.Graph(id="win-loss-pie", config={"displayModeBar": False},
                                       style={"height": "36vh","border": f"1px solid {PALETTE['grid']}",
                                              "borderRadius": "10px","background": PALETTE["panel"]}), width=4)],
                    className="g-3"
                ),
                dbc.Row(
                    dbc.Col(html.Div(id="trades-table-wrap",
                                     style={"border": f"1px solid {PALETTE['grid']}",
                                            "borderRadius": "10px","padding": "8px","background": PALETTE["panel"]}),
                            width=12),
                    className="g-3"
                ),
                dcc.Interval(id="graph-update", interval=1200, n_intervals=0),
                dbc.Row(dbc.Col(html.Div("Close the browser or press Ctrl-C to stop the bot.",
                                         style={"textAlign": "center","marginTop": "8px","color": PALETTE["muted"]}),
                                width=12))
            ], fluid=True, className="p-3", style={"backgroundColor": PALETTE["page_bg"]}
        )
        @app.callback(Output("live-graph", "figure"), Input("graph-update", "n_intervals"))
        def _update_graph(_): return self._build_price_figure()
        @app.callback([Output("plan-card", "children"), Output("position-card", "children"), Output("win-loss-pie", "figure")],
                      Input("graph-update", "n_intervals"))
        def _update_cards(_): return self._build_plan_card(), self._build_position_card(), self._build_winloss_pie()
        @app.callback(Output("trades-table-wrap", "children"),
                      [Input("graph-update", "n_intervals"), Input("close-button", "n_clicks"),
                       Input("start-button", "n_clicks"), Input("shorttrig-button", "n_clicks"),
                       Input("longtrig-button", "n_clicks")])
        def _update_table_and_handle_buttons(_, close_clicks, start_clicks, shorttrig_clicks, longtrig_clicks):
            trig = ctx.triggered_id
            if trig == "close-button" and close_clicks:
                price = self._get_price(); ts = datetime.utcnow().replace(tzinfo=timezone.utc)
                if price is not None: self._close_position(price, ts, closed_by="Manual Close")
            if trig == "start-button" and start_clicks:
                self.trading_enabled = True; logger.info("Trading enabled via Start Trade button.")
            if trig == "shorttrig-button" and shorttrig_clicks:
                price = self._get_price(); ts = datetime.utcnow().replace(tzinfo=timezone.utc)
                if price is not None and self.position_side == "Flat": self._open_short(price, ts, trigger="manual_short")
            if trig == "longtrig-button" and longtrig_clicks:
                price = self._get_price(); ts = datetime.utcnow().replace(tzinfo=timezone.utc)
                if price is not None and self.position_side == "Flat": self._open_long(price, ts, trigger="manual_long")
            return self._build_trades_table()

    @staticmethod
    def _tf_seconds(tf: str) -> int:
        if tf.endswith("m"): return int(tf[:-1]) * 60
        if tf.endswith("h"): return int(tf[:-1]) * 3600
        if tf.endswith("d"): return int(tf[:-1]) * 86400
        return 60

    @staticmethod
    def _floor_time(ts: datetime, seconds: int) -> datetime:
        epoch = int(ts.replace(tzinfo=timezone.utc).timestamp())
        floored = epoch - (epoch % seconds)
        return datetime.fromtimestamp(floored, tz=timezone.utc)

    def _prime_live_candles(self, limit: int = 200):
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol_ex, '1m', limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms'); df.set_index('timestamp', inplace=True)
            with self._dash_lock:
                self.candle_history = [
                    {"timestamp": ts.to_pydatetime().replace(tzinfo=timezone.utc),
                     "open": float(r["open"]), "high": float(r["high"]),
                     "low": float(r["low"]), "close": float(r["close"])}
                    for ts, r in df.iterrows()
                ]
        except Exception: logger.exception("Prime live candles failed")

    def _update_live_candle(self, now: datetime, price: float):
        secs = 60; bucket = self._floor_time(now, secs)
        with self._dash_lock:
            if not self.candle_history:
                self.candle_history.append({"timestamp": bucket,"open": price,"high": price,"low": price,"close": price}); return
            last = self.candle_history[-1]; last_bucket = last["timestamp"]
            if bucket > last_bucket:
                self.candle_history.append({"timestamp": bucket,"open": price,"high": price,"low": price,"close": price})
            else:
                last["close"] = price
                if price > last["high"]: last["high"] = price
                if price < last["low"]:  last["low"]  = price

    def _start_dash_server(self):
        if self._dash_server: return
        @app.server.route("/_shutdown")
        def _shutdown():
            try:
                from flask import request
                func = request.environ.get('werkzeug.server.shutdown')
                if func: func()
            except Exception: pass
            return "Shutting down..."
        def run():
            self._dash_server = app.server
            logger.info("Dash server started – http://127.0.0.1:8050")
            app.run(host="127.0.0.1", port=8050, debug=False, use_reloader=False)
        self._dash_thread = threading.Thread(target=run, daemon=True); self._dash_thread.start()
        print("\nDashboard launched – open http://127.0.0.1:8050"); logger.info("Dashboard launched")

    def _shutdown_dash_server(self):
        if self._dash_server:
            try: requests.get("http://127.0.0.1:8050/_shutdown", timeout=1); logger.info("Dashboard shutdown request sent")
            except Exception: logger.debug("Dashboard shutdown endpoint not available")
            finally: self._dash_server = None

    def run_live_paper(self):
        print(f"{Colors.CYAN}Futures Scalper – 1m/5m – {self.symbol_ex} – {self.leverage}x{Colors.END}")
        self._start_dash_server(); self._prime_live_candles(limit=300)
        while True:
            price = self._get_price()
            if price is None: time.sleep(1); continue
            now = datetime.utcnow().replace(tzinfo=timezone.utc)
            self.last_price = price; self._update_live_candle(now, price)
            with self._dash_lock:
                self.portfolio_history.append({"timestamp": now, "price": price, "portfolio_value": self.equity_usdt})
            if self._check_sl_tp_on_tick(now, price): time.sleep(1); continue
            quick_sig = self.strategy.quick_signal()
            self._process_quick_signal(quick_sig, price, now)
            time.sleep(1.2)

    def run(self):
        if MODE == "backtest":
            print(f"{Colors.YELLOW}Backtest mode not implemented for futures scalper. Use MODE=paper or live.{Colors.END}")
            return
        self.run_live_paper()

    def stop(self):
        print("Stopping bot..."); self._shutdown_dash_server(); logger.info("Bot stopped"); print("Bot stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Futures Scalping Bot (1m/5m)")
    parser.add_argument("--symbol", default=SYMBOL)
    args = parser.parse_args()
    bot = FuturesScalperBot(symbol=args.symbol)
    try: bot.run()
    except KeyboardInterrupt:
        print(f"{Colors.YELLOW}Interrupted by user.{Colors.END}"); logger.info("Interrupted by user")
    finally: bot.stop()
