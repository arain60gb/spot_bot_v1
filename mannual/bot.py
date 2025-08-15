# bot.py  — updated to show Plan, Position, Win/Loss in one row (Position center)
import sys
import os
import json
import csv
import time
import atexit
import threading
import argparse
import logging
import re
import requests
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timezone, timedelta
from pathlib import Path
from logging.handlers import RotatingFileHandler

# ── Config ──────────────────────────────────────────────────────────────────────
from config import (
    SYMBOL,
    INITIAL_USDT,
    INITIAL_TAO,
    FIXED_ORDER_SIZE_USDT,
    MODE,
    STOP_LOSS_PERCENT,
    TAKE_PROFIT_PERCENT,
    TRADE_HISTORY_FILE,
    LOG_FILE,
)

# ── Technical analysis utilities ───────────────────────────────────────────────
from technical_analyzer import (
    TechnicalAnalyzer,
    DataFetcher,
    Colors,
    color_signal,
    compute_trade_plan_1h_5m,   # <— plan builder
)

# ── Historical data fetcher (back-test) ────────────────────────────────────────
from data_fetcher import BacktestDataFetcher, fetch_multiple_timeframes_concurrently

# ── Dash & Plotly ─────────────────────────────────────────────────────────────
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, ctx
from dash.dependencies import Output, Input, State
from dash.dash_table import DataTable
import plotly.graph_objs as go


# Light Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# Light palette
PALETTE = {
    "page_bg": "#f7f9fc",
    "panel": "#ffffff",
    "text": "#1f2937",
    "muted": "#6b7280",
    "grid": "#e5e7eb",
    "accent": "#0d6efd",
    "entry": "#22c55e",
    "exit": "#ef4444",
    "candle_up": "#16a34a",
    "candle_down": "#dc2626",
    "trade_win": "#15803d",
    "trade_loss": "#b91c1c",
    "halo": "rgba(0,0,0,0.08)",
    "button_bg": "#dc3545",
    "button_hover": "#bb2d3b",
    "start_button_bg": "#198754",
    "start_button_hover": "#157347",
}

# Marker density controls
MAX_MARKERS_ON_CHART = 120
SPARSE_MARKER_SIZE = 10
DENSE_MARKER_SIZE = 6


# ── Logging setup ------------------------------------------------------------
def setup_logging(log_path: str | Path):
    log_path = Path(log_path) if log_path else Path("trading_bot.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    class AnsiStrippingFileHandler(RotatingFileHandler):
        ansi_re = re.compile(r"\x1b\[[0-9;]*m")
        def emit(self, record):
            if isinstance(record.msg, str):
                record.msg = self.ansi_re.sub("", record.msg)
            super().emit(record)

    fh = AnsiStrippingFileHandler(log_path, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt); fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt); ch.setLevel(logging.INFO)

    logger.handlers.clear()
    logger.addHandler(fh); logger.addHandler(ch)

    class _LoggerWriter:
        def __init__(self, level_func): self.level_func = level_func
        def write(self, message):
            if message.strip():
                for line in message.rstrip().splitlines():
                    self.level_func(line)
        def flush(self): pass

    sys.stdout = _LoggerWriter(logger.info)
    sys.stderr = _LoggerWriter(logger.error)


setup_logging(LOG_FILE)
logger = logging.getLogger(__name__)
logger.info("Logger initialised – MODE=%s – SYMBOL=%s", MODE, SYMBOL)


# ── Trading strategy ---------------------------------------------------------
class TradingStrategy:
    def __init__(self, symbol, primary_tf="15m"):
        self.analyzer = TechnicalAnalyzer(symbol)
        self.symbol = symbol
        self.primary_tf = primary_tf

    def get_signals(self):
        if MODE in ["paper", "live"]:
            try:
                return self.analyzer.analyze_all_timeframes()
            except Exception:
                logger.exception("Error getting live signals")
                print("Error getting live signals")
                return {}
        return {}

    def should_buy(self, signals):
        return signals.get(self.primary_tf, {}).get("overall_recommendation", "") == "Buy"

    def should_sell(self, signals):
        return signals.get(self.primary_tf, {}).get("overall_recommendation", "") == "Sell"


# ── Core bot ---------------------------------------------------------------
class CryptoTradingBot:
    def __init__(self, symbol, primary_tf="15m"):
        self.symbol = symbol
        self.base_asset, self.quote_asset = symbol.split("/")
        self.primary_tf = primary_tf

        # position state
        self.position_state = "Flat"       # Flat | Long | Short
        self.entry_price = None
        self.amount_base = 0.0
        self.sl_level = None               # dynamic SL for open position
        self.tp_level = None               # dynamic TP for open position

        # optional trade-logic state (kept for future rules/UI)
        self.seen_strong_follow_through = False
        self.open_signal = None

        # trading state
        self.trading_enabled = True

        # price (for pill & chart)
        self.last_price = None

        self.simulated_balance = { self.quote_asset: INITIAL_USDT, self.base_asset: INITIAL_TAO }
        self.strategy = TradingStrategy(symbol, primary_tf)
        self.trade_history = self._load_trade_history()

        # dashboard data
        self.completed_trades = []   # dicts include side, entry/exit, pnl, sl, tp, open_signal, close_reason
        self.portfolio_history = []  # snapshots
        self.candle_history = []     # OHLC for primary tf
        self._open_trades_csv = {}

        # current signal (for dashboard)
        self.current_signal = "Neutral"

        # last plan for cards
        self.last_plan = None

        # live mode data fetcher & candle aggregator
        self.live_fetcher = None
        if MODE in ["paper", "live"]:
            try:
                self.live_fetcher = DataFetcher(symbol)
            except Exception:
                logger.exception("Could not init DataFetcher")

        self._dash_lock = threading.Lock()
        self._dash_server = None
        self._dash_thread = None
        atexit.register(self._shutdown_dash_server)

        if MODE in ["paper", "live"]:
            cfg = {"enableRateLimit": True, "timeout": 30_000}
            self.exchange = ccxt.binance(cfg)
        else:
            self.exchange = None

        if MODE == "backtest":
            self.backtest_fetcher = BacktestDataFetcher(symbol)

        print(f"Bot initialised for {symbol} – mode={MODE} – TF={primary_tf}")
        logger.info("Bot init – %s – %s – %s", symbol, MODE, primary_tf)

        self._setup_dashboard()

    # ── Helpers for markers ---------------------------------------------------
    def _thin_points(self, xs, ys, max_points):
        """Evenly sample down to <= max_points, preserving endpoints."""
        n = len(xs)
        if n <= max_points:
            return xs, ys
        idx = np.linspace(0, n - 1, max_points, dtype=int)
        return [xs[i] for i in idx], [ys[i] for i in idx]

    # ── Small UI helpers -----------------------------------------------------
    def _signal_pill(self, rec: str):
        rec = rec or "Neutral"
        cmap = {
            "Strong Buy": ("#065f46", "#ecfdf5"),
            "Buy":        ("#15803d", "#ecfdf5"),
            "Neutral":    ("#374151", "#f3f4f6"),
            "Sell":       ("#b91c1c", "#fee2e2"),
            "Strong Sell":("#7f1d1d", "#fee2e2"),
        }
        fg, bg = cmap.get(rec, cmap["Neutral"])
        return html.Span(
            rec,
            style={
                "display": "inline-block",
                "padding": "4px 10px",
                "borderRadius": "9999px",
                "fontWeight": 600,
                "color": fg,
                "backgroundColor": bg,
                "border": f"1px solid {PALETTE['grid']}",
                "marginLeft": "6px",
            },
        )

    def _price_pill(self, price):
        txt = f"{price:,.4f} {self.quote_asset}" if price is not None else "—"
        return html.Span(
            txt,
            style={
                "display": "inline-block",
                "padding": "4px 10px",
                "borderRadius": "8px",
                "fontWeight": 600,
                "color": "#111827",
                "backgroundColor": "#eef2ff",
                "border": f"1px solid {PALETTE['grid']}",
                "marginLeft": "6px",
            },
        )

    # ── NEW: Plan card (left) -------------------------------------------------
    def _build_plan_card(self):
        with self._dash_lock:
            pricepill = self._price_pill(self.last_price)
            pill = self._signal_pill(self.current_signal)
            plan = self.last_plan

        items = []
        items.append(html.Div([html.Span("Current signal:"), pill], className="mb-2"))
        items.append(html.Div([html.Span("Current price:"), pricepill], className="mb-2"))

        if plan:
            items += [
                html.P(f"Trend: Long={plan.get('trend_long', False)} | Short={plan.get('trend_short', False)}", className="mb-1"),
                html.P(f"1H EMA20: {plan.get('ema20_1h', float('nan')):.4f}", className="mb-1"),
                html.P(f"LongTrig: {plan.get('long_trigger', float('nan')):.4f} • PullLong: {plan.get('pullback_long', float('nan')):.4f}", className="mb-1"),
                html.P(f"ShortTrig: {plan.get('short_trigger', float('nan')):.4f} • PullShort: {plan.get('pullback_short', float('nan')):.4f}", className="mb-1"),
                html.P(f"ATR1h: {plan.get('atr1h', float('nan')):.4f} • SLx: {plan.get('sl_mult', float('nan')):.4f} • TPx: {plan.get('tp_mult', float('nan')):.4f}", className="mb-1"),
            ]
        else:
            items.append(html.P("Plan not computed yet…", className="mb-1", style={"color": PALETTE["muted"]}))

        return dbc.Card(
            [
                dbc.CardHeader("Plan-Driven Bot"),
                dbc.CardBody(items),
            ],
            className="mt-2",
            style={"backgroundColor": PALETTE["panel"], "color": PALETTE["text"],
                   "border": f"1px solid {PALETTE['grid']}", "borderRadius": "10px"}
        )

    # ── NEW: Position card (center) ------------------------------------------
    def _build_position_card(self):
        with self._dash_lock:
            pill = self._signal_pill(self.current_signal)
            pricepill = self._price_pill(self.last_price)
            pos = self.position_state
            entry = self.entry_price
            sl = self.sl_level
            tp = self.tp_level
            unreal = 0.0
            if pos != "Flat" and entry:
                p = self.last_price if self.last_price is not None else entry
                unreal = ((p - entry) * self.amount_base) if pos == "Long" else ((entry - p) * self.amount_base)
            seen_strong = self.seen_strong_follow_through
            arm_neutral = True
            post_neutral = False

        items = [
            html.Div([html.Span("Current signal:"), pill], className="mb-2"),
            html.Div([html.Span("Current price:"), pricepill], className="mb-2"),
            html.P(f"Timeframe: {self.primary_tf}", className="mb-1"),
            html.P(f"Side: {pos}", className="mb-1"),
        ]

        if pos != "Flat" and entry:
            items += [
                html.P(f"Entry: {entry:.4f} {self.quote_asset}", className="mb-1"),
                html.P(f"SL: {sl:.4f}", className="mb-1"),
                html.P(f"TP: {tp:.4f}", className="mb-1"),
                html.P(f"Strong seen after entry: {seen_strong}", className="mb-1"),
                html.P(f"Arm exit on Neutral: {arm_neutral}", className="mb-1"),
                html.P(f"Post-Neutral State: {post_neutral}", className="mb-1"),
                html.P(f"Unrealised PnL: {unreal:.4f} {self.quote_asset}", className="mb-1"),
            ]
        else:
            items.append(html.P("No open position.", className="mb-1", style={"color": PALETTE["muted"]}))

        items.append(
            html.Div(
                [
                    dbc.Button("Close Trade", id="close-button", color="danger",
                               className="mt-2 me-2",
                               style={"backgroundColor": PALETTE["button_bg"], "borderColor": PALETTE["button_bg"]}),
                    dbc.Button("Start Trade", id="start-button", color="success",
                               className="mt-2",
                               style={"backgroundColor": PALETTE["start_button_bg"], "borderColor": PALETTE["start_button_bg"]}),
                ]
            )
        )

        return dbc.Card(
            [dbc.CardHeader("Position"), dbc.CardBody(items)],
            className="mt-2",
            style={"backgroundColor": PALETTE["panel"], "color": PALETTE["text"],
                   "border": f"1px solid {PALETTE['grid']}", "borderRadius": "10px"}
        )

    # ── Manual Close Position Method ──────────────────────────────────────────
    def close_position(self, ts, price, closed_by="Manual Close"):
        if self.position_state == "Flat":
            logger.info("Manual close requested, but no position is open.")
            print("No open position to close.")
            return
        logger.info("Manual close initiated – %s @ %s", self.position_state, price)
        print(f"{Colors.YELLOW}[{ts}] Manual close initiated – {self.position_state} @ {price:.4f}{Colors.END}")

        if self.position_state == "Long":
            self._execute_sell(price, ts, closed_by=closed_by)
        elif self.position_state == "Short":
            self._execute_buy(price, ts, closed_by=closed_by)

        self.trading_enabled = False
        logger.info("Trading disabled after manual close. Waiting for Start Trade button.")

    def enable_trading(self):
        if not self.trading_enabled:
            self.trading_enabled = True
            logger.info("Trading enabled via Start Trade button.")
            print("Trading enabled. Bot will now respond to signals.")

    # ── Price figure ---------------------------------------------------------
    def _build_price_figure(self):
        with self._dash_lock:
            candles = pd.DataFrame(self.candle_history)
            completed = list(self.completed_trades)
            sig = self.current_signal
            cur_price = self.last_price

        candle_trace = go.Candlestick(
            x=candles["timestamp"] if not candles.empty else [],
            open=candles["open"] if not candles.empty else [],
            high=candles["high"] if not candles.empty else [],
            low=candles["low"] if not candles.empty else [],
            close=candles["close"] if not candles.empty else [],
            name="Price",
            increasing=dict(line=dict(color=PALETTE["candle_up"]), fillcolor=PALETTE["candle_up"]),
            decreasing=dict(line=dict(color=PALETTE["candle_down"]), fillcolor=PALETTE["candle_down"]),
            whiskerwidth=0.4
        )

        entry_x = [t["entry_time"] for t in completed]
        entry_y = [t["entry_price"] for t in completed]
        exit_x  = [t["exit_time"] for t in completed]
        exit_y  = [t["exit_price"] for t in completed]

        total_markers = len(entry_x) + len(exit_x)
        dense = total_markers > MAX_MARKERS_ON_CHART
        if dense:
            budget_each = max(1, MAX_MARKERS_ON_CHART // 2)
            entry_x, entry_y = self._thin_points(entry_x, entry_y, budget_each)
            exit_x,  exit_y  = self._thin_points(exit_x,  exit_y,  budget_each)

        msize = DENSE_MARKER_SIZE if dense else SPARSE_MARKER_SIZE
        outline = dict(width=1, color=PALETTE["panel"])

        fig = go.Figure()
        fig.add_trace(candle_trace)

        if not dense:
            fig.add_trace(
                go.Scatter(
                    x=entry_x, y=entry_y, mode="markers", name="Entry halo",
                    marker=dict(size=msize+8, color=PALETTE["halo"], line=dict(width=0)),
                    hoverinfo="skip", showlegend=False
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=exit_x, y=exit_y, mode="markers", name="Exit halo",
                    marker=dict(size=msize+8, color=PALETTE["halo"], line=dict(width=0)),
                    hoverinfo="skip", showlegend=False
                )
            )

        fig.add_trace(
            go.Scatter(
                x=entry_x, y=entry_y, mode="markers", name="Entry",
                marker=dict(size=msize, color=PALETTE["entry"], line=outline, symbol="circle"),
                hovertemplate="Entry @ %{y:.4f}<br>%{x|%Y-%m-%d %H:%M}<extra></extra>"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=exit_x, y=exit_y, mode="markers", name="Exit",
                marker=dict(size=msize, color=PALETTE["exit"], line=outline, symbol="circle"),
                hovertemplate="Exit @ %{y:.4f}<br>%{x|%Y-%m-%d %H:%M}<extra></extra>"
            )
        )

        for t in completed:
            is_win = (t.get("pnl", 0) or 0) >= 0
            col = PALETTE["trade_win"] if is_win else PALETTE["trade_loss"]
            fig.add_trace(
                go.Scatter(
                    x=[t["entry_time"], t["exit_time"]],
                    y=[t["entry_price"], t["exit_price"]],
                    mode="lines",
                    line=dict(color=col, width=3),
                    hoverinfo="skip",
                    name="Trade",
                    showlegend=False,
                )
            )

        total_pnl = sum((t.get("pnl") or 0) for t in completed)
        pnl_color = PALETTE["trade_win"] if total_pnl >= 0 else PALETTE["trade_loss"]
        price_txt = f" • Current: {cur_price:,.4f} {self.quote_asset}" if cur_price is not None else ""
        density_note = " • Markers thinned" if dense else ""
        status = "ENABLED" if self.trading_enabled else "DISABLED"
        status_color = "#15803d" if self.trading_enabled else "#b91c1c"

        fig.update_layout(
            title=dict(
                text=(
                    f"Total PnL: {total_pnl:,.2f} {self.quote_asset} • "
                    f"Signal: {sig}{price_txt}{density_note}<br>"
                    f"Trading: <span style='color:{status_color}'>{status}</span>"
                ),
                x=0.5,
                font=dict(color=pnl_color, size=20),
            ),
            template="plotly_white",
            paper_bgcolor=PALETTE["panel"],
            plot_bgcolor=PALETTE["panel"],
            font=dict(color=PALETTE["text"]),
            margin=dict(l=40, r=20, t=90, b=40),
            legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
            hovermode="x unified",
        )
        fig.update_xaxes(title_text="Time", gridcolor=PALETTE["grid"], showspikes=True)
        fig.update_yaxes(title_text=f"Price ({self.quote_asset})", gridcolor=PALETTE["grid"], showspikes=True)
        return fig

    # ── Win/Loss pie ----------------------------------------------------------
    def _build_winloss_pie(self):
        with self._dash_lock:
            completed = list(self.completed_trades)

        win = sum(1 for t in completed if (t.get("pnl") or 0) > 0)
        loss = sum(1 for t in completed if (t.get("pnl") or 0) < 0)
        if win == 0 and loss == 0:
            win, loss = 1, 1

        vals = [win, loss]
        labels = ["Winning", "Losing"]
        total = sum(vals)
        texts = [f"{lbl} {v/total:.0%}" if v > 0 else "" for lbl, v in zip(labels, vals)]

        fig = go.Figure(
            data=[go.Pie(
                labels=labels, values=vals, hole=0.55,
                text=texts, textinfo="text", textposition="inside",
                insidetextorientation="horizontal",
                marker=dict(colors=[PALETTE["trade_win"], PALETTE["trade_loss"]]),
                sort=False
            )]
        )
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor=PALETTE["panel"],
            plot_bgcolor=PALETTE["panel"],
            font=dict(color=PALETTE["text"]),
            title=dict(text="Win / Loss", x=0.5, y=0.97),
            margin=dict(l=10, r=10, t=60, b=10),
        )
        return fig

    # ── Trades table ---------------------------------------------------------
    def _build_trades_table(self):
        base = self.base_asset
        quote = self.quote_asset
        with self._dash_lock:
            rows = []
            for t in self.completed_trades:
                amt_base = t.get("amount")
                entry_price = t.get("entry_price")
                amt_quote = (entry_price or 0.0) * (amt_base or 0.0)
                rows.append({
                    "Timestamp (open)": t["entry_time"].strftime("%Y-%m-%d %H:%M"),
                    "Side": t["side"],
                    "Entry": round(entry_price, 4) if entry_price is not None else "-",
                    "Exit": round(t.get("exit_price", np.nan), 4) if t.get("exit_price") is not None else "-",
                    "PnL": round(t.get("pnl", 0.0), 4),
                    f"Amount {base}": round(amt_base, 6) if amt_base is not None else "-",
                    f"Amount {quote}": round(amt_quote, 2) if amt_base is not None else "-",
                    "SL": round(t.get("sl"), 4) if t.get("sl") is not None else "-",
                    "TP": round(t.get("tp"), 4) if t.get("tp") is not None else "-",
                    "Open Signal": t.get("open_signal", "-"),
                    "Close Reason": t.get("close_reason", "signal"),
                })

        cols_order = [
            "Timestamp (open)", "Side", "Entry", "Exit", "PnL",
            f"Amount {base}", f"Amount {quote}", "SL", "TP", "Open Signal", "Close Reason"
        ]

        return DataTable(
            id="trades-table",
            columns=[{"name": c, "id": c} for c in cols_order],
            data=rows,
            style_header={
                "backgroundColor": PALETTE["panel"],
                "color": PALETTE["text"],
                "border": f"1px solid {PALETTE['grid']}",
                "fontWeight": "600",
            },
            style_cell={
                "backgroundColor": PALETTE["panel"],
                "color": PALETTE["text"],
                "border": f"1px solid {PALETTE['grid']}",
                "padding": "8px",
                "fontFamily": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
                "textAlign": "center",
            },
            style_table={"overflowX": "auto"},
            style_data_conditional=[
                {"if": {"filter_query": "{PnL} >= 0", "column_id": "PnL"},
                 "color": PALETTE["trade_win"], "fontWeight": "600"},
                {"if": {"filter_query": "{PnL} < 0", "column_id": "PnL"},
                 "color": PALETTE["trade_loss"], "fontWeight": "600"},
            ],
            page_size=100,
        )

    # ── Dashboard layout & callbacks -----------------------------------------
    def _setup_dashboard(self):
        app.bot_instance = self
        app.layout = dbc.Container(
            [
                # Row 1: Price graph (full width)
                dbc.Row(
                    dbc.Col(
                        dcc.Graph(
                            id="live-graph",
                            config={"displayModeBar": True},
                            style={"height": "62vh", "border": f"1px solid {PALETTE['grid']}",
                                   "borderRadius": "10px", "background": PALETTE["panel"]},
                        ),
                        width=12,
                    ),
                    className="g-3",
                    style={"marginTop": "8px"}
                ),

                # Row 2: Plan (left) + Position (center) + Win/Loss (right)
                dbc.Row(
                    [
                        dbc.Col(html.Div(id="plan-card"), width=4),
                        dbc.Col(html.Div(id="position-card"), width=4),
                        dbc.Col(
                            dcc.Graph(
                                id="win-loss-pie",
                                config={"displayModeBar": False},
                                style={"height": "36vh", "border": f"1px solid {PALETTE['grid']}",
                                       "borderRadius": "10px", "background": PALETTE["panel"]},
                            ),
                            width=4,
                        ),
                    ],
                    className="g-3"
                ),

                # Row 3: Trades table
                dbc.Row(
                    dbc.Col(
                        html.Div(
                            id="trades-table-wrap",
                            style={
                                "border": f"1px solid {PALETTE['grid']}",
                                "borderRadius": "10px",
                                "padding": "8px",
                                "background": PALETTE["panel"],
                            },
                        ),
                        width=12,
                    ),
                    className="g-3"
                ),

                dcc.Store(id='close-button-store', data=0),
                dcc.Store(id='start-button-store', data=0),

                dcc.Interval(id="graph-update", interval=2000, n_intervals=0),
                dbc.Row(
                    dbc.Col(
                        html.Div(
                            "Close the browser or press Ctrl-C to stop the bot.",
                            style={"textAlign": "center", "marginTop": "8px", "color": PALETTE["muted"]},
                        ),
                        width=12,
                    )
                ),
            ],
            fluid=True,
            className="p-3",
            style={"backgroundColor": PALETTE["page_bg"]}
        )

        @app.callback(Output("live-graph", "figure"), Input("graph-update", "n_intervals"))
        def _update_graph(_):
            return self._build_price_figure()

        # NEW: one callback for Plan + Position + Pie
        @app.callback(
            [Output("plan-card", "children"),
             Output("position-card", "children"),
             Output("win-loss-pie", "figure")],
            Input("graph-update", "n_intervals"),
        )
        def _update_cards(_):
            return self._build_plan_card(), self._build_position_card(), self._build_winloss_pie()

        @app.callback(
            Output("trades-table-wrap", "children"),
            [Input("graph-update", "n_intervals"),
             Input('close-button', 'n_clicks'),
             Input('start-button', 'n_clicks')],
        )
        def _update_table_and_handle_buttons(_, close_clicks, start_clicks):
            trig = ctx.triggered_id
            if trig == "close-button" and close_clicks:
                price = self._get_price()
                ts = datetime.utcnow().replace(tzinfo=timezone.utc)
                if price is not None:
                    self.close_position(ts, price)
                else:
                    logger.warning("Could not fetch price for manual close.")

            if trig == "start-button" and start_clicks:
                self.enable_trading()

            return self._build_trades_table()

    # ── Trade-history helpers ------------------------------------------------
    def _get_trade_history_file(self):
        if TRADE_HISTORY_FILE:
            p = Path(TRADE_HISTORY_FILE)
        else:
            p = Path(f"trade_history_{self.symbol.replace('/', '_')}_{MODE}.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)

    def _load_trade_history(self):
        fn = self._get_trade_history_file()
        if os.path.exists(fn):
            try:
                with open(fn, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        print(f"Loaded {len(data)} trade records")
                        return data
            except Exception:
                logger.exception("Error loading trade history")
        print("No trade-history file – starting fresh.")
        return []

    def _save_trade_history(self):
        fn = self._get_trade_history_file()
        try:
            def ser(o):
                if isinstance(o, (datetime, pd.Timestamp)):
                    return o.isoformat()
                raise TypeError
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(self.trade_history, f, indent=4, default=ser)
        except Exception:
            logger.exception("Error saving trade history")
            print(f"{Colors.RED}Error saving trade history{Colors.END}")

    def _log_event(self, event_type, details):
        ts = details.get("timestamp", datetime.utcnow().isoformat())
        entry = {"timestamp": ts, "event": event_type, "details": details}
        self.trade_history.append(entry)
        print(f"[{ts}] {event_type.upper()}: {details}")
        logger.info("Trade event – %s – %s", event_type, details)
        self._save_trade_history()
        if event_type in ["BUY_EXECUTED", "SELL_EXECUTED"]:
            self._log_to_csv(details)
        return entry

    # ── CSV logger ------------------------------------------------------------
    def _get_csv_file(self):
        p = Path(f"trade_cycle_{self.symbol.replace('/', '_')}_{MODE}.csv")
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)

    def _log_to_csv(self, details):
        csv_fn = self._get_csv_file()
        file_exists = os.path.isfile(csv_fn)

        typ = details.get("type", "")
        is_open = typ in ["BUY_OPEN_LONG", "SELL_OPEN_SHORT"]
        is_close = typ in ["SELL_CLOSE_LONG", "BUY_CLOSE_SHORT"]
        if not (is_open or is_close):
            return

        trade_id = f"{typ}_{details.get('amount_base', 0)}_{details.get('timestamp', '')}"
        if is_open:
            self._open_trades_csv[trade_id] = details.copy()
            return

        counterpart = "BUY_OPEN_LONG" if typ == "SELL_CLOSE_LONG" else "SELL_OPEN_SHORT"
        match_key, open_detail = None, None
        for k, v in list(self._open_trades_csv.items()):
            if v.get("type") == counterpart and abs(v.get("amount_base", 0) - details.get("amount_base", 0)) < 1e-9:
                match_key, open_detail = k, v
                break
        if not open_detail:
            logger.warning("No matching open trade for %s", typ)
            return

        row = {
            "timestamp": open_detail.get("timestamp", ""),
            "type": f"{open_detail['type'].split('_')[2]}_{typ.split('_')[2]}",
            "symbol": self.symbol,
            "entry_price": open_detail.get("price", ""),
            "exit_price": details.get("price", ""),
            "amount_base": open_detail.get("amount_base", ""),
            "pnl": details.get("pnl", ""),
            "fee": (open_detail.get("fee_quote", 0) or 0) + (details.get("fee_quote", 0) or 0),
            "is_sl_tp": details.get("closed_by", "signal"),
        }

        fields = ["timestamp", "type", "symbol", "entry_price", "exit_price", "amount_base", "pnl", "fee", "is_sl_tp"]
        with open(csv_fn, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not file_exists:
                writer.writeheader()
                logger.info("Created CSV %s", csv_fn)
            writer.writerow(row)

        self._register_completed_trade(open_detail, details)
        self._open_trades_csv.pop(match_key, None)

    # ── Dashboard data registration -------------------------------------------
    def _register_completed_trade(self, open_d, close_d):
        side = "Long" if close_d["type"] == "SELL_CLOSE_LONG" else "Short"
        entry_price = float(open_d.get("price", np.nan))
        sl = open_d.get("sl")
        tp = open_d.get("tp")

        closed_by = close_d.get("closed_by", "signal")
        close_reason = closed_by
        if isinstance(closed_by, str) and closed_by.startswith("signal:"):
            close_reason = closed_by.split("signal:", 1)[1]

        with self._dash_lock:
            self.completed_trades.append({
                "side": side,
                "entry_time": pd.to_datetime(open_d.get("timestamp")),
                "entry_price": entry_price,
                "exit_time": pd.to_datetime(close_d.get("timestamp")),
                "exit_price": float(close_d.get("price", np.nan)),
                "amount": float(open_d.get("amount_base", np.nan)),
                "pnl": float(close_d.get("pnl", np.nan)),
                "sl": float(sl) if sl is not None else None,
                "tp": float(tp) if tp is not None else None,
                "open_signal": open_d.get("trigger_signal", "-"),
                "close_reason": close_reason,
            })

    # ── Balance & price helpers ----------------------------------------------
    def _get_balance(self):
        if MODE == "live":
            try:
                return self.exchange.fetch_balance()["total"]
            except Exception:
                logger.exception("Live balance fetch error")
        return self.simulated_balance

    def _update_balance(self, asset, delta):
        if MODE != "live":
            self.simulated_balance[asset] = self.simulated_balance.get(asset, 0) + delta

    def _get_price(self):
        try:
            if MODE == "live":
                return self.exchange.fetch_ticker(self.symbol)["last"]
            if MODE == "paper":
                return ccxt.binance().fetch_ticker(self.symbol)["last"]
        except Exception:
            logger.exception("Price fetch error")
        if hasattr(self, "historical_data") and not self.historical_data.empty:
            idx = getattr(self, "current_backtest_index", None)
            if idx is not None and 0 <= idx < len(self.historical_data):
                return self.historical_data["close"].iloc[idx]
        return None

    # ── Signal processing (TABLE-DRIVEN) --------------------------------------
    def _process_signal(self, rec, price, ts):
        """
        Apply trade actions per decision table:

        Current Pos | Signal                 | Action            | Function
        ------------|------------------------|-------------------|------------------------------
        Flat        | Buy / Strong Buy       | Open Long         | _execute_buy()
        Flat        | Sell / Strong Sell     | Open Short        | _execute_sell()
        Long        | Sell / Strong Sell     | Close Long        | _execute_sell(closed_by=signal)
        Short       | Buy / Strong Buy       | Close Short       | _execute_buy(closed_by=signal)
        Long        | Neutral / Buy / SB     | Hold              | —
        Short       | Neutral / Sell / SS    | Hold              | —

        SL/TP exits are handled earlier and take priority.
        """
        self.current_signal = rec or "Neutral"
        self.last_price = price
        print(f"{Colors.BOLD}[{ts}] SIGNAL: {rec}{Colors.END}")

        if not self.trading_enabled:
            print(f"{Colors.YELLOW}[{ts}] Trading is disabled. Ignoring signal: {rec}{Colors.END}")
            return

        rec = rec or "Neutral"

        # When Flat: open positions on Buy/SB or Sell/SS
        if self.position_state == "Flat":
            if rec in ("Buy", "Strong Buy"):
                self._execute_buy(price, ts)  # OPEN LONG
                self.open_signal = rec
                self.seen_strong_follow_through = False
            elif rec in ("Sell", "Strong Sell"):
                self._execute_sell(price, ts)  # OPEN SHORT
                self.open_signal = rec
                self.seen_strong_follow_through = False
            # Neutral -> Hold
            return

        # When Long: close on Sell/Strong Sell; else Hold
        if self.position_state == "Long":
            if rec in ("Sell", "Strong Sell"):
                self._execute_sell(price, ts, closed_by=f"signal:{rec}")  # CLOSE LONG
                self.open_signal = None
                self.seen_strong_follow_through = False
            return

        # When Short: close on Buy/Strong Buy; else Hold
        if self.position_state == "Short":
            if rec in ("Buy", "Strong Buy"):
                self._execute_buy(price, ts, closed_by=f"signal:{rec}")   # CLOSE SHORT
                self.open_signal = None
                self.seen_strong_follow_through = False
            return

    # ── Execution helpers -----------------------------------------------------
    def _execute_buy(self, price, ts, closed_by=None):
        if MODE not in ["paper", "backtest"]:
            print("Live buy not implemented."); logger.warning("Live buy not implemented"); return

        if self.position_state == "Flat":
            usdt = self.simulated_balance.get(self.quote_asset, 0)
            spend = min(FIXED_ORDER_SIZE_USDT, usdt)
            if spend <= 0:
                print("Insufficient USDT to open long.")
                return

            base_qty = spend / price
            fee = base_qty * 0.001
            base_qty -= fee

            self._update_balance(self.quote_asset, -spend)
            self._update_balance(self.base_asset, base_qty)

            self.position_state = "Long"
            self.entry_price = price
            self.amount_base = base_qty
            self.sl_level = price * (1 - STOP_LOSS_PERCENT / 100)
            self.tp_level = price * (1 + TAKE_PROFIT_PERCENT / 100)

            print(f"{Colors.GREEN}[{ts}] BUY LONG @ {price:.4f} – {base_qty:.6f} {self.base_asset}{Colors.END}")
            print(f"   Entry: {price:.4f} | SL: {self.sl_level:.4f} | TP: {self.tp_level:.4f}")

            trade = {
                "type": "BUY_OPEN_LONG",
                "timestamp": ts.isoformat(),
                "symbol": self.symbol,
                "price": price,
                "amount_quote": spend,
                "amount_base": base_qty,
                "fee_base": fee,
                "sl": self.sl_level,
                "tp": self.tp_level,
                "trigger_signal": self.current_signal or "Buy",
            }
            self._log_event("BUY_EXECUTED", trade)

        elif self.position_state == "Short":
            amount = self.amount_base
            cost = amount * price
            fee = cost * 0.001
            cost += fee
            pnl = (self.entry_price - price) * amount

            self._update_balance(self.quote_asset, -cost)
            self._update_balance(self.base_asset, amount)

            self.position_state = "Flat"
            self.entry_price = None
            self.amount_base = 0.0
            self.sl_level = None
            self.tp_level = None

            print(f"{Colors.GREEN}[{ts}] BUY CLOSE SHORT @ {price:.4f}{Colors.END}")
            print(f"   PnL: {pnl:.4f} {self.quote_asset}")

            trade = {
                "type": "BUY_CLOSE_SHORT",
                "timestamp": ts.isoformat(),
                "symbol": self.symbol,
                "price": price,
                "amount_base": amount,
                "pnl": pnl,
                "trigger_signal": "Buy",
                "closed_by": closed_by or "signal",
            }
            self._log_event("BUY_EXECUTED", trade)

    def _execute_sell(self, price, ts, closed_by=None):
        if MODE not in ["paper", "backtest"]:
            print("Live sell not implemented."); logger.warning("Live sell not implemented"); return

        if self.position_state == "Flat":
            base_bal = self.simulated_balance.get(self.base_asset, 0)
            amount = min(FIXED_ORDER_SIZE_USDT / price, base_bal)
            if amount <= 0:
                print("Insufficient base asset to open short.")
                return

            received = amount * price
            fee = received * 0.001
            received -= fee

            self._update_balance(self.base_asset, -amount)
            self._update_balance(self.quote_asset, received)

            self.position_state = "Short"
            self.entry_price = price
            self.amount_base = amount
            self.sl_level = price * (1 + STOP_LOSS_PERCENT / 100)
            self.tp_level = price * (1 - TAKE_PROFIT_PERCENT / 100)

            print(f"{Colors.RED}[{ts}] SELL SHORT @ {price:.4f} – {amount:.6f} {self.base_asset}{Colors.END}")
            print(f"   Entry: {price:.4f} | SL: {self.sl_level:.4f} | TP: {self.tp_level:.4f}")

            trade = {
                "type": "SELL_OPEN_SHORT",
                "timestamp": ts.isoformat(),
                "symbol": self.symbol,
                "price": price,
                "amount_base": amount,
                "fee_quote": fee,
                "sl": self.sl_level,
                "tp": self.tp_level,
                "trigger_signal": self.current_signal or "Sell",
            }
            self._log_event("SELL_EXECUTED", trade)

        elif self.position_state == "Long":
            amount = self.amount_base
            received = amount * price
            fee = received * 0.001
            received -= fee
            pnl = (price - self.entry_price) * amount

            self._update_balance(self.base_asset, -amount)
            self._update_balance(self.quote_asset, received)

            self.position_state = "Flat"
            self.entry_price = None
            self.amount_base = 0.0
            self.sl_level = None
            self.tp_level = None

            print(f"{Colors.RED}[{ts}] SELL CLOSE LONG @ {price:.4f}{Colors.END}")
            print(f"   PnL: {pnl:.4f} {self.quote_asset}")

            trade = {
                "type": "SELL_CLOSE_LONG",
                "timestamp": ts.isoformat(),
                "symbol": self.symbol,
                "price": price,
                "amount_base": amount,
                "pnl": pnl,
                "trigger_signal": "Sell",
                "closed_by": closed_by or "signal",
            }
            self._log_event("SELL_EXECUTED", trade)

    # ── SL/TP checks ----------------------------------------------------------
    def _check_sl_tp_on_candle(self, candle_ts, high, low):
        if self.position_state == "Flat" or self.entry_price is None:
            return False

        eps = 1e-9
        if self.position_state == "Long":
            if self.sl_level is not None and low <= self.sl_level + eps:
                self._execute_sell(self.sl_level, candle_ts, closed_by="SL")
                self.open_signal = None; self.seen_strong_follow_through = False
                return True
            if self.tp_level is not None and high >= self.tp_level - eps:
                self._execute_sell(self.tp_level, candle_ts, closed_by="TP")
                self.open_signal = None; self.seen_strong_follow_through = False
                return True

        elif self.position_state == "Short":
            if self.sl_level is not None and high >= self.sl_level - eps:
                self._execute_buy(self.sl_level, candle_ts, closed_by="SL")
                self.open_signal = None; self.seen_strong_follow_through = False
                return True
            if self.tp_level is not None and low <= self.tp_level + eps:
                self._execute_buy(self.tp_level, candle_ts, closed_by="TP")
                self.open_signal = None; self.seen_strong_follow_through = False
                return True
        return False

    def _check_sl_tp_on_tick(self, ts, price):
        if self.position_state == "Flat" or self.entry_price is None:
            return False

        if self.position_state == "Long":
            if self.sl_level is not None and price <= self.sl_level:
                self._execute_sell(self.sl_level, ts, closed_by="SL")
                self.open_signal = None; self.seen_strong_follow_through = False
                return True
            if self.tp_level is not None and price >= self.tp_level:
                self._execute_sell(self.tp_level, ts, closed_by="TP")
                self.open_signal = None; self.seen_strong_follow_through = False
                return True
        else:
            if self.sl_level is not None and price >= self.sl_level:
                self._execute_buy(self.sl_level, ts, closed_by="SL")
                self.open_signal = None; self.seen_strong_follow_through = False
                return True
            if self.tp_level is not None and price <= self.tp_level:
                self._execute_buy(self.tp_level, ts, closed_by="TP")
                self.open_signal = None; self.seen_strong_follow_through = False
                return True
        return False

    # ── Live candle utilities -------------------------------------------------
    @staticmethod
    def _tf_seconds(tf: str) -> int:
        if tf.endswith("m"):
            return int(tf[:-1]) * 60
        if tf.endswith("h"):
            return int(tf[:-1]) * 3600
        if tf.endswith("d"):
            return int(tf[:-1]) * 86400
        if tf.endswith("w"):
            return int(tf[:-1]) * 7 * 86400
        return 60

    @staticmethod
    def _floor_time(ts: datetime, seconds: int) -> datetime:
        epoch = int(ts.replace(tzinfo=timezone.utc).timestamp())
        floored = epoch - (epoch % seconds)
        return datetime.fromtimestamp(floored, tz=timezone.utc)

    def _prime_live_candles(self, limit: int = 200):
        if not self.live_fetcher:
            return
        try:
            df = self.live_fetcher.fetch_data(self.primary_tf, limit=limit)
            if df is not None and not df.empty:
                with self._dash_lock:
                    self.candle_history = [
                        {"timestamp": ts.to_pydatetime().replace(tzinfo=timezone.utc),
                         "open": float(r["open"]), "high": float(r["high"]),
                         "low": float(r["low"]), "close": float(r["close"])}
                        for ts, r in df.iterrows()
                    ]
        except Exception:
            logger.exception("Prime live candles failed")

    def _update_live_candle(self, now: datetime, price: float):
        secs = self._tf_seconds(self.primary_tf)
        bucket = self._floor_time(now, secs)
        with self._dash_lock:
            if not self.candle_history:
                self.candle_history.append(
                    {"timestamp": bucket, "open": price, "high": price, "low": price, "close": price}
                )
                return
            last = self.candle_history[-1]
            last_bucket = last["timestamp"]
            if isinstance(last_bucket, pd.Timestamp):
                last_bucket = last_bucket.to_pydatetime().replace(tzinfo=timezone.utc)

            if bucket > last_bucket:
                self.candle_history.append(
                    {"timestamp": bucket, "open": price, "high": price, "low": price, "close": price}
                )
            else:
                last["close"] = price
                if price > last["high"]:
                    last["high"] = price
                if price < last["low"]:
                    last["low"] = price

    # ── Back-test driver -------------------------------------------------------
    def run_backtest(self, timeframe=None, days=30):
        tf = timeframe or self.primary_tf
        print(f"{Colors.MAGENTA}Back-test start – {self.symbol} – TF={tf} – {days}d{Colors.END}")
        logger.info("Back-test start – %s – %s – %d days", self.symbol, tf, days)

        self.historical_data = self.backtest_fetcher.fetch_historical_data(tf, days)
        if self.historical_data.empty:
            print("No data – abort."); return

        tf_map = {t: days + 20 for t in ["5m", "15m", "1h", "1d", "1w"]}
        all_tf = fetch_multiple_timeframes_concurrently(self.backtest_fetcher, tf_map)

        warmup = {"5m": 200, "15m": 150, "1h": 100, "1d": 10, "1w": 2}.get(tf, 100)
        if len(self.historical_data) <= warmup:
            print("Not enough candles for warm-up."); return
        start_idx = warmup

        self._start_dash_server()

        start_price = self.historical_data["close"].iloc[0]
        peak_val = self.simulated_balance[self.quote_asset] + self.simulated_balance[self.base_asset] * start_price
        max_dd = 0.0

        for i in range(start_idx, len(self.historical_data)):
            self.current_backtest_index = i
            candle = self.historical_data.iloc[i]
            ts = candle.name
            open_ = candle["open"]; high = candle["high"]; low = candle["low"]; close = candle["close"]

            with self._dash_lock:
                self.candle_history.append(
                    {"timestamp": ts, "open": open_, "high": high, "low": low, "close": close}
                )
                port_val = (self.simulated_balance[self.quote_asset] +
                            self.simulated_balance[self.base_asset] * close)
                self.portfolio_history.append({"timestamp": ts, "price": close, "portfolio_value": port_val})
                self.last_price = close

            # SL/TP first
            if self._check_sl_tp_on_candle(ts, high, low):
                pass

            # Build slices for strategy
            slices = {k: v[v.index <= ts] for k, v in all_tf.items()}
            temp_analyzer = TechnicalAnalyzer(self.symbol)
            temp_analyzer.data_fetcher = type(
                "SliceFetcher",
                (),
                {"fetch_data": lambda self, tf, limit=500: slices.get(tf, pd.DataFrame()).tail(limit)},
            )()

            # Plan for cards: compute from slices (if available)
            try:
                df1h = slices.get("1h", pd.DataFrame()).tail(200)
                df5m = slices.get("5m", pd.DataFrame()).tail(300)
                if not df1h.empty and not df5m.empty:
                    news_score = 0.71 if getattr(self, "use_sentiment", True) else 0.0
                    self.last_plan = compute_trade_plan_1h_5m(df1h, df5m, sentiment=news_score)
            except Exception:
                logger.exception("Plan compute failed (backtest)")

            try:
                signals = temp_analyzer.analyze_all_timeframes()
            except Exception:
                logger.exception("Signal error at %s", ts)
                signals = {}

            rec = signals.get(tf, {}).get("overall_recommendation", "")
            self._process_signal(rec, close, ts)

            if port_val > peak_val:
                peak_val = port_val
            else:
                dd = (peak_val - port_val) / peak_val * 100
                max_dd = max(max_dd, dd)

        final_price = self.historical_data["close"].iloc[-1]
        final_ts = self.historical_data.index[-1]
        final_port = (self.simulated_balance[self.quote_asset] +
                      self.simulated_balance[self.base_asset] * final_price)
        with self._dash_lock:
            self.portfolio_history.append(
                {"timestamp": final_ts, "price": final_price, "portfolio_value": final_port}
            )

        net = final_port - peak_val
        net_pct = net / peak_val * 100
        print(f"\n{Colors.CYAN}Back-test completed – Net PnL: {net:.2f} ({net_pct:.2f}%) – Max DD: {max_dd:.2f}%{Colors.END}")
        logger.info("Back-test finished – Net PnL %.2f – Max DD %.2f", net, max_dd)
        self._save_trade_history()

    # ── Live / paper loop -----------------------------------------------------
    def run_live_paper(self):
        print(f"{Colors.CYAN}Live/Paper mode – TF={self.primary_tf}{Colors.END}")
        self._start_dash_server()
        self._prime_live_candles(limit=200)

        iteration = 0
        while True:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")
            price = self._get_price()
            if price is None:
                print("No price – wait.")
                time.sleep(3)
                continue

            now = datetime.utcnow().replace(tzinfo=timezone.utc)
            self.last_price = price
            self._update_live_candle(now, price)

            port_val = (self.simulated_balance[self.quote_asset] +
                        self.simulated_balance[self.base_asset] * price)

            with self._dash_lock:
                self.portfolio_history.append(
                    {"timestamp": now, "price": price, "portfolio_value": port_val}
                )

            # update plan regularly for cards
            try:
                df_1h = self.live_fetcher.fetch_data("1h", limit=200)
                df_5m = self.live_fetcher.fetch_data("5m", limit=300)
                news_score = 0.71 if getattr(self, "use_sentiment", True) else 0.0
                self.last_plan = compute_trade_plan_1h_5m(df_1h, df_5m, sentiment=news_score)
            except Exception:
                logger.exception("Plan compute failed (live)")

            if self._check_sl_tp_on_tick(now, price):
                time.sleep(2)
                continue

            signals = self.strategy.get_signals()
            rec = signals.get(self.primary_tf, {}).get("overall_recommendation", "")
            self._process_signal(rec, price, now)

            time.sleep(2)

    # ── Server helpers --------------------------------------------------------
    def _start_dash_server(self):
        if self._dash_server:
            return
        def run():
            self._dash_server = app.server
            logger.info("Dash server started – http://127.0.0.1:8050")
            app.run(host="127.0.0.1", port=8050, debug=False, use_reloader=False)
        self._dash_thread = threading.Thread(target=run, daemon=True)
        self._dash_thread.start()
        print("\nDashboard launched – open http://127.0.0.1:8050")
        logger.info("Dashboard launched")

    def _shutdown_dash_server(self):
        if self._dash_server:
            try:
                requests.get("http://127.0.0.1:8050/_shutdown", timeout=1)
                logger.info("Dashboard shutdown request sent")
            except Exception:
                logger.debug("Dashboard shutdown endpoint not available")
            finally:
                self._dash_server = None

    # ── Runner ---------------------------------------------------------------
    def run(self, backtest_tf=None, backtest_days=30):
        if MODE == "backtest":
            self.run_backtest(backtest_tf, backtest_days)
        else:
            self.run_live_paper()

    def stop(self):
        print("Stopping bot...")
        self._save_trade_history()
        self._shutdown_dash_server()
        logger.info("Bot stopped")
        print("Bot stopped.")


# ── CLI ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto Trading Bot (Light Mode Dashboard)")
    parser.add_argument("--timeframe", default="15m", choices=["5m", "15m", "1h", "1d", "1w"])
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--no-sentiment", dest="no_sentiment", action="store_true",
                        help="Disable sentiment bias in plan")
    args = parser.parse_args()

    bot = CryptoTradingBot(symbol=args.symbol, primary_tf=args.timeframe)
    bot.use_sentiment = not args.no_sentiment  # toggle sentiment used in plan

    try:
        bot.run(backtest_tf=args.timeframe, backtest_days=args.days)
    except KeyboardInterrupt:
        print(f"{Colors.YELLOW}Interrupted by user.{Colors.END}")
        logger.info("Interrupted by user")
    finally:
        bot.stop()
