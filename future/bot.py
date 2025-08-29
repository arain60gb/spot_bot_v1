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
        self.trading_enabled = True                     # governs auto entries when FLAT
        self.manual_lock_active = False                 # when True, signals CANNOT close/flip; requires manual close
        self.last_price = None; self.current_signal = "Neutral"
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

    # --- OPEN/CLOSE with manual flag -----------------------------------------
    def _open_long(self, price, ts, trigger="signal", manual: bool=False):
        if self.position_side != "Flat": return
        qty, notional = self._qty_for_price(price); fee = self._taker_fee_quote(notional)
        margin = notional / self.leverage
        if MODE != "live":
            if self.equity_usdt < (margin + fee): print("Insufficient equity to open long."); return
            self.equity_usdt -= (margin + fee); self.margin_locked = margin
        self.position_side = "Long"; self.entry_price = price; self.qty_base = qty
        self.sl_level = price * (1 - STOP_LOSS_PERCENT / 100.0)
        self.tp_level = price * (1 + TAKE_PROFIT_PERCENT / 100.0) if TAKE_PROFIT_PERCENT > 0 else None
        self.manual_lock_active = bool(manual)  # lock if manual entry
        logger.info("OPEN LONG @ %.4f qty=%.6f notional=%.2f fee=%.4f manual=%s", price, qty, notional, fee, manual)
        self._log_event("FUTURES_OPEN_LONG", {
            "timestamp": ts.isoformat(),"price": price,"qty": qty,"notional": notional,
            "fee_quote": fee,"sl": self.sl_level,"tp": self.tp_level,"trigger": trigger,
            "manual": manual
        })

    def _open_short(self, price, ts, trigger="signal", manual: bool=False):
        if self.position_side != "Flat": return
        qty, notional = self._qty_for_price(price); fee = self._taker_fee_quote(notional)
        margin = notional / self.leverage
        if MODE != "live":
            if self.equity_usdt < (margin + fee): print("Insufficient equity to open short."); return
            self.equity_usdt -= (margin + fee); self.margin_locked = margin
        self.position_side = "Short"; self.entry_price = price; self.qty_base = qty
        self.sl_level = price * (1 + STOP_LOSS_PERCENT / 100.0)
        self.tp_level = price * (1 - TAKE_PROFIT_PERCENT / 100.0) if TAKE_PROFIT_PERCENT > 0 else None
        self.manual_lock_active = bool(manual)  # lock if manual entry
        logger.info("OPEN SHORT @ %.4f qty=%.6f notional=%.2f fee=%.4f manual=%s", price, qty, notional, fee, manual)
        self._log_event("FUTURES_OPEN_SHORT", {
            "timestamp": ts.isoformat(),"price": price,"qty": qty,"notional": notional,
            "fee_quote": fee,"sl": self.sl_level,"tp": self.tp_level,"trigger": trigger,
            "manual": manual
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
        # Reset position state; manual lock is cleared when position is no longer open
        self.position_side = "Flat"; self.entry_price = None; self.qty_base = 0.0
        self.sl_level = None; self.tp_level = None; self.margin_locked = 0.0
        self.manual_lock_active = False

    # --- Signal engine --------------------------------------------------------
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
        """
        - If manual_lock_active: DO NOT flip or close based on signals. Hold until manual close (SL/TP still apply).
        - If not manual_lock_active:
            • When FLAT: only open if trading_enabled.
            • When IN POSITION: allow signal-based flips as before.
        """
        rec = quick_sig.signal or "Neutral"; self.current_signal = rec; self.last_price = price

        # Manual lock: ignore signal-based exits/flip logic entirely
        if self.manual_lock_active:
            return

        # Normal mode
        if not self.trading_enabled:
            # No auto-entries when flat
            if self.position_side == "Flat":
                return

        if self.position_side == "Flat":
            if rec == "Buy": self._open_long(price, ts, trigger="quick_1m_5m", manual=False)
            if rec == "Sell": self._open_short(price, ts, trigger="quick_1m_5m", manual=False)
            return

        # In-position & not manual-locked: allow flips
        if self.position_side == "Long" and rec == "Sell":
            self._close_position(price, ts, closed_by="signal:flip_to_short"); self._open_short(price, ts, trigger="flip", manual=False); return
        if self.position_side == "Short" and rec == "Buy":
            self._close_position(price, ts, closed_by="signal:flip_to_long"); self._open_long(price, ts, trigger="flip", manual=False); return

    # --- Persistence ----------------------------------------------------------
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

    # --- UI helpers -----------------------------------------------------------
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

    # --- Dash layout & callbacks ---------------------------------------------
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
            """
            New behavior:
              • 'Close Trade' closes (if any) AND pauses auto-entries until Start/LongTrig/ShortTrig.
              • LongTrig/ShortTrig open MANUAL positions (no signal-based flips/closes) and resume trading_enabled=True.
              • Start Trade only resumes auto-entries when flat; it does not override manual lock if position is open.
            """
            trig = ctx.triggered_id

            if trig == "close-button" and close_clicks:
                price = self._get_price(); ts = datetime.utcnow().replace(tzinfo=timezone.utc)
                if self.position_side != "Flat" and price is not None:
                    self._close_position(price, ts, closed_by="Manual Close")
                # Pause after manual close; also ensure manual lock is off
                self.trading_enabled = False
                self.manual_lock_active = False
                logger.info("Manual Close pressed: trade closed (if any) and auto-trading is now PAUSED.")

            if trig == "start-button" and start_clicks:
                # Only resume auto entries. If a manual position is open, manual lock still blocks signal flips.
                self.trading_enabled = True
                logger.info("Trading enabled via Start Trade button (auto entries allowed when flat).")

            if trig == "shorttrig-button" and shorttrig_clicks:
                price = self._get_price(); ts = datetime.utcnow().replace(tzinfo=timezone.utc)
                if price is not None and self.position_side == "Flat":
                    self.trading_enabled = True  # resume
                    self._open_short(price, ts, trigger="manual_short", manual=True)
                    logger.info("Manual SHORT opened; signals cannot flip/close until manual close (SL/TP still active).")

            if trig == "longtrig-button" and longtrig_clicks:
                price = self._get_price(); ts = datetime.utcnow().replace(tzinfo=timezone.utc)
                if price is not None and self.position_side == "Flat":
                    self.trading_enabled = True  # resume
                    self._open_long(price, ts, trigger="manual_long", manual=True)
                    logger.info("Manual LONG opened; signals cannot flip/close until manual close (SL/TP still active).")

            return self._build_trades_table()

    # --- Time utils & data feed ----------------------------------------------
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

    # --- Runner ---------------------------------------------------------------
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
