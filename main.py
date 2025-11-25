# main.py
import os
import pandas as pd
import glob
from datetime import datetime
from colorama import Fore, Style, init
from engine.data_feed import DataFeed
from engine.trade_manager import TradeManager
from engine.core.structure_logger import CSLStructureLogger
from engine.erm_module import EnergyResonanceMapping
from engine.structure import MultiTimeframeResonanceManager
from engine.adaptive_decision_engine import AdaptiveDecisionEngine
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from typing import Optional, List, Dict, Any

# LIVE MODE SWITCHES
LIVE_TRADING = False           # Enable for live trading via broker (CCXT/Binance)
LIVE_MT5_TRADING = False       # Enable for live trading via MetaTrader5

init(autoreset=True)

# Initialize structure logger
structure_logger = CSLStructureLogger()

# MT5 Broker (lazy-loaded if needed)
_mt5_broker = None

def get_mt5_broker():
    """Lazy-load MT5 broker on demand."""
    global _mt5_broker
    if _mt5_broker is None and LIVE_MT5_TRADING:
        try:
            from engine.mt5_broker import MT5Broker
            _mt5_broker = MT5Broker(initialize=True)
        except Exception as e:
            print(f"Failed to initialize MT5Broker: {e}")
    return _mt5_broker

# === FULL CSLStrategy CLASS (NO calibrate() call) ===
class CSLStrategy:
    def __init__(self, trade_manager=None, ade_rrl_bridge=None, params=None, log_fn=None):
        self.trade_manager = trade_manager
        self.ade_rrl_bridge = ade_rrl_bridge
        self.params = params or {}
        self.log_fn = log_fn or print

        # ----- CSL state ------------------------------------------------
        self.arm_history: List[Dict[str, Any]] = []
        self.current_streak = 0
        self.corner_flag = False
        self.lol_level: Optional[float] = None

        # ----- Position tracking -----------------------------------------
        self.position: Optional[str] = None
        self.entry_price: Optional[float] = None
        self.trade_history: List[Dict[str, Any]] = []

        # ----- ADE -------------------------------------------------------
        self.ade = None

        # ----- Live status ------------------------------------------------
        self.status_path = Path("data/rrl_status.json")
        self.status_path.parent.mkdir(parents=True, exist_ok=True)

        # ----- RRL feedback tracking --------------------------------------
        self.last_eps_vec = None
        self.last_pred_score = 0.0

        # ----- Risk Management Params -------------------------------------
        self.max_risk_per_trade = 0.10
        self.account_balance = 10000.0
        self.risk_reward_ratio = 2.0
        self.atr_multiplier = 1.5
        self.atr_period = 14

    def generate_signal(self, data):
        return "NO_SIGNAL", "Warming up|Data<15", None, None

    def _corner_detected(self, candle):
        return False

    def _compute_lol(self):
        return 0.0

    def _lol_confirms(self, direction, candle, lol):
        return False


class Backtester:
    def __init__(self, data_path="data/candles_*.csv", log_file="data/trade_log.csv", summary_file="data/backtest_summary.csv"):
        self.data_path = data_path
        self.log_file = log_file
        self.summary_file = summary_file
        self.trade_manager = TradeManager(log_file=log_file)
        self.resonance_manager = MultiTimeframeResonanceManager()
        self.strategies = {"BTCUSDT": CSLStrategy(self.trade_manager)}
        self.ade = None
        try:
            self.ade = AdaptiveDecisionEngine(
                resonance_manager=self.resonance_manager,
                trade_manager=self.trade_manager,
                strategies=self.strategies,
                config={"debug": False, "risk_per_trade_pct": 0.01},
                log_fn=print
            )
            # NO calibrate() — ADE does not have it
            print("ADE initialized (no calibration needed).")
            # Write status manually
            with open(self.strategies["BTCUSDT"].status_path, "w") as f:
                f.write("INITIALIZED")
        except Exception as e:
            print(f"ADE init failed: {e}")
        self.data_feed = None
        self.data = None
        os.makedirs("data", exist_ok=True)
        self.erm = EnergyResonanceMapping()

    def load_data(self, symbol_file):
        self.data_feed = DataFeed(symbol_file)
        self.data = self.data_feed.get_dataframe()
        if "datetime" in self.data.columns:
            self.data.set_index("datetime", inplace=True)
        elif "time" in self.data.columns:
            self.data.set_index("time", inplace=True)
        print(f"Loaded {len(self.data)} candles for {symbol_file}")

    def update_trailing_sl(self, trade, candle):
        direction = trade["direction"]
        momentum_candles = [h["candle"] for h in self.trade_manager.high_momentum_history if h["direction"] == direction]
        if momentum_candles:
            last_momentum = momentum_candles[-1]
            lol = (last_momentum["high"] + last_momentum["low"]) / 2.0
            if direction == "long" and lol > trade["sl"]:
                trade["sl"] = lol
                trade["reason"] += "; Trailing SL updated"
            elif direction == "short" and lol < trade["sl"]:
                trade["sl"] = lol
                trade["reason"] += "; Trailing SL updated"
        return trade

    def place_trade(self, signal, entry_price, sl, tp, reason, candle):
        if LIVE_TRADING and hasattr(self, 'broker'):
            symbol = "BTC/USDT"
            amount = 0.001
            side = "buy" if signal == "BUY" else "sell"
            order = self.broker.place_order(symbol=symbol, side=side, amount=amount, price=entry_price)
            if order:
                print(Fore.CYAN + f"LIVE ORDER: {side.upper()} {amount} {symbol} @ {entry_price}")
            else:
                print(Fore.RED + "LIVE ORDER FAILED")
        elif LIVE_MT5_TRADING:
            # Live trading via MT5
            mt5_broker = get_mt5_broker()
            if mt5_broker:
                symbol = "EURUSD"  # Default MT5 symbol
                amount = 0.01      # Default MT5 lot size
                side = "buy" if signal == "BUY" else "sell"
                order = mt5_broker.place_order(symbol=symbol, side=side, amount=amount, price=entry_price)
                if order:
                    print(Fore.CYAN + f"MT5 LIVE ORDER: {side.upper()} {amount} {symbol} @ {entry_price}")
                else:
                    print(Fore.RED + "MT5 LIVE ORDER FAILED")
            else:
                print(Fore.YELLOW + "MT5Broker not initialized, skipping order")
        else:
            # Backtesting mode
            direction = "long" if signal == "BUY" else "short"
            tid = self.trade_manager.open_trade(
                direction=direction,
                entry_price=entry_price,
                candle_time=candle.name,
                signal_time=candle.name,
                signal=signal,
                sl=sl,
                tp=tp,
                momentum_streak=self.strategies["BTCUSDT"].current_streak,
                corner=self.strategies["BTCUSDT"]._corner_detected(candle),
                reason=reason
            )
            self.trade_manager.add_high_momentum_candle(candle, direction)
            print(Fore.CYAN + f"Opened {tid} ({signal}) at {entry_price:.5f} | TP={tp:.5f}, SL={sl:.5f}")

    def run(self, symbol_file):
        self.load_data(symbol_file)
        symbol = os.path.basename(symbol_file).replace("candles_", "").replace(".csv", "")
        print(f"\nStarting Backtest for {symbol}...\n")

        previous_candle = None

        for i in range(3, len(self.data)):
            candle = self.data.iloc[i]
            current_time = candle.name

            signal, reason, sl, tp = self.strategies["BTCUSDT"].generate_signal(self.data.iloc[:i + 1])

            minute_point = "3rd-4th" if current_time.minute % 5 in [2, 3] else None
            structure_logger.log(
                candle_time=current_time,
                index=i,
                direction="BUY" if candle["close"] > candle["open"] else "SELL",
                momentum=self.strategies["BTCUSDT"].current_streak,
                corner=self.strategies["BTCUSDT"]._corner_detected(candle),
                lol_status="Confirm" if self.strategies["BTCUSDT"]._lol_confirms("bullish" if signal == "BUY" else "bearish", candle, self.strategies["BTCUSDT"]._compute_lol()) else "Reject",
                minute_point=minute_point,
                meme_arm=self.strategies["BTCUSDT"].current_streak,
                structure_note=""
            )

            print(f"{current_time} | O:{candle['open']} H:{candle['high']} L:{candle['low']} C:{candle['close']}")
            if signal.startswith("BUY"):
                print(Fore.GREEN + f"{signal}: {reason}")
            elif signal.startswith("SELL"):
                print(Fore.RED + f"{signal}: {reason}")
            elif signal.startswith("EXIT"):
                print(Fore.MAGENTA + f"{signal}: {reason}")
            else:
                print(Fore.YELLOW + f"{signal}: {reason}")

            if previous_candle is not None:
                eri_info = self.erm.compute_eri(candle, previous_candle)
                event = self.erm.detect_energy_resonance(eri_info)
                print(f"{eri_info['time']} | ERI={eri_info['ERI']:.3f} | Phase={eri_info['phase']} | {event or ''}")
            previous_candle = candle

            for tid, trade in list(self.trade_manager.get_open_trades().items()):
                trade = self.update_trailing_sl(trade, candle)
                if trade["direction"] == "long":
                    self.trade_manager.long_trades[tid] = trade
                else:
                    self.trade_manager.short_trades[tid] = trade

                exit_price = None
                exit_reason = None
                if trade["sl"] and trade["tp"]:
                    if trade["direction"] == "long":
                        if candle["low"] <= trade["sl"]:
                            exit_price, exit_reason = trade["sl"], "SL hit"
                        elif candle["high"] >= trade["tp"]:
                            exit_price, exit_reason = trade["tp"], "TP hit"
                    else:
                        if candle["high"] >= trade["sl"]:
                            exit_price, exit_reason = trade["sl"], "SL hit"
                        elif candle["low"] <= trade["tp"]:
                            exit_price, exit_reason = trade["tp"], "TP hit"
                if signal.startswith("EXIT"):
                    exit_price, exit_reason = candle["close"], "EXIT signal"

                if exit_price:
                    closed = self.trade_manager.close_trade(tid, exit_price, current_time, current_time, signal, exit_reason)
                    if closed:
                        print(Fore.MAGENTA + f"Closed {tid} at {exit_price:.5f} | P/L: {closed['pnl']:.5f} | {exit_reason}")

            if signal in ["BUY", "SELL"]:
                entry_price = candle["close"]
                if sl is None or tp is None:
                    sl = entry_price * (0.998 if signal == "BUY" else 1.002)
                    tp = entry_price * (1.002 if signal == "BUY" else 0.998)
                self.place_trade(signal, entry_price, sl, tp, reason, candle)

        self._summarize_results(symbol)
        self.plot_candlestick(symbol_file)

    def run_multi_symbol(self):
        symbol_files = glob.glob(self.data_path) if "candles_*.csv" in self.data_path else [self.data_path]
        if not symbol_files:
            print(f"No data files found at {self.data_path}")
            return
        for symbol_file in symbol_files:
            log_file = f"data/trade_log_{os.path.basename(symbol_file).replace('.csv', '')}.csv"
            self.trade_manager = TradeManager(log_file=log_file)
            self.strategies = {"BTCUSDT": CSLStrategy(self.trade_manager)}
            if self.ade:
                self.ade.strategies = self.strategies
            self.run(symbol_file)

    def _summarize_results(self, symbol):
        closed = [t for t in self.trade_manager.get_all_trades().values() if t["status"] == "closed"]
        if not closed:
            print(f"No closed trades for {symbol}.")
            return
        df = pd.DataFrame(closed)
        total_pnl = df["pnl"].sum()
        win_rate = len(df[df["pnl"] > 0]) / len(df) * 100
        print(f"\n--- Summary for {symbol} ---")
        print(f"Trades: {len(closed)} | PnL: {total_pnl:.5f} | Win Rate: {win_rate:.1f}%")
        df.to_csv(f"data/backtest_summary_{symbol}.csv", index=False)

    def plot_candlestick(self, symbol_file):
        symbol = os.path.basename(symbol_file).replace("candles_", "").replace(".csv", "")
        df = self.data.copy()
        trades = pd.DataFrame([t for t in self.trade_manager.get_all_trades().values() if t["status"] == "closed"])
        if trades.empty:
            print(f"No trades to plot for {symbol}.")
            return
        plt.figure(figsize=(14, 7))
        for idx, row in df.iterrows():
            color = 'green' if row['close'] >= row['open'] else 'red'
            plt.plot([idx, idx], [row['low'], row['high']], color='black')
            plt.bar(idx, abs(row['close'] - row['open']), bottom=min(row['open'], row['close']), color=color, width=0.6)
        for _, t in trades.iterrows():
            try:
                entry_idx = df.index.get_loc(t['entry_time'])
                exit_idx = df.index.get_loc(t['exit_time'])
                color = 'green' if t['pnl'] > 0 else 'red'
                plt.plot([entry_idx, exit_idx], [t['entry'], t['exit']], color=color, linestyle='--')
                plt.scatter(entry_idx, t['entry'], color='blue', marker='^', s=120)
                plt.scatter(exit_idx, t['exit'], color='purple', marker='v', s=120)
            except:
                pass
        plt.title(f"{symbol} Trades")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"data/chart_{symbol}.png", dpi=150)
        plt.close()
        print(f"Chart saved: data/chart_{symbol}.png")


if __name__ == "__main__":
    backtester = Backtester()
    backtester.run_multi_symbol()