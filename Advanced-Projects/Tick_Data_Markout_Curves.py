#!/usr/bin/env python3
import os
import json
import logging
import time
import polars as pl
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import skew, kurtosis

# Attempt to import CuPy for GPU acceleration.
try:
    import cupy as cp
except ImportError:
    cp = None
USE_GPU = cp is not None

# Configure logging for detailed runtime info.
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


# =============================================================================
# TRADE ANALYZER CLASS
# =============================================================================
class TradeAnalyzer:
    def __init__(self, instrument: str, data_dir: str = "data", reports_dir: str = "reports", use_gpu: bool = False):
        self.instrument = instrument
        self.data_dir = data_dir
        self.reports_dir = reports_dir
        self.use_gpu = use_gpu and (cp is not None)
        self.meta = {}
        self.horizon_ticks = []
        self.mkt_df = pl.DataFrame()
        self.trades_df = pl.DataFrame()

    # -------------------------------------------------------------------------
    # Data Loading and Preprocessing
    # -------------------------------------------------------------------------
    def load_data(self) -> bool:
        md_file = os.path.join(self.data_dir, f"{self.instrument}_md.csv")
        trades_file = os.path.join(self.data_dir, f"{self.instrument}_trades.csv")
        meta_file = os.path.join(self.data_dir, f"{self.instrument}.json")
        ticks_file = os.path.join(self.data_dir, "horizon_ticks")

        for fp in [md_file, trades_file, meta_file, ticks_file]:
            if not os.path.exists(fp):
                logging.error(f"Missing required file: {fp}")
                return False

        try:
            self.mkt_df = pl.read_csv(md_file, try_parse_dates=False)
            self.trades_df = pl.read_csv(trades_file, try_parse_dates=False)
            with open(meta_file, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
            with open(ticks_file, "r", encoding="utf-8") as f:
                ticks = json.load(f)
                self.horizon_ticks = [int(tick) for tick in ticks]

            logging.info("Data loaded successfully.")
            return True
        except Exception as e:
            logging.error(f"Error during data loading: {e}")
            return False

    def preprocess_data(self):
        try:
            # Preprocess market data.
            self.mkt_df = self.mkt_df.with_columns([
                pl.col("ts_ms").cast(pl.Int64),
                ((pl.col("bid") + pl.col("ask")) / 2).alias("mid_price")
            ]).sort("ts_ms")
            # Preprocess trades data.
            self.trades_df = self.trades_df.with_columns([
                pl.col("ts_ms").cast(pl.Int64),
                pl.col("px").cast(pl.Float64),
                pl.col("size").cast(pl.Float64),
                pl.col("side").str.to_lowercase()
            ]).sort("ts_ms")
            logging.info("Data preprocessed successfully.")
        except Exception as e:
            logging.error(f"Error in data preprocessing: {e}")

    # -------------------------------------------------------------------------
    # One‑Hour Period Selection
    # -------------------------------------------------------------------------
    def select_one_hour_period(self) -> (pl.DataFrame, int):
        # Use the appropriate module (CuPy if enabled, else NumPy)
        xp = cp if self.use_gpu else np

        # Convert columns to arrays using xp
        mkt_ts = xp.array(self.mkt_df["ts_ms"].to_numpy())
        candidate_starts = mkt_ts[::300]
        trade_ts = xp.array(self.trades_df["ts_ms"].to_numpy())

        # Calculate number of trades in each candidate period
        left_indices = xp.searchsorted(trade_ts, candidate_starts, side="left")
        right_indices = xp.searchsorted(trade_ts, candidate_starts + 3600000, side="left")
        trade_counts = right_indices - left_indices

        mid_prices = xp.array(self.mkt_df["mid_price"].to_numpy())

        # Compute volatility for each candidate start.
        def compute_std(start):
            mask = (mkt_ts >= start) & (mkt_ts < start + 3600000)
            if xp.sum(mask) > 0:
                return float(mid_prices[mask].std())
            else:
                return 0.0

        # Compute volatilities
        vol_list = [compute_std(start) for start in candidate_starts]
        volatilities = xp.array(vol_list)

        # Compute scores for each candidate period
        scores = trade_counts * volatilities
        best_index = int(xp.argmax(scores))
        best_start = int(candidate_starts[best_index])
        logging.info(f"Selected one‑hour period starting at {pd.to_datetime(best_start, unit='ms')} with score {float(scores[best_index]):.2f}")

        one_hour_df = self.mkt_df.filter((pl.col("ts_ms") >= best_start) & (pl.col("ts_ms") < best_start + 3600000))
        return one_hour_df, best_start

    # -------------------------------------------------------------------------
    # Trade Markout Curve Computation
    # -------------------------------------------------------------------------
    def compute_trade_markout_curves(self) -> (np.ndarray, np.ndarray):
        # Use the appropriate module (CuPy if enabled, else NumPy)
        xp = cp if self.use_gpu else np
        # Convert market data
        mkt_pdf = self.mkt_df.to_pandas()
        mkt_ts = mkt_pdf["ts_ms"].values
        mid_prices = mkt_pdf["mid_price"].values
        # Convert trade data
        trades_pdf = self.trades_df.to_pandas()
        trade_times = trades_pdf["ts_ms"].values
        trade_px = trades_pdf["px"].values
        trade_size = trades_pdf["size"].values
        trade_side = np.where(trades_pdf["side"].values == "b", 1, -1)
        # Use GPU arrays if applicable
        if self.use_gpu:
            mkt_ts = xp.array(mkt_ts)
            mid_prices = xp.array(mid_prices)
            trade_times = xp.array(trade_times)
            trade_px = xp.array(trade_px)
            trade_size = xp.array(trade_size)
            trade_side = xp.array(trade_side)
            horizon_ticks = xp.array(self.horizon_ticks)
        else:
            horizon_ticks = np.array(self.horizon_ticks)
        # Compute lookup times for each trade and horizon tick
        lookup_times = trade_times[:, None] + horizon_ticks[None, :]
        indices = xp.searchsorted(mkt_ts, lookup_times, side="left")
        indices = xp.clip(indices, 0, len(mkt_ts) - 1)
        market_prices = mid_prices[indices]
        # Compute pnl for each trade at each horizon tick
        pnl = trade_side[:, None] * (market_prices - trade_px[:, None]) * trade_size[:, None]
        # Convert back to numpy if GPU was used
        if self.use_gpu:
            pnl = cp.asnumpy(pnl)
            trade_size = cp.asnumpy(trade_size)
        return pnl, trade_size

    def compute_aggregate_markout(self) -> np.ndarray:
        pnl, trade_size = self.compute_trade_markout_curves()
        # Compute overall margin at each horizon tick
        total_pnl = np.sum(pnl, axis=0)
        total_size = np.sum(trade_size[:, None], axis=0)
        agg_curve = np.where(total_size != 0, total_pnl / total_size, 0)
        logging.info("Computed weighted aggregate markout curve.")
        return agg_curve

    # -------------------------------------------------------------------------
    # Interesting Strategy Characteristics
    # -------------------------------------------------------------------------
    def compute_characteristics(self) -> (dict, pd.DataFrame):
        # Compute strategy pnl and cumulative pnl

        # Convert data to Pandas for easier processing
        mkt_pdf = self.mkt_df.to_pandas()
        trades_pdf = self.trades_df.to_pandas()
        xp = cp if self.use_gpu else np

        # Convert data to NumPy arrays
        mkt_times = xp.array(mkt_pdf["ts_ms"].values)
        market_mid_prices = xp.array(mkt_pdf["mid_price"].values)
        trade_times = xp.array(trades_pdf["ts_ms"].values)
        order_side = xp.array(np.where(trades_pdf["side"].values == "b", 1, -1))
        trade_px = xp.array(trades_pdf["px"].values)
        trade_size = xp.array(trades_pdf["size"].values)

        # Compute trade quantities, cash flows, and cumulative PnL
        trade_qty = trade_size * order_side
        cum_qty = xp.cumsum(trade_qty)
        cash_flow = xp.cumsum(-trade_qty * trade_px)
        trade_idx = xp.searchsorted(trade_times, mkt_times, side="right") - 1
        open_qty = xp.where(trade_idx >= 0, cum_qty[trade_idx], 0)
        cash_flow_at_trade = xp.where(trade_idx >= 0, cash_flow[trade_idx], 0)
        cumulative_pnl = open_qty * market_mid_prices + cash_flow_at_trade
        pnl = xp.diff(cumulative_pnl, prepend=0)

        if self.use_gpu:
            cumulative_pnl = cp.asnumpy(cumulative_pnl)
            pnl = cp.asnumpy(pnl)
            mkt_times = cp.asnumpy(mkt_times)
        mkt_pdf["cumulative_pnl"] = cumulative_pnl

        # Compute strategy metrics
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = running_max - cumulative_pnl
        max_dd = drawdown.max()
        dd_time = mkt_times[drawdown.argmax()]
        mean_pnl = np.mean(pnl)
        std_pnl = np.std(pnl, ddof=1)
        sharpe_ratio = mean_pnl / std_pnl if std_pnl != 0 else np.nan
        skewness = skew(pnl)
        kurt_val = kurtosis(pnl)
        var_95 = np.percentile(pnl, 5)
        cvar_95 = np.mean(pnl[pnl <= var_95])
        profits = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        mean_profit = np.mean(profits) if profits.size > 0 else 0
        mean_loss = np.mean(losses) if losses.size > 0 else 0
        profit_factor = np.sum(profits) / abs(np.sum(losses)) if np.sum(losses) != 0 else np.nan
        p95 = np.percentile(pnl, 95)
        p5 = np.percentile(pnl, 5)
        tail_ratio = p95 / abs(p5) if p5 != 0 else np.nan
        target = 0.0
        omega_numerator = np.sum(np.maximum(pnl - target, 0))
        omega_denominator = np.sum(np.maximum(target - pnl, 0))
        omega_ratio = omega_numerator / omega_denominator if omega_denominator != 0 else np.nan
        calmar_ratio = mean_pnl / max_dd if max_dd != 0 else np.nan
        downside_markouts = pnl[pnl < target]
        if downside_markouts.size > 1:
            downside_std = np.std(downside_markouts, ddof=1)
        else:
            downside_std = np.nan
        sortino_ratio = mean_pnl / downside_std if downside_std and downside_std != 0 else np.nan

        # Consolidate all metrics in a dictionary
        metrics = {
            "mean_pnl": mean_pnl,
            "std_pnl": std_pnl,
            "sharpe_ratio": sharpe_ratio,
            "skewness": skewness,
            "excess_kurtosis": kurt_val,
            "VaR_95": var_95,
            "CVaR_95": cvar_95,
            "mean_profit": mean_profit,
            "mean_loss": mean_loss,
            "profit_factor": profit_factor,
            "max_drawdown": max_dd,
            "drawdown_time": dd_time,
            "tail_ratio": tail_ratio,
            "omega_ratio": omega_ratio,
            "calmar_ratio": calmar_ratio,
            "sortino_ratio": sortino_ratio
        }

        logging.info("Computed strategy interesting characteristics.")
        return metrics, mkt_pdf

    # -------------------------------------------------------------------------
    # Plotting and Reporting
    # -------------------------------------------------------------------------
    @staticmethod
    def save_plotly_figure(fig, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.write_html(filepath)
        logging.info(f"Interactive plot saved to {filepath}")

    def generate_market_trade_plot(self, one_hour_pdf: pd.DataFrame, trades_pdf: pd.DataFrame):
        trades_in_period = trades_pdf[
            (trades_pdf["ts_ms"] >= one_hour_pdf["ts_ms"].min()) &
            (trades_pdf["ts_ms"] <= one_hour_pdf["ts_ms"].max())
        ].copy()
        one_hour_pdf["datetime"] = pd.to_datetime(one_hour_pdf["ts_ms"], unit="ms")
        trades_in_period["datetime"] = pd.to_datetime(trades_in_period["ts_ms"], unit="ms")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=one_hour_pdf["datetime"],
            y=one_hour_pdf["bid"],
            mode="lines",
            name="Bid Price",
            line=dict(color="green", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=one_hour_pdf["datetime"],
            y=one_hour_pdf["ask"],
            mode="lines",
            name="Ask Price",
            line=dict(color="red", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=one_hour_pdf["datetime"],
            y=one_hour_pdf["mid_price"],
            mode="lines",
            name="Mid Price",
            line=dict(color="blue", width=2)
        ))

        buy_trades = trades_in_period[trades_in_period["side"] == "b"]
        sell_trades = trades_in_period[trades_in_period["side"] == "s"]
        max_volume = trades_in_period["size"].max() or 1
        scaled_sizes = (trades_in_period["size"] / max_volume) * 10

        fig.add_trace(go.Scatter(
            x=buy_trades["datetime"],
            y=buy_trades["px"],
            mode="markers",
            name="Buy Trade",
            marker=dict(
                symbol="triangle-up",
                color="blue",
                size=2 * scaled_sizes[buy_trades.index],
                line=dict(color="black", width=1)
            ),
            text=buy_trades["size"],
            hovertemplate="<b>Time:</b> %{x}<br><b>Price:</b> %{y}<br><b>Volume:</b> %{text}"
        ))
        fig.add_trace(go.Scatter(
            x=sell_trades["datetime"],
            y=sell_trades["px"],
            mode="markers",
            name="Sell Trade",
            marker=dict(
                symbol="triangle-down",
                color="orange",
                size=2 * scaled_sizes[sell_trades.index],
                line=dict(color="black", width=1)
            ),
            text=sell_trades["size"],
            hovertemplate="<b>Time:</b> %{x}<br><b>Price:</b> %{y}<br><b>Volume:</b> %{text}"
        ))

        fig.update_layout(
            title=f"{self.instrument} Market Data & Trades (1 Hour)",
            xaxis_title="Datetime",
            yaxis_title="Price",
            legend_title="Market Data",
            font=dict(size=12),
            template="plotly_white"
        )
        report_file = os.path.join(self.reports_dir, f"{self.instrument}_market_trades.html")
        self.save_plotly_figure(fig, report_file)

    def generate_markout_plot(self, agg_curve: np.ndarray):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.horizon_ticks,
            y=agg_curve,
            mode="lines+markers",
            line=dict(color="royalblue", width=3, dash="dot"),
            marker=dict(color="tomato", size=8, symbol="circle-open"),
            name="Aggregate Markout"
        ))
        fig.update_layout(
            title=f"{self.instrument} Aggregate Markout Curve",
            xaxis_title="Horizon Tick (ms)",
            yaxis_title="Margin (PnL per Unit Volume)",
            template="plotly_white",
            font=dict(family="Arial, sans-serif", size=14),
            xaxis=dict(showgrid=True, gridcolor="lightgrey", zeroline=True, zerolinecolor="grey"),
            yaxis=dict(showgrid=True, gridcolor="lightgrey", zeroline=True, zerolinecolor="grey"),
            plot_bgcolor="white",
            margin=dict(l=50, r=50, t=80, b=50)
        )
        fig.add_hline(y=0, line_dash="dash", line_color="grey")
        report_file = os.path.join(self.reports_dir, f"{self.instrument}_aggregate_markout.html")
        self.save_plotly_figure(fig, report_file)

    def generate_cumulative_pnl_plot(self, mkt_pnl_pdf: pd.DataFrame):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(mkt_pnl_pdf["ts_ms"], unit="ms"),
            y=mkt_pnl_pdf["cumulative_pnl"],
            mode="lines",
            name="Cumulative PnL"
        ))
        fig.update_layout(
            title=f"{self.instrument} Cumulative PnL",
            xaxis_title="Datetime",
            yaxis_title="Cumulative PnL"
        )
        report_file = os.path.join(self.reports_dir, f"{self.instrument}_cumulative_pnl.html")
        self.save_plotly_figure(fig, report_file)

    def generate_text_report(self, adv_metrics: dict):
        report_text = (
            "Quantitative FX Market Analysis Report\n"
            "========================================\n\n"
            f"Instrument: {self.instrument}\n"
            f"FX Pair: {self.meta['lhs_ccy']}/{self.meta['rhs_ccy']}\n\n"
            "Strategy Metrics:\n"
            f" * Mean PnL: {adv_metrics['mean_pnl']:.4f}\n"
            f" * PnL Volatility (Std. Dev): {adv_metrics['std_pnl']:.4f}\n"
            f" * Sharpe Ratio: {adv_metrics['sharpe_ratio']:.4f}\n"
            f" * Tail Ratio: {adv_metrics['tail_ratio']:.4f}\n"
            f" * Omega Ratio: {adv_metrics['omega_ratio']:.4f}\n"
            f" * Calmar Ratio: {adv_metrics['calmar_ratio']:.4f}\n"
            f" * Sortino Ratio: {adv_metrics['sortino_ratio']:.4f}\n"
            f" * Skewness: {adv_metrics['skewness']:.4f}\n"
            f" * Excess Kurtosis: {adv_metrics['excess_kurtosis']:.4f}\n"
            f" * 95% Value-at-Risk (VaR): {adv_metrics['VaR_95']:.4f}\n"
            f" * 95% Conditional VaR (CVaR): {adv_metrics['CVaR_95']:.4f}\n"
            f" * Mean Profit: {adv_metrics['mean_profit']:.4f}\n"
            f" * Mean Loss: {adv_metrics['mean_loss']:.4f}\n"
            f" * Profit Factor: {adv_metrics['profit_factor']:.4f}\n"
            f" * Maximum Drawdown: {adv_metrics['max_drawdown']:.2f}\n"
            f" * Drawdown Time: {pd.to_datetime(adv_metrics['drawdown_time'], unit='ms')}\n\n"
        )
        report_file = os.path.join(self.reports_dir, f"{self.instrument}_analysis_report.txt")
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        logging.info(f"Analysis report saved to {report_file}")

    # -------------------------------------------------------------------------
    # Main Workflow Execution
    # -------------------------------------------------------------------------
    def run_analysis(self):
        start_time = time.perf_counter()
        logging.getLogger().handlers[0].stream.write("\n")
        logging.info(f"Starting analysis for {self.instrument}")

        if not self.load_data():
            logging.error("Data loading failed. Exiting analysis.")
            return

        self.preprocess_data()
        one_hour_df, best_start = self.select_one_hour_period()
        if one_hour_df.height == 0:
            logging.error("Selected one‑hour market period is empty. Exiting analysis.")
            return

        agg_curve = self.compute_aggregate_markout()
        adv_metrics, mkt_pnl_pdf = self.compute_characteristics()

        one_hour_pdf = one_hour_df.to_pandas()
        trades_pdf = self.trades_df.to_pandas()

        self.generate_market_trade_plot(one_hour_pdf, trades_pdf)
        self.generate_markout_plot(agg_curve)
        self.generate_cumulative_pnl_plot(mkt_pnl_pdf)
        self.generate_text_report(adv_metrics)

        elapsed_time = time.perf_counter() - start_time
        logging.info(f"Execution time for {self.instrument} analysis: {elapsed_time:.2f} seconds")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main():
    instruments = ["GMMAUSD", "BTAUSD", "ZTAUSD", "LMDAUSD"]  # Adjust instrument list as needed.
    for instrument in instruments:
        analyzer = TradeAnalyzer(instrument, data_dir="data", reports_dir="reports", use_gpu=USE_GPU)
        analyzer.run_analysis()


if __name__ == "__main__":
    main()