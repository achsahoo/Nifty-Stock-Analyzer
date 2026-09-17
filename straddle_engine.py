"""
Straddle Analytics Backend Engine
=================================
High-performance quantitative engine analyzing Nifty 1-minute ATM Straddles
across 1,255 historical sessions (2021-2026) using Polars & Pandas.
"""

import os
import time
import pandas as pd
import polars as pl
import numpy as np
import streamlit as st
from dte_manager import DTEManager

# Parquet path resolution
PARQUET_FILE_CANDIDATES = [
    r"D:\NIFTY_MODEL\Nifty_1min_spot_1min_option\nifty_options_1min.parquet",
    r"D:\NIFTY_MODEL\GridOpt\nifty_options_1min.parquet",
    r"D:\NIFTY_MODEL\backtest_result_optimizer\nifty_options_1min.parquet"
]

def resolve_option_parquet():
    for p in PARQUET_FILE_CANDIDATES:
        if os.path.exists(p):
            return p
    return PARQUET_FILE_CANDIDATES[0]

OPTION_PARQUET_PATH = resolve_option_parquet()


class StraddleEngine:
    def __init__(self, parquet_path=OPTION_PARQUET_PATH):
        self.parquet_path = parquet_path
        self.dte_mgr = DTEManager()
        self._all_atm_df = None
        self._grouped_days = None
        self._available_dates_cache = None

    def load_data(self):
        """Loads and pivots all ATM 1-minute Call & Put data across all 1255 days."""
        if self._all_atm_df is not None:
            return self._all_atm_df

        if not os.path.exists(self.parquet_path):
            return pd.DataFrame()

        # Polars lazy scan on ATM strike label
        atm_lazy = pl.scan_parquet(self.parquet_path).filter(
            pl.col("strike_label") == "ATM"
        ).select([
            "datetime", "option_type", "close", "open", "high", "low", "spot", "strike_price"
        ])

        # Deduplicate and pivot to combine CALL and PUT
        df = (
            atm_lazy
            .collect()
            .unique(subset=["datetime", "option_type"])
            .pivot(
                values=["close", "strike_price", "open", "high", "low"],
                index=["datetime", "spot"],
                on="option_type",
                aggregate_function="first"
            )
            .sort("datetime")
            .with_columns([
                (pl.col("close_CALL") + pl.col("close_PUT")).alias("straddle"),
                pl.col("datetime").dt.date().alias("date"),
                pl.col("datetime").dt.strftime("%H:%M:00").alias("time_str"),
                pl.col("datetime").dt.strftime("%Y-%m-%d").alias("date_str")
            ])
            .to_pandas()
        )

        # Map DTE label to each date
        df['dte'] = df['date_str'].apply(self.dte_mgr.get_dte_label_for_date)
        self._all_atm_df = df

        # Group by date for rapid lookup
        self._grouped_days = {
            d: g.drop_duplicates('time_str').set_index('time_str')
            for d, g in df.groupby('date_str')
        }
        return df

    def get_available_dates(self):
        """Returns sorted list of trading dates with DTE metadata."""
        if self._available_dates_cache is not None:
            return self._available_dates_cache

        self.load_data()
        if not self._grouped_days:
            return []

        date_records = []
        for d in sorted(self._grouped_days.keys(), reverse=True):
            dte = self.dte_mgr.get_dte_label_for_date(d)
            date_records.append({"date_str": d, "dte": dte})

        self._available_dates_cache = date_records
        return date_records

    def get_day_straddle(self, date_str, mode="fixed_at_entry", entry_time="09:20:00"):
        """
        Retrieves 1-minute straddle series for a single date:
        - mode='fixed_at_entry': locks the ATM strike chosen at entry_time
        - mode='rolling_atm': follows dynamic ATM at each minute
        """
        self.load_data()
        if date_str not in self._grouped_days:
            return pd.DataFrame(), {}

        day_df = self._grouped_days[date_str].copy().reset_index()
        day_dte = self.dte_mgr.get_dte_label_for_date(date_str)

        if entry_time not in day_df['time_str'].values:
            entry_time = day_df['time_str'].iloc[0]

        entry_row = day_df[day_df['time_str'] == entry_time].iloc[0]
        p_entry = float(entry_row['straddle'])
        strike_at_entry = int(entry_row['strike_price_CALL']) if not pd.isna(entry_row['strike_price_CALL']) else 0

        # Sub-slice from entry to end of day
        sub_df = day_df[day_df['time_str'] >= entry_time]
        p_open = float(day_df['straddle'].iloc[0])
        p_close = float(day_df['straddle'].iloc[-1])
        p_min = float(sub_df['straddle'].min()) if len(sub_df) > 0 else p_open
        p_max = float(sub_df['straddle'].max()) if len(sub_df) > 0 else p_open

        # Min and Max time
        min_row = sub_df[sub_df['straddle'] == p_min].iloc[0] if len(sub_df) > 0 else day_df.iloc[0]
        max_row = sub_df[sub_df['straddle'] == p_max].iloc[0] if len(sub_df) > 0 else day_df.iloc[0]

        total_decay_pts = p_entry - p_close
        total_decay_pct = (total_decay_pts / p_entry * 100.0) if p_entry > 0 else 0.0
        mae_pct = ((p_max - p_entry) / p_entry * 100.0) if p_entry > 0 else 0.0
        mfe_pct = ((p_entry - p_min) / p_entry * 100.0) if p_entry > 0 else 0.0

        summary = {
            "date_str": date_str,
            "dte": day_dte,
            "entry_time": entry_time,
            "entry_price": p_entry,
            "day_open": p_open,
            "day_close": p_close,
            "day_min": p_min,
            "day_max": p_max,
            "strike_at_entry": strike_at_entry,
            "decay_pts": round(total_decay_pts, 2),
            "decay_pct": round(total_decay_pct, 2),
            "mae_pct": round(mae_pct, 2),
            "mfe_pct": round(mfe_pct, 2),
            "time_of_min": min_row['time_str'],
            "time_of_max": max_row['time_str'],
            "spot_open": float(day_df['spot'].iloc[0]),
            "spot_close": float(day_df['spot'].iloc[-1]),
            "spot_change": round(float(day_df['spot'].iloc[-1] - day_df['spot'].iloc[0]), 2)
        }

        return day_df, summary

    def get_hourly_window_stats(self, selected_dtes=None):
        """Computes decay velocity across intraday time windows."""
        self.load_data()
        windows = [
            ("09:15:00", "10:00:00", "09:15 - 10:00 (Opening Rush)"),
            ("10:00:00", "11:00:00", "10:00 - 11:00 (Morning Setup)"),
            ("11:00:00", "12:00:00", "11:00 - 12:00 (Mid-Day 1)"),
            ("12:00:00", "13:00:00", "12:00 - 13:00 (Lunch Lull)"),
            ("13:00:00", "14:00:00", "13:00 - 14:00 (Afternoon Trend)"),
            ("14:00:00", "15:00:00", "14:00 - 15:00 (Pre-Close Gamma / Theta Surge)"),
            ("15:00:00", "15:25:00", "15:00 - 15:25 (Final Settlement)"),
            ("09:20:00", "15:15:00", "Full Session (09:20 - 15:15)")
        ]

        results = []
        for t_start, t_end, label in windows:
            pts_list = []
            pct_list = []

            for d_str, df_day in self._grouped_days.items():
                if selected_dtes:
                    day_dte = self.dte_mgr.get_dte_label_for_date(d_str)
                    if day_dte not in selected_dtes:
                        continue

                if t_start in df_day.index and t_end in df_day.index:
                    p_start = float(df_day.loc[t_start, 'straddle'])
                    p_end = float(df_day.loc[t_end, 'straddle'])
                    if p_start > 0 and p_end > 0 and not np.isnan(p_start) and not np.isnan(p_end):
                        pts = p_start - p_end
                        pct = (pts / p_start) * 100.0
                        pts_list.append(pts)
                        pct_list.append(pct)

            if pts_list:
                pts_arr = np.array(pts_list)
                pct_arr = np.array(pct_list)
                win_rate = float(np.mean(pts_arr > 0) * 100.0)
                avg_pts = float(np.mean(pts_arr))
                avg_pct = float(np.mean(pct_arr))
                med_pct = float(np.median(pct_arr))
                results.append({
                    "Window": label,
                    "Days": len(pts_arr),
                    "Decay Win Rate (%)": round(win_rate, 1),
                    "Avg Decay (Pts)": round(avg_pts, 2),
                    "Avg Decay (%)": round(avg_pct, 2),
                    "Median Decay (%)": round(med_pct, 2)
                })
        return pd.DataFrame(results)

    def get_adverse_excursion_stats(self, entry_time="09:20:00", exit_time="15:15:00", selected_dtes=None):
        """
        Calculates cool-down probabilities for spike thresholds (+5% to +50%)
        and MAE percentile distribution.
        """
        self.load_data()
        thresholds = [5, 10, 15, 20, 25, 30, 40, 50]
        spike_stats = []
        all_spikes = []
        total_days = 0

        for d_str, df_day in self._grouped_days.items():
            if selected_dtes:
                if self.dte_mgr.get_dte_label_for_date(d_str) not in selected_dtes:
                    continue
            if entry_time in df_day.index and exit_time in df_day.index:
                p_entry = float(df_day.loc[entry_time, 'straddle'])
                if p_entry <= 0 or np.isnan(p_entry):
                    continue
                sub = df_day.loc[entry_time:exit_time]
                max_p = float(sub['straddle'].max())
                spike_pct = ((max_p - p_entry) / p_entry) * 100.0
                all_spikes.append(spike_pct)
                total_days += 1

        for th in thresholds:
            breached = 0
            cool_down = 0  # finished < entry price at exit_time
            final_decays = []

            for d_str, df_day in self._grouped_days.items():
                if selected_dtes:
                    if self.dte_mgr.get_dte_label_for_date(d_str) not in selected_dtes:
                        continue
                if entry_time in df_day.index and exit_time in df_day.index:
                    p_entry = float(df_day.loc[entry_time, 'straddle'])
                    if p_entry <= 0 or np.isnan(p_entry):
                        continue
                    sub = df_day.loc[entry_time:exit_time]
                    max_p = float(sub['straddle'].max())
                    spike_pct = ((max_p - p_entry) / p_entry) * 100.0
                    p_exit = float(df_day.loc[exit_time, 'straddle'])

                    if spike_pct >= th:
                        breached += 1
                        if p_exit < p_entry:
                            cool_down += 1
                        final_decays.append(((p_entry - p_exit) / p_entry) * 100.0)

            if breached > 0:
                cool_rate = (cool_down / breached) * 100.0
                avg_final = float(np.mean(final_decays))
                spike_stats.append({
                    "Spike Threshold": f"+{th}%",
                    "Breach Sessions": breached,
                    "Frequency (% of Sessions)": round((breached / total_days) * 100.0, 1),
                    "Cool-Down Win Rate (%)": round(cool_rate, 1),
                    "Avg Final Decay (%)": round(avg_final, 2)
                })

        all_spikes = np.array(all_spikes) if all_spikes else np.array([0.0])
        pctiles = {
            "Median (50th)": round(float(np.percentile(all_spikes, 50)), 2),
            "75th Percentile": round(float(np.percentile(all_spikes, 75)), 2),
            "90th Percentile": round(float(np.percentile(all_spikes, 90)), 2),
            "95th Percentile": round(float(np.percentile(all_spikes, 95)), 2),
            "Max Spike Seen": round(float(np.max(all_spikes)), 2),
            "Total Sessions": total_days
        }

        return pd.DataFrame(spike_stats), pctiles

    def get_entry_time_matrix(self, exit_time="15:15:00", selected_dtes=None):
        """Compares decay performance across entry times ranging from 09:16 to 13:00."""
        self.load_data()
        test_entries = [
            "09:16:00", "09:20:00", "09:30:00", "09:45:00", 
            "10:00:00", "10:15:00", "10:30:00", "11:00:00", 
            "11:30:00", "12:00:00", "12:30:00", "13:00:00"
        ]

        matrix_rows = []
        for e_time in test_entries:
            decays_pts = []
            decays_pct = []
            maes = []

            for d_str, df_day in self._grouped_days.items():
                if selected_dtes:
                    if self.dte_mgr.get_dte_label_for_date(d_str) not in selected_dtes:
                        continue
                if e_time in df_day.index and exit_time in df_day.index:
                    p_entry = float(df_day.loc[e_time, 'straddle'])
                    p_exit = float(df_day.loc[exit_time, 'straddle'])
                    sub = df_day.loc[e_time:exit_time]
                    max_p = float(sub['straddle'].max())

                    if p_entry > 0 and p_exit > 0:
                        pts = p_entry - p_exit
                        pct = (pts / p_entry) * 100.0
                        mae = ((max_p - p_entry) / p_entry) * 100.0
                        decays_pts.append(pts)
                        decays_pct.append(pct)
                        maes.append(mae)

            if decays_pts:
                pts_arr = np.array(decays_pts)
                pct_arr = np.array(decays_pct)
                mae_arr = np.array(maes)
                win_rate = float(np.mean(pts_arr > 0) * 100.0)
                avg_pts = float(np.mean(pts_arr))
                avg_pct = float(np.mean(pct_arr))
                med_pct = float(np.median(pct_arr))
                med_mae = float(np.median(mae_arr))

                gross_profit = np.sum(pts_arr[pts_arr > 0])
                gross_loss = abs(np.sum(pts_arr[pts_arr < 0]))
                pf = (gross_profit / gross_loss) if gross_loss > 0 else 99.9

                matrix_rows.append({
                    "Entry Time": e_time[:5],
                    "Sessions": len(pts_arr),
                    "Win Rate (%)": round(win_rate, 1),
                    "Avg Decay (Pts)": round(avg_pts, 2),
                    "Avg Decay (%)": round(avg_pct, 2),
                    "Median Decay (%)": round(med_pct, 2),
                    "Median Spike (MAE %)": round(med_mae, 2),
                    "Profit Factor": round(pf, 2)
                })

        return pd.DataFrame(matrix_rows)

    def get_sl_target_matrix(self, entry_time="09:20:00", exit_time="15:15:00", selected_dtes=None):
        """
        Simulates combinations of Stop Loss % vs Target % on Short Straddles
        to find optimal risk/reward parameters.
        """
        self.load_data()
        sl_options = [10, 15, 20, 25, 30, 40, 50, 0]     # 0 = No SL
        target_options = [10, 15, 20, 25, 30, 40, 50, 0] # 0 = Open-Ended (Close at 15:15)

        results = []
        for sl_pct in sl_options:
            for tgt_pct in target_options:
                pnls_pct = []
                wins = 0
                losses = 0

                for d_str, df_day in self._grouped_days.items():
                    if selected_dtes:
                        if self.dte_mgr.get_dte_label_for_date(d_str) not in selected_dtes:
                            continue
                    if entry_time in df_day.index and exit_time in df_day.index:
                        p_entry = float(df_day.loc[entry_time, 'straddle'])
                        if p_entry <= 0:
                            continue
                        sub = df_day.loc[entry_time:exit_time]['straddle'].values

                        trade_exit_pct = None
                        sl_price = p_entry * (1.0 + sl_pct / 100.0) if sl_pct > 0 else 1e9
                        tgt_price = p_entry * (1.0 - tgt_pct / 100.0) if tgt_pct > 0 else -1e9

                        for p in sub:
                            if tgt_pct > 0 and p <= tgt_price:
                                trade_exit_pct = tgt_pct
                                break
                            if sl_pct > 0 and p >= sl_price:
                                trade_exit_pct = -sl_pct
                                break

                        if trade_exit_pct is None:
                            p_exit = sub[-1]
                            trade_exit_pct = ((p_entry - p_exit) / p_entry) * 100.0

                        pnls_pct.append(trade_exit_pct)
                        if trade_exit_pct > 0:
                            wins += 1
                        elif trade_exit_pct < 0:
                            losses += 1

                if pnls_pct:
                    pnls_arr = np.array(pnls_pct)
                    n_trades = len(pnls_arr)
                    win_rate = (wins / n_trades) * 100.0
                    avg_pnl = float(np.mean(pnls_arr))
                    
                    gross_win = np.sum(pnls_arr[pnls_arr > 0])
                    gross_loss = abs(np.sum(pnls_arr[pnls_arr < 0]))
                    pf = (gross_win / gross_loss) if gross_loss > 0 else 99.9

                    results.append({
                        "SL_Pct": sl_pct,
                        "SL_Label": f"{sl_pct}% SL" if sl_pct > 0 else "No SL",
                        "Target_Pct": tgt_pct,
                        "Target_Label": f"+{tgt_pct}% Target" if tgt_pct > 0 else "Open-Ended",
                        "Trades": n_trades,
                        "Win Rate (%)": round(win_rate, 1),
                        "Avg PnL (%)": round(avg_pnl, 2),
                        "Profit Factor": round(pf, 2)
                    })

        return pd.DataFrame(results)

    def get_dte_summary_stats(self):
        """Compares decay metrics across DTEs (0 DTE, 1 DTE, 2 DTE, 3 DTE, 4 DTE)."""
        self.load_data()
        dtes = ["0 DTE", "1 DTE", "2 DTE", "3 DTE", "4 DTE"]
        rows = []

        for dte in dtes:
            decays_pts = []
            decays_pct = []
            maes = []

            for d_str, df_day in self._grouped_days.items():
                if self.dte_mgr.get_dte_label_for_date(d_str) != dte:
                    continue
                if "09:20:00" in df_day.index and "15:15:00" in df_day.index:
                    p_entry = float(df_day.loc["09:20:00", 'straddle'])
                    p_exit = float(df_day.loc["15:15:00", 'straddle'])
                    sub = df_day.loc["09:20:00":"15:15:00"]
                    max_p = float(sub['straddle'].max())

                    if p_entry > 0 and p_exit > 0:
                        pts = p_entry - p_exit
                        pct = (pts / p_entry) * 100.0
                        mae = ((max_p - p_entry) / p_entry) * 100.0
                        decays_pts.append(pts)
                        decays_pct.append(pct)
                        maes.append(mae)

            if decays_pts:
                pts_arr = np.array(decays_pts)
                pct_arr = np.array(decays_pct)
                mae_arr = np.array(maes)

                rows.append({
                    "DTE": dte,
                    "Sessions": len(pts_arr),
                    "Win Rate (%)": round(float(np.mean(pts_arr > 0) * 100.0), 1),
                    "Avg Decay (Pts)": round(float(np.mean(pts_arr)), 2),
                    "Avg Decay (%)": round(float(np.mean(pct_arr)), 2),
                    "Median Decay (%)": round(float(np.median(pct_arr)), 2),
                    "Median Spike (MAE %)": round(float(np.median(mae_arr)), 2)
                })

        return pd.DataFrame(rows)


@st.cache_resource(show_spinner=False)
def get_straddle_engine():
    """Cached singleton instance of StraddleEngine."""
    engine = StraddleEngine()
    engine.load_data()
    return engine
