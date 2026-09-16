import os
import pandas as pd
import polars as pl
import numpy as np

def _resolve_spot_file():
    candidates = [
        os.path.join(os.path.dirname(__file__), "data", "nifty_1min_2015_2026.parquet"),
        os.path.join(os.path.dirname(__file__), "nifty_1min_2015_2026.parquet"),
        r"D:\NIFTY_MODEL\nifty_1min_2015_2026.parquet",
        r"D:\NIFTY_MODEL\Nifty_1min_spot_1min_option\nifty_1min_2015_2026.parquet",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]

SPOT_FILE = _resolve_spot_file()

class NiftyAnalyticsEngine:
    def __init__(self, parquet_path=None):
        self.parquet_path = parquet_path or _resolve_spot_file()
        self._df_raw = None
        self._daily_df = None
        self._weekly_df = None
        self._monthly_df = None
        self._grouped_cache = {}
        
    def load_data(self):
        if self._df_raw is None:
            df = pl.read_parquet(self.parquet_path)
            # Add date, week, month, and time columns
            df = df.with_columns([
                pl.col("date").dt.date().alias("day"),
                pl.col("date").dt.truncate("1w").alias("week"),
                pl.col("date").dt.truncate("1mo").alias("month"),
                pl.col("date").dt.strftime("%H:%M:00").alias("time_str")
            ])
            self._df_raw = df
            
            # 1. Daily Aggregation
            daily = df.group_by("day").agg([
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("date").first().alias("datetime")
            ]).sort("day").to_pandas()
            
            daily['date'] = pd.to_datetime(daily['day'])
            daily['range_pct'] = (daily['high'] - daily['low']) / daily['open'] * 100.0
            daily['move_pct'] = (daily['close'] - daily['open']) / daily['open'] * 100.0
            daily['day_name'] = daily['date'].dt.day_name()
            self._daily_df = daily
            
            # 2. Weekly Aggregation
            daily_indexed = daily.set_index('date')
            weekly = daily_indexed.resample('W-FRI').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).dropna().reset_index()
            weekly['range_pct'] = (weekly['high'] - weekly['low']) / weekly['open'] * 100.0
            weekly['move_pct'] = (weekly['close'] - weekly['open']) / weekly['open'] * 100.0
            self._weekly_df = weekly
            
            # 3. Monthly Aggregation
            monthly = daily_indexed.resample('ME').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).dropna().reset_index()
            monthly['range_pct'] = (monthly['high'] - monthly['low']) / monthly['open'] * 100.0
            monthly['move_pct'] = (monthly['close'] - monthly['open']) / monthly['open'] * 100.0
            self._monthly_df = monthly
            
    def get_range_and_move_stats(self, timeframe="Daily", threshold_pct=1.0):
        self.load_data()
        
        if timeframe == "Weekly":
            df = self._weekly_df
            period_label = "Weeks"
        elif timeframe == "Monthly":
            df = self._monthly_df
            period_label = "Months"
        else:
            df = self._daily_df
            period_label = "Days"
            
        total_periods = len(df)
        
        # Range stats
        within_range = (df['range_pct'] <= threshold_pct)
        pct_within_range = within_range.mean() * 100.0
        cnt_within_range = int(within_range.sum())
        
        # Close Move stats
        within_move = (df['move_pct'].abs() <= threshold_pct)
        pct_within_move = within_move.mean() * 100.0
        cnt_within_move = int(within_move.sum())
        
        above_pos = (df['move_pct'] > threshold_pct)
        pct_above_pos = above_pos.mean() * 100.0
        cnt_above_pos = int(above_pos.sum())
        
        below_neg = (df['move_pct'] < -threshold_pct)
        pct_below_neg = below_neg.mean() * 100.0
        cnt_below_neg = int(below_neg.sum())
        
        stats = {
            'timeframe': timeframe,
            'period_label': period_label,
            'total_periods': total_periods,
            'threshold_pct': threshold_pct,
            # Range
            'pct_within_range': pct_within_range,
            'cnt_within_range': cnt_within_range,
            'avg_range_pct': df['range_pct'].mean(),
            'median_range_pct': df['range_pct'].median(),
            'max_range_pct': df['range_pct'].max(),
            'min_range_pct': df['range_pct'].min(),
            # Move
            'pct_within_move': pct_within_move,
            'cnt_within_move': cnt_within_move,
            'pct_above_pos': pct_above_pos,
            'cnt_above_pos': cnt_above_pos,
            'pct_below_neg': pct_below_neg,
            'cnt_below_neg': cnt_below_neg,
            'avg_move_pct': df['move_pct'].mean(),
            'median_move_pct': df['move_pct'].median(),
            'df': df
        }
        return stats

    def _get_period_groups(self, timeframe="Daily"):
        self.load_data()
        col = {"Daily": "day", "Weekly": "week", "Monthly": "month"}.get(timeframe, "day")
        if col not in self._grouped_cache:
            df_raw = self._df_raw
            period_opens = df_raw.group_by(col).agg(pl.col("open").first().alias("period_open"))
            df_joined = df_raw.join(period_opens, on=col).with_columns([
                ((pl.col("high") - pl.col("period_open")) / pl.col("period_open") * 100.0).alias("move_high_pct"),
                ((pl.col("low") - pl.col("period_open")) / pl.col("period_open") * 100.0).alias("move_low_pct"),
                ((pl.col("close") - pl.col("period_open")) / pl.col("period_open") * 100.0).alias("move_close_pct")
            ]).to_pandas()
            self._grouped_cache[col] = [group for _, group in df_joined.groupby(col)]
        return self._grouped_cache[col]

    def get_stretch_reversal_stats(self, timeframe="Daily", stretch_threshold=None, retrace_pct=50.0, ext_steps=None):
        if stretch_threshold is None:
            stretch_threshold = 0.50 if timeframe == "Daily" else (1.50 if timeframe == "Weekly" else 3.00)
            
        if ext_steps is None:
            if timeframe == "Daily":
                ext_steps = [0.25, 0.50, 0.75, 1.00]
            elif timeframe == "Weekly":
                ext_steps = [0.50, 1.00, 1.50, 2.00]
            else:
                ext_steps = [1.00, 2.00, 3.00, 4.00]
                
        unit_singular, unit_plural = {
            "Daily": ("Day", "Days"),
            "Weekly": ("Week", "Weeks"),
            "Monthly": ("Month", "Months")
        }.get(timeframe, ("Period", "Periods"))
        
        periods_grouped = self._get_period_groups(timeframe)
        total_periods = len(periods_grouped)
        
        # Upside (+X% from Open)
        pos_count = 0
        pos_first_touch_times = []
        pos_extensions = []
        pos_step_counts = [0] * len(ext_steps)
        pos_pulled_below_trigger = 0
        pos_retrace_goal_count = 0
        pos_back_to_open_count = 0
        pos_closed_trend_count = 0
        pos_closed_consol_count = 0
        pos_closed_red_count = 0
        pos_peak_pullbacks = []
        pos_retrace_peak_goal_count = 0
        pos_peak_highs = []
        pos_final_closes = []
        
        # Post-Retrace & Post-Open forward sequence counters (Upside)
        pos_after_retrace_retest_cnt = 0
        pos_after_retrace_open_cnt = 0
        pos_after_retrace_close_trend_cnt = 0
        pos_after_retrace_close_consol_cnt = 0
        pos_after_retrace_close_red_cnt = 0
        
        pos_after_open_plunge_cnt = 0
        pos_after_open_bounce_trig_cnt = 0
        pos_after_open_close_red_cnt = 0
        pos_after_open_close_green_cnt = 0
        
        # Downside (-X% from Open)
        neg_count = 0
        neg_first_touch_times = []
        neg_extensions = []
        neg_step_counts = [0] * len(ext_steps)
        neg_bounced_above_trigger = 0
        neg_retrace_goal_count = 0
        neg_bounce_trough_goal_count = 0
        neg_back_to_open_count = 0
        neg_closed_trend_count = 0
        neg_closed_consol_count = 0
        neg_closed_green_count = 0
        neg_trough_bounces = []
        neg_trough_lows = []
        neg_final_closes = []
        
        # Post-Bounce & Post-Open forward sequence counters (Downside)
        neg_after_retrace_retest_cnt = 0
        neg_after_retrace_open_cnt = 0
        neg_after_retrace_close_trend_cnt = 0
        neg_after_retrace_close_consol_cnt = 0
        neg_after_retrace_close_green_cnt = 0
        
        neg_after_open_plunge_cnt = 0
        neg_after_open_bounce_trig_cnt = 0
        neg_after_open_close_green_cnt = 0
        neg_after_open_close_red_cnt = 0
        
        pos_goal_level = stretch_threshold * (1.0 - retrace_pct / 100.0)
        neg_goal_level = -stretch_threshold * (1.0 - retrace_pct / 100.0)
        
        for p_df in periods_grouped:
            # 1. Upside Stretch Check
            pos_mask = (p_df['move_high_pct'] >= stretch_threshold)
            if pos_mask.any():
                pos_count += 1
                first_idx = pos_mask.idxmax()
                pos_first_touch_times.append(p_df.loc[first_idx, 'time_str'])
                
                after_touch = p_df.loc[first_idx:]
                
                # Maximum high established AFTER the first touch
                sub_high = after_touch['move_high_pct'].max()
                pos_peak_highs.append(sub_high)
                ext = sub_high - stretch_threshold
                pos_extensions.append(ext)
                for s_i, s_val in enumerate(ext_steps):
                    if ext >= s_val:
                        pos_step_counts[s_i] += 1
                
                # Minimum low established AFTER the first touch
                sub_low = after_touch['move_low_pct'].min()
                
                if sub_low < stretch_threshold:
                    pos_pulled_below_trigger += 1
                    
                # Post-Retrace tracking (50% goal)
                retrace_mask = (after_touch['move_low_pct'] <= pos_goal_level)
                if retrace_mask.any():
                    pos_retrace_goal_count += 1
                    retrace_idx = retrace_mask.idxmax()
                    after_retrace = p_df.loc[retrace_idx:]
                    if (after_retrace['move_high_pct'] >= stretch_threshold).any():
                        pos_after_retrace_retest_cnt += 1
                    if (after_retrace['move_low_pct'] <= 0.0).any():
                        pos_after_retrace_open_cnt += 1
                        
                    final_c = p_df.iloc[-1]['move_close_pct']
                    if final_c >= stretch_threshold:
                        pos_after_retrace_close_trend_cnt += 1
                    elif final_c > 0.0:
                        pos_after_retrace_close_consol_cnt += 1
                    else:
                        pos_after_retrace_close_red_cnt += 1
                    
                if sub_high > 0 and (sub_high - sub_low) >= (sub_high * (retrace_pct / 100.0)):
                    pos_retrace_peak_goal_count += 1
                    
                # Post-Open tracking (Day Open <= 0.0%)
                open_mask = (after_touch['move_low_pct'] <= 0.0)
                if open_mask.any():
                    pos_back_to_open_count += 1
                    open_idx = open_mask.idxmax()
                    after_open = p_df.loc[open_idx:]
                    if (after_open['move_low_pct'] <= -stretch_threshold).any():
                        pos_after_open_plunge_cnt += 1
                    if (after_open['move_high_pct'] >= stretch_threshold).any():
                        pos_after_open_bounce_trig_cnt += 1
                        
                    final_c = p_df.iloc[-1]['move_close_pct']
                    if final_c < 0.0:
                        pos_after_open_close_red_cnt += 1
                    else:
                        pos_after_open_close_green_cnt += 1
                    
                pos_peak_pullbacks.append(sub_high - sub_low)
                
                final_move = p_df.iloc[-1]['move_close_pct']
                pos_final_closes.append(final_move)
                if final_move >= stretch_threshold:
                    pos_closed_trend_count += 1
                elif final_move > 0.0:
                    pos_closed_consol_count += 1
                else:
                    pos_closed_red_count += 1
                    
            # 2. Downside Stretch Check
            neg_mask = (p_df['move_low_pct'] <= -stretch_threshold)
            if neg_mask.any():
                neg_count += 1
                first_idx = neg_mask.idxmax()
                neg_first_touch_times.append(p_df.loc[first_idx, 'time_str'])
                
                after_touch = p_df.loc[first_idx:]
                
                # Maximum drop established AFTER first touch
                sub_low = after_touch['move_low_pct'].min()
                neg_trough_lows.append(sub_low)
                ext = (-sub_low) - stretch_threshold
                neg_extensions.append(ext)
                for s_i, s_val in enumerate(ext_steps):
                    if ext >= s_val:
                        neg_step_counts[s_i] += 1
                
                # Maximum high established AFTER first touch
                sub_high = after_touch['move_high_pct'].max()
                
                if sub_high > -stretch_threshold:
                    neg_bounced_above_trigger += 1
                    
                # Post-Bounce tracking (50% goal)
                bounce_mask = (after_touch['move_high_pct'] >= neg_goal_level)
                if bounce_mask.any():
                    neg_retrace_goal_count += 1
                    bounce_idx = bounce_mask.idxmax()
                    after_bounce = p_df.loc[bounce_idx:]
                    if (after_bounce['move_low_pct'] <= -stretch_threshold).any():
                        neg_after_retrace_retest_cnt += 1
                    if (after_bounce['move_high_pct'] >= 0.0).any():
                        neg_after_retrace_open_cnt += 1
                        
                    final_c = p_df.iloc[-1]['move_close_pct']
                    if final_c <= -stretch_threshold:
                        neg_after_retrace_close_trend_cnt += 1
                    elif final_c < 0.0:
                        neg_after_retrace_close_consol_cnt += 1
                    else:
                        neg_after_retrace_close_green_cnt += 1
                    
                if sub_low < 0 and (sub_high - sub_low) >= (abs(sub_low) * (retrace_pct / 100.0)):
                    neg_bounce_trough_goal_count += 1
                    
                # Post-Open tracking (Day Open >= 0.0%)
                open_mask = (after_touch['move_high_pct'] >= 0.0)
                if open_mask.any():
                    neg_back_to_open_count += 1
                    open_idx = open_mask.idxmax()
                    after_open = p_df.loc[open_idx:]
                    if (after_open['move_high_pct'] >= stretch_threshold).any():
                        neg_after_open_plunge_cnt += 1
                    if (after_open['move_low_pct'] <= -stretch_threshold).any():
                        neg_after_open_bounce_trig_cnt += 1
                        
                    final_c = p_df.iloc[-1]['move_close_pct']
                    if final_c > 0.0:
                        neg_after_open_close_green_cnt += 1
                    else:
                        neg_after_open_close_red_cnt += 1
                    
                neg_trough_bounces.append(sub_high - sub_low)
                
                final_move = p_df.iloc[-1]['move_close_pct']
                neg_final_closes.append(final_move)
                if final_move <= -stretch_threshold:
                    neg_closed_trend_count += 1
                elif final_move < 0.0:
                    neg_closed_consol_count += 1
                else:
                    neg_closed_green_count += 1
                    
        pos_step_pcts = [(c / pos_count * 100.0) if pos_count > 0 else 0 for c in pos_step_counts]
        neg_step_pcts = [(c / neg_count * 100.0) if neg_count > 0 else 0 for c in neg_step_counts]
        
        return {
            'timeframe': timeframe,
            'unit_singular': unit_singular,
            'unit_plural': unit_plural,
            'total_periods': total_periods,
            'total_days': total_periods, # backward compatibility
            'stretch_threshold': stretch_threshold,
            'retrace_pct': retrace_pct,
            'ext_steps': ext_steps,
            'pos_goal_level': pos_goal_level,
            'neg_goal_level': neg_goal_level,
            'pos': {
                'count': pos_count,
                'pct_of_all_days': (pos_count / total_periods * 100.0) if total_periods > 0 else 0,
                'pct_of_all_periods': (pos_count / total_periods * 100.0) if total_periods > 0 else 0,
                'first_touch_times': pos_first_touch_times,
                'final_closes': pos_final_closes,
                'ext_step_counts': pos_step_counts,
                'ext_step_pcts': pos_step_pcts,
                'ext_25_count': pos_step_counts[0] if len(pos_step_counts) > 0 else 0,
                'ext_25_pct': pos_step_pcts[0] if len(pos_step_pcts) > 0 else 0,
                'ext_50_count': pos_step_counts[1] if len(pos_step_counts) > 1 else 0,
                'ext_50_pct': pos_step_pcts[1] if len(pos_step_pcts) > 1 else 0,
                'ext_75_count': pos_step_counts[2] if len(pos_step_counts) > 2 else 0,
                'ext_75_pct': pos_step_pcts[2] if len(pos_step_pcts) > 2 else 0,
                'ext_100_count': pos_step_counts[3] if len(pos_step_counts) > 3 else 0,
                'ext_100_pct': pos_step_pcts[3] if len(pos_step_pcts) > 3 else 0,
                'avg_extension': np.mean(pos_extensions) if len(pos_extensions) > 0 else 0,
                'median_extension': np.median(pos_extensions) if len(pos_extensions) > 0 else 0,
                'avg_peak_high': np.mean(pos_peak_highs) if len(pos_peak_highs) > 0 else stretch_threshold,
                'pulled_below_trigger_count': pos_pulled_below_trigger,
                'pulled_below_trigger_pct': (pos_pulled_below_trigger / pos_count * 100.0) if pos_count > 0 else 0,
                'retrace_goal_count': pos_retrace_goal_count,
                'retrace_goal_pct': (pos_retrace_goal_count / pos_count * 100.0) if pos_count > 0 else 0,
                'retrace_peak_goal_count': pos_retrace_peak_goal_count,
                'retrace_peak_goal_pct': (pos_retrace_peak_goal_count / pos_count * 100.0) if pos_count > 0 else 0,
                'back_to_open_count': pos_back_to_open_count,
                'back_to_open_pct': (pos_back_to_open_count / pos_count * 100.0) if pos_count > 0 else 0,
                'closed_red_count': pos_closed_red_count,
                'closed_red_pct': (pos_closed_red_count / pos_count * 100.0) if pos_count > 0 else 0,
                'avg_retrace_from_peak': np.mean(pos_peak_pullbacks) if len(pos_peak_pullbacks) > 0 else 0,
                'closed_trend_count': pos_closed_trend_count,
                'closed_trend_pct': (pos_closed_trend_count / pos_count * 100.0) if pos_count > 0 else 0,
                'closed_consol_count': pos_closed_consol_count,
                'closed_consol_pct': (pos_closed_consol_count / pos_count * 100.0) if pos_count > 0 else 0,
                
                # New Post-Retrace Sequence Keys
                'after_retrace_retest_cnt': pos_after_retrace_retest_cnt,
                'after_retrace_retest_pct': (pos_after_retrace_retest_cnt / pos_retrace_goal_count * 100.0) if pos_retrace_goal_count > 0 else 0,
                'after_retrace_open_cnt': pos_after_retrace_open_cnt,
                'after_retrace_open_pct': (pos_after_retrace_open_cnt / pos_retrace_goal_count * 100.0) if pos_retrace_goal_count > 0 else 0,
                'after_retrace_close_trend_cnt': pos_after_retrace_close_trend_cnt,
                'after_retrace_close_trend_pct': (pos_after_retrace_close_trend_cnt / pos_retrace_goal_count * 100.0) if pos_retrace_goal_count > 0 else 0,
                'after_retrace_close_consol_cnt': pos_after_retrace_close_consol_cnt,
                'after_retrace_close_consol_pct': (pos_after_retrace_close_consol_cnt / pos_retrace_goal_count * 100.0) if pos_retrace_goal_count > 0 else 0,
                'after_retrace_close_red_cnt': pos_after_retrace_close_red_cnt,
                'after_retrace_close_red_pct': (pos_after_retrace_close_red_cnt / pos_retrace_goal_count * 100.0) if pos_retrace_goal_count > 0 else 0,
                
                # New Post-Open Sequence Keys
                'after_open_plunge_cnt': pos_after_open_plunge_cnt,
                'after_open_plunge_pct': (pos_after_open_plunge_cnt / pos_back_to_open_count * 100.0) if pos_back_to_open_count > 0 else 0,
                'after_open_bounce_trig_cnt': pos_after_open_bounce_trig_cnt,
                'after_open_bounce_trig_pct': (pos_after_open_bounce_trig_cnt / pos_back_to_open_count * 100.0) if pos_back_to_open_count > 0 else 0,
                'after_open_close_red_cnt': pos_after_open_close_red_cnt,
                'after_open_close_red_pct': (pos_after_open_close_red_cnt / pos_back_to_open_count * 100.0) if pos_back_to_open_count > 0 else 0,
                'after_open_close_green_cnt': pos_after_open_close_green_cnt,
                'after_open_close_green_pct': (pos_after_open_close_green_cnt / pos_back_to_open_count * 100.0) if pos_back_to_open_count > 0 else 0,
            },
            'neg': {
                'count': neg_count,
                'pct_of_all_days': (neg_count / total_periods * 100.0) if total_periods > 0 else 0,
                'pct_of_all_periods': (neg_count / total_periods * 100.0) if total_periods > 0 else 0,
                'first_touch_times': neg_first_touch_times,
                'final_closes': neg_final_closes,
                'ext_step_counts': neg_step_counts,
                'ext_step_pcts': neg_step_pcts,
                'ext_25_count': neg_step_counts[0] if len(neg_step_counts) > 0 else 0,
                'ext_25_pct': neg_step_pcts[0] if len(neg_step_pcts) > 0 else 0,
                'ext_50_count': neg_step_counts[1] if len(neg_step_counts) > 1 else 0,
                'ext_50_pct': neg_step_pcts[1] if len(neg_step_pcts) > 1 else 0,
                'ext_75_count': neg_step_counts[2] if len(neg_step_counts) > 2 else 0,
                'ext_75_pct': neg_step_pcts[2] if len(neg_step_pcts) > 2 else 0,
                'ext_100_count': neg_step_counts[3] if len(neg_step_counts) > 3 else 0,
                'ext_100_pct': neg_step_pcts[3] if len(neg_step_pcts) > 3 else 0,
                'avg_extension': np.mean(neg_extensions) if len(neg_extensions) > 0 else 0,
                'median_extension': np.median(neg_extensions) if len(neg_extensions) > 0 else 0,
                'avg_trough_low': np.mean(neg_trough_lows) if len(neg_trough_lows) > 0 else -stretch_threshold,
                'bounced_above_trigger_count': neg_bounced_above_trigger,
                'bounced_above_trigger_pct': (neg_bounced_above_trigger / neg_count * 100.0) if neg_count > 0 else 0,
                'retrace_goal_count': neg_retrace_goal_count,
                'retrace_goal_pct': (neg_retrace_goal_count / neg_count * 100.0) if neg_count > 0 else 0,
                'bounce_trough_goal_count': neg_bounce_trough_goal_count,
                'bounce_trough_goal_pct': (neg_bounce_trough_goal_count / neg_count * 100.0) if neg_count > 0 else 0,
                'back_to_open_count': neg_back_to_open_count,
                'back_to_open_pct': (neg_back_to_open_count / neg_count * 100.0) if neg_count > 0 else 0,
                'closed_green_count': neg_closed_green_count,
                'closed_green_pct': (neg_closed_green_count / neg_count * 100.0) if neg_count > 0 else 0,
                'avg_bounce_from_trough': np.mean(neg_trough_bounces) if len(neg_trough_bounces) > 0 else 0,
                'closed_trend_count': neg_closed_trend_count,
                'closed_trend_pct': (neg_closed_trend_count / neg_count * 100.0) if neg_count > 0 else 0,
                'closed_consol_count': neg_closed_consol_count,
                'closed_consol_pct': (neg_closed_consol_count / neg_count * 100.0) if neg_count > 0 else 0,
                
                # New Post-Bounce Sequence Keys
                'after_retrace_retest_cnt': neg_after_retrace_retest_cnt,
                'after_retrace_retest_pct': (neg_after_retrace_retest_cnt / neg_retrace_goal_count * 100.0) if neg_retrace_goal_count > 0 else 0,
                'after_retrace_open_cnt': neg_after_retrace_open_cnt,
                'after_retrace_open_pct': (neg_after_retrace_open_cnt / neg_retrace_goal_count * 100.0) if neg_retrace_goal_count > 0 else 0,
                'after_retrace_close_trend_cnt': neg_after_retrace_close_trend_cnt,
                'after_retrace_close_trend_pct': (neg_after_retrace_close_trend_cnt / neg_retrace_goal_count * 100.0) if neg_retrace_goal_count > 0 else 0,
                'after_retrace_close_consol_cnt': neg_after_retrace_close_consol_cnt,
                'after_retrace_close_consol_pct': (neg_after_retrace_close_consol_cnt / neg_retrace_goal_count * 100.0) if neg_retrace_goal_count > 0 else 0,
                'after_retrace_close_green_cnt': neg_after_retrace_close_green_cnt,
                'after_retrace_close_green_pct': (neg_after_retrace_close_green_cnt / neg_retrace_goal_count * 100.0) if neg_retrace_goal_count > 0 else 0,
                
                # New Post-Open Sequence Keys
                'after_open_plunge_cnt': neg_after_open_plunge_cnt,
                'after_open_plunge_pct': (neg_after_open_plunge_cnt / neg_back_to_open_count * 100.0) if neg_back_to_open_count > 0 else 0,
                'after_open_bounce_trig_cnt': neg_after_open_bounce_trig_cnt,
                'after_open_bounce_trig_pct': (neg_after_open_bounce_trig_cnt / neg_back_to_open_count * 100.0) if neg_back_to_open_count > 0 else 0,
                'after_open_close_green_cnt': neg_after_open_close_green_cnt,
                'after_open_close_green_pct': (neg_after_open_close_green_cnt / neg_back_to_open_count * 100.0) if neg_back_to_open_count > 0 else 0,
                'after_open_close_red_cnt': neg_after_open_close_red_cnt,
                'after_open_close_red_pct': (neg_after_open_close_red_cnt / neg_back_to_open_count * 100.0) if neg_back_to_open_count > 0 else 0,
            }
        }

    def get_intraday_reversal_stats(self, stretch_threshold=0.50, retrace_pct=50.0, timeframe="Daily"):
        return self.get_stretch_reversal_stats(timeframe=timeframe, stretch_threshold=stretch_threshold, retrace_pct=retrace_pct)
