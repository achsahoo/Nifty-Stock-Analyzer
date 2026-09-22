import os
import datetime
import calendar
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

def compute_monthly_expiries(trading_days):
    """
    Computes monthly contract expiry dates based on official NSE schedule:
    - June 2000 – Aug 31, 2025: Last Thursday of each month.
    - Sept 1, 2025 – Present: Last Tuesday of each month.
    Holiday shifting rule: If the scheduled day is an exchange holiday (no daily data),
    shifts backwards to the immediately preceding trading day in that month.
    """
    if not trading_days:
        return []
    min_date, max_date = trading_days[0], trading_days[-1]
    months = pd.date_range(min_date.replace(day=1), max_date, freq='MS')
    monthly_expiries = []
    for m in months:
        year, month = m.year, m.month
        target_weekday = 3 if m < pd.Timestamp('2025-09-01') else 1  # Thu=3, Tue=1
        _, last_day = calendar.monthrange(year, month)
        sched_date = None
        for day in range(last_day, 0, -1):
            dt = datetime.date(year, month, day)
            if dt.weekday() == target_weekday:
                sched_date = dt
                break
        if sched_date:
            actual = [d for d in trading_days if d <= sched_date and d.year == year and d.month == month]
            if actual:
                act = max(actual)
                monthly_expiries.append({
                    'sched_date': sched_date,
                    'actual_date': act,
                    'shifted': (sched_date != act)
                })
    return monthly_expiries

def compute_weekly_expiries(trading_days, start_date=None):
    """
    Computes weekly contract expiry dates based on official NSE schedule:
    - Feb 11, 2019 – Aug 31, 2025: Every Thursday.
    - Sept 1, 2025 – Present: Every Tuesday.
    (If start_date < 2019-02-11, uses Thursday schedule for extended backtests).
    Holiday shifting rule: If scheduled expiry is a holiday, shifts back to the preceding available trading day.
    """
    if not trading_days:
        return []
    if start_date is None:
        start_date = trading_days[0]
    end_date = trading_days[-1]
    
    candidates = []
    cur = start_date
    while cur <= end_date:
        if cur < datetime.date(2025, 9, 1):
            if cur.weekday() == 3:  # Thursday
                candidates.append(cur)
        else:
            if cur.weekday() == 1:  # Tuesday
                candidates.append(cur)
        cur += datetime.timedelta(days=1)
        
    actual_expiries = []
    prev_exp = None
    for cand in candidates:
        valid_days = [d for d in trading_days if d <= cand and (prev_exp is None or d > prev_exp)]
        if valid_days:
            act = max(valid_days)
            actual_expiries.append({
                'sched_date': cand,
                'actual_date': act,
                'shifted': (cand != act)
            })
            prev_exp = act
    return actual_expiries

def build_expiry_cycles(daily_df, expiry_meta_list, start_filter=None):
    """
    Aggregates daily candles into complete expiry cycles bounded by contract settlement dates.
    - Cycle Open: First trading day immediately following previous expiry (or dataset start).
    - Cycle Close: Expiry trading day.
    OHLC:
      open = first day open
      high = max high across all days in cycle
      low = min low across all days in cycle
      close = expiry day close
    """
    if not expiry_meta_list or daily_df.empty:
        return pd.DataFrame()
        
    df = daily_df.sort_values('date').copy()
    if start_filter:
        df = df[df['date'].dt.date >= start_filter]
        
    actual_dates = [item['actual_date'] for item in expiry_meta_list]
    exp_ts = [pd.Timestamp(d) for d in actual_dates]
    
    indices = np.searchsorted(exp_ts, df['date'])
    valid_mask = indices < len(exp_ts)
    df = df[valid_mask]
    indices = indices[valid_mask]
    
    df['cycle_idx'] = indices
    grouped = df.groupby('cycle_idx').agg(
        cycle_start=('date', 'first'),
        date=('date', 'last'),
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        trading_days=('date', 'count')
    ).reset_index()
    
    sched_dates = [item['sched_date'] for item in expiry_meta_list]
    shifted_flags = [item['shifted'] for item in expiry_meta_list]
    
    grouped['sched_date'] = grouped['cycle_idx'].apply(lambda idx: pd.Timestamp(sched_dates[idx]) if idx < len(sched_dates) else None)
    grouped['is_shifted'] = grouped['cycle_idx'].apply(lambda idx: shifted_flags[idx] if idx < len(shifted_flags) else False)
    grouped['range_pct'] = (grouped['high'] - grouped['low']) / grouped['open'] * 100.0
    grouped['move_pct'] = (grouped['close'] - grouped['open']) / grouped['open'] * 100.0
    grouped['start_day_name'] = grouped['cycle_start'].dt.day_name()
    grouped['expiry_day_name'] = grouped['date'].dt.day_name()
    
    return grouped.drop(columns=['cycle_idx'])

class NiftyAnalyticsEngine:
    def __init__(self, parquet_path=None):
        self.parquet_path = parquet_path or _resolve_spot_file()
        self._df_raw = None
        self._daily_df = None
        self._weekly_df = None
        self._monthly_df = None
        self._monthly_expiry_df = None
        self._weekly_expiry_official_df = None
        self._weekly_expiry_full_df = None
        self._monthly_expiries_meta = []
        self._weekly_expiries_official_meta = []
        self._weekly_expiries_full_meta = []
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
            
            # 4. Expiry Aggregations (NSE F&O Schedule & Holiday Shifts)
            trading_days = sorted(daily['date'].dt.date.unique())
            
            # Monthly Expiries (2015 – 2026)
            self._monthly_expiries_meta = compute_monthly_expiries(trading_days)
            self._monthly_expiry_df = build_expiry_cycles(daily, self._monthly_expiries_meta)
            
            # Weekly Expiries (Official F&O Era: Feb 11, 2019+)
            start_official = datetime.date(2019, 2, 11)
            self._weekly_expiries_official_meta = compute_weekly_expiries(trading_days, start_official)
            self._weekly_expiry_official_df = build_expiry_cycles(daily, self._weekly_expiries_official_meta, start_official)
            
            # Weekly Expiries (Full Extended: 2015+)
            self._weekly_expiries_full_meta = compute_weekly_expiries(trading_days, trading_days[0])
            self._weekly_expiry_full_df = build_expiry_cycles(daily, self._weekly_expiries_full_meta)
            
    def get_range_and_move_stats(self, timeframe="Daily", threshold_pct=1.0, close_type="TradingView", weekly_scope="Official"):
        self.load_data()
        
        is_expiry = ("expiry" in str(close_type).lower())
        
        if timeframe == "Weekly":
            if is_expiry:
                if "full" in str(weekly_scope).lower() or "extended" in str(weekly_scope).lower():
                    df = self._weekly_expiry_full_df
                    period_label = "Expiry Weeks (2015–Present)"
                else:
                    df = self._weekly_expiry_official_df
                    period_label = "Expiry Weeks (Feb 2019–Present)"
            else:
                df = self._weekly_df
                period_label = "Weeks"
        elif timeframe == "Monthly":
            if is_expiry:
                df = self._monthly_expiry_df
                period_label = "Expiry Months"
            else:
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
