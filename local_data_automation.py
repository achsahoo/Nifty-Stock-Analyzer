"""
Local Data Automation: Nifty 500 Downloader (Section 4) & Multi-Stock Scanner (Section 5)
========================================================================================
Dedicated local tools for automated end-of-day data maintenance and real-time market scans.
"""

import os
import time
import shutil
import pandas as pd
import numpy as np
import streamlit as st
try:
    import polars as pl
except ImportError:
    pl = None

# Local & Cloud-Portable Data Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Resolves to ./data/... if present (Cloud & Local standalone), falls back to D:\NIFTY_MODEL
LOCAL_STOCK_DAILY_DIR = os.path.join(DATA_DIR, "stock_daily_parquet") if os.path.exists(os.path.join(DATA_DIR, "stock_daily_parquet")) else r"D:\NIFTY_MODEL\stock_daily_parquet"
LOCAL_STOCK_WEEKLY_DIR = os.path.join(DATA_DIR, "stock_weekly_parquet") if os.path.exists(os.path.join(DATA_DIR, "stock_weekly_parquet")) else r"D:\NIFTY_MODEL\stock_weekly_parquet"
LOCAL_STOCK_MONTHLY_DIR = os.path.join(DATA_DIR, "stock_monthly_parquet") if os.path.exists(os.path.join(DATA_DIR, "stock_monthly_parquet")) else r"D:\NIFTY_MODEL\stock_monthly_parquet"

# Secondary sync destination (if running on local machine with separate dev folder)
ALT_SYNC_DAILY = r"D:\NIFTY_MODEL\stock_daily_parquet"
ALT_SYNC_WEEKLY = r"D:\NIFTY_MODEL\stock_weekly_parquet"
ALT_SYNC_MONTHLY = r"D:\NIFTY_MODEL\stock_monthly_parquet"

# Deployment sync destination (if repository present)
DEPLOY_DIR = r"D:\NIFTY_MODEL\Nifty Move & Reversal Analytics\data"
DEPLOY_STOCK_DAILY_DIR = os.path.join(DEPLOY_DIR, "stock_daily_parquet")
DEPLOY_STOCK_WEEKLY_DIR = os.path.join(DEPLOY_DIR, "stock_weekly_parquet")
DEPLOY_STOCK_MONTHLY_DIR = os.path.join(DEPLOY_DIR, "stock_monthly_parquet")
has_deploy = os.path.exists(DEPLOY_DIR)

# NIFTY 1-Minute Precision Spot Data Paths
LOCAL_NIFTY_1MIN_FILE = os.path.join(DATA_DIR, "nifty_1min_2015_2026.parquet")
ALT_NIFTY_1MIN_FILES = [
    r"D:\NIFTY_MODEL\nifty_1min_2015_2026.parquet",
    r"D:\NIFTY_MODEL\Nifty_1min_spot_1min_option\nifty_1min_2015_2026.parquet",
    os.path.join(DEPLOY_DIR, "nifty_1min_2015_2026.parquet")
]

# NIFTY Higher Timeframes (Daily, Weekly)
LOCAL_NIFTY_HTF_DIR = os.path.join(DATA_DIR, "nifty_daily_weekly_parquet")
ALT_NIFTY_HTF_DIRS = [
    r"D:\NIFTY_MODEL\nifty_daily_weekly_parquet",
    os.path.join(DEPLOY_DIR, "nifty_daily_weekly_parquet")
]

NIFTY500_CSV_PATH = os.path.join(BASE_DIR, "ind_nifty500list.csv")
if not os.path.exists(NIFTY500_CSV_PATH):
    NIFTY500_CSV_PATH = r"D:\NIFTY_MODEL\backtest_result_optimizer\ind_nifty500list.csv"

# TradingView Credentials
TV_USERNAME = 'viline9842@adambra.com'
TV_PASSWORD = 'Trade@!751024'

@st.cache_data(show_spinner=False)
def get_nifty500_symbols():
    if os.path.exists(NIFTY500_CSV_PATH):
        try:
            df = pd.read_csv(NIFTY500_CSV_PATH)
            if 'Symbol' in df.columns:
                symbols = df['Symbol'].dropna().str.strip().unique().tolist()
                return sorted(symbols)
        except Exception:
            pass
    return ["INFY", "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL", "ITC", "LT", "AXISBANK"]


def get_stock_parquet_path(sym, timeframe="daily"):
    """Locates existing parquet file for a symbol in primary or alternate directories."""
    tf = timeframe.lower()
    fname = f"{sym}_{tf}_data_raw.parquet"
    if tf == "daily":
        dirs = [LOCAL_STOCK_DAILY_DIR, ALT_SYNC_DAILY]
    elif tf == "weekly":
        dirs = [LOCAL_STOCK_WEEKLY_DIR, ALT_SYNC_WEEKLY]
    else:
        dirs = [LOCAL_STOCK_MONTHLY_DIR, ALT_SYNC_MONTHLY]
    for d in dirs:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    return os.path.join(dirs[0], fname)


def get_parquet_latest_date(filepath):
    """Efficiently retrieves the maximum datetime from an existing parquet file."""
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_parquet(filepath, columns=['datetime'])
        if df.empty or 'datetime' not in df.columns:
            return None
        dt_col = pd.to_datetime(df['datetime'])
        if dt_col.dt.tz is not None:
            dt_col = dt_col.dt.tz_localize(None)
        return dt_col.max()
    except Exception:
        return None


def get_dataset_overview_metrics():
    """Returns total available parquet files and latest date from benchmark symbols."""
    target_dir = LOCAL_STOCK_DAILY_DIR
    if not os.path.exists(target_dir):
        return {"count": 0, "latest_date": None, "sample_sym": None}
    files = [f for f in os.listdir(target_dir) if f.endswith("_daily_data_raw.parquet")]
    if not files:
        return {"count": 0, "latest_date": None, "sample_sym": None}
    benchmarks = ["RELIANCE", "INFY", "TCS", "HDFCBANK", "ICICIBANK", "SBIN"]
    latest_dt = None
    found_sym = None
    for sym in benchmarks:
        p = os.path.join(target_dir, f"{sym}_daily_data_raw.parquet")
        if os.path.exists(p):
            dt = get_parquet_latest_date(p)
            if dt is not None and (latest_dt is None or dt > latest_dt):
                latest_dt = dt
                found_sym = sym
    if latest_dt is None and files:
        sample_p = os.path.join(target_dir, files[0])
        latest_dt = get_parquet_latest_date(sample_p)
        found_sym = files[0].replace("_daily_data_raw.parquet", "")
    return {"count": len(files), "latest_date": latest_dt, "sample_sym": found_sym}


def calculate_bars_to_fetch(last_dt, timeframe="daily", full_bars_count=8000):
    """Calculates minimal number of bars to fetch based on elapsed calendar time."""
    if last_dt is None:
        return full_bars_count
    now = pd.Timestamp.now()
    days_diff = (now.date() - last_dt.date()).days
    tf = timeframe.lower()
    if tf == "daily":
        if days_diff <= 0:
            return 5  # Refresh today's latest candle
        needed = days_diff + 5
        return min(max(needed, 5), full_bars_count)
    elif tf == "weekly":
        weeks_diff = max(0, days_diff // 7)
        needed = weeks_diff + 3
        return min(max(needed, 3), full_bars_count)
    else:  # monthly
        months_diff = max(0, (now.year - last_dt.year) * 12 + (now.month - last_dt.month))
        needed = months_diff + 2
        return min(max(needed, 2), full_bars_count)


def save_parquet_to_destinations(df, sym, timeframe):
    """Saves dataframe to local, alternate, and deployment parquet directories."""
    tf = timeframe.lower()
    filename = f"{sym}_{tf}_data_raw.parquet"
    has_alt = os.path.exists(ALT_SYNC_DAILY) and ALT_SYNC_DAILY != LOCAL_STOCK_DAILY_DIR
    if tf == "daily":
        loc_p = os.path.join(LOCAL_STOCK_DAILY_DIR, filename)
        alt_p = os.path.join(ALT_SYNC_DAILY, filename)
        dep_p = os.path.join(DEPLOY_STOCK_DAILY_DIR, filename) if has_deploy else None
    elif tf == "weekly":
        loc_p = os.path.join(LOCAL_STOCK_WEEKLY_DIR, filename)
        alt_p = os.path.join(ALT_SYNC_WEEKLY, filename)
        dep_p = os.path.join(DEPLOY_STOCK_WEEKLY_DIR, filename) if has_deploy else None
    else:
        loc_p = os.path.join(LOCAL_STOCK_MONTHLY_DIR, filename)
        alt_p = os.path.join(ALT_SYNC_MONTHLY, filename)
        dep_p = os.path.join(DEPLOY_STOCK_MONTHLY_DIR, filename) if has_deploy else None
        
    df.to_parquet(loc_p, index=False)
    if has_alt and os.path.abspath(loc_p) != os.path.abspath(alt_p):
        try:
            df.to_parquet(alt_p, index=False)
        except Exception:
            pass
    if dep_p and os.path.exists(os.path.dirname(dep_p)):
        try:
            df.to_parquet(dep_p, index=False)
        except Exception:
            pass


def sync_timeframe_data(tv, sym, tf_name, interval, full_bars, sync_mode="incremental"):
    """
    Downloads and synchronizes OHLCV data for a single timeframe.
    Returns: (is_success, detail_msg, new_bars_count, error_msg)
    """
    existing_path = get_stock_parquet_path(sym, tf_name)
    df_existing = None
    last_dt = None
    
    if sync_mode == "incremental" and os.path.exists(existing_path):
        try:
            df_existing = pd.read_parquet(existing_path)
            if not df_existing.empty and 'datetime' in df_existing.columns:
                df_existing['datetime'] = pd.to_datetime(df_existing['datetime'])
                if df_existing['datetime'].dt.tz is not None:
                    df_existing['datetime'] = df_existing['datetime'].dt.tz_localize(None)
                last_dt = df_existing['datetime'].max()
            else:
                df_existing = None
        except Exception:
            df_existing = None
            last_dt = None
            
    n_bars = calculate_bars_to_fetch(last_dt, tf_name, full_bars) if sync_mode == "incremental" else full_bars
    
    try:
        df_new = tv.get_hist(symbol=sym, exchange='NSE', interval=interval, n_bars=n_bars)
    except Exception as e:
        return False, "", 0, f"{tf_name} error: {str(e)[:60]}"
        
    if df_new is None or df_new.empty:
        return False, "", 0, f"{tf_name}: No data returned from NSE"
        
    df_new = df_new.reset_index()
    if 'datetime' not in df_new.columns:
        return False, "", 0, f"{tf_name}: Missing datetime column in data"
        
    df_new['datetime'] = pd.to_datetime(df_new['datetime'])
    if df_new['datetime'].dt.tz is not None:
        df_new['datetime'] = df_new['datetime'].dt.tz_localize(None)
        
    if df_existing is not None and last_dt is not None:
        # Incremental diff merge
        new_bars_only = df_new[df_new['datetime'] > last_dt]
        new_bars_count = len(new_bars_only)
        
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.drop_duplicates(subset=['datetime'], keep='last', inplace=True)
        df_combined.sort_values('datetime', inplace=True)
        df_combined.reset_index(drop=True, inplace=True)
        new_last_dt = df_combined['datetime'].max()
        
        save_parquet_to_destinations(df_combined, sym, tf_name)
        
        if new_bars_count > 0:
            detail = f"+{new_bars_count} new bars (to {new_last_dt.strftime('%d-%b-%Y')})"
        else:
            detail = f"Updated close ({new_last_dt.strftime('%d-%b-%Y')})"
        return True, detail, new_bars_count, None
    else:
        # Full sync or new stock
        df_combined = df_new.sort_values('datetime').reset_index(drop=True)
        new_last_dt = df_combined['datetime'].max()
        new_bars_count = len(df_combined)
        
        save_parquet_to_destinations(df_combined, sym, tf_name)
        detail = f"Full {new_bars_count} bars (to {new_last_dt.strftime('%d-%b-%Y')})"
        return True, detail, new_bars_count, None


def run_stock_download(selected_symbols, download_daily=True, download_weekly=True, download_monthly=True, sleep_sec=0.3, sync_mode="incremental"):
    """
    Downloads historical OHLCV data using TvDatafeed, directly saves as Parquet,
    and updates both local and deployment folders. Tracks successes and failures.
    Supports partial incremental sync (fetching diff only) and full re-download.
    """
    from tvDatafeed import TvDatafeed, Interval
    
    tv = TvDatafeed(TV_USERNAME, TV_PASSWORD)
    
    os.makedirs(LOCAL_STOCK_DAILY_DIR, exist_ok=True)
    os.makedirs(LOCAL_STOCK_WEEKLY_DIR, exist_ok=True)
    if download_monthly:
        os.makedirs(LOCAL_STOCK_MONTHLY_DIR, exist_ok=True)
        
    has_alt_sync = os.path.exists(ALT_SYNC_DAILY) and ALT_SYNC_DAILY != LOCAL_STOCK_DAILY_DIR
    if has_deploy:
        os.makedirs(DEPLOY_STOCK_DAILY_DIR, exist_ok=True)
        os.makedirs(DEPLOY_STOCK_WEEKLY_DIR, exist_ok=True)
        if download_monthly:
            os.makedirs(DEPLOY_STOCK_MONTHLY_DIR, exist_ok=True)
    
    total = len(selected_symbols)
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    
    success_list = []
    failed_list = []
    mode_label = "⚡ Partial Diff Sync" if sync_mode == "incremental" else "🔄 Full Re-download"
    
    for idx, sym in enumerate(selected_symbols):
        status_text.markdown(f"⏳ **[{idx+1}/{total}] Processing:** `{sym}` ({mode_label})...")
        errors = []
        synced_details = {}
        
        # 1. Weekly Download
        if download_weekly:
            succ_w, det_w, _, err_w = sync_timeframe_data(tv, sym, "Weekly", Interval.in_weekly, full_bars=2000, sync_mode=sync_mode)
            if succ_w:
                synced_details['Weekly'] = det_w
            else:
                errors.append(err_w)
                
        # 2. Daily Download
        if download_daily:
            succ_d, det_d, _, err_d = sync_timeframe_data(tv, sym, "Daily", Interval.in_daily, full_bars=8000, sync_mode=sync_mode)
            if succ_d:
                synced_details['Daily'] = det_d
            else:
                errors.append(err_d)
                
        # 3. Monthly Download
        if download_monthly:
            succ_m, det_m, _, err_m = sync_timeframe_data(tv, sym, "Monthly", Interval.in_monthly, full_bars=600, sync_mode=sync_mode)
            if succ_m:
                synced_details['Monthly'] = det_m
            else:
                errors.append(err_m)
                
        if errors:
            failed_list.append({
                'Symbol': sym,
                'Mode': mode_label,
                'Errors': "; ".join(errors),
                'Synced': ", ".join([f"{k}: {v}" for k, v in synced_details.items()]) if synced_details else "None"
            })
        else:
            success_list.append({
                'Symbol': sym,
                'Mode': mode_label,
                'Daily Diff': synced_details.get('Daily', '-'),
                'Weekly Diff': synced_details.get('Weekly', '-'),
                'Monthly Diff': synced_details.get('Monthly', '-'),
                'Status': '✅ Synced & Merged'
            })
            
        progress_bar.progress((idx + 1) / total)
        time.sleep(sleep_sec)
        
    status_text.markdown(f"🎉 **Batch Sync Complete!** Updated **{len(success_list)}** stocks. ({len(failed_list)} with errors/skipped)")
    st.cache_data.clear()
    
    st.session_state['last_download_success'] = pd.DataFrame(success_list)
    st.session_state['last_download_failed'] = pd.DataFrame(failed_list)


# =============================================================
# SECTION 4: AUTOMATED NIFTY 500 DOWNLOADER
# =============================================================
def render_downloader_section():
    st.header("📥 Section 4: Nifty 500 Automated Data Downloader & Parquet Sync")
    st.markdown("""
    One-click automated end-of-day market data updater for all **500 Nifty Stocks** from `ind_nifty500list.csv`.
    Downloads directly from TradingView, converts in-memory to **Apache Parquet**, and automatically syncs to:
    - `D:\\NIFTY_MODEL\\stock_daily_parquet`, `stock_weekly_parquet`, `stock_monthly_parquet`
    - `D:\\NIFTY_MODEL\\Nifty Move & Reversal Analytics\\data\\...` (Ready for Cloud deployment)
    """)
    
    # Dataset Health Card
    overview = get_dataset_overview_metrics()
    now_date = pd.Timestamp.now().date()
    latest_dt = overview.get("latest_date")
    total_files = overview.get("count", 0)
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.markdown(f"""<div class='metric-card'><span class='metric-label'>Parquet Files</span><div class='metric-value' style='color:#38bdf8;'>{total_files} / 500</div><span style='font-size:11px;color:#94a3b8;'>Available Stocks</span></div>""", unsafe_allow_html=True)
    with col_stat2:
        date_str = latest_dt.strftime('%d-%b-%Y') if latest_dt is not None else "No data"
        st.markdown(f"""<div class='metric-card'><span class='metric-label'>Dataset Last Date</span><div class='metric-value' style='color:#a78bfa;'>{date_str}</div><span style='font-size:11px;color:#94a3b8;'>Sample: {overview.get('sample_sym', 'N/A')}</span></div>""", unsafe_allow_html=True)
    with col_stat3:
        st.markdown(f"""<div class='metric-card'><span class='metric-label'>Current Date</span><div class='metric-value' style='color:#f59e0b;'>{now_date.strftime('%d-%b-%Y')}</div><span style='font-size:11px;color:#94a3b8;'>Today's Date</span></div>""", unsafe_allow_html=True)
    with col_stat4:
        if latest_dt is not None:
            days_behind = (now_date - latest_dt.date()).days
            if days_behind <= 0:
                health_color = "#00d26a"
                health_txt = "Up to Date"
                health_sub = "Today's candles ready to refresh"
            else:
                health_color = "#38bdf8"
                health_txt = f"{days_behind} Day(s) Behind"
                health_sub = f"Diff to fetch: {days_behind} day(s)"
        else:
            health_color = "#ef4444"
            health_txt = "Initial Setup Needed"
            health_sub = "No local parquets detected"
        st.markdown(f"""<div class='metric-card'><span class='metric-label'>Sync Health</span><div class='metric-value' style='color:{health_color};'>{health_txt}</div><span style='font-size:11px;color:#94a3b8;'>{health_sub}</span></div>""", unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    all_symbols = get_nifty500_symbols()
    
    # Sync Mode Selector
    row1_c1, row1_c2 = st.columns([2.3, 1.7])
    with row1_c1:
        sync_mode_choice = st.radio(
            "⚡ Select Download & Sync Mode:",
            [
                "⚡ Incremental / Partial Sync (Only download difference between last date and today - Fast & Recommended)",
                "🔄 Full Historical Re-download (Re-download multi-year data from scratch - Slow)"
            ],
            index=0,
            key="sec4_sync_mode"
        )
    with row1_c2:
        if "Incremental" in sync_mode_choice:
            st.success("⚡ **Partial Mode Active**: Inspects existing parquet, calculates missing days/weeks, downloads only difference bars, and merges seamlessly.")
        else:
            st.warning("⚠️ **Full Mode Active**: Will re-fetch up to 8,000 bars per stock from TradingView and overwrite existing files.")
            
    mode_arg = "incremental" if "Incremental" in sync_mode_choice else "full"
    
    c1, c2, c3 = st.columns([1.8, 2.0, 1.4])
    with c1:
        dl_scope = st.radio(
            "Select Symbols to Download:",
            [
                f"All Nifty 500 Stocks ({len(all_symbols)} symbols)",
                "Top 50 Nifty Stocks Only",
                "Custom Symbol List"
            ],
            index=0,
            key="sec4_scope"
        )
    with c2:
        dl_tf = st.radio(
            "Timeframes to Download:",
            [
                "All 3 Timeframes (Daily, Weekly, Monthly)",
                "Daily & Weekly Only",
                "Weekly Only",
                "Daily Only",
                "Monthly Only"
            ],
            index=0,
            key="sec4_tf"
        )
    with c3:
        def_pause = 0.2 if mode_arg == "incremental" else 0.4
        pause_sec = st.number_input("Pause Between Stocks (sec):", min_value=0.05, max_value=3.0, value=def_pause, step=0.05, key="sec4_pause")
        
    if dl_scope.startswith("All Nifty 500"):
        target_symbols = all_symbols
    elif dl_scope.startswith("Top 50"):
        target_symbols = all_symbols[:50]
    else:
        target_symbols = st.multiselect("Pick Specific Symbols:", all_symbols, default=["INFY", "RELIANCE", "TCS", "HDFCBANK"], key="sec4_custom_syms")
        
    est_per_stock = pause_sec + (0.15 if mode_arg == "incremental" else 0.8)
    est_time = len(target_symbols) * est_per_stock / 60
    st.info(f"📋 Target: **{len(target_symbols)}** stocks ({dl_tf}) | Mode: **{'Partial Diff' if mode_arg == 'incremental' else 'Full'}** | Estimated runtime: ~{est_time:.1f} minutes.")
    
    col_btn, col_retry = st.columns([2.0, 2.0])
    with col_btn:
        btn_start = st.button("🚀 Start Automated Download & Sync", type="primary", key="btn_sec4_start")
    
    if btn_start:
        is_d = "Daily" in dl_tf or "All 3" in dl_tf
        is_w = "Weekly" in dl_tf or "All 3" in dl_tf
        is_m = "Monthly" in dl_tf or "All 3" in dl_tf
        run_stock_download(target_symbols, download_daily=is_d, download_weekly=is_w, download_monthly=is_m, sleep_sec=pause_sec, sync_mode=mode_arg)
        
    # Display Success & Failure Reports
    if 'last_download_success' in st.session_state or 'last_download_failed' in st.session_state:
        df_succ = st.session_state.get('last_download_success', pd.DataFrame())
        df_fail = st.session_state.get('last_download_failed', pd.DataFrame())
        
        st.markdown("---")
        st.markdown("### 📊 Download & Sync Audit Report")
        
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown(f"""<div class='metric-card'><span class='metric-label'>Total Processed</span><div class='metric-value' style='color:#38bdf8;'>{len(df_succ) + len(df_fail)}</div><span style='font-size:11px;color:#94a3b8;'>Requested Stocks</span></div>""", unsafe_allow_html=True)
        with sc2:
            st.markdown(f"""<div class='metric-card'><span class='metric-label'>Successfully Synced</span><div class='metric-value' style='color:#00d26a;'>{len(df_succ)}</div><span style='font-size:11px;color:#94a3b8;'>✅ Saved to Parquet</span></div>""", unsafe_allow_html=True)
        with sc3:
            fail_color = "#ef4444" if len(df_fail) > 0 else "#94a3b8"
            st.markdown(f"""<div class='metric-card'><span class='metric-label'>Failed / Skipped</span><div class='metric-value' style='color:{fail_color};'>{len(df_fail)}</div><span style='font-size:11px;color:#94a3b8;'>⚠️ Requires Attention</span></div>""", unsafe_allow_html=True)
            
        tab_f, tab_s = st.tabs([f"❌ Failed / Skipped Stocks ({len(df_fail)})", f"✅ Successfully Synced Stocks ({len(df_succ)})"])
        
        with tab_f:
            if not df_fail.empty:
                st.dataframe(df_fail, width="stretch", hide_index=True)
                failed_syms_list = df_fail['Symbol'].tolist()
                if st.button(f"🔄 Re-try {len(failed_syms_list)} Failed Stocks Only", key="btn_retry_fail"):
                    is_d = "Daily" in dl_tf or "All 3" in dl_tf
                    is_w = "Weekly" in dl_tf or "All 3" in dl_tf
                    is_m = "Monthly" in dl_tf or "All 3" in dl_tf
                    run_stock_download(failed_syms_list, download_daily=is_d, download_weekly=is_w, download_monthly=is_m, sleep_sec=pause_sec, sync_mode=mode_arg)
                    st.rerun()
            else:
                st.success("🎉 All stocks downloaded and synced with 0 errors!")
                
        with tab_s:
            if not df_succ.empty:
                st.dataframe(df_succ, width="stretch", hide_index=True)
            else:
                st.info("No successful downloads recorded yet.")


# =============================================================
# SECTION 5: NIFTY 500 DIVERGENCE SCANNER
# =============================================================
def run_divergence_scanner(
    timeframe="Weekly",
    recency_mode="Recent (2026 - Present)",
    direction_filter="All",
    price_basis="close",
    fast_ema=9,
    slow_ema=21,
    ema_filter_mode="Trend Aligned"
):
    """
    Scans all available stock parquets for divergence signals matching recency, direction, and Dual EMA filters.
    Supports Daily, Weekly, and Monthly timeframes with configurable Price Basis (Candle Close vs Open) and customizable 2-EMA rules.
    """
    from rsi_divergence_ui import detect_pivots_and_signals
    
    if timeframe == "Weekly":
        target_dir = LOCAL_STOCK_WEEKLY_DIR
        pattern_suffix = "_weekly_data_raw.parquet"
    elif timeframe == "Monthly":
        target_dir = LOCAL_STOCK_MONTHLY_DIR if os.path.exists(LOCAL_STOCK_MONTHLY_DIR) else LOCAL_STOCK_DAILY_DIR
        pattern_suffix = "_monthly_data_raw.parquet" if os.path.exists(LOCAL_STOCK_MONTHLY_DIR) else "_daily_data_raw.parquet"
    else: # Daily
        target_dir = LOCAL_STOCK_DAILY_DIR
        pattern_suffix = "_daily_data_raw.parquet"
        
    if not os.path.exists(target_dir):
        st.error(f"Data directory `{target_dir}` not found.")
        return pd.DataFrame()
        
    files = [f for f in os.listdir(target_dir) if f.endswith(pattern_suffix)]
    if not files:
        st.warning(f"No parquets found in `{target_dir}` for timeframe {timeframe}.")
        return pd.DataFrame()
        
    total_files = len(files)
    scan_prog = st.progress(0.0)
    scan_status = st.empty()
    
    signals_found = []
    fast_p = min(int(fast_ema), int(slow_ema))
    slow_p = max(int(fast_ema), int(slow_ema))
    
    for idx, f in enumerate(files):
        sym = f.replace(pattern_suffix, "")
        p = os.path.join(target_dir, f)
        try:
            df = pd.read_parquet(p)
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # If monthly requested but loading from daily parquet
            if timeframe == "Monthly" and pattern_suffix == "_daily_data_raw.parquet":
                df = df.set_index('datetime').resample('ME').agg({
                    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                }).dropna().reset_index()
                
            df['date'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('date').reset_index(drop=True)
            if len(df) < max(slow_p + 10, 50):
                continue
                
            # Calculate Dual EMAs
            df['fast_ema'] = df['close'].ewm(span=fast_p, adjust=False).mean()
            df['slow_ema'] = df['close'].ewm(span=slow_p, adjust=False).mean()
                
            _, sig_df = detect_pivots_and_signals(df, rsi_len=14, lb_left=5, lb_right=5, tol=0.002, price_basis=price_basis)
            if not sig_df.empty:
                # Filter by recency
                if recency_mode.startswith("🔥 Today") or "Today" in recency_mode:
                    last_candle_day = df['date'].dt.date.iloc[-1]
                    if "Confirmed at p+lbR" in recency_mode:
                        sig_recent = sig_df[sig_df['confirm_date'].dt.date == last_candle_day]
                    else:
                        sig_recent = sig_df[sig_df['signal_date'].dt.date == last_candle_day]
                elif recency_mode.startswith("Recent (2026"):
                    sig_recent = sig_df[sig_df['confirm_date'] >= '2026-01-01']
                elif recency_mode.startswith("Last 4 Weeks / 20 Days"):
                    cutoff = df['date'].max() - pd.Timedelta(days=28)
                    sig_recent = sig_df[sig_df['confirm_date'] >= cutoff]
                elif recency_mode.startswith("Last 8 Weeks / 40 Days"):
                    cutoff = df['date'].max() - pd.Timedelta(days=56)
                    sig_recent = sig_df[sig_df['confirm_date'] >= cutoff]
                else: # All history
                    sig_recent = sig_df
                    
                # Filter by direction
                if direction_filter == "🟢 Bullish Only (BUY)":
                    sig_recent = sig_recent[sig_recent['signal'] == 'BUY']
                elif direction_filter == "🔴 Bearish Only (SELL)":
                    sig_recent = sig_recent[sig_recent['signal'] == 'SELL']
                    
                for _, r in sig_recent.iterrows():
                    sl_ref = r['signal_low'] if r['signal'] == 'BUY' else r['signal_high']
                    c_idx = int(r['confirm_idx']) if 'confirm_idx' in r and not pd.isna(r['confirm_idx']) else len(df) - 1
                    ref_price_col = 'open' if price_basis == 'open' else 'close'
                    confirm_ref_price = df[ref_price_col].iloc[c_idx] if c_idx < len(df) else r['price']
                    
                    val_fast = df['fast_ema'].iloc[c_idx] if c_idx < len(df) else np.nan
                    val_slow = df['slow_ema'].iloc[c_idx] if c_idx < len(df) else np.nan
                    
                    is_bullish_ema = val_fast > val_slow
                    ema_trend_str = f"Bullish ({fast_p} > {slow_p})" if is_bullish_ema else f"Bearish ({fast_p} < {slow_p})"
                    
                    # Apply Dual EMA Confluence Filter
                    if "Trend Aligned" in ema_filter_mode:
                        if r['signal'] == 'BUY' and not is_bullish_ema:
                            continue
                        if r['signal'] == 'SELL' and is_bullish_ema:
                            continue
                    elif "Price Aligned" in ema_filter_mode:
                        if r['signal'] == 'BUY' and not (confirm_ref_price > val_fast and confirm_ref_price > val_slow):
                            continue
                        if r['signal'] == 'SELL' and not (confirm_ref_price < val_fast and confirm_ref_price < val_slow):
                            continue
                    elif "Fresh EMA Cross" in ema_filter_mode:
                        w_start = max(0, c_idx - 5)
                        diff = df['fast_ema'].iloc[w_start:c_idx+1] - df['slow_ema'].iloc[w_start:c_idx+1]
                        if not (diff.min() <= 0 and diff.max() >= 0):
                            continue
                            
                    risk_pct = abs((confirm_ref_price - sl_ref) / confirm_ref_price) * 100.0 if confirm_ref_price > 0 else 0.0
                    signals_found.append({
                        'Symbol': sym,
                        'Timeframe': timeframe,
                        'Basis': 'Open' if price_basis == 'open' else 'Close',
                        'Signal': '🟢 BUY' if r['signal'] == 'BUY' else '🔴 SELL',
                        'raw_signal': r['signal'],
                        'Signal_Date': r['signal_date'].strftime('%Y-%m-%d'),
                        'Confirm_Date': r['confirm_date'].strftime('%Y-%m-%d'),
                        'Current / Entry Price': f"₹{confirm_ref_price:,.2f}",
                        f'Fast EMA ({fast_p})': f"₹{val_fast:,.2f}",
                        f'Slow EMA ({slow_p})': f"₹{val_slow:,.2f}",
                        'EMA Trend': ema_trend_str,
                        'Pivot Price (p)': f"₹{r['price']:,.2f}",
                        'raw_price': confirm_ref_price,
                        'RSI': f"{r['rsi']:.1f}",
                        'Pattern': r['reason'],
                        'Dynamic SL': f"₹{sl_ref:,.2f}",
                        'SL Risk': f"{risk_pct:.1f}%"
                    })
        except Exception:
            pass
            
        if idx % 10 == 0 or idx == total_files - 1:
            scan_prog.progress((idx + 1) / total_files)
            scan_status.markdown(f"🔍 Scanned **{idx+1}/{total_files}** stocks | Found **{len(signals_found)}** divergences...")
            
    scan_status.empty()
    scan_prog.empty()
    
    if signals_found:
        df_out = pd.DataFrame(signals_found)
        return df_out.sort_values('Confirm_Date', ascending=False).reset_index(drop=True)
    return pd.DataFrame()


def render_scanner_section():
    st.header("🔍 Section 5: Nifty 500 Multi-Stock Divergence Scanner")
    st.markdown("""
    High-speed scanner analyzing all **500 Nifty Equities** across **Daily, Weekly, and Monthly** candles.
    Instantly detects genuine **Bullish and Bearish RSI Divergences** on **Candle Close** (Standard EOD) or **Candle Open** (Early Morning Signal) filtered by **Dual EMA Confluence**.
    """)
    
    sc1, sc2, sc3, sc4 = st.columns([1.3, 1.8, 1.8, 1.6])
    with sc1:
        scan_tf = st.selectbox("Scan Timeframe:", ["Weekly", "Daily", "Monthly"], index=0, key="sec5_scan_tf")
    with sc2:
        scan_basis = st.selectbox(
            "Signal Timing / Basis:",
            [
                "📊 Candle Close Divergence (Standard)",
                "🌅 Candle Open Divergence (Early Signal)"
            ],
            index=0,
            key="sec5_scan_basis",
            help="Choose whether divergence is confirmed on Candle Close or evaluated on Candle Open."
        )
        basis_val = "open" if "Open" in scan_basis else "close"
    with sc3:
        today_label = "🔥 Today (Signal on Latest Open)" if basis_val == "open" else "🔥 Today (Signal on Latest Close)"
        scan_recency = st.selectbox(
            "Signal Recency Filter:",
            [
                today_label,
                "⚡ Today (Alert Confirmed at p+lbR)",
                "Last 4 Weeks / 20 Days",
                "Recent (2026 - Present)",
                "Last 8 Weeks / 40 Days",
                "All Confirmed Signals"
            ],
            index=0,
            key="sec5_scan_rec"
        )
    with sc4:
        scan_dir = st.selectbox("Direction Filter:", ["All", "🟢 Bullish Only (BUY)", "🔴 Bearish Only (SELL)"], index=0, key="sec5_scan_dir")
        
    st.markdown("##### ⚙️ Dual EMA Confluence Filter")
    em1, em2, em3 = st.columns([1.2, 1.2, 2.6])
    with em1:
        scan_fast_ema = st.number_input("⚡ Fast EMA Period:", min_value=2, max_value=200, value=9, step=1, key="sec5_fast_ema")
    with em2:
        scan_slow_ema = st.number_input("🐢 Slow EMA Period:", min_value=3, max_value=500, value=21, step=1, key="sec5_slow_ema")
    with em3:
        scan_ema_mode = st.selectbox(
            "🎯 EMA Confluence Rule:",
            [
                "🟢 Trend Aligned (BUY: Fast > Slow | SELL: Fast < Slow)",
                "💎 Price Aligned (BUY: Price > Both EMAs | SELL: Price < Both EMAs)",
                "⚡ Fresh EMA Cross (Cross within last 5 bars)",
                "⚪ None (Show All Divergences - Display EMA values)"
            ],
            index=0,
            key="sec5_ema_filter_mode",
            help="Filter RSI divergences using the two specified EMAs. 'Trend Aligned' only shows Bullish divergences when Fast EMA > Slow EMA and Bearish when Fast < Slow."
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    bcol1, bcol2, bcol3 = st.columns([1.8, 1.8, 2.0])
    with bcol1:
        btn_scan_close = st.button("📊 Scan Candle Close Divergences", type="primary", key="btn_sec5_close")
    with bcol2:
        btn_scan_open = st.button("🌅 Scan Candle Open Divergences", key="btn_sec5_open")
    with bcol3:
        btn_scan_custom = st.button("🚀 Run RSI Scanner (Active Filters)", key="btn_sec5_run")
        
    run_scan = False
    chosen_basis = basis_val
    if btn_scan_close:
        run_scan = True
        chosen_basis = "close"
    elif btn_scan_open:
        run_scan = True
        chosen_basis = "open"
    elif btn_scan_custom:
        run_scan = True
        chosen_basis = basis_val
        
    if run_scan:
        with st.spinner(f"Scanning 500 stocks for {scan_tf} divergences ({chosen_basis.upper()} basis, {scan_fast_ema}/{scan_slow_ema} EMA)..."):
            t_start = time.time()
            scan_results = run_divergence_scanner(
                scan_tf,
                scan_recency,
                scan_dir,
                price_basis=chosen_basis,
                fast_ema=scan_fast_ema,
                slow_ema=scan_slow_ema,
                ema_filter_mode=scan_ema_mode
            )
            t_elapsed = time.time() - t_start
            st.session_state['sec5_scan_results'] = scan_results
            st.session_state['sec5_scan_time'] = t_elapsed
            st.session_state['sec5_active_tf'] = scan_tf
            st.session_state['sec5_active_basis'] = chosen_basis
            st.session_state['sec5_active_fast_ema'] = scan_fast_ema
            st.session_state['sec5_active_slow_ema'] = scan_slow_ema
            
    if 'sec5_scan_results' in st.session_state:
        res_df = st.session_state['sec5_scan_results']
        t_el = st.session_state.get('sec5_scan_time', 0.0)
        cur_tf = st.session_state.get('sec5_active_tf', scan_tf)
        cur_fast = st.session_state.get('sec5_active_fast_ema', scan_fast_ema)
        cur_slow = st.session_state.get('sec5_active_slow_ema', scan_slow_ema)
        fast_p_disp = min(int(cur_fast), int(cur_slow))
        slow_p_disp = max(int(cur_fast), int(cur_slow))
        
        if not res_df.empty:
            buys = len(res_df[res_df['raw_signal'] == 'BUY'])
            sells = len(res_df[res_df['raw_signal'] == 'SELL'])
            
            st.markdown("---")
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(f"""<div class='metric-card'><span class='metric-label'>Total Matches</span><div class='metric-value' style='color:#38bdf8;'>{len(res_df)}</div><span style='font-size:11px;color:#94a3b8;'>Scan Time: {t_el:.2f}s</span></div>""", unsafe_allow_html=True)
            with k2:
                st.markdown(f"""<div class='metric-card'><span class='metric-label'>Bullish Setups</span><div class='metric-value' style='color:#00d26a;'>{buys}</div><span style='font-size:11px;color:#94a3b8;'>🟢 BUY Divergences</span></div>""", unsafe_allow_html=True)
            with k3:
                st.markdown(f"""<div class='metric-card'><span class='metric-label'>Bearish Setups</span><div class='metric-value' style='color:#ef4444;'>{sells}</div><span style='font-size:11px;color:#94a3b8;'>🔴 SELL Divergences</span></div>""", unsafe_allow_html=True)
            with k4:
                basis_tag = st.session_state.get('sec5_active_basis', 'close').upper()
                st.markdown(f"""<div class='metric-card'><span class='metric-label'>Active Setup</span><div class='metric-value' style='color:#fbbf24;'>{cur_tf} ({basis_tag})</div><span style='font-size:11px;color:#94a3b8;'>EMAs: {fast_p_disp}/{slow_p_disp} | {scan_recency}</span></div>""", unsafe_allow_html=True)
            
            display_cols = [
                'Symbol', 'Timeframe', 'Basis', 'Signal', 'Signal_Date', 'Confirm_Date',
                'Current / Entry Price', f'Fast EMA ({fast_p_disp})', f'Slow EMA ({slow_p_disp})',
                'EMA Trend', 'RSI', 'Dynamic SL', 'SL Risk', 'Pattern'
            ]
            st.dataframe(res_df[display_cols], width="stretch", hide_index=True)
            
            st.markdown("#### 🎯 Quick-Load Into Section 3 Backtest Engine")
            matched_symbols = sorted(res_df['Symbol'].unique().tolist())
            qcol1, qcol2 = st.columns([2.5, 1.5])
            with qcol1:
                picked_stock = st.selectbox("Select Detected Stock to Analyze in Detail:", matched_symbols, key="sec5_picked_stock")
            with qcol2:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                if st.button(f"📊 Open {picked_stock} in Section 3", key="btn_sec5_load"):
                    st.session_state['selected_stock_override'] = picked_stock
                    st.session_state['nav_selection_override'] = "📐 Section 3: RSI Custom Divergence Backtest"
                    st.rerun()
        else:
            st.info(f"No divergence signals found on {cur_tf} matching the selected filter `{scan_recency}`.")


# =============================================================
# SECTION 6: AUTOMATED NIFTY DATA DOWNLOADER (1-MIN & HTF)
# =============================================================
def get_nifty_dataset_overview():
    """Inspects all Nifty datasets (1-Min, Daily, Weekly) and returns summary metrics."""
    now_date = pd.Timestamp.now().date()
    
    # 1-Min Spot
    one_min_path = LOCAL_NIFTY_1MIN_FILE if os.path.exists(LOCAL_NIFTY_1MIN_FILE) else ALT_NIFTY_1MIN_FILES[0]
    one_min_dt = None
    one_min_rows = 0
    if os.path.exists(one_min_path):
        try:
            if pl is not None:
                df_scan = pl.scan_parquet(one_min_path)
                one_min_rows = df_scan.select(pl.len()).collect().item()
                tail_date = df_scan.select(pl.col("date").tail(1)).collect().item()
                one_min_dt = pd.to_datetime(tail_date)
            else:
                df_tail = pd.read_parquet(one_min_path, columns=['date'])
                one_min_rows = len(df_tail)
                one_min_dt = pd.to_datetime(df_tail['date'].iloc[-1])
            if one_min_dt.tzinfo is not None:
                one_min_dt = one_min_dt.tz_localize(None)
        except Exception:
            one_min_dt = None
            
    # Daily
    daily_path = os.path.join(LOCAL_NIFTY_HTF_DIR, "nifty_daily_data_raw.parquet")
    if not os.path.exists(daily_path) and ALT_NIFTY_HTF_DIRS:
        daily_path = os.path.join(ALT_NIFTY_HTF_DIRS[0], "nifty_daily_data_raw.parquet")
    daily_dt = get_parquet_latest_date(daily_path)
    daily_rows = 0
    if os.path.exists(daily_path):
        try:
            daily_rows = len(pd.read_parquet(daily_path, columns=['datetime']))
        except Exception:
            pass

    # Weekly
    weekly_path = os.path.join(LOCAL_NIFTY_HTF_DIR, "nifty_weekly_data_raw.parquet")
    if not os.path.exists(weekly_path) and ALT_NIFTY_HTF_DIRS:
        weekly_path = os.path.join(ALT_NIFTY_HTF_DIRS[0], "nifty_weekly_data_raw.parquet")
    weekly_dt = get_parquet_latest_date(weekly_path)
    weekly_rows = 0
    if os.path.exists(weekly_path):
        try:
            weekly_rows = len(pd.read_parquet(weekly_path, columns=['datetime']))
        except Exception:
            pass

    return {
        "1min": {"dt": one_min_dt, "rows": one_min_rows, "path": one_min_path},
        "daily": {"dt": daily_dt, "rows": daily_rows, "path": daily_path},
        "weekly": {"dt": weekly_dt, "rows": weekly_rows, "path": weekly_path},
        "now_date": now_date
    }


def sync_nifty_1min_data(tv, sync_mode="incremental"):
    """
    Synchronizes NIFTY 1-minute precision spot data into nifty_1min_2015_2026.parquet.
    Appends new missing candles, deduplicates by timestamp, and synchronizes to all directories.
    """
    primary_path = LOCAL_NIFTY_1MIN_FILE if os.path.exists(LOCAL_NIFTY_1MIN_FILE) else ALT_NIFTY_1MIN_FILES[0]
    if not os.path.exists(primary_path):
        return False, "File nifty_1min_2015_2026.parquet not found", 0, "File missing"

    try:
        if pl is not None:
            df_existing = pl.read_parquet(primary_path)
            last_dt = pd.to_datetime(df_existing['date'].max())
        else:
            df_existing = pd.read_parquet(primary_path)
            last_dt = pd.to_datetime(df_existing['date'].max())
        if last_dt.tzinfo is not None:
            last_dt = last_dt.tz_localize(None)
    except Exception as e:
        return False, f"Error reading existing 1-min parquet: {str(e)[:60]}", 0, str(e)

    now = pd.Timestamp.now()
    days_diff = (now.date() - last_dt.date()).days
    
    # Calculate needed bars (375 bars per trading day)
    if sync_mode == "incremental":
        needed_bars = min(max((days_diff + 2) * 375, 200), 5000)
    else:
        needed_bars = 5000  # Max available on TradingView 1-min feed

    try:
        from tvDatafeed import Interval
        df_new = tv.get_hist('NIFTY', 'NSE', interval=Interval.in_1_minute, n_bars=needed_bars)
    except Exception as e:
        return False, f"TradingView 1-min fetch failed: {str(e)[:60]}", 0, str(e)

    if df_new is None or df_new.empty:
        return False, "No 1-minute data returned from NSE", 0, "Empty data"

    df_new = df_new.reset_index()
    if 'datetime' not in df_new.columns:
        return False, "Missing datetime column in 1-min data", 0, "Missing datetime"

    # Schema alignment: ['open', 'high', 'low', 'close', 'volume', 'date']
    df_new['date'] = pd.to_datetime(df_new['datetime'])
    if df_new['date'].dt.tz is not None:
        df_new['date'] = df_new['date'].dt.tz_localize(None)
    df_new['open'] = df_new['open'].astype('float64')
    df_new['high'] = df_new['high'].astype('float64')
    df_new['low'] = df_new['low'].astype('float64')
    df_new['close'] = df_new['close'].astype('float64')
    df_new['volume'] = df_new['volume'].fillna(0).astype('int64')
    df_new = df_new[['open', 'high', 'low', 'close', 'volume', 'date']]

    new_candles = df_new[df_new['date'] > last_dt]
    new_bars_count = len(new_candles)

    # Merge and deduplicate
    try:
        if pl is not None:
            try:
                df_new_pl = pl.from_pandas(df_new)
                # Auto-align schema types (e.g. Datetime('us') vs Datetime('ns')) to match existing dataset
                for col, dtype in df_existing.schema.items():
                    if col in df_new_pl.columns and df_new_pl.schema[col] != dtype:
                        df_new_pl = df_new_pl.with_columns(pl.col(col).cast(dtype))

                df_combined = pl.concat([df_existing, df_new_pl]).unique(subset=['date'], keep='last').sort('date')
                new_last_dt = pd.to_datetime(df_combined['date'].max())
                total_rows = len(df_combined)
                
                # Save to primary
                os.makedirs(os.path.dirname(LOCAL_NIFTY_1MIN_FILE), exist_ok=True)
                df_combined.write_parquet(LOCAL_NIFTY_1MIN_FILE)
                
                # Save to secondary / alternate paths
                for alt_p in ALT_NIFTY_1MIN_FILES:
                    try:
                        if os.path.exists(os.path.dirname(alt_p)) and os.path.abspath(alt_p) != os.path.abspath(LOCAL_NIFTY_1MIN_FILE):
                            df_combined.write_parquet(alt_p)
                    except Exception:
                        pass
            except Exception as e_pl:
                # Resilient fallback to pandas if polars encounters any schema/conversion conflict
                df_ex_pd = df_existing.to_pandas() if hasattr(df_existing, "to_pandas") else df_existing
                df_combined = pd.concat([df_ex_pd, df_new], ignore_index=True)
                df_combined.drop_duplicates(subset=['date'], keep='last', inplace=True)
                df_combined.sort_values('date', inplace=True)
                df_combined.reset_index(drop=True, inplace=True)
                new_last_dt = df_combined['date'].max()
                total_rows = len(df_combined)
                
                os.makedirs(os.path.dirname(LOCAL_NIFTY_1MIN_FILE), exist_ok=True)
                df_combined.to_parquet(LOCAL_NIFTY_1MIN_FILE, index=False)
                for alt_p in ALT_NIFTY_1MIN_FILES:
                    try:
                        if os.path.exists(os.path.dirname(alt_p)) and os.path.abspath(alt_p) != os.path.abspath(LOCAL_NIFTY_1MIN_FILE):
                            df_combined.to_parquet(alt_p, index=False)
                    except Exception:
                        pass
        else:
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.drop_duplicates(subset=['date'], keep='last', inplace=True)
            df_combined.sort_values('date', inplace=True)
            df_combined.reset_index(drop=True, inplace=True)
            new_last_dt = df_combined['date'].max()
            total_rows = len(df_combined)
            
            os.makedirs(os.path.dirname(LOCAL_NIFTY_1MIN_FILE), exist_ok=True)
            df_combined.to_parquet(LOCAL_NIFTY_1MIN_FILE, index=False)
            for alt_p in ALT_NIFTY_1MIN_FILES:
                try:
                    if os.path.exists(os.path.dirname(alt_p)) and os.path.abspath(alt_p) != os.path.abspath(LOCAL_NIFTY_1MIN_FILE):
                        df_combined.to_parquet(alt_p, index=False)
                except Exception:
                    pass
    except Exception as esave:
        return False, f"Failed saving 1-min parquet: {str(esave)[:60]}", 0, str(esave)

    if new_bars_count > 0:
        detail = f"+{new_bars_count:,} new 1-min candles (to {new_last_dt.strftime('%d-%b-%Y %H:%M')}, total: {total_rows:,})"
    else:
        detail = f"Updated latest candle (up to {new_last_dt.strftime('%d-%b-%Y %H:%M')}, total: {total_rows:,})"
        
    return True, detail, new_bars_count, None


def sync_nifty_htf_data(tv, tf_name, interval, full_bars=8000, sync_mode="incremental"):
    """
    Synchronizes NIFTY daily or weekly parquet data (nifty_daily_data_raw.parquet / nifty_weekly_data_raw.parquet).
    """
    filename = f"nifty_{tf_name.lower()}_data_raw.parquet"
    primary_dir = LOCAL_NIFTY_HTF_DIR
    os.makedirs(primary_dir, exist_ok=True)
    primary_path = os.path.join(primary_dir, filename)
    
    df_existing = None
    last_dt = None
    if os.path.exists(primary_path):
        try:
            df_existing = pd.read_parquet(primary_path)
            if not df_existing.empty and 'datetime' in df_existing.columns:
                df_existing['datetime'] = pd.to_datetime(df_existing['datetime'])
                if df_existing['datetime'].dt.tz is not None:
                    df_existing['datetime'] = df_existing['datetime'].dt.tz_localize(None)
                last_dt = df_existing['datetime'].max()
        except Exception:
            df_existing = None
            last_dt = None
    elif ALT_NIFTY_HTF_DIRS:
        for alt_d in ALT_NIFTY_HTF_DIRS:
            alt_p = os.path.join(alt_d, filename)
            if os.path.exists(alt_p):
                try:
                    df_existing = pd.read_parquet(alt_p)
                    if not df_existing.empty and 'datetime' in df_existing.columns:
                        df_existing['datetime'] = pd.to_datetime(df_existing['datetime'])
                        if df_existing['datetime'].dt.tz is not None:
                            df_existing['datetime'] = df_existing['datetime'].dt.tz_localize(None)
                        last_dt = df_existing['datetime'].max()
                    break
                except Exception:
                    pass

    n_bars = calculate_bars_to_fetch(last_dt, tf_name, full_bars) if sync_mode == "incremental" else full_bars

    try:
        df_new = tv.get_hist('NIFTY', 'NSE', interval=interval, n_bars=n_bars)
    except Exception as e:
        return False, "", 0, f"{tf_name} error: {str(e)[:60]}"

    if df_new is None or df_new.empty:
        return False, "", 0, f"{tf_name}: No data returned from NSE"

    df_new = df_new.reset_index()
    if 'datetime' not in df_new.columns:
        return False, "", 0, f"{tf_name}: Missing datetime column in data"

    df_new['datetime'] = pd.to_datetime(df_new['datetime'])
    if df_new['datetime'].dt.tz is not None:
        df_new['datetime'] = df_new['datetime'].dt.tz_localize(None)

    if df_existing is not None and last_dt is not None:
        new_bars_only = df_new[df_new['datetime'] > last_dt]
        new_bars_count = len(new_bars_only)

        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.drop_duplicates(subset=['datetime'], keep='last', inplace=True)
        df_combined.sort_values('datetime', inplace=True)
        df_combined.reset_index(drop=True, inplace=True)
        new_last_dt = df_combined['datetime'].max()
    else:
        df_combined = df_new.sort_values('datetime').reset_index(drop=True)
        new_last_dt = df_combined['datetime'].max()
        new_bars_count = len(df_combined)

    # Save to primary
    df_combined.to_parquet(primary_path, index=False)

    # Save to secondary directories
    for alt_d in ALT_NIFTY_HTF_DIRS:
        try:
            if os.path.exists(alt_d) and os.path.abspath(alt_d) != os.path.abspath(primary_dir):
                alt_file = os.path.join(alt_d, filename)
                df_combined.to_parquet(alt_file, index=False)
        except Exception:
            pass

    if df_existing is not None and last_dt is not None:
        if new_bars_count > 0:
            detail = f"+{new_bars_count} new bars (to {new_last_dt.strftime('%d-%b-%Y')})"
        else:
            detail = f"Updated close ({new_last_dt.strftime('%d-%b-%Y')})"
    else:
        detail = f"Full {new_bars_count} bars (to {new_last_dt.strftime('%d-%b-%Y')})"

    return True, detail, new_bars_count, None


def render_nifty_downloader_section():
    st.header("⚡ Section 6: Nifty Automated Data Downloader & Parquet Sync")
    st.markdown("""
    One-click dedicated data synchronization engine for **NIFTY 50 (Spot & Index)**.
    Maintains **1-Minute Precision Spot History** (`nifty_1min_2015_2026.parquet`) and **Daily / Weekly Index Data** (`nifty_daily_data_raw.parquet`, `nifty_weekly_data_raw.parquet`).
    - Automatically checks the latest candle dates and performs **lightning-fast incremental partial syncs**.
    - Syncs seamlessly across `./data`, `D:\\NIFTY_MODEL`, and deployment directories.
    """)
    
    # Overview
    overview = get_nifty_dataset_overview()
    now_date = overview["now_date"]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        dt_1m = overview["1min"]["dt"]
        dt_1m_str = dt_1m.strftime('%d-%b-%Y %H:%M') if dt_1m else "Missing"
        rows_1m = f"{overview['1min']['rows']:,}" if overview['1min']['rows'] else "0"
        st.markdown(f"""<div class='metric-card'><span class='metric-label'>1-Min Spot Archive</span><div class='metric-value' style='color:#38bdf8;'>{rows_1m}</div><span style='font-size:11px;color:#94a3b8;'>Last: {dt_1m_str}</span></div>""", unsafe_allow_html=True)
    with col2:
        dt_d = overview["daily"]["dt"]
        dt_d_str = dt_d.strftime('%d-%b-%Y') if dt_d else "Missing"
        rows_d = f"{overview['daily']['rows']:,}" if overview['daily']['rows'] else "0"
        st.markdown(f"""<div class='metric-card'><span class='metric-label'>Daily Index Parquet</span><div class='metric-value' style='color:#a78bfa;'>{rows_d}</div><span style='font-size:11px;color:#94a3b8;'>Last: {dt_d_str}</span></div>""", unsafe_allow_html=True)
    with col3:
        dt_w = overview["weekly"]["dt"]
        dt_w_str = dt_w.strftime('%d-%b-%Y') if dt_w else "Missing"
        rows_w = f"{overview['weekly']['rows']:,}" if overview['weekly']['rows'] else "0"
        st.markdown(f"""<div class='metric-card'><span class='metric-label'>Weekly Index Parquet</span><div class='metric-value' style='color:#fbbf24;'>{rows_w}</div><span style='font-size:11px;color:#94a3b8;'>Last: {dt_w_str}</span></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class='metric-card'><span class='metric-label'>Today's Date</span><div class='metric-value' style='color:#00d26a;'>{now_date.strftime('%d-%b-%Y')}</div><span style='font-size:11px;color:#94a3b8;'>Live Trading Session</span></div>""", unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Configuration
    c_mode, c_scope = st.columns([2.2, 1.8])
    with c_mode:
        sync_mode = st.radio(
            "⚡ Sync Mode:",
            [
                "⚡ Incremental / Partial Sync (Only download difference between last date and current - Fast & Safe)",
                "🔄 Full Re-download (Higher Timeframes only; 1-min fetches latest 5,000 bars)"
            ],
            index=0,
            key="sec6_sync_mode"
        )
    with c_scope:
        scope = st.radio(
            "Select Datasets to Synchronize:",
            [
                "All Nifty Datasets (1-Minute Spot + Daily + Weekly)",
                "1-Minute Spot Only (nifty_1min_2015_2026.parquet)",
                "Daily & Weekly Index Only (nifty_daily_weekly_parquet)",
                "Daily Only",
                "Weekly Only"
            ],
            index=0,
            key="sec6_scope"
        )
        
    btn_sync = st.button("🚀 Start Nifty Data Synchronization", type="primary", key="btn_sec6_start")
    
    mode_arg = "incremental" if "Incremental" in sync_mode else "full"
    
    if btn_sync:
        from tvDatafeed import TvDatafeed, Interval
        tv = TvDatafeed(TV_USERNAME, TV_PASSWORD)
        
        do_1m = "1-Minute" in scope or "All Nifty" in scope
        do_d = "Daily" in scope or "All Nifty" in scope
        do_w = "Weekly" in scope or "All Nifty" in scope
        
        results = []
        with st.spinner("Synchronizing NIFTY data with TradingView..."):
            prog = st.progress(0.0)
            status_t = st.empty()
            
            # 1-Minute
            if do_1m:
                status_t.markdown("⏳ **Syncing NIFTY 1-Minute Spot Data** (`nifty_1min_2015_2026.parquet`)...")
                s_1m, d_1m, c_1m, e_1m = sync_nifty_1min_data(tv, sync_mode=mode_arg)
                results.append({
                    "Dataset": "NIFTY 1-Minute Spot",
                    "Target File": "nifty_1min_2015_2026.parquet",
                    "Result": "✅ Success" if s_1m else "❌ Failed",
                    "Details": d_1m if s_1m else e_1m,
                    "New Bars": c_1m
                })
                prog.progress(0.4 if (do_d or do_w) else 1.0)
                
            # Daily
            if do_d:
                status_t.markdown("⏳ **Syncing NIFTY Daily Index Data** (`nifty_daily_data_raw.parquet`)...")
                s_d, d_d, c_d, e_d = sync_nifty_htf_data(tv, "Daily", Interval.in_daily, full_bars=8000, sync_mode=mode_arg)
                results.append({
                    "Dataset": "NIFTY Daily Index",
                    "Target File": "nifty_daily_data_raw.parquet",
                    "Result": "✅ Success" if s_d else "❌ Failed",
                    "Details": d_d if s_d else e_d,
                    "New Bars": c_d
                })
                prog.progress(0.7 if do_w else 1.0)
                
            # Weekly
            if do_w:
                status_t.markdown("⏳ **Syncing NIFTY Weekly Index Data** (`nifty_weekly_data_raw.parquet`)...")
                s_w, d_w, c_w, e_w = sync_nifty_htf_data(tv, "Weekly", Interval.in_weekly, full_bars=2000, sync_mode=mode_arg)
                results.append({
                    "Dataset": "NIFTY Weekly Index",
                    "Target File": "nifty_weekly_data_raw.parquet",
                    "Result": "✅ Success" if s_w else "❌ Failed",
                    "Details": d_w if s_w else e_w,
                    "New Bars": c_w
                })
                prog.progress(1.0)
                
            status_t.markdown("🎉 **Nifty Synchronization Complete!**")
            st.cache_data.clear()
            st.session_state['last_nifty_sync'] = pd.DataFrame(results)
            st.rerun()

    # Display Report
    if 'last_nifty_sync' in st.session_state:
        st.markdown("---")
        st.markdown("### 📊 Nifty Synchronization Audit Report")
        df_audit = st.session_state['last_nifty_sync']
        st.dataframe(df_audit, width="stretch", hide_index=True)
        st.success("✅ NIFTY data successfully refreshed and synced across all local and cloud deployment directories.")
