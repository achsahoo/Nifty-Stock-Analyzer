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


def run_stock_download(selected_symbols, download_daily=True, download_weekly=True, download_monthly=True, sleep_sec=0.4):
    """
    Downloads historical OHLCV data using TvDatafeed, directly saves as Parquet,
    and updates both local and deployment folders. Tracks both successes and failures.
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
    
    for idx, sym in enumerate(selected_symbols):
        status_text.markdown(f"⏳ **[{idx+1}/{total}] Processing:** `{sym}`...")
        errors = []
        synced_tf = []
        
        # 1. Weekly Download
        if download_weekly:
            try:
                df_w = tv.get_hist(symbol=sym, exchange='NSE', interval=Interval.in_weekly, n_bars=2000)
                if df_w is not None and not df_w.empty:
                    df_w = df_w.reset_index()
                    df_w.to_parquet(os.path.join(LOCAL_STOCK_WEEKLY_DIR, f"{sym}_weekly_data_raw.parquet"), index=False)
                    if has_alt_sync:
                        df_w.to_parquet(os.path.join(ALT_SYNC_WEEKLY, f"{sym}_weekly_data_raw.parquet"), index=False)
                    synced_tf.append("Weekly")
                else:
                    errors.append("Weekly: No data returned from NSE")
            except Exception as ew:
                errors.append(f"Weekly error: {str(ew)[:60]}")
                
        # 2. Daily Download
        if download_daily:
            try:
                df_d = tv.get_hist(symbol=sym, exchange='NSE', interval=Interval.in_daily, n_bars=8000)
                if df_d is not None and not df_d.empty:
                    df_d = df_d.reset_index()
                    df_d.to_parquet(os.path.join(LOCAL_STOCK_DAILY_DIR, f"{sym}_daily_data_raw.parquet"), index=False)
                    if has_alt_sync:
                        df_d.to_parquet(os.path.join(ALT_SYNC_DAILY, f"{sym}_daily_data_raw.parquet"), index=False)
                    synced_tf.append("Daily")
                else:
                    errors.append("Daily: No data returned from NSE")
            except Exception as ed:
                errors.append(f"Daily error: {str(ed)[:60]}")
                
        # 3. Monthly Download
        if download_monthly:
            try:
                df_m = tv.get_hist(symbol=sym, exchange='NSE', interval=Interval.in_monthly, n_bars=600)
                if df_m is not None and not df_m.empty:
                    df_m = df_m.reset_index()
                    df_m.to_parquet(os.path.join(LOCAL_STOCK_MONTHLY_DIR, f"{sym}_monthly_data_raw.parquet"), index=False)
                    if has_alt_sync:
                        df_m.to_parquet(os.path.join(ALT_SYNC_MONTHLY, f"{sym}_monthly_data_raw.parquet"), index=False)
                    synced_tf.append("Monthly")
                else:
                    errors.append("Monthly: No data returned from NSE")
            except Exception as em:
                errors.append(f"Monthly error: {str(em)[:60]}")
                
        if errors:
            failed_list.append({
                'Symbol': sym,
                'Errors': "; ".join(errors),
                'Synced': ", ".join(synced_tf) if synced_tf else "None"
            })
        else:
            success_list.append({
                'Symbol': sym,
                'Timeframes': ", ".join(synced_tf),
                'Status': 'Synced Local & Cloud'
            })
            
        progress_bar.progress((idx + 1) / total)
        time.sleep(sleep_sec)
        
    status_text.markdown(f"🎉 **Batch Download Complete!** Updated **{len(success_list)}** stocks. ({len(failed_list)} with errors/skipped)")
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
    
    all_symbols = get_nifty500_symbols()
    
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
        pause_sec = st.number_input("Pause Between Stocks (sec):", min_value=0.1, max_value=3.0, value=0.4, step=0.1, key="sec4_pause")
        
    if dl_scope.startswith("All Nifty 500"):
        target_symbols = all_symbols
    elif dl_scope.startswith("Top 50"):
        target_symbols = all_symbols[:50]
    else:
        target_symbols = st.multiselect("Pick Specific Symbols:", all_symbols, default=["INFY", "RELIANCE", "TCS", "HDFCBANK"], key="sec4_custom_syms")
        
    st.info(f"📋 Target: **{len(target_symbols)}** stocks ({dl_tf}). Estimated runtime: ~{len(target_symbols) * pause_sec / 60:.1f} minutes.")
    
    col_btn, col_retry = st.columns([2.0, 2.0])
    with col_btn:
        btn_start = st.button("🚀 Start Automated Download & Sync", type="primary", key="btn_sec4_start")
    
    if btn_start:
        is_d = "Daily" in dl_tf or "All 3" in dl_tf
        is_w = "Weekly" in dl_tf or "All 3" in dl_tf
        is_m = "Monthly" in dl_tf or "All 3" in dl_tf
        run_stock_download(target_symbols, download_daily=is_d, download_weekly=is_w, download_monthly=is_m, sleep_sec=pause_sec)
        
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
                    run_stock_download(failed_syms_list, download_daily=is_d, download_weekly=is_w, download_monthly=is_m, sleep_sec=pause_sec)
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
def run_divergence_scanner(timeframe="Weekly", recency_mode="Recent (2026 - Present)", direction_filter="All"):
    """
    Scans all available stock parquets for divergence signals matching recency & direction filters.
    Supports Daily, Weekly, and Monthly timeframes.
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
            if len(df) < 50:
                continue
                
            _, sig_df = detect_pivots_and_signals(df, rsi_len=14, lb_left=5, lb_right=5, tol=0.002)
            if not sig_df.empty:
                # Filter by recency
                if recency_mode.startswith("🔥 Today (Signal on Latest Close)"):
                    # Strictly filter by signal formed on the latest closed candle (not confirmation day)
                    last_candle_day = df['date'].dt.date.iloc[-1]
                    sig_recent = sig_df[sig_df['signal_date'].dt.date == last_candle_day]
                elif recency_mode.startswith("⚡ Today (Alert Confirmed at p+lbR)"):
                    # Filter by confirmation day (candle p+lbR closed on the latest date, triggering the alert)
                    last_candle_day = df['date'].dt.date.iloc[-1]
                    sig_recent = sig_df[sig_df['confirm_date'].dt.date == last_candle_day]
                elif "Today" in recency_mode:
                    # Default Today filter: strictly by signal formed on latest close price as requested
                    last_candle_day = df['date'].dt.date.iloc[-1]
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
                    confirm_close_price = df['close'].iloc[c_idx] if c_idx < len(df) else r['price']
                    risk_pct = abs((confirm_close_price - sl_ref) / confirm_close_price) * 100.0 if confirm_close_price > 0 else 0.0
                    signals_found.append({
                        'Symbol': sym,
                        'Timeframe': timeframe,
                        'Signal': '🟢 BUY' if r['signal'] == 'BUY' else '🔴 SELL',
                        'raw_signal': r['signal'],
                        'Signal_Date': r['signal_date'].strftime('%Y-%m-%d'),
                        'Confirm_Date': r['confirm_date'].strftime('%Y-%m-%d'),
                        'Current / Entry Price': f"₹{confirm_close_price:,.2f}",
                        'Pivot Price (p)': f"₹{r['price']:,.2f}",
                        'raw_price': confirm_close_price,
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
    Instantly detects genuine **Bullish and Bearish RSI Divergences** based on TradingView-exact pivot parameters.
    """)
    
    sc1, sc2, sc3 = st.columns([1.5, 2.0, 2.0])
    with sc1:
        scan_tf = st.selectbox("Scan Timeframe:", ["Weekly", "Daily", "Monthly"], index=0, key="sec5_scan_tf")
    with sc2:
        scan_recency = st.selectbox(
            "Signal Recency Filter:",
            [
                "🔥 Today (Signal on Latest Close)",
                "⚡ Today (Alert Confirmed at p+lbR)",
                "Last 4 Weeks / 20 Days",
                "Recent (2026 - Present)",
                "Last 8 Weeks / 40 Days",
                "All Confirmed Signals"
            ],
            index=0,
            key="sec5_scan_rec"
        )
    with sc3:
        scan_dir = st.selectbox("Direction Filter:", ["All", "🟢 Bullish Only (BUY)", "🔴 Bearish Only (SELL)"], index=0, key="sec5_scan_dir")
        
    btn_scan = st.button("🚀 Run Divergence Scan Across Nifty 500", type="primary", key="btn_sec5_run")
    
    if btn_scan:
        with st.spinner(f"Scanning 500 stocks for {scan_tf} divergences..."):
            t_start = time.time()
            scan_results = run_divergence_scanner(scan_tf, scan_recency, scan_dir)
            t_elapsed = time.time() - t_start
            st.session_state['sec5_scan_results'] = scan_results
            st.session_state['sec5_scan_time'] = t_elapsed
            st.session_state['sec5_active_tf'] = scan_tf
            
    if 'sec5_scan_results' in st.session_state:
        res_df = st.session_state['sec5_scan_results']
        t_el = st.session_state.get('sec5_scan_time', 0.0)
        cur_tf = st.session_state.get('sec5_active_tf', scan_tf)
        
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
                st.markdown(f"""<div class='metric-card'><span class='metric-label'>Active Timeframe</span><div class='metric-value' style='color:#fbbf24;'>{cur_tf}</div><span style='font-size:11px;color:#94a3b8;'>{scan_recency}</span></div>""", unsafe_allow_html=True)
            
            display_cols = ['Symbol', 'Timeframe', 'Signal', 'Signal_Date', 'Confirm_Date', 'Current / Entry Price', 'Pivot Price (p)', 'RSI', 'Dynamic SL', 'SL Risk', 'Pattern']
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
