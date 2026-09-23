import sys
import os
from streamlit.runtime.scriptrunner import get_script_run_ctx

# Auto-launch via streamlit CLI if executed directly with "python <script>.py"
if get_script_run_ctx() is None:
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
    sys.exit(stcli.main())

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import importlib
import nifty_analytics
try:
    nifty_analytics = importlib.reload(nifty_analytics)
except Exception:
    pass
from nifty_analytics import NiftyAnalyticsEngine, SPOT_FILE

# Hot-reload safety for dynamically modified modules on Streamlit Cloud
try:
    import local_data_automation
    if not hasattr(local_data_automation, "render_nifty_downloader_section"):
        local_data_automation = importlib.reload(local_data_automation)
except Exception:
    pass

# =============================================================
# PAGE CONFIGURATION & SLEEK THEME
# =============================================================
st.set_page_config(
    page_title="Nifty & Stock Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Institutional Dark Theme CSS
st.markdown("""
<style>
    .main { background-color: #0b0f19; }
    .metric-card {
        background-color: #151c2c;
        border: 1px solid #232f48;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-value {
        font-size: 22px;
        font-weight: 700;
        color: #00d26a;
    }
    .metric-value-neg {
        font-size: 22px;
        font-weight: 700;
        color: #f8312f;
    }
    .metric-label {
        font-size: 13px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: bold;
    }
    .badge-primary { background: #1e3a8a; color: #93c5fd; }
    .badge-success { background: #064e3b; color: #6ee7b7; }
    
    /* Mobile & Tablet touch-friendly adjustments */
    @media (max-width: 768px) {
        div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 10px !important;
        }
    }
    div[data-testid="stToggle"] {
        cursor: pointer !important;
        touch-action: manipulation !important;
        user-select: none !important;
    }
    div[data-testid="stToggle"] label {
        cursor: pointer !important;
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================
# LIGHTWEIGHT IN-MEMORY DATA ENGINE
# =============================================================
import os

def _get_engine_cache_key():
    keys = []
    try:
        if os.path.exists(SPOT_FILE):
            keys.append(os.path.getmtime(SPOT_FILE))
    except Exception:
        pass
    try:
        if hasattr(nifty_analytics, '__file__') and os.path.exists(nifty_analytics.__file__):
            keys.append(os.path.getmtime(nifty_analytics.__file__))
    except Exception:
        pass
    return tuple(keys)

@st.cache_resource(show_spinner="Loading Nifty 1-minute spot data (< 0.5s)...")
def get_analytics_engine(cache_key):
    engine = NiftyAnalyticsEngine()
    engine.load_data()
    return engine

analytics_engine = get_analytics_engine(_get_engine_cache_key())

# Self-healing safeguard: ensure cached engine has expiry structures loaded
if not hasattr(analytics_engine, '_monthly_expiry_df') or analytics_engine._monthly_expiry_df is None:
    try:
        st.cache_resource.clear()
    except Exception:
        pass
    analytics_engine = NiftyAnalyticsEngine()
    analytics_engine.load_data()

# Dynamic Dataset Metrics
total_days = len(analytics_engine._daily_df)
total_weeks = len(analytics_engine._weekly_df)
total_months = len(analytics_engine._monthly_df)
min_date = analytics_engine._daily_df['date'].min()
max_date = analytics_engine._daily_df['date'].max()
min_year = min_date.year
max_year = max_date.year
min_date_str = min_date.strftime("%d-%b-%Y")
max_date_str = max_date.strftime("%d-%b-%Y")

# =============================================================
# SIDEBAR NAVIGATION & DATASET METRICS
# =============================================================
with st.sidebar:
    st.markdown("### 📈 Nifty Analytics")
    st.markdown("<span style='font-size: 12px; color: #94a3b8;'>Ultra-Fast Dedicated Move & Reversal Terminal</span>", unsafe_allow_html=True)
    st.markdown("---")
    
    nav_options = [
        "📊 Section 1: Range & Move Distribution",
        "🔄 Section 2: First-Touch Stretch & Reversals",
        "📥 Section 3: Nifty 500 Automated Data Downloader",
        "⚡ Section 4: Nifty Automated Data Downloader"
    ]
    
    default_nav_idx = 0
    if "nav_selection_override" in st.session_state:
        override_nav = st.session_state.pop("nav_selection_override")
        if override_nav in nav_options:
            default_nav_idx = nav_options.index(override_nav)
            
    nav_selection = st.radio(
        "Navigation:",
        nav_options,
        index=default_nav_idx
    )
    
    st.markdown("---")
    st.markdown("#### ⚡ Dataset Specs")
    st.markdown(f"""
    <div style="font-size: 12px; color: #94a3b8; line-height: 1.8;">
        • <strong>Asset:</strong> Nifty 50 Index (Spot)<br>
        • <strong>Granularity:</strong> 1-Minute Precision<br>
        • <strong>History:</strong> {min_year} – {max_year} ({min_date_str} to {max_date_str})<br>
        • <strong>Total Trading Days:</strong> {total_days:,} Days<br>
        • <strong>Calendar Weeks:</strong> {total_weeks:,} Weeks<br>
        • <strong>Calendar Months:</strong> {total_months:,} Months<br>
        • <strong>Options Data:</strong> None (Zero Lag)
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<span class='badge badge-success'>🟢 Instant RAM Cache Active</span>", unsafe_allow_html=True)

# =============================================================
# SECTION 1: RANGE & MOVE PERCENTAGE DISTRIBUTION
# =============================================================
if nav_selection == "📊 Section 1: Range & Move Distribution":
    st.header("📊 Section 1: How Far Does Nifty Move? (Daily, Weekly & Monthly Distribution)")
    st.markdown("Type or toggle any percentage threshold (e.g. **1.00%**). The terminal instantly computes how many **Days**, **Weeks**, and **Months** Nifty stayed within that range vs breached it.")
    
    # Preset Quick Buttons
    st.markdown("**Quick Range Presets:**")
    q_cols = st.columns(6)
    if 'stat_thresh_pct' not in st.session_state:
        st.session_state.stat_thresh_pct = 1.00
        
    def set_stat_thresh(val):
        st.session_state.stat_thresh_pct = float(val)
        
    presets_s1 = [0.50, 0.75, 1.00, 1.25, 1.50, 2.00]
    for i, p_val in enumerate(presets_s1):
        with q_cols[i]:
            st.button(f"{p_val:.2f}% Range", key=f"s1_p_{p_val}", on_click=set_stat_thresh, args=(p_val,), width="stretch")
            
    def _change_stat_thresh(delta):
        cur = float(st.session_state.get('stat_thresh_pct', 1.00))
        new_v = round(cur + delta, 2)
        st.session_state.stat_thresh_pct = max(0.10, min(10.00, new_v))

    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px; margin-bottom: 2px;">
        <span style="font-weight: 600; color: #f1f5f9; font-size: 13.5px;">🎯 Selected Move & Range Threshold (% from Open):</span>
        <span style="background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.4); color: #38bdf8; font-weight: 700; font-size: 14px; padding: 2px 10px; border-radius: 6px;">
            &plusmn;{st.session_state.stat_thresh_pct:.2f}%
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    c_dec, c_slider, c_inc = st.columns([0.6, 8.8, 0.6], vertical_alignment="center")
    with c_dec:
        st.button("➖", key="btn_dec_s1", on_click=_change_stat_thresh, args=(-0.10,), help="Decrease threshold by 0.10%", width="stretch")
    with c_slider:
        target_pct = st.slider(
            "Target Move Range Threshold (%):",
            min_value=0.10,
            max_value=10.00,
            step=0.05,
            key="stat_thresh_pct",
            label_visibility="collapsed",
            help="Select the benchmark percentage threshold for move and range evaluation"
        )
    with c_inc:
        st.button("➕", key="btn_inc_s1", on_click=_change_stat_thresh, args=(0.10,), help="Increase threshold by 0.10%", width="stretch")

    # Close Aggregation Mode Selector (TradingView Close vs Expiry Close)
    st.markdown("---")
    c_mode1, c_mode2 = st.columns([1.8, 3.2])
    with c_mode1:
        close_type = st.radio(
            "🎯 Candle Close Aggregation Mode:",
            [
                "📈 TradingView Close (Standard Calendar)",
                "⚡ Expiry Close (NSE F&O Schedule)"
            ],
            index=0,
            key="s1_close_type",
            help="• TradingView Close: Standard calendar aggregation (Monday open to Friday close for Weekly, 1st to Month-end for Monthly).\n• Expiry Close: F&O contract settlement cycles (Thursdays up to Aug 2025, Tuesdays from Sept 2025 onwards). Preceding trading day is used on holidays."
        )
    with c_mode2:
        is_expiry_mode = ("expiry" in str(close_type).lower())
        if is_expiry_mode:
            st.markdown("""
            <div style='background:rgba(56, 189, 248, 0.08); border-left: 3px solid #38bdf8; padding: 10px 14px; border-radius: 6px; font-size: 13px; color: #cbd5e1; margin-top: 4px;'>
                <strong>⚡ NSE F&O Expiry Settlement Rules Active:</strong><br>
                • <strong>Monthly Expiry:</strong> Last Thursday (up to Aug 31, 2025) ➔ Last Tuesday (from Sept 1, 2025 onwards).<br>
                • <strong>Weekly Expiry:</strong> Every Thursday (up to Aug 31, 2025) ➔ Every Tuesday (from Sept 1, 2025 onwards).<br>
                • <strong>Holiday Shift Rule:</strong> If expiry day is an exchange holiday, settlement shifts to the preceding trading day (e.g. Wednesday or Monday/Friday). Next cycle opens immediately on the next trading session.
            </div>
            """, unsafe_allow_html=True)
            weekly_scope_choice = st.radio(
                "🗓️ Weekly Expiry Contract Scope:",
                [
                    "🎯 Official F&O Weekly Era (Feb 2019 – Present: ~397 Weeks)",
                    "🕰️ Extended History (Synthetic Thursday 1996 – Present: ~1,603 Weeks)"
                ],
                index=0,
                key="s1_weekly_scope",
                horizontal=True,
                help="• Official F&O Weekly Era: Matches NSE regulatory history (weekly options officially launched on Feb 11, 2019: ~397 Weeks).\n• Extended History: Evaluates Thursday-to-Thursday weekly cycles back to 1996 for 30+ years historical backtesting (~1,603 Weeks)."
            )
            weekly_scope = "Official" if "official" in weekly_scope_choice.lower() else "Full"
        else:
            weekly_scope = "Official"
            st.caption("ℹ️ Standard TradingView weekly candles (Monday open to Friday close) and monthly candles (1st trading day to last trading day of the calendar month).")
            
    # Compute Daily, Weekly, and Monthly stats (with hot-reload fallback for Streamlit Cloud)
    try:
        stats_daily = analytics_engine.get_range_and_move_stats("Daily", target_pct, close_type=close_type, weekly_scope=weekly_scope)
        stats_weekly = analytics_engine.get_range_and_move_stats("Weekly", target_pct, close_type=close_type, weekly_scope=weekly_scope)
        stats_monthly = analytics_engine.get_range_and_move_stats("Monthly", target_pct, close_type=close_type, weekly_scope=weekly_scope)
    except TypeError:
        st.cache_resource.clear()
        import nifty_analytics
        nifty_analytics = importlib.reload(nifty_analytics)
        analytics_engine = nifty_analytics.NiftyAnalyticsEngine()
        analytics_engine.load_data()
        stats_daily = analytics_engine.get_range_and_move_stats("Daily", target_pct, close_type=close_type, weekly_scope=weekly_scope)
        stats_weekly = analytics_engine.get_range_and_move_stats("Weekly", target_pct, close_type=close_type, weekly_scope=weekly_scope)
        stats_monthly = analytics_engine.get_range_and_move_stats("Monthly", target_pct, close_type=close_type, weekly_scope=weekly_scope)
        
    mode_tag = " [Expiry Close]" if is_expiry_mode else " [TradingView Close]"
    st.markdown(f"### 🌐 Multi-Timeframe Move Comparison (Threshold: &plusmn;{target_pct:.2f}%){mode_tag}")
    
    # 3 Side-by-Side Timeframe Columns
    col_d, col_w, col_m = st.columns(3)
    
    def _build_metric_card(title, title_color, stats, target_pct, unit_name):
        return f"""<div class="metric-card">
<div class="metric-label" style="font-size: 14.5px; color: {title_color}; font-weight: 700;">{title}</div>
<div style="height: 1px; background: #232f48; margin: 6px 0;"></div>
<div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 2px;">
<span style="font-size: 12.5px; color: #94a3b8;">Close Move within &plusmn;{target_pct:.2f}%:</span>
<span style="font-size: 15px; font-weight: 700; color: {title_color};">{stats['pct_within_move']:.1f}% <span style="font-size: 11.5px; color: #94a3b8; font-weight: normal;">({stats['cnt_within_move']:,})</span></span>
</div>
<div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 4px;">
<span style="font-size: 12.5px; color: #94a3b8;">Total High-Low Range &le; {target_pct:.2f}%:</span>
<span style="font-size: 15px; font-weight: 700; color: #00d26a;">{stats['pct_within_range']:.1f}% <span style="font-size: 11.5px; color: #94a3b8; font-weight: normal;">({stats['cnt_within_range']:,})</span></span>
</div>
<div style="height: 1px; background: #1e293b; margin: 8px 0;"></div>
<div style="font-size: 12.5px; font-weight: 700; color: #00d26a; margin-bottom: 2px;">🟢 Upside Stretch & Close (+{target_pct:.2f}%):</div>
<div style="font-size: 12px; color: #cbd5e1; line-height: 1.6; padding-left: 4px;">
• High Touched/Crossed &ge; +{target_pct:.2f}%: <strong style="color: #00d26a;">{stats['pct_high_breach_pos']:.1f}%</strong> ({stats['cnt_high_breach_pos']:,} {unit_name})<br>
• Closed ABOVE +{target_pct:.2f}% (Breakout): <strong style="color: #38bdf8;">{stats['pct_above_pos']:.1f}%</strong> ({stats['cnt_above_pos']:,} {unit_name})<br>
• Closed BELOW +{target_pct:.2f}%: <strong style="color: #f1f5f9;">{stats['pct_close_below_pos']:.1f}%</strong> ({stats['cnt_close_below_pos']:,} {unit_name})
</div>
<div style="height: 1px; background: #1e293b; margin: 8px 0;"></div>
<div style="font-size: 12.5px; font-weight: 700; color: #f87171; margin-bottom: 2px;">🔴 Downside Drop & Close (-{target_pct:.2f}%):</div>
<div style="font-size: 12px; color: #cbd5e1; line-height: 1.6; padding-left: 4px;">
• Low Touched/Crossed &le; -{target_pct:.2f}%: <strong style="color: #f87171;">{stats['pct_low_breach_neg']:.1f}%</strong> ({stats['cnt_low_breach_neg']:,} {unit_name})<br>
• Closed BELOW -{target_pct:.2f}% (Breakdown): <strong style="color: #f8312f;">{stats['pct_below_neg']:.1f}%</strong> ({stats['cnt_below_neg']:,} {unit_name})<br>
• Closed ABOVE -{target_pct:.2f}%: <strong style="color: #f1f5f9;">{stats['pct_close_above_neg']:.1f}%</strong> ({stats['cnt_close_above_neg']:,} {unit_name})
</div>
<div style="height: 1px; background: #1e293b; margin: 8px 0;"></div>
<div style="font-size: 12px; color: #94a3b8;">
Average Range: <strong style="color: #f8fafc;">{stats['avg_range_pct']:.2f}%</strong> (Median: {stats['median_range_pct']:.2f}%)
</div>
</div>"""

    with col_d:
        st.markdown(_build_metric_card(
            f"📅 DAILY ({stats_daily['total_periods']:,} Days)",
            "#38bdf8",
            stats_daily,
            target_pct,
            "days"
        ), unsafe_allow_html=True)
        
    with col_w:
        w_title = f"📆 WEEKLY - {stats_weekly['period_label']} ({stats_weekly['total_periods']:,})" if is_expiry_mode else f"📆 WEEKLY ({stats_weekly['total_periods']:,} Weeks)"
        st.markdown(_build_metric_card(
            w_title,
            "#a78bfa",
            stats_weekly,
            target_pct,
            "periods"
        ), unsafe_allow_html=True)
        
    with col_m:
        m_title = f"🗓️ MONTHLY - Expiry ({stats_monthly['total_periods']:,} Months)" if is_expiry_mode else f"🗓️ MONTHLY ({stats_monthly['total_periods']:,} Months)"
        st.markdown(_build_metric_card(
            m_title,
            "#f43f5e",
            stats_monthly,
            target_pct,
            "months"
        ), unsafe_allow_html=True)
        
    # Unified Comparative Summary Table
    tf_w_name = f"Weekly ({stats_weekly['period_label']})" if is_expiry_mode else "Weekly"
    tf_m_name = f"Monthly ({stats_monthly['period_label']})" if is_expiry_mode else "Monthly"
    comp_df = pd.DataFrame([
        {
            "Timeframe": "Daily",
            "Total Periods": stats_daily['total_periods'],
            f"High Touched ≥ +{target_pct:.2f}%": f"{stats_daily['pct_high_breach_pos']:.1f}% ({stats_daily['cnt_high_breach_pos']:,})",
            f"Close Above +{target_pct:.2f}% (Bullish)": f"{stats_daily['pct_above_pos']:.1f}% ({stats_daily['cnt_above_pos']:,})",
            f"Close Below +{target_pct:.2f}%": f"{stats_daily['pct_close_below_pos']:.1f}% ({stats_daily['cnt_close_below_pos']:,})",
            f"Low Touched ≤ -{target_pct:.2f}%": f"{stats_daily['pct_low_breach_neg']:.1f}% ({stats_daily['cnt_low_breach_neg']:,})",
            f"Close Below -{target_pct:.2f}% (Bearish)": f"{stats_daily['pct_below_neg']:.1f}% ({stats_daily['cnt_below_neg']:,})",
            f"Close Above -{target_pct:.2f}%": f"{stats_daily['pct_close_above_neg']:.1f}% ({stats_daily['cnt_close_above_neg']:,})",
            f"Within ±{target_pct:.2f}% Move": f"{stats_daily['pct_within_move']:.1f}% ({stats_daily['cnt_within_move']:,})",
            f"Range ≤ {target_pct:.2f}%": f"{stats_daily['pct_within_range']:.1f}% ({stats_daily['cnt_within_range']:,})",
            "Avg Range (%)": f"{stats_daily['avg_range_pct']:.2f}%",
            "Median Range (%)": f"{stats_daily['median_range_pct']:.2f}%"
        },
        {
            "Timeframe": tf_w_name,
            "Total Periods": stats_weekly['total_periods'],
            f"High Touched ≥ +{target_pct:.2f}%": f"{stats_weekly['pct_high_breach_pos']:.1f}% ({stats_weekly['cnt_high_breach_pos']:,})",
            f"Close Above +{target_pct:.2f}% (Bullish)": f"{stats_weekly['pct_above_pos']:.1f}% ({stats_weekly['cnt_above_pos']:,})",
            f"Close Below +{target_pct:.2f}%": f"{stats_weekly['pct_close_below_pos']:.1f}% ({stats_weekly['cnt_close_below_pos']:,})",
            f"Low Touched ≤ -{target_pct:.2f}%": f"{stats_weekly['pct_low_breach_neg']:.1f}% ({stats_weekly['cnt_low_breach_neg']:,})",
            f"Close Below -{target_pct:.2f}% (Bearish)": f"{stats_weekly['pct_below_neg']:.1f}% ({stats_weekly['cnt_below_neg']:,})",
            f"Close Above -{target_pct:.2f}%": f"{stats_weekly['pct_close_above_neg']:.1f}% ({stats_weekly['cnt_close_above_neg']:,})",
            f"Within ±{target_pct:.2f}% Move": f"{stats_weekly['pct_within_move']:.1f}% ({stats_weekly['cnt_within_move']:,})",
            f"Range ≤ {target_pct:.2f}%": f"{stats_weekly['pct_within_range']:.1f}% ({stats_weekly['cnt_within_range']:,})",
            "Avg Range (%)": f"{stats_weekly['avg_range_pct']:.2f}%",
            "Median Range (%)": f"{stats_weekly['median_range_pct']:.2f}%"
        },
        {
            "Timeframe": tf_m_name,
            "Total Periods": stats_monthly['total_periods'],
            f"High Touched ≥ +{target_pct:.2f}%": f"{stats_monthly['pct_high_breach_pos']:.1f}% ({stats_monthly['cnt_high_breach_pos']:,})",
            f"Close Above +{target_pct:.2f}% (Bullish)": f"{stats_monthly['pct_above_pos']:.1f}% ({stats_monthly['cnt_above_pos']:,})",
            f"Close Below +{target_pct:.2f}%": f"{stats_monthly['pct_close_below_pos']:.1f}% ({stats_monthly['cnt_close_below_pos']:,})",
            f"Low Touched ≤ -{target_pct:.2f}%": f"{stats_monthly['pct_low_breach_neg']:.1f}% ({stats_monthly['cnt_low_breach_neg']:,})",
            f"Close Below -{target_pct:.2f}% (Bearish)": f"{stats_monthly['pct_below_neg']:.1f}% ({stats_monthly['cnt_below_neg']:,})",
            f"Close Above -{target_pct:.2f}%": f"{stats_monthly['pct_close_above_neg']:.1f}% ({stats_monthly['cnt_close_above_neg']:,})",
            f"Within ±{target_pct:.2f}% Move": f"{stats_monthly['pct_within_move']:.1f}% ({stats_monthly['cnt_within_move']:,})",
            f"Range ≤ {target_pct:.2f}%": f"{stats_monthly['pct_within_range']:.1f}% ({stats_monthly['cnt_within_range']:,})",
            "Avg Range (%)": f"{stats_monthly['avg_range_pct']:.2f}%",
            "Median Range (%)": f"{stats_monthly['median_range_pct']:.2f}%"
        }
    ])
    st.dataframe(comp_df, hide_index=True, width="stretch")
    
    # Drill-down Tabs for Distribution Charts
    st.markdown("---")
    st.markdown("#### 📊 Timeframe Distribution Drill-down")
    d_tab1, d_tab2, d_tab3 = st.tabs(["📅 Daily Analysis", "📆 Weekly Analysis", "🗓️ Monthly Analysis"])
    
    with d_tab1:
        df_d = stats_daily['df']
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Histogram(
            x=df_d['move_pct'], nbinsx=80, marker_color='#38bdf8', opacity=0.75, name='Daily Move %'
        ))
        fig_dd.add_vline(x=-target_pct, line_dash="dash", line_color="#ef4444", line_width=2,
                         annotation_text=f"-{target_pct:.2f}% Threshold", annotation_position="top left")
        fig_dd.add_vline(x=target_pct, line_dash="dash", line_color="#00d26a", line_width=2,
                         annotation_text=f"+{target_pct:.2f}% Threshold", annotation_position="top right")
        fig_dd.update_layout(title=f"Daily Open-to-Close Move Distribution ({stats_daily['total_periods']:,} Days)",
                             xaxis_title="Move %", yaxis_title="Number of Days", template="plotly_dark", height=380)
        st.plotly_chart(fig_dd, width="stretch")
        
    with d_tab2:
        df_w = stats_weekly['df']
        w_chart_label = f"Weekly ({stats_weekly['period_label']})" if is_expiry_mode else "Weekly"
        fig_ww = go.Figure()
        fig_ww.add_trace(go.Histogram(
            x=df_w['move_pct'], nbinsx=60, marker_color='#a78bfa', opacity=0.75, name=f'{w_chart_label} Move %'
        ))
        fig_ww.add_vline(x=-target_pct, line_dash="dash", line_color="#ef4444", line_width=2,
                         annotation_text=f"-{target_pct:.2f}% Threshold", annotation_position="top left")
        fig_ww.add_vline(x=target_pct, line_dash="dash", line_color="#00d26a", line_width=2,
                         annotation_text=f"+{target_pct:.2f}% Threshold", annotation_position="top right")
        fig_ww.update_layout(title=f"{w_chart_label} Open-to-Close Move Distribution ({stats_weekly['total_periods']:,} Periods)",
                             xaxis_title="Move %", yaxis_title="Number of Periods", template="plotly_dark", height=380)
        st.plotly_chart(fig_ww, width="stretch")
        
    with d_tab3:
        df_m = stats_monthly['df']
        m_chart_label = f"Monthly ({stats_monthly['period_label']})" if is_expiry_mode else "Monthly"
        fig_mm = go.Figure()
        fig_mm.add_trace(go.Histogram(
            x=df_m['move_pct'], nbinsx=40, marker_color='#f43f5e', opacity=0.75, name=f'{m_chart_label} Move %'
        ))
        fig_mm.add_vline(x=-target_pct, line_dash="dash", line_color="#ef4444", line_width=2,
                         annotation_text=f"-{target_pct:.2f}% Threshold", annotation_position="top left")
        fig_mm.add_vline(x=target_pct, line_dash="dash", line_color="#00d26a", line_width=2,
                         annotation_text=f"+{target_pct:.2f}% Threshold", annotation_position="top right")
        fig_mm.update_layout(title=f"{m_chart_label} Open-to-Close Move Distribution ({stats_monthly['total_periods']:,} Periods)",
                             xaxis_title="Move %", yaxis_title="Number of Periods", template="plotly_dark", height=380)
        st.plotly_chart(fig_mm, width="stretch")

    # Itemized Expiry Cycles & Holiday Settlements Audit Ledger
    if is_expiry_mode:
        st.markdown("---")
        with st.expander("🔍 Inspect Expiry Cycles & Holiday Settlements Audit Ledger", expanded=True):
            exp_tab1, exp_tab2 = st.tabs(["📆 Weekly Expiry Cycles", "🗓️ Monthly Expiry Cycles"])
            with exp_tab1:
                df_w_exp = stats_weekly['df'].copy()
                st.markdown(f"**Itemized Weekly Expiry Cycles ({len(df_w_exp)} Contracts):**")
                show_w = pd.DataFrame({
                    'Cycle Start': df_w_exp['cycle_start'].dt.strftime('%Y-%m-%d') + ' (' + df_w_exp['start_day_name'] + ')',
                    'Expiry Date': df_w_exp['date'].dt.strftime('%Y-%m-%d') + ' (' + df_w_exp['expiry_day_name'] + ')',
                    'Scheduled Expiry': df_w_exp['sched_date'].dt.strftime('%Y-%m-%d'),
                    'Settlement Status': df_w_exp['is_shifted'].apply(lambda x: '⚠️ Holiday Shift (Preceding Trading Day)' if x else '✅ Normal Settlement'),
                    'Trading Days': df_w_exp['trading_days'],
                    'Cycle Open': df_w_exp['open'].apply(lambda x: f"₹{x:,.2f}"),
                    'Cycle High': df_w_exp['high'].apply(lambda x: f"₹{x:,.2f}"),
                    'Cycle Low': df_w_exp['low'].apply(lambda x: f"₹{x:,.2f}"),
                    'Expiry Close': df_w_exp['close'].apply(lambda x: f"₹{x:,.2f}"),
                    'Range %': df_w_exp['range_pct'].apply(lambda x: f"{x:.2f}%"),
                    'Move %': df_w_exp['move_pct'].apply(lambda x: f"{x:+.2f}%")
                })
                st.dataframe(show_w, hide_index=True, width="stretch")
            with exp_tab2:
                df_m_exp = stats_monthly['df'].copy()
                st.markdown(f"**Itemized Monthly Expiry Cycles ({len(df_m_exp)} Contracts):**")
                show_m = pd.DataFrame({
                    'Cycle Start': df_m_exp['cycle_start'].dt.strftime('%Y-%m-%d') + ' (' + df_m_exp['start_day_name'] + ')',
                    'Expiry Date': df_m_exp['date'].dt.strftime('%Y-%m-%d') + ' (' + df_m_exp['expiry_day_name'] + ')',
                    'Scheduled Expiry': df_m_exp['sched_date'].dt.strftime('%Y-%m-%d'),
                    'Settlement Status': df_m_exp['is_shifted'].apply(lambda x: '⚠️ Holiday Shift (Preceding Trading Day)' if x else '✅ Normal Settlement'),
                    'Trading Days': df_m_exp['trading_days'],
                    'Cycle Open': df_m_exp['open'].apply(lambda x: f"₹{x:,.2f}"),
                    'Cycle High': df_m_exp['high'].apply(lambda x: f"₹{x:,.2f}"),
                    'Cycle Low': df_m_exp['low'].apply(lambda x: f"₹{x:,.2f}"),
                    'Expiry Close': df_m_exp['close'].apply(lambda x: f"₹{x:,.2f}"),
                    'Range %': df_m_exp['range_pct'].apply(lambda x: f"{x:.2f}%"),
                    'Move %': df_m_exp['move_pct'].apply(lambda x: f"{x:+.2f}%")
                })
                st.dataframe(show_m, hide_index=True, width="stretch")

# =============================================================
# SECTION 2: FIRST-TOUCH STRETCH, FORWARD EXTENSIONS & REVERSALS
# =============================================================
elif nav_selection == "🔄 Section 2: First-Touch Stretch & Reversals":
    st.header("🔄 Section 2: First-Touch Stretch, Forward Extensions & Mean-Reversion")
    top_banner_html = """<div style="background-color: #151c2c; border-left: 4px solid #38bdf8; padding: 12px 16px; border-radius: 6px; margin-bottom: 16px;">
<strong style="color: #38bdf8; font-size: 14px;">🎯 Real-Time Execution Point Analysis (Zero Hindsight Bias across Daily, Weekly & Monthly):</strong><br>
<span style="color: #cbd5e1; font-size: 13px;">
When Nifty reaches a stretch level from the period's opening price for the <strong>FIRST TIME</strong>, a trader in live market cannot know if that is the ultimate extreme or if it will run much further.<br>
This engine anchors to that <strong>exact first-touch minute</strong> and evaluates the forward progression until the period close:
<strong>1. Forward Extension / Breakout</strong> (Did it continue running further?) •
<strong>2. Retracement / Pullback</strong> (Did it give back 50% of the stretch or return to Period Open?) •
<strong>3. Final Settlement</strong> (Where did it close: Trend Continuation, Consolidation, or Full Reversal?)
</span>
</div>"""
    st.markdown(top_banner_html, unsafe_allow_html=True)
    
    # Close Aggregation Mode Selector (TradingView Close vs Expiry Close)
    c_mode1, c_mode2 = st.columns([1.8, 3.2])
    with c_mode1:
        sec2_close_type = st.radio(
            "🎯 Candle Close Aggregation Mode:",
            [
                "📈 TradingView Close (Standard Calendar)",
                "⚡ Expiry Close (NSE F&O Schedule)"
            ],
            index=0,
            key="s2_close_type",
            help="• TradingView Close: Standard calendar aggregation (Monday open to Friday close for Weekly, 1st to Month-end for Monthly).\n• Expiry Close: F&O contract settlement cycles (Thursdays up to Aug 2025, Tuesdays from Sept 2025 onwards). Preceding trading day is used on holidays."
        )
    with c_mode2:
        is_expiry_mode = ("expiry" in str(sec2_close_type).lower())
        if is_expiry_mode:
            st.markdown("""
            <div style='background:rgba(56, 189, 248, 0.08); border-left: 3px solid #38bdf8; padding: 10px 14px; border-radius: 6px; font-size: 13px; color: #cbd5e1; margin-top: 4px;'>
                <strong>⚡ NSE F&O Expiry Settlement Rules Active:</strong><br>
                • <strong>Monthly Expiry:</strong> Last Thursday (up to Aug 31, 2025) ➔ Last Tuesday (from Sept 1, 2025 onwards).<br>
                • <strong>Weekly Expiry:</strong> Every Thursday (up to Aug 31, 2025) ➔ Every Tuesday (from Sept 1, 2025 onwards).<br>
                • <strong>Holiday Shift Rule:</strong> If expiry day is an exchange holiday, settlement shifts to the preceding trading day (e.g. Wednesday or Monday/Friday). Next cycle opens immediately on the next trading session.
            </div>
            """, unsafe_allow_html=True)
            sec2_weekly_scope_choice = st.radio(
                "🗓️ Weekly Expiry Contract Scope:",
                [
                    "🎯 Official F&O Weekly Era (Feb 2019 – Present: ~397 Weeks)",
                    "🕰️ Extended History (Synthetic Thursday 1996 – Present: ~1,603 Weeks)"
                ],
                index=0,
                key="s2_weekly_scope",
                horizontal=True,
                help="• Official F&O Weekly Era: Matches NSE regulatory history (weekly options officially launched on Feb 11, 2019: ~397 Weeks).\n• Extended History: Evaluates Thursday-to-Thursday weekly cycles back to 1996 for 30+ years historical backtesting (~1,603 Weeks)."
            )
            sec2_weekly_scope = "Official" if "official" in sec2_weekly_scope_choice.lower() else "Full"
        else:
            sec2_weekly_scope = "Official"
            st.caption("ℹ️ Standard TradingView weekly candles (Monday open to Friday close) and monthly candles (1st trading day to last trading day of the calendar month).")
            
    # Multi-timeframe Quick Matrix Expander
    with st.expander("🌐 View Multi-Timeframe Stretch Summary Matrix (Daily vs Weekly vs Monthly)", expanded=False):
        try:
            c_d = analytics_engine.get_stretch_reversal_stats("Daily", 0.50, 50.0, close_type=sec2_close_type, weekly_scope=sec2_weekly_scope)
            c_w = analytics_engine.get_stretch_reversal_stats("Weekly", 1.50, 50.0, close_type=sec2_close_type, weekly_scope=sec2_weekly_scope)
            c_m = analytics_engine.get_stretch_reversal_stats("Monthly", 3.00, 50.0, close_type=sec2_close_type, weekly_scope=sec2_weekly_scope)
        except TypeError:
            st.cache_resource.clear()
            import nifty_analytics
            nifty_analytics = importlib.reload(nifty_analytics)
            analytics_engine = nifty_analytics.NiftyAnalyticsEngine()
            analytics_engine.load_data()
            c_d = analytics_engine.get_stretch_reversal_stats("Daily", 0.50, 50.0, close_type=sec2_close_type, weekly_scope=sec2_weekly_scope)
            c_w = analytics_engine.get_stretch_reversal_stats("Weekly", 1.50, 50.0, close_type=sec2_close_type, weekly_scope=sec2_weekly_scope)
            c_m = analytics_engine.get_stretch_reversal_stats("Monthly", 3.00, 50.0, close_type=sec2_close_type, weekly_scope=sec2_weekly_scope)
        
        w_tf_label = "📆 Weekly (Expiry Cycle)" if is_expiry_mode else "📆 Weekly"
        m_tf_label = "🗓️ Monthly (Expiry Cycle)" if is_expiry_mode else "🗓️ Monthly"
        
        summary_comp_df = pd.DataFrame([
            {
                "Timeframe": "📅 Daily",
                "Total Periods": f"{c_d['total_periods']:,} Days",
                "Benchmark Stretch": "±0.50%",
                "Bullish Triggered (%)": f"{c_d['pos']['pct_of_all_periods']:.1f}% ({c_d['pos']['count']:,})",
                "Avg Peak High": f"+{c_d['pos']['avg_peak_high']:.2f}%",
                "Retraced 50% of Stretch (%)": f"{c_d['pos']['retrace_goal_pct']:.1f}%",
                "Returned to Open (0%)": f"{c_d['pos']['back_to_open_pct']:.1f}%",
                "Trend Close (%)": f"{c_d['pos']['closed_trend_pct']:.1f}%",
                "Failed Reversal / Red (%)": f"{c_d['pos']['closed_red_pct']:.1f}%"
            },
            {
                "Timeframe": w_tf_label,
                "Total Periods": f"{c_w['total_periods']:,} {c_w['unit_plural']}",
                "Benchmark Stretch": "±1.50%",
                "Bullish Triggered (%)": f"{c_w['pos']['pct_of_all_periods']:.1f}% ({c_w['pos']['count']:,})",
                "Avg Peak High": f"+{c_w['pos']['avg_peak_high']:.2f}%",
                "Retraced 50% of Stretch (%)": f"{c_w['pos']['retrace_goal_pct']:.1f}%",
                "Returned to Open (0%)": f"{c_w['pos']['back_to_open_pct']:.1f}%",
                "Trend Close (%)": f"{c_w['pos']['closed_trend_pct']:.1f}%",
                "Failed Reversal / Red (%)": f"{c_w['pos']['closed_red_pct']:.1f}%"
            },
            {
                "Timeframe": m_tf_label,
                "Total Periods": f"{c_m['total_periods']:,} {c_m['unit_plural']}",
                "Benchmark Stretch": "±3.00%",
                "Bullish Triggered (%)": f"{c_m['pos']['pct_of_all_periods']:.1f}% ({c_m['pos']['count']:,})",
                "Avg Peak High": f"+{c_m['pos']['avg_peak_high']:.2f}%",
                "Retraced 50% of Stretch (%)": f"{c_m['pos']['retrace_goal_pct']:.1f}%",
                "Returned to Open (0%)": f"{c_m['pos']['back_to_open_pct']:.1f}%",
                "Trend Close (%)": f"{c_m['pos']['closed_trend_pct']:.1f}%",
                "Failed Reversal / Red (%)": f"{c_m['pos']['closed_red_pct']:.1f}%"
            }
        ])
        st.dataframe(summary_comp_df, hide_index=True, width="stretch")
    
    # Timeframe Selector
    tf_radio_options = [
        "📅 Daily (Intraday Stretch)",
        "📆 Weekly (Expiry Cycle Stretch)" if is_expiry_mode else "📆 Weekly (Intraweek Stretch)",
        "🗓️ Monthly (Expiry Cycle Stretch)" if is_expiry_mode else "🗓️ Monthly (Intramonth Stretch)"
    ]
    sec2_tf_choice = st.radio(
        "Select Stretch Analysis Timeframe:",
        tf_radio_options,
        horizontal=True
    )
    
    if "Daily" in sec2_tf_choice:
        selected_tf = "Daily"
    elif "Weekly" in sec2_tf_choice:
        selected_tf = "Weekly"
    else:
        selected_tf = "Monthly"
    
    # Timeframe-specific configurations
    if selected_tf == "Daily":
        presets = [0.50, 0.75, 1.00, 1.25, 1.50]
        default_stretch = 0.50
        s_min, s_max, s_step = 0.20, 2.50, 0.05
        ext_steps = [0.25, 0.50, 0.75, 1.00]
        tf_desc = f"2,896 trading sessions (2015 – 2026, 1-Minute Precision), measured from Day Open (09:15) to 15:30 Close"
        tf_period_name = "days"
        tf_close_label = "Day Close at 15:30"
        rev_label_pos = "Closed RED (< 0.0%)"
        rev_label_neg = "Closed GREEN (> 0.0%)"
        selected_tf_display = "Daily"
    elif selected_tf == "Weekly":
        presets = [1.00, 1.50, 2.00, 2.50, 3.00]
        default_stretch = 1.50
        s_min, s_max, s_step = 0.50, 5.00, 0.10
        ext_steps = [0.50, 1.00, 1.50, 2.00]
        if is_expiry_mode:
            tf_period_name = "expiry weeks"
            total_exp_w = len(analytics_engine._weekly_expiry_full_df) if sec2_weekly_scope == "Full" else len(analytics_engine._weekly_expiry_official_df)
            scope_desc = "1996 – Present" if sec2_weekly_scope == "Full" else "Feb 2019 – Present"
            tf_desc = f"{total_exp_w:,} F&O weekly expiry cycles ({scope_desc}), measured from Post-Expiry Open (09:15) to Expiry 15:30 Close"
            tf_close_label = "Thursday/Tuesday Expiry Close at 15:30"
            rev_label_pos = "Closed RED Expiry Week (< 0.0%)"
            rev_label_neg = "Closed GREEN Expiry Week (> 0.0%)"
            selected_tf_display = "Expiry Week"
        else:
            tf_desc = f"{total_weeks:,} calendar weeks ({min_year} – {max_year}), measured from Monday Open to Friday 15:30 Close"
            tf_period_name = "weeks"
            tf_close_label = "Friday Close at 15:30"
            rev_label_pos = "Closed RED Week (< 0.0%)"
            rev_label_neg = "Closed GREEN Week (> 0.0%)"
            selected_tf_display = "Weekly"
    else: # Monthly
        presets = [2.00, 3.00, 4.00, 5.00, 6.00]
        default_stretch = 3.00
        s_min, s_max, s_step = 1.00, 10.00, 0.25
        ext_steps = [1.00, 2.00, 3.00, 4.00]
        if is_expiry_mode:
            tf_period_name = "expiry months"
            total_exp_m = len(analytics_engine._monthly_expiry_df)
            tf_desc = f"{total_exp_m:,} F&O monthly expiry cycles (1996 – Present), measured from Post-Expiry Open (09:15) to Monthly Expiry 15:30 Close"
            tf_close_label = "Last Thursday/Tuesday Expiry Close at 15:30"
            rev_label_pos = "Closed RED Expiry Month (< 0.0%)"
            rev_label_neg = "Closed GREEN Expiry Month (> 0.0%)"
            selected_tf_display = "Expiry Month"
        else:
            tf_desc = f"{total_months:,} calendar months ({min_year} – {max_year}), measured from Month Open (1st trading minute) to Month-End Close"
            tf_period_name = "months"
            tf_close_label = "Month-End Close at 15:30"
            rev_label_pos = "Closed RED Month (< 0.0%)"
            rev_label_neg = "Closed GREEN Month (> 0.0%)"
            selected_tf_display = "Monthly"
        
    session_key = f'sec2_thresh_{selected_tf}'
    if session_key not in st.session_state:
        st.session_state[session_key] = default_stretch
        
    retrace_key = 'sec2_retrace_goal'
    if retrace_key not in st.session_state:
        st.session_state[retrace_key] = 50.0

    def _change_thresh(delta, s_k, min_v, max_v, step_v):
        cur = float(st.session_state.get(s_k, default_stretch))
        new_v = round(cur + delta, 2)
        st.session_state[s_k] = max(min_v, min(max_v, new_v))

    def _set_preset_thresh(val, s_k):
        st.session_state[s_k] = float(val)

    def _change_retrace(delta, r_k, min_v, max_v, step_v):
        cur = float(st.session_state.get(r_k, 50.0))
        new_v = round(cur + delta, 1)
        st.session_state[r_k] = max(min_v, min(max_v, new_v))

    # Preset Buttons
    st.markdown(f"**Quick {selected_tf_display} Stretch Presets:**")
    s_cols = st.columns(5)
    for i, p_val in enumerate(presets):
        with s_cols[i]:
            st.button(f"{p_val:.2f}% Stretch", key=f"btn_{selected_tf}_{p_val}",
                      on_click=_set_preset_thresh, args=(p_val, session_key), width="stretch")
                
    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
            <span style="font-weight: 600; color: #f1f5f9; font-size: 13.5px;">{selected_tf_display} Stretch Threshold (% from Period Open):</span>
            <span style="background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.4); color: #38bdf8; font-weight: 700; font-size: 13.5px; padding: 2px 8px; border-radius: 6px;">
                &plusmn;{st.session_state[session_key]:.2f}%
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        c_dec, c_slider, c_inc = st.columns([1.1, 7.8, 1.1], vertical_alignment="center")
        with c_dec:
            st.button("➖", key=f"btn_dec_{selected_tf}", on_click=_change_thresh, args=(-s_step, session_key, s_min, s_max, s_step),
                      help=f"Decrease {selected_tf_display.lower()} threshold by {s_step:.2f}%", width="stretch")
        with c_slider:
            stretch_thresh = st.slider(
                f"{selected_tf_display} Stretch Threshold",
                min_value=s_min, max_value=s_max,
                step=s_step,
                key=session_key,
                label_visibility="collapsed",
                help=f"Trigger level from {selected_tf_display.lower()} open"
            )
        with c_inc:
            st.button("➕", key=f"btn_inc_{selected_tf}", on_click=_change_thresh, args=(s_step, session_key, s_min, s_max, s_step),
                      help=f"Increase {selected_tf_display.lower()} threshold by {s_step:.2f}%", width="stretch")
                      
    with rc2:
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
            <span style="font-weight: 600; color: #f1f5f9; font-size: 13.5px;">Retracement Goal (% of initial stretch):</span>
            <span style="background: rgba(168, 85, 247, 0.15); border: 1px solid rgba(168, 85, 247, 0.4); color: #c084fc; font-weight: 700; font-size: 13.5px; padding: 2px 8px; border-radius: 6px;">
                {st.session_state[retrace_key]:.0f}%
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        cr_dec, cr_slider, cr_inc = st.columns([1.1, 7.8, 1.1], vertical_alignment="center")
        with cr_dec:
            st.button("➖", key="btn_dec_retrace", on_click=_change_retrace, args=(-5.0, retrace_key, 20.0, 100.0, 5.0),
                      help="Decrease retracement goal by 5%", width="stretch")
        with cr_slider:
            retrace_goal = st.slider(
                "Retracement Goal (% of initial stretch)",
                min_value=20.0, max_value=100.0, step=5.0,
                key=retrace_key,
                label_visibility="collapsed",
                help="e.g. 50% means the market gives back half of the initial stretch move"
            )
        with cr_inc:
            st.button("➕", key="btn_inc_retrace", on_click=_change_retrace, args=(5.0, retrace_key, 20.0, 100.0, 5.0),
                      help="Increase retracement goal by 5%", width="stretch")
                                 
    try:
        rev_stats = analytics_engine.get_stretch_reversal_stats(
            timeframe=selected_tf,
            stretch_threshold=stretch_thresh,
            retrace_pct=retrace_goal,
            ext_steps=ext_steps,
            close_type=sec2_close_type,
            weekly_scope=sec2_weekly_scope
        )
    except TypeError:
        st.cache_resource.clear()
        import nifty_analytics
        nifty_analytics = importlib.reload(nifty_analytics)
        analytics_engine = nifty_analytics.NiftyAnalyticsEngine()
        analytics_engine.load_data()
        rev_stats = analytics_engine.get_stretch_reversal_stats(
            timeframe=selected_tf,
            stretch_threshold=stretch_thresh,
            retrace_pct=retrace_goal,
            ext_steps=ext_steps,
            close_type=sec2_close_type,
            weekly_scope=sec2_weekly_scope
        )
        
    pos = rev_stats['pos']
    neg = rev_stats['neg']
    total_p = rev_stats['total_periods']
    pos_goal = rev_stats['pos_goal_level']
    neg_goal = rev_stats['neg_goal_level']
    e_s = rev_stats['ext_steps']
    
    mode_badge = "<span style='background:rgba(56, 189, 248, 0.2); border:1px solid #38bdf8; color:#38bdf8; padding:2px 8px; border-radius:4px; font-size:11.5px; font-weight:700; margin-left:6px;'>⚡ EXPIRY CLOSE</span>" if is_expiry_mode else "<span style='background:rgba(148, 163, 184, 0.15); border:1px solid #64748b; color:#94a3b8; padding:2px 8px; border-radius:4px; font-size:11.5px; font-weight:700; margin-left:6px;'>📈 TRADINGVIEW CLOSE</span>"
    
    st.markdown(f"""
    <div style="font-size: 13px; color: #94a3b8; margin-bottom: 12px;">
        Timeframe: <strong style="color: #38bdf8;">{selected_tf_display}</strong> ({tf_desc}) {mode_badge} | 
        Trigger Level: <strong style="color: #f8fafc;">&plusmn;{stretch_thresh:.2f}%</strong> | 
        {retrace_goal:.0f}% Retracement Level: <strong style="color: #f8fafc;">+{pos_goal:.2f}%</strong> (Upside) / <strong style="color: #f8fafc;">{neg_goal:.2f}%</strong> (Downside) | 
        Total Historical Sample: <strong style="color: #f8fafc;">{total_p:,} {tf_period_name}</strong>
    </div>
    """, unsafe_allow_html=True)
    
    col_up, col_down = st.columns(2)
    with col_up:
        card_up_html = f"""<div class="metric-card" style="border-top: 3px solid #00d26a;">
<div class="metric-label" style="font-size: 16px; color: #00d26a; font-weight: bold;">🟢 UPSIDE FIRST TOUCH (+{stretch_thresh:.2f}% from {selected_tf_display} Open)</div>
<div style="font-size: 13px; color: #94a3b8; margin-top: 4px;">Triggered in <strong>{pos['count']:,} {tf_period_name}</strong> ({pos['pct_of_all_periods']:.1f}% of all {total_p:,} {tf_period_name})</div>
<div style="font-size: 12px; color: #cbd5e1; margin-top: 2px;">Average Peak High reached: <strong style="color: #00d26a;">+{pos['avg_peak_high']:.2f}%</strong> (Avg extra run: +{pos['avg_extension']:.2f}%, Median: +{pos['median_extension']:.2f}%)</div>
<hr style="margin: 10px 0; border-color: #232f48;">
<div style="font-size: 13px; font-weight: bold; color: #38bdf8;">1. 🚀 Forward Breakout Extension (Did it run more?)</div>
<div style="font-size: 12px; color: #cbd5e1; margin-top: 4px; line-height: 1.6;">
• Ran &ge; +{e_s[0]:.2f}% further (reached &ge; +{stretch_thresh+e_s[0]:.2f}%): <strong style="color: #38bdf8;">{pos['ext_step_pcts'][0]:.1f}%</strong> ({pos['ext_step_counts'][0]:,} {tf_period_name})<br>
• Ran &ge; +{e_s[1]:.2f}% further (reached &ge; +{stretch_thresh+e_s[1]:.2f}%): <strong style="color: #38bdf8;">{pos['ext_step_pcts'][1]:.1f}%</strong> ({pos['ext_step_counts'][1]:,} {tf_period_name})<br>
• Ran &ge; +{e_s[2]:.2f}% further (reached &ge; +{stretch_thresh+e_s[2]:.2f}%): <strong style="color: #38bdf8;">{pos['ext_step_pcts'][2]:.1f}%</strong> ({pos['ext_step_counts'][2]:,} {tf_period_name})<br>
• Ran &ge; +{e_s[3]:.2f}% further (reached &ge; +{stretch_thresh+e_s[3]:.2f}%): <strong style="color: #38bdf8;">{pos['ext_step_pcts'][3]:.1f}%</strong> ({pos['ext_step_counts'][3]:,} {tf_period_name})
</div>
<hr style="margin: 10px 0; border-color: #232f48;">
<div style="font-size: 13px; font-weight: bold; color: #fbbf24;">2. 🔄 Pullback & Mean-Reversion (Did it retrace?)</div>
<div style="font-size: 12px; color: #cbd5e1; margin-top: 4px; line-height: 1.6;">
• Dropped Below Trigger Level (< +{stretch_thresh:.2f}%): <strong style="color: #f8fafc;">{pos['pulled_below_trigger_pct']:.1f}%</strong> ({pos['pulled_below_trigger_count']:,} {tf_period_name})<br>
• Retraced &ge; {retrace_goal:.0f}% of Initial Stretch (to &le; +{pos_goal:.2f}%): <strong style="color: #fbbf24;">{pos['retrace_goal_pct']:.1f}%</strong> ({pos['retrace_goal_count']:,} {tf_period_name})<br>
<div style="margin: 4px 0 6px 14px; padding: 6px 10px; background: rgba(251, 191, 36, 0.08); border-left: 2px solid #fbbf24; border-radius: 0 6px 6px 0; font-size: 11.5px; line-height: 1.5; color: #cbd5e1;">
    <span style="color: #fbbf24; font-weight: 700;">↳ After {retrace_goal:.0f}% Retracement — What Happens Next?</span><br>
    • Re-tested Trigger (+{stretch_thresh:.2f}%): <strong style="color: #38bdf8;">{pos['after_retrace_retest_pct']:.1f}%</strong> ({pos['after_retrace_retest_cnt']:,} {tf_period_name})<br>
    • Dropped Further to Open (&le; 0.0%): <strong style="color: #f87171;">{pos['after_retrace_open_pct']:.1f}%</strong> ({pos['after_retrace_open_cnt']:,} {tf_period_name})<br>
    • Post-Retrace Close Settlement: <strong style="color: #00d26a;">{pos['after_retrace_close_trend_pct']:.1f}% Trend</strong> | <strong style="color: #38bdf8;">{pos['after_retrace_close_consol_pct']:.1f}% Consol</strong> | <strong style="color: #f8312f;">{pos['after_retrace_close_red_pct']:.1f}% Red</strong>
</div>
• Reversed All the Way to Open (&le; 0.0%): <strong style="color: #f87171;">{pos['back_to_open_pct']:.1f}%</strong> ({pos['back_to_open_count']:,} {tf_period_name})<br>
<div style="margin: 4px 0 6px 14px; padding: 6px 10px; background: rgba(248, 113, 113, 0.08); border-left: 2px solid #f87171; border-radius: 0 6px 6px 0; font-size: 11.5px; line-height: 1.5; color: #cbd5e1;">
    <span style="color: #f87171; font-weight: 700;">↳ After Reversing to Open (0.0%) — What Happens Next?</span><br>
    • Plunged into Negative Breakdown (&le; -{stretch_thresh:.2f}%): <strong style="color: #f8312f;">{pos['after_open_plunge_pct']:.1f}%</strong> ({pos['after_open_plunge_cnt']:,} {tf_period_name})<br>
    • Re-bounced Back to Stretch (+{stretch_thresh:.2f}%): <strong style="color: #38bdf8;">{pos['after_open_bounce_trig_pct']:.1f}%</strong> ({pos['after_open_bounce_trig_cnt']:,} {tf_period_name})<br>
    • Post-Open Close Settlement: <strong style="color: #f8312f;">{pos['after_open_close_red_pct']:.1f}% Closed Red</strong> | <strong style="color: #00d26a;">{pos['after_open_close_green_pct']:.1f}% Re-bounced Green</strong>
</div>
• Retraced &ge; {retrace_goal:.0f}% from Period Peak High: <strong style="color: #fbbf24;">{pos['retrace_peak_goal_pct']:.1f}%</strong> ({pos['retrace_peak_goal_count']:,} {tf_period_name})<br>
• Average Retracement Depth from Peak: <strong style="color: #f8fafc;">{pos['avg_retrace_from_peak']:.2f}%</strong>
</div>
<hr style="margin: 10px 0; border-color: #232f48;">
<div style="font-size: 13px; font-weight: bold; color: #a78bfa;">3. 🎯 Final Close Destination ({tf_close_label})</div>
<div style="font-size: 12px; color: #cbd5e1; margin-top: 4px; line-height: 1.6;">
• <strong>Trend Continuation Close</strong> (> +{stretch_thresh:.2f}%): <strong style="color: #00d26a;">{pos['closed_trend_pct']:.1f}%</strong> ({pos['closed_trend_count']:,} {tf_period_name})<br>
• <strong>Consolidation Close</strong> (0.0% to +{stretch_thresh:.2f}%): <strong style="color: #38bdf8;">{pos['closed_consol_pct']:.1f}%</strong> ({pos['closed_consol_count']:,} {tf_period_name})<br>
• <strong>Failed Breakout / {rev_label_pos}</strong>: <strong style="color: #f8312f;">{pos['closed_red_pct']:.1f}%</strong> ({pos['closed_red_count']:,} {tf_period_name})
</div>
</div>"""
        st.markdown(card_up_html, unsafe_allow_html=True)
        
    with col_down:
        card_down_html = f"""<div class="metric-card" style="border-top: 3px solid #f8312f;">
<div class="metric-label" style="font-size: 16px; color: #f8312f; font-weight: bold;">🔴 DOWNSIDE FIRST TOUCH (-{stretch_thresh:.2f}% from {selected_tf_display} Open)</div>
<div style="font-size: 13px; color: #94a3b8; margin-top: 4px;">Triggered in <strong>{neg['count']:,} {tf_period_name}</strong> ({neg['pct_of_all_periods']:.1f}% of all {total_p:,} {tf_period_name})</div>
<div style="font-size: 12px; color: #cbd5e1; margin-top: 2px;">Average Trough Low reached: <strong style="color: #f8312f;">{neg['avg_trough_low']:.2f}%</strong> (Avg extra drop: -{neg['avg_extension']:.2f}%, Median: -{neg['median_extension']:.2f}%)</div>
<hr style="margin: 10px 0; border-color: #232f48;">
<div style="font-size: 13px; font-weight: bold; color: #38bdf8;">1. 📉 Forward Breakdown Extension (Did it drop more?)</div>
<div style="font-size: 12px; color: #cbd5e1; margin-top: 4px; line-height: 1.6;">
• Dropped &ge; -{e_s[0]:.2f}% further (reached &le; -{stretch_thresh+e_s[0]:.2f}%): <strong style="color: #38bdf8;">{neg['ext_step_pcts'][0]:.1f}%</strong> ({neg['ext_step_counts'][0]:,} {tf_period_name})<br>
• Dropped &ge; -{e_s[1]:.2f}% further (reached &le; -{stretch_thresh+e_s[1]:.2f}%): <strong style="color: #38bdf8;">{neg['ext_step_pcts'][1]:.1f}%</strong> ({neg['ext_step_counts'][1]:,} {tf_period_name})<br>
• Dropped &ge; -{e_s[2]:.2f}% further (reached &le; -{stretch_thresh+e_s[2]:.2f}%): <strong style="color: #38bdf8;">{neg['ext_step_pcts'][2]:.1f}%</strong> ({neg['ext_step_counts'][2]:,} {tf_period_name})<br>
• Dropped &ge; -{e_s[3]:.2f}% further (reached &le; -{stretch_thresh+e_s[3]:.2f}%): <strong style="color: #38bdf8;">{neg['ext_step_pcts'][3]:.1f}%</strong> ({neg['ext_step_counts'][3]:,} {tf_period_name})
</div>
<hr style="margin: 10px 0; border-color: #232f48;">
<div style="font-size: 13px; font-weight: bold; color: #fbbf24;">2. 🔄 Bounce & Mean-Reversion (Did it bounce back?)</div>
<div style="font-size: 12px; color: #cbd5e1; margin-top: 4px; line-height: 1.6;">
• Bounced Above Trigger Level (> -{stretch_thresh:.2f}%): <strong style="color: #f8fafc;">{neg['bounced_above_trigger_pct']:.1f}%</strong> ({neg['bounced_above_trigger_count']:,} {tf_period_name})<br>
• Bounced &ge; {retrace_goal:.0f}% of Initial Drop (to &ge; -{neg_goal:.2f}%): <strong style="color: #fbbf24;">{neg['retrace_goal_pct']:.1f}%</strong> ({neg['retrace_goal_count']:,} {tf_period_name})<br>
<div style="margin: 4px 0 6px 14px; padding: 6px 10px; background: rgba(251, 191, 36, 0.08); border-left: 2px solid #fbbf24; border-radius: 0 6px 6px 0; font-size: 11.5px; line-height: 1.5; color: #cbd5e1;">
    <span style="color: #fbbf24; font-weight: 700;">↳ After {retrace_goal:.0f}% Bounce — What Happens Next?</span><br>
    • Re-tested Breakdown Level (-{stretch_thresh:.2f}%): <strong style="color: #f8312f;">{neg['after_retrace_retest_pct']:.1f}%</strong> ({neg['after_retrace_retest_cnt']:,} {tf_period_name})<br>
    • Bounced Further to Open (&ge; 0.0%): <strong style="color: #38bdf8;">{neg['after_retrace_open_pct']:.1f}%</strong> ({neg['after_retrace_open_cnt']:,} {tf_period_name})<br>
    • Post-Bounce Close Settlement: <strong style="color: #f8312f;">{neg['after_retrace_close_trend_pct']:.1f}% Trend</strong> | <strong style="color: #38bdf8;">{neg['after_retrace_close_consol_pct']:.1f}% Consol</strong> | <strong style="color: #00d26a;">{neg['after_retrace_close_green_pct']:.1f}% Green</strong>
</div>
• Bounced All the Way to Open (&ge; 0.0%): <strong style="color: #6ee7b7;">{neg['back_to_open_pct']:.1f}%</strong> ({neg['back_to_open_count']:,} {tf_period_name})<br>
<div style="margin: 4px 0 6px 14px; padding: 6px 10px; background: rgba(110, 231, 183, 0.08); border-left: 2px solid #00d26a; border-radius: 0 6px 6px 0; font-size: 11.5px; line-height: 1.5; color: #cbd5e1;">
    <span style="color: #00d26a; font-weight: 700;">↳ After Bouncing to Open (0.0%) — What Happens Next?</span><br>
    • Surged into Positive Stretch (&ge; +{stretch_thresh:.2f}%): <strong style="color: #00d26a;">{neg['after_open_plunge_pct']:.1f}%</strong> ({neg['after_open_plunge_cnt']:,} {tf_period_name})<br>
    • Plunged Back to Breakdown (-{stretch_thresh:.2f}%): <strong style="color: #f8312f;">{neg['after_open_bounce_trig_pct']:.1f}%</strong> ({neg['after_open_bounce_trig_cnt']:,} {tf_period_name})<br>
    • Post-Open Close Settlement: <strong style="color: #00d26a;">{neg['after_open_close_green_pct']:.1f}% Closed Green</strong> | <strong style="color: #f8312f;">{neg['after_open_close_red_pct']:.1f}% Plunged Red</strong>
</div>
• Bounced &ge; {retrace_goal:.0f}% from Period Trough Low: <strong style="color: #fbbf24;">{neg['bounce_trough_goal_pct']:.1f}%</strong> ({neg['bounce_trough_goal_count']:,} {tf_period_name})<br>
• Average Bounce Height from Trough: <strong style="color: #f8fafc;">{neg['avg_bounce_from_trough']:.2f}%</strong>
</div>
<hr style="margin: 10px 0; border-color: #232f48;">
<div style="font-size: 13px; font-weight: bold; color: #a78bfa;">3. 🎯 Final Close Destination ({tf_close_label})</div>
<div style="font-size: 12px; color: #cbd5e1; margin-top: 4px; line-height: 1.6;">
• <strong>Trend Continuation Close</strong> (< -{stretch_thresh:.2f}%): <strong style="color: #f8312f;">{neg['closed_trend_pct']:.1f}%</strong> ({neg['closed_trend_count']:,} {tf_period_name})<br>
• <strong>Consolidation Close</strong> (-{stretch_thresh:.2f}% to 0.0%): <strong style="color: #38bdf8;">{neg['closed_consol_pct']:.1f}%</strong> ({neg['closed_consol_count']:,} {tf_period_name})<br>
• <strong>Failed Breakdown / {rev_label_neg}</strong>: <strong style="color: #00d26a;">{neg['closed_green_pct']:.1f}%</strong> ({neg['closed_green_count']:,} {tf_period_name})
</div>
</div>"""
        st.markdown(card_down_html, unsafe_allow_html=True)
        
    # Comparison Bar Chart
    st.markdown("---")
    st.markdown(f"#### 📊 Comparative Progression Chart: {selected_tf_display} (Extension vs Retracement vs Close)")
    
    fig_rev = go.Figure()
    categories = [
        f"Run +{e_s[0]:.2f}% Further",
        f"Run +{e_s[1]:.2f}% Further",
        f"Retrace {retrace_goal:.0f}% of Trigger",
        "Return to Open (0.0%)",
        "Trend Close",
        "Consolidation Close",
        "Opposite Color Close"
    ]
    
    pos_vals = [
        pos['ext_step_pcts'][0],
        pos['ext_step_pcts'][1],
        pos['retrace_goal_pct'],
        pos['back_to_open_pct'],
        pos['closed_trend_pct'],
        pos['closed_consol_pct'],
        pos['closed_red_pct']
    ]
    neg_vals = [
        neg['ext_step_pcts'][0],
        neg['ext_step_pcts'][1],
        neg['retrace_goal_pct'],
        neg['back_to_open_pct'],
        neg['closed_trend_pct'],
        neg['closed_consol_pct'],
        neg['closed_green_pct']
    ]
    
    fig_rev.add_trace(go.Bar(
        x=categories,
        y=pos_vals,
        name=f"Upside First Touch (+{stretch_thresh:.2f}%)",
        marker_color='#00d26a',
        text=[f"{v:.1f}%" for v in pos_vals],
        textposition='auto'
    ))
    fig_rev.add_trace(go.Bar(
        x=categories,
        y=neg_vals,
        name=f"Downside First Touch (-{stretch_thresh:.2f}%)",
        marker_color='#ef4444',
        text=[f"{v:.1f}%" for v in neg_vals],
        textposition='auto'
    ))
    fig_rev.update_layout(
        title=f"Post-Trigger Behavioral Distribution (First Touch = &plusmn;{stretch_thresh:.2f}% from {selected_tf_display} Open)",
        yaxis_title="Probability (%)",
        barmode='group',
        template="plotly_dark",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_rev, width="stretch")
    
    # Dual Distribution Histogram of Final Closes
    if len(pos['final_closes']) > 0 and len(neg['final_closes']) > 0:
        st.markdown(f"#### 🎯 Where Did Nifty End the {selected_tf_display}? (Final {tf_close_label} Move Distribution on Trigger {tf_period_name})")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=pos['final_closes'],
            nbinsx=40 if selected_tf != "Daily" else 50,
            name=f"{selected_tf_display}s touching +{stretch_thresh:.2f}% ({len(pos['final_closes']):,} {tf_period_name})",
            marker_color='#00d26a',
            opacity=0.65
        ))
        fig_hist.add_trace(go.Histogram(
            x=neg['final_closes'],
            nbinsx=40 if selected_tf != "Daily" else 50,
            name=f"{selected_tf_display}s touching -{stretch_thresh:.2f}% ({len(neg['final_closes']):,} {tf_period_name})",
            marker_color='#ef4444',
            opacity=0.65
        ))
        fig_hist.add_vline(x=0.0, line_dash="dash", line_color="#94a3b8", line_width=1.5, annotation_text=f"{selected_tf_display} Open (0.0%)")
        fig_hist.add_vline(x=stretch_thresh, line_dash="dot", line_color="#00d26a", line_width=1.5, annotation_text=f"+{stretch_thresh:.2f}%")
        fig_hist.add_vline(x=-stretch_thresh, line_dash="dot", line_color="#ef4444", line_width=1.5, annotation_text=f"-{stretch_thresh:.2f}%")
        fig_hist.update_layout(
            title=f"Final Open-to-Close % on {selected_tf_display}s When First Touch &plusmn;{stretch_thresh:.2f}% Was Reached",
            xaxis_title=f"Final {selected_tf_display} Close Move (%)",
            yaxis_title=f"Number of {tf_period_name}",
            barmode='overlay',
            template="plotly_dark",
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_hist, width="stretch")


# =============================================================
# SECTION 3: NIFTY 500 AUTOMATED DATA DOWNLOADER
# =============================================================
elif "Nifty 500 Automated Data Downloader" in nav_selection:
    try:
        import local_data_automation
        if not hasattr(local_data_automation, "render_downloader_section"):
            import importlib
            local_data_automation = importlib.reload(local_data_automation)
        local_data_automation.render_downloader_section()
    except Exception as e:
        st.error(f"### ⚠️ Failed to load Section 3: {e}")
        st.info("💡 **Tip**: If you recently updated the app, please click **'Manage app'** (lower-right corner) ➔ **'Reboot app'**.")
        import traceback
        st.code(traceback.format_exc(), language="python")

# =============================================================
# SECTION 4: AUTOMATED NIFTY DATA DOWNLOADER
# =============================================================
elif "Nifty Automated Data Downloader" in nav_selection:
    try:
        import local_data_automation
        if not hasattr(local_data_automation, "render_nifty_downloader_section"):
            import importlib
            local_data_automation = importlib.reload(local_data_automation)
        local_data_automation.render_nifty_downloader_section()
    except Exception as e:
        st.error(f"### ⚠️ Failed to load Section 4: {e}")
        st.info("💡 **Tip**: If you recently updated the app on Streamlit Cloud, please click **'Manage app'** (lower-right corner) ➔ **'Reboot app'** to reload newly deployed modules.")
        import traceback
        st.code(traceback.format_exc(), language="python")


