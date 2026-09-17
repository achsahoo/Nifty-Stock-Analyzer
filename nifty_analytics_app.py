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
from nifty_analytics import NiftyAnalyticsEngine

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
</style>
""", unsafe_allow_html=True)

# =============================================================
# LIGHTWEIGHT IN-MEMORY DATA ENGINE
# =============================================================
import os
from nifty_analytics import SPOT_FILE

def _get_spot_mtime():
    try:
        if os.path.exists(SPOT_FILE):
            return os.path.getmtime(SPOT_FILE)
    except Exception:
        pass
    return 0

@st.cache_resource(show_spinner="Loading Nifty 1-minute spot data (< 0.5s)...")
def get_analytics_engine(mtime):
    engine = NiftyAnalyticsEngine()
    engine.load_data()
    return engine

analytics_engine = get_analytics_engine(_get_spot_mtime())

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
        "📐 Section 3: RSI Custom Divergence Backtest",
        "📥 Section 4: Nifty 500 Automated Data Downloader",
        "🔍 Section 5: Nifty 500 Multi-Stock Divergence Scanner",
        "⚡ Section 6: Nifty Automated Data Downloader"
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
            
    c_in1, c_in2 = st.columns([1.5, 2.5])
    with c_in1:
        target_pct = st.number_input("Target Move Range Threshold (%):", min_value=0.10, max_value=10.0,
                                     value=float(st.session_state.stat_thresh_pct), step=0.10, format="%.2f")
        st.session_state.stat_thresh_pct = target_pct
    with c_in2:
        target_pct_slider = st.slider("Adjust Range Slider (%):", min_value=0.25, max_value=5.0,
                                      value=float(st.session_state.stat_thresh_pct), step=0.05)
        if target_pct_slider != st.session_state.stat_thresh_pct:
            st.session_state.stat_thresh_pct = target_pct_slider
            target_pct = target_pct_slider
            
    # Compute Daily, Weekly, and Monthly stats
    stats_daily = analytics_engine.get_range_and_move_stats("Daily", target_pct)
    stats_weekly = analytics_engine.get_range_and_move_stats("Weekly", target_pct)
    stats_monthly = analytics_engine.get_range_and_move_stats("Monthly", target_pct)
        
    st.markdown(f"### 🌐 Multi-Timeframe Move Comparison (Threshold: &plusmn;{target_pct:.2f}%)")
    
    # 3 Side-by-Side Timeframe Columns
    col_d, col_w, col_m = st.columns(3)
    
    with col_d:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label" style="font-size: 14px; color: #38bdf8;">📅 DAILY ({stats_daily['total_periods']:,} Days)</div>
            <hr style="margin: 6px 0; border-color: #232f48;">
            <div style="font-size: 13px; color: #94a3b8;">Close Move within &plusmn;{target_pct:.2f}%:</div>
            <div class="metric-value" style="color: #38bdf8;">{stats_daily['pct_within_move']:.1f}% <span style="font-size: 13px; color: #94a3b8;">({stats_daily['cnt_within_move']:,} days)</span></div>
            <div style="font-size: 13px; color: #94a3b8; margin-top: 6px;">Total High-Low Range &le; {target_pct:.2f}%:</div>
            <div class="metric-value" style="color: #00d26a;">{stats_daily['pct_within_range']:.1f}% <span style="font-size: 13px; color: #94a3b8;">({stats_daily['cnt_within_range']:,} days)</span></div>
            <div style="font-size: 12px; color: #cbd5e1; margin-top: 8px; line-height: 1.6;">
                • Bullish Breakout (> +{target_pct:.2f}%): <strong>{stats_daily['pct_above_pos']:.1f}%</strong> ({stats_daily['cnt_above_pos']:,} days)<br>
                • Bearish Breakdown (< -{target_pct:.2f}%): <strong>{stats_daily['pct_below_neg']:.1f}%</strong> ({stats_daily['cnt_below_neg']:,} days)<br>
                • Average Daily Range: <strong>{stats_daily['avg_range_pct']:.2f}%</strong> (Median: {stats_daily['median_range_pct']:.2f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_w:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label" style="font-size: 14px; color: #a78bfa;">📆 WEEKLY ({stats_weekly['total_periods']:,} Weeks)</div>
            <hr style="margin: 6px 0; border-color: #232f48;">
            <div style="font-size: 13px; color: #94a3b8;">Close Move within &plusmn;{target_pct:.2f}%:</div>
            <div class="metric-value" style="color: #a78bfa;">{stats_weekly['pct_within_move']:.1f}% <span style="font-size: 13px; color: #94a3b8;">({stats_weekly['cnt_within_move']:,} weeks)</span></div>
            <div style="font-size: 13px; color: #94a3b8; margin-top: 6px;">Total High-Low Range &le; {target_pct:.2f}%:</div>
            <div class="metric-value" style="color: #00d26a;">{stats_weekly['pct_within_range']:.1f}% <span style="font-size: 13px; color: #94a3b8;">({stats_weekly['cnt_within_range']:,} weeks)</span></div>
            <div style="font-size: 12px; color: #cbd5e1; margin-top: 8px; line-height: 1.6;">
                • Bullish Breakout (> +{target_pct:.2f}%): <strong>{stats_weekly['pct_above_pos']:.1f}%</strong> ({stats_weekly['cnt_above_pos']:,} weeks)<br>
                • Bearish Breakdown (< -{target_pct:.2f}%): <strong>{stats_weekly['pct_below_neg']:.1f}%</strong> ({stats_weekly['cnt_below_neg']:,} weeks)<br>
                • Average Weekly Range: <strong>{stats_weekly['avg_range_pct']:.2f}%</strong> (Median: {stats_weekly['median_range_pct']:.2f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_m:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label" style="font-size: 14px; color: #f43f5e;">🗓️ MONTHLY ({stats_monthly['total_periods']:,} Months)</div>
            <hr style="margin: 6px 0; border-color: #232f48;">
            <div style="font-size: 13px; color: #94a3b8;">Close Move within &plusmn;{target_pct:.2f}%:</div>
            <div class="metric-value" style="color: #f43f5e;">{stats_monthly['pct_within_move']:.1f}% <span style="font-size: 13px; color: #94a3b8;">({stats_monthly['cnt_within_move']:,} months)</span></div>
            <div style="font-size: 13px; color: #94a3b8; margin-top: 6px;">Total High-Low Range &le; {target_pct:.2f}%:</div>
            <div class="metric-value" style="color: #00d26a;">{stats_monthly['pct_within_range']:.1f}% <span style="font-size: 13px; color: #94a3b8;">({stats_monthly['cnt_within_range']:,} months)</span></div>
            <div style="font-size: 12px; color: #cbd5e1; margin-top: 8px; line-height: 1.6;">
                • Bullish Breakout (> +{target_pct:.2f}%): <strong>{stats_monthly['pct_above_pos']:.1f}%</strong> ({stats_monthly['cnt_above_pos']:,} months)<br>
                • Bearish Breakdown (< -{target_pct:.2f}%): <strong>{stats_monthly['pct_below_neg']:.1f}%</strong> ({stats_monthly['cnt_below_neg']:,} months)<br>
                • Average Monthly Range: <strong>{stats_monthly['avg_range_pct']:.2f}%</strong> (Median: {stats_monthly['median_range_pct']:.2f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # Unified Comparative Summary Table
    comp_df = pd.DataFrame([
        {
            "Timeframe": "Daily",
            "Total Periods": stats_daily['total_periods'],
            f"Within ±{target_pct:.2f}% Move (%)": f"{stats_daily['pct_within_move']:.1f}% ({stats_daily['cnt_within_move']:,})",
            f"High-Low Range ≤ {target_pct:.2f}% (%)": f"{stats_daily['pct_within_range']:.1f}% ({stats_daily['cnt_within_range']:,})",
            f"Bullish (> +{target_pct:.2f}%)": f"{stats_daily['pct_above_pos']:.1f}% ({stats_daily['cnt_above_pos']:,})",
            f"Bearish (< -{target_pct:.2f}%)": f"{stats_daily['pct_below_neg']:.1f}% ({stats_daily['cnt_below_neg']:,})",
            "Avg Range (%)": f"{stats_daily['avg_range_pct']:.2f}%",
            "Median Range (%)": f"{stats_daily['median_range_pct']:.2f}%"
        },
        {
            "Timeframe": "Weekly",
            "Total Periods": stats_weekly['total_periods'],
            f"Within ±{target_pct:.2f}% Move (%)": f"{stats_weekly['pct_within_move']:.1f}% ({stats_weekly['cnt_within_move']:,})",
            f"High-Low Range ≤ {target_pct:.2f}% (%)": f"{stats_weekly['pct_within_range']:.1f}% ({stats_weekly['cnt_within_range']:,})",
            f"Bullish (> +{target_pct:.2f}%)": f"{stats_weekly['pct_above_pos']:.1f}% ({stats_weekly['cnt_above_pos']:,})",
            f"Bearish (< -{target_pct:.2f}%)": f"{stats_weekly['pct_below_neg']:.1f}% ({stats_weekly['cnt_below_neg']:,})",
            "Avg Range (%)": f"{stats_weekly['avg_range_pct']:.2f}%",
            "Median Range (%)": f"{stats_weekly['median_range_pct']:.2f}%"
        },
        {
            "Timeframe": "Monthly",
            "Total Periods": stats_monthly['total_periods'],
            f"Within ±{target_pct:.2f}% Move (%)": f"{stats_monthly['pct_within_move']:.1f}% ({stats_monthly['cnt_within_move']:,})",
            f"High-Low Range ≤ {target_pct:.2f}% (%)": f"{stats_monthly['pct_within_range']:.1f}% ({stats_monthly['cnt_within_range']:,})",
            f"Bullish (> +{target_pct:.2f}%)": f"{stats_monthly['pct_above_pos']:.1f}% ({stats_monthly['cnt_above_pos']:,})",
            f"Bearish (< -{target_pct:.2f}%)": f"{stats_monthly['pct_below_neg']:.1f}% ({stats_monthly['cnt_below_neg']:,})",
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
        fig_ww = go.Figure()
        fig_ww.add_trace(go.Histogram(
            x=df_w['move_pct'], nbinsx=60, marker_color='#a78bfa', opacity=0.75, name='Weekly Move %'
        ))
        fig_ww.add_vline(x=-target_pct, line_dash="dash", line_color="#ef4444", line_width=2,
                         annotation_text=f"-{target_pct:.2f}% Threshold", annotation_position="top left")
        fig_ww.add_vline(x=target_pct, line_dash="dash", line_color="#00d26a", line_width=2,
                         annotation_text=f"+{target_pct:.2f}% Threshold", annotation_position="top right")
        fig_ww.update_layout(title=f"Weekly Open-to-Close Move Distribution ({stats_weekly['total_periods']:,} Weeks)",
                             xaxis_title="Move %", yaxis_title="Number of Weeks", template="plotly_dark", height=380)
        st.plotly_chart(fig_ww, width="stretch")
        
    with d_tab3:
        df_m = stats_monthly['df']
        fig_mm = go.Figure()
        fig_mm.add_trace(go.Histogram(
            x=df_m['move_pct'], nbinsx=40, marker_color='#f43f5e', opacity=0.75, name='Monthly Move %'
        ))
        fig_mm.add_vline(x=-target_pct, line_dash="dash", line_color="#ef4444", line_width=2,
                         annotation_text=f"-{target_pct:.2f}% Threshold", annotation_position="top left")
        fig_mm.add_vline(x=target_pct, line_dash="dash", line_color="#00d26a", line_width=2,
                         annotation_text=f"+{target_pct:.2f}% Threshold", annotation_position="top right")
        fig_mm.update_layout(title=f"Monthly Open-to-Close Move Distribution ({stats_monthly['total_periods']:,} Months)",
                             xaxis_title="Move %", yaxis_title="Number of Months", template="plotly_dark", height=380)
        st.plotly_chart(fig_mm, width="stretch")

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
    
    # Multi-timeframe Quick Matrix Expander
    with st.expander("🌐 View Multi-Timeframe Stretch Summary Matrix (Daily vs Weekly vs Monthly)", expanded=False):
        c_d = analytics_engine.get_stretch_reversal_stats("Daily", 0.50, 50.0)
        c_w = analytics_engine.get_stretch_reversal_stats("Weekly", 1.50, 50.0)
        c_m = analytics_engine.get_stretch_reversal_stats("Monthly", 3.00, 50.0)
        
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
                "Timeframe": "📆 Weekly",
                "Total Periods": f"{c_w['total_periods']:,} Weeks",
                "Benchmark Stretch": "±1.50%",
                "Bullish Triggered (%)": f"{c_w['pos']['pct_of_all_periods']:.1f}% ({c_w['pos']['count']:,})",
                "Avg Peak High": f"+{c_w['pos']['avg_peak_high']:.2f}%",
                "Retraced 50% of Stretch (%)": f"{c_w['pos']['retrace_goal_pct']:.1f}%",
                "Returned to Open (0%)": f"{c_w['pos']['back_to_open_pct']:.1f}%",
                "Trend Close (%)": f"{c_w['pos']['closed_trend_pct']:.1f}%",
                "Failed Reversal / Red (%)": f"{c_w['pos']['closed_red_pct']:.1f}%"
            },
            {
                "Timeframe": "🗓️ Monthly",
                "Total Periods": f"{c_m['total_periods']:,} Months",
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
    sec2_tf_choice = st.radio(
        "Select Stretch Analysis Timeframe:",
        ["📅 Daily (Intraday Stretch)", "📆 Weekly (Intraweek Stretch)", "🗓️ Monthly (Intramonth Stretch)"],
        horizontal=True
    )
    
    tf_map = {
        "📅 Daily (Intraday Stretch)": "Daily",
        "📆 Weekly (Intraweek Stretch)": "Weekly",
        "🗓️ Monthly (Intramonth Stretch)": "Monthly"
    }
    selected_tf = tf_map[sec2_tf_choice]
    
    # Timeframe-specific configurations
    if selected_tf == "Daily":
        presets = [0.50, 0.75, 1.00, 1.25, 1.50]
        default_stretch = 0.50
        s_min, s_max, s_step = 0.20, 2.50, 0.05
        ext_steps = [0.25, 0.50, 0.75, 1.00]
        tf_desc = f"{total_days:,} trading sessions ({min_year} – {max_year}), measured from Day Open (09:15) to 15:30 Close"
        tf_period_name = "days"
        tf_close_label = "Day Close at 15:30"
        rev_label_pos = "Closed RED (< 0.0%)"
        rev_label_neg = "Closed GREEN (> 0.0%)"
    elif selected_tf == "Weekly":
        presets = [1.00, 1.50, 2.00, 2.50, 3.00]
        default_stretch = 1.50
        s_min, s_max, s_step = 0.50, 5.00, 0.10
        ext_steps = [0.50, 1.00, 1.50, 2.00]
        tf_desc = f"{total_weeks:,} calendar weeks ({min_year} – {max_year}), measured from Monday Open to Friday 15:30 Close"
        tf_period_name = "weeks"
        tf_close_label = "Friday Close at 15:30"
        rev_label_pos = "Closed RED Week (< 0.0%)"
        rev_label_neg = "Closed GREEN Week (> 0.0%)"
    else: # Monthly
        presets = [2.00, 3.00, 4.00, 5.00, 6.00]
        default_stretch = 3.00
        s_min, s_max, s_step = 1.00, 10.00, 0.25
        ext_steps = [1.00, 2.00, 3.00, 4.00]
        tf_desc = f"{total_months:,} calendar months ({min_year} – {max_year}), measured from Month Open (1st trading minute) to Month-End Close"
        tf_period_name = "months"
        tf_close_label = "Month-End Close at 15:30"
        rev_label_pos = "Closed RED Month (< 0.0%)"
        rev_label_neg = "Closed GREEN Month (> 0.0%)"
        
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
    st.markdown(f"**Quick {selected_tf} Stretch Presets:**")
    s_cols = st.columns(5)
    for i, p_val in enumerate(presets):
        with s_cols[i]:
            st.button(f"{p_val:.2f}% Stretch", key=f"btn_{selected_tf}_{p_val}",
                      on_click=_set_preset_thresh, args=(p_val, session_key), width="stretch")
                
    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
            <span style="font-weight: 600; color: #f1f5f9; font-size: 13.5px;">{selected_tf} Stretch Threshold (% from Period Open):</span>
            <span style="background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.4); color: #38bdf8; font-weight: 700; font-size: 13.5px; padding: 2px 8px; border-radius: 6px;">
                &plusmn;{st.session_state[session_key]:.2f}%
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        c_dec, c_slider, c_inc = st.columns([1.1, 7.8, 1.1], vertical_alignment="center")
        with c_dec:
            st.button("➖", key=f"btn_dec_{selected_tf}", on_click=_change_thresh, args=(-s_step, session_key, s_min, s_max, s_step),
                      help=f"Decrease {selected_tf.lower()} threshold by {s_step:.2f}%", width="stretch")
        with c_slider:
            stretch_thresh = st.slider(
                f"{selected_tf} Stretch Threshold",
                min_value=s_min, max_value=s_max,
                step=s_step,
                key=session_key,
                label_visibility="collapsed",
                help=f"Trigger level from {selected_tf.lower()} open"
            )
        with c_inc:
            st.button("➕", key=f"btn_inc_{selected_tf}", on_click=_change_thresh, args=(s_step, session_key, s_min, s_max, s_step),
                      help=f"Increase {selected_tf.lower()} threshold by {s_step:.2f}%", width="stretch")
                      
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
                                 
    rev_stats = analytics_engine.get_stretch_reversal_stats(timeframe=selected_tf, stretch_threshold=stretch_thresh, retrace_pct=retrace_goal, ext_steps=ext_steps)
        
    pos = rev_stats['pos']
    neg = rev_stats['neg']
    total_p = rev_stats['total_periods']
    pos_goal = rev_stats['pos_goal_level']
    neg_goal = rev_stats['neg_goal_level']
    e_s = rev_stats['ext_steps']
    
    st.markdown(f"""
    <div style="font-size: 13px; color: #94a3b8; margin-bottom: 12px;">
        Timeframe: <strong style="color: #38bdf8;">{selected_tf}</strong> ({tf_desc}) | 
        Trigger Level: <strong style="color: #f8fafc;">&plusmn;{stretch_thresh:.2f}%</strong> | 
        {retrace_goal:.0f}% Retracement Level: <strong style="color: #f8fafc;">+{pos_goal:.2f}%</strong> (Upside) / <strong style="color: #f8fafc;">{neg_goal:.2f}%</strong> (Downside) | 
        Total Historical Sample: <strong style="color: #f8fafc;">{total_p:,} {tf_period_name}</strong>
    </div>
    """, unsafe_allow_html=True)
    
    col_up, col_down = st.columns(2)
    with col_up:
        card_up_html = f"""<div class="metric-card" style="border-top: 3px solid #00d26a;">
<div class="metric-label" style="font-size: 16px; color: #00d26a; font-weight: bold;">🟢 UPSIDE FIRST TOUCH (+{stretch_thresh:.2f}% from {selected_tf} Open)</div>
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
<div class="metric-label" style="font-size: 16px; color: #f8312f; font-weight: bold;">🔴 DOWNSIDE FIRST TOUCH (-{stretch_thresh:.2f}% from {selected_tf} Open)</div>
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
    st.markdown(f"#### 📊 Comparative Progression Chart: {selected_tf} (Extension vs Retracement vs Close)")
    
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
        title=f"Post-Trigger Behavioral Distribution (First Touch = &plusmn;{stretch_thresh:.2f}% from {selected_tf} Open)",
        yaxis_title="Probability (%)",
        barmode='group',
        template="plotly_dark",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_rev, width="stretch")
    
    # Dual Distribution Histogram of Final Closes
    if len(pos['final_closes']) > 0 and len(neg['final_closes']) > 0:
        st.markdown(f"#### 🎯 Where Did Nifty End the {selected_tf}? (Final {tf_close_label} Move Distribution on Trigger {selected_tf}s)")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=pos['final_closes'],
            nbinsx=40 if selected_tf != "Daily" else 50,
            name=f"{selected_tf}s touching +{stretch_thresh:.2f}% ({len(pos['final_closes']):,} {tf_period_name})",
            marker_color='#00d26a',
            opacity=0.65
        ))
        fig_hist.add_trace(go.Histogram(
            x=neg['final_closes'],
            nbinsx=40 if selected_tf != "Daily" else 50,
            name=f"{selected_tf}s touching -{stretch_thresh:.2f}% ({len(neg['final_closes']):,} {tf_period_name})",
            marker_color='#ef4444',
            opacity=0.65
        ))
        fig_hist.add_vline(x=0.0, line_dash="dash", line_color="#94a3b8", line_width=1.5, annotation_text=f"{selected_tf} Open (0.0%)")
        fig_hist.add_vline(x=stretch_thresh, line_dash="dot", line_color="#00d26a", line_width=1.5, annotation_text=f"+{stretch_thresh:.2f}%")
        fig_hist.add_vline(x=-stretch_thresh, line_dash="dot", line_color="#ef4444", line_width=1.5, annotation_text=f"-{stretch_thresh:.2f}%")
        fig_hist.update_layout(
            title=f"Final Open-to-Close % on {selected_tf}s When First Touch &plusmn;{stretch_thresh:.2f}% Was Reached",
            xaxis_title=f"Final {selected_tf} Close Move (%)",
            yaxis_title=f"Number of {selected_tf}s",
            barmode='overlay',
            template="plotly_dark",
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_hist, width="stretch")


# =============================================================
# SECTION 3: RSI CUSTOM DIVERGENCE BACKTEST ENGINE
# =============================================================
elif nav_selection == "📐 Section 3: RSI Custom Divergence Backtest":
    from rsi_divergence_ui import render_rsi_divergence_section
    render_rsi_divergence_section(analytics_engine)

elif nav_selection == "📥 Section 4: Nifty 500 Automated Data Downloader":
    try:
        import local_data_automation
        if not hasattr(local_data_automation, "render_downloader_section"):
            import importlib
            local_data_automation = importlib.reload(local_data_automation)
        local_data_automation.render_downloader_section()
    except Exception as e:
        st.error(f"### ⚠️ Failed to load Section 4: {e}")
        st.info("💡 **Tip**: If you recently updated the app, please click **'Manage app'** (lower-right corner) ➔ **'Reboot app'**.")
        import traceback
        st.code(traceback.format_exc(), language="python")

elif nav_selection == "🔍 Section 5: Nifty 500 Multi-Stock Divergence Scanner":
    try:
        import local_data_automation
        if not hasattr(local_data_automation, "render_scanner_section"):
            import importlib
            local_data_automation = importlib.reload(local_data_automation)
        local_data_automation.render_scanner_section()
    except Exception as e:
        st.error(f"### ⚠️ Failed to load Section 5: {e}")
        st.info("💡 **Tip**: If you recently updated the app, please click **'Manage app'** (lower-right corner) ➔ **'Reboot app'**.")
        import traceback
        st.code(traceback.format_exc(), language="python")

elif nav_selection == "⚡ Section 6: Nifty Automated Data Downloader":
    try:
        import local_data_automation
        if not hasattr(local_data_automation, "render_nifty_downloader_section"):
            import importlib
            local_data_automation = importlib.reload(local_data_automation)
        local_data_automation.render_nifty_downloader_section()
    except Exception as e:
        st.error(f"### ⚠️ Failed to load Section 6: {e}")
        st.info("💡 **Tip**: If you recently updated the app on Streamlit Cloud, please click **'Manage app'** (lower-right corner) ➔ **'Reboot app'** to reload newly deployed modules.")
        import traceback
        st.code(traceback.format_exc(), language="python")
