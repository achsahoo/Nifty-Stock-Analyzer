"""
Seasonality & Month-Wise Historical Performance Analytics Engine
===============================================================
Comprehensive monthly seasonality terminal for NIFTY 50 Index, 20 Market Sectors,
Nifty 50 constituents, and all 500 Nifty Equities.
"""

import os
import glob
import time
import pandas as pd
import polars as pl
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Data Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOCAL_STOCK_MONTHLY_DIR = os.path.join(DATA_DIR, "stock_monthly_parquet") if os.path.exists(os.path.join(DATA_DIR, "stock_monthly_parquet")) else r"D:\NIFTY_MODEL\stock_monthly_parquet"
CSV_PATH = os.path.join(BASE_DIR, "ind_nifty500list.csv")

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_FULL_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

NIFTY50_CONSTITUENTS = [
    'ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK',
    'BAJAJ_AUTO', 'BAJFINANCE', 'BAJAJFINSV', 'BEL', 'BHARTIARTL',
    'BPCL', 'BRITANNIA', 'CIPLA', 'COALINDIA', 'DRREDDY',
    'EICHERMOT', 'GRASIM', 'HCLTECH', 'HDFCBANK', 'HDFCLIFE',
    'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR', 'ICICIBANK', 'INDUSINDBK',
    'INFY', 'ITC', 'JSWSTEEL', 'KOTAKBANK', 'LT',
    'M&M', 'MARUTI', 'NESTLEIND', 'NTPC', 'ONGC',
    'POWERGRID', 'RELIANCE', 'SBILIFE', 'SBIN', 'SHRIRAMFIN',
    'SUNPHARMA', 'TATACONSUM', 'TMCV', 'TATASTEEL', 'TCS',
    'TECHM', 'TITAN', 'TRENT', 'ULTRACEMCO', 'WIPRO'
]


# =============================================================
# DATA LOADERS & CACHING
# =============================================================
@st.cache_data(ttl=3600)
def load_metadata():
    """Loads symbol, company name, and industry mapping from ind_nifty500list.csv."""
    if os.path.exists(CSV_PATH):
        df_csv = pd.read_csv(CSV_PATH)
        sym_to_industry = dict(zip(df_csv['Symbol'], df_csv['Industry']))
        sym_to_company = dict(zip(df_csv['Symbol'], df_csv['Company Name']))
        return df_csv, sym_to_industry, sym_to_company
    return pd.DataFrame(), {}, {}


@st.cache_data(show_spinner="Loading Monthly Historical Stock Parquets (< 1s)...", ttl=3600)
def load_all_monthly_stocks_df():
    """
    Rapidly loads all 500 stock monthly parquets using Polars in ~0.5s.
    Returns a unified Polars DataFrame with year, month, move_pct, and is_positive.
    """
    _, sym_to_industry, sym_to_company = load_metadata()
    target_dir = LOCAL_STOCK_MONTHLY_DIR
    if not os.path.exists(target_dir):
        target_dir = r"D:\NIFTY_MODEL\stock_monthly_parquet"

    pattern = os.path.join(target_dir, "*_monthly_data_raw.parquet")
    files = glob.glob(pattern)
    if not files:
        return pl.DataFrame()

    dfs = []
    nifty50_set = set(NIFTY50_CONSTITUENTS)

    for f in files:
        sym = os.path.basename(f).replace("_monthly_data_raw.parquet", "")
        ind = sym_to_industry.get(sym, "Other")
        comp = sym_to_company.get(sym, sym)
        is_n50 = sym in nifty50_set

        try:
            d = pl.read_parquet(f).select([
                pl.col("datetime"),
                pl.lit(sym).alias("symbol"),
                pl.lit(comp).alias("company_name"),
                pl.lit(ind).alias("industry"),
                pl.lit(is_n50).alias("is_nifty50"),
                pl.col("open"),
                pl.col("high"),
                pl.col("low"),
                pl.col("close"),
                pl.col("volume")
            ])
            dfs.append(d)
        except Exception:
            pass

    if not dfs:
        return pl.DataFrame()

    combined = pl.concat(dfs)
    combined = combined.with_columns([
        combined["datetime"].dt.year().alias("year"),
        combined["datetime"].dt.month().alias("month"),
        ((combined["close"] - combined["open"]) / combined["open"] * 100.0).alias("move_pct"),
        ((combined["high"] - combined["open"]) / combined["open"] * 100.0).alias("high_stretch_pct"),
        ((combined["low"] - combined["open"]) / combined["open"] * 100.0).alias("low_drop_pct"),
        (combined["close"] > combined["open"]).alias("is_positive")
    ])
    return combined


@st.cache_data(show_spinner="Loading Nifty 50 Historical Monthly Data...", ttl=3600)
def load_nifty_monthly_seasonality_df():
    """
    Loads clean Nifty 50 monthly candles (1996 to 2026: 369 months) from NiftyAnalyticsEngine.
    """
    from nifty_analytics import NiftyAnalyticsEngine
    engine = NiftyAnalyticsEngine()
    engine.load_data()
    df = engine._monthly_df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['is_positive'] = df['close'] > df['open']
    df['high_stretch_pct'] = (df['high'] - df['open']) / df['open'] * 100.0
    df['low_drop_pct'] = (df['low'] - df['open']) / df['open'] * 100.0
    return df


# =============================================================
# STATISTICAL CALCULATION ENGINES
# =============================================================
def compute_monthly_summary(df, date_col='date', move_col='move_pct'):
    """
    Computes month-wise stats across Jan to Dec for a single asset (Nifty or individual stock).
    Returns a DataFrame with 12 rows (Jan..Dec) containing Win Rate, Avg, Median, Best, Worst.
    """
    stats = []
    for m in range(1, 13):
        m_sub = df[df['month'] == m]
        tot = len(m_sub)
        if tot == 0:
            stats.append({
                'month_idx': m,
                'Month': MONTH_NAMES[m - 1],
                'Month_Full': MONTH_FULL_NAMES[m - 1],
                'Total_Years': 0,
                'Win_Rate': 0.0,
                'Avg_Move': 0.0,
                'Median_Move': 0.0,
                'Pos_Count': 0,
                'Neg_Count': 0,
                'Best_Year': '-',
                'Best_Move': 0.0,
                'Worst_Year': '-',
                'Worst_Move': 0.0,
                'Avg_High_Stretch': 0.0,
                'Avg_Low_Drop': 0.0
            })
            continue

        pos_mask = m_sub[move_col] > 0
        pos_cnt = int(pos_mask.sum())
        neg_cnt = tot - pos_cnt
        win_rate = (pos_cnt / tot) * 100.0
        avg_move = float(m_sub[move_col].mean())
        med_move = float(m_sub[move_col].median())

        best_idx = m_sub[move_col].idxmax()
        worst_idx = m_sub[move_col].idxmin()
        best_row = m_sub.loc[best_idx]
        worst_row = m_sub.loc[worst_idx]

        avg_high = float(m_sub['high_stretch_pct'].mean()) if 'high_stretch_pct' in m_sub else 0.0
        avg_low = float(m_sub['low_drop_pct'].mean()) if 'low_drop_pct' in m_sub else 0.0

        stats.append({
            'month_idx': m,
            'Month': MONTH_NAMES[m - 1],
            'Month_Full': MONTH_FULL_NAMES[m - 1],
            'Total_Years': tot,
            'Win_Rate': win_rate,
            'Avg_Move': avg_move,
            'Median_Move': med_move,
            'Pos_Count': pos_cnt,
            'Neg_Count': neg_cnt,
            'Best_Year': str(int(best_row['year'])),
            'Best_Move': float(best_row[move_col]),
            'Worst_Year': str(int(worst_row['year'])),
            'Worst_Move': float(worst_row[move_col]),
            'Avg_High_Stretch': avg_high,
            'Avg_Low_Drop': avg_low
        })
    return pd.DataFrame(stats)


def build_year_matrix(df, move_col='move_pct'):
    """
    Builds a Year x Month pivot table with annual return.
    """
    if df.empty:
        return pd.DataFrame()

    pivot = df.pivot(index='year', columns='month', values=move_col)
    pivot.columns = [MONTH_NAMES[m - 1] for m in pivot.columns]

    # Reindex to ensure all 12 months present
    for m_name in MONTH_NAMES:
        if m_name not in pivot.columns:
            pivot[m_name] = np.nan
    pivot = pivot[MONTH_NAMES]

    # Calculate Annual Return from Jan Open to Dec Close (or sum of compounding returns)
    annual_moves = []
    for yr in pivot.index:
        yr_df = df[df['year'] == yr].sort_values('month')
        if not yr_df.empty:
            yr_open = yr_df['open'].iloc[0]
            yr_close = yr_df['close'].iloc[-1]
            ann_ret = (yr_close - yr_open) / yr_open * 100.0 if yr_open > 0 else 0.0
        else:
            ann_ret = np.nan
        annual_moves.append(ann_ret)

    pivot['Full Year'] = annual_moves
    return pivot.sort_index(ascending=False)


# =============================================================
# PLOTLY VISUALIZATION BUILDERS
# =============================================================
def plot_seasonality_profile_chart(stats_df, title="Monthly Seasonality Profile"):
    """
    Creates a dual-metric chart showing Positive Win Rate (%) bars and Average Move (%) overlay.
    """
    fig = go.Figure()

    # Win Rate Bars
    colors = [
        '#00d26a' if wr >= 60.0 else ('#38bdf8' if wr >= 50.0 else ('#fbbf24' if wr >= 40.0 else '#f43f5e'))
        for wr in stats_df['Win_Rate']
    ]

    fig.add_trace(go.Bar(
        x=stats_df['Month'],
        y=stats_df['Win_Rate'],
        name='Positive Win Rate (%)',
        marker_color=colors,
        text=stats_df['Win_Rate'].apply(lambda x: f"{x:.1f}%"),
        textposition='outside',
        yaxis='y'
    ))

    # Average Move Line
    fig.add_trace(go.Scatter(
        x=stats_df['Month'],
        y=stats_df['Avg_Move'],
        name='Avg Return (%)',
        mode='lines+markers+text',
        line=dict(color='#fbbf24', width=3),
        marker=dict(size=8, color='#fbbf24'),
        text=stats_df['Avg_Move'].apply(lambda x: f"{x:+.1f}%"),
        textposition='top center',
        yaxis='y2'
    ))

    # 50% Baseline reference
    fig.add_hline(y=50.0, line_dash="dash", line_color="#94a3b8", line_width=1.5,
                  annotation_text="50% Coin Toss Baseline", annotation_position="top left")

    fig.update_layout(
        title=f"<b>{title}</b>",
        xaxis=dict(title="Calendar Month"),
        yaxis=dict(
            title="Positive Win Rate (%)",
            range=[0, 100],
            gridcolor="#232f48"
        ),
        yaxis2=dict(
            title="Average Return (%)",
            overlaying='y',
            side='right',
            gridcolor="#1e293b",
            showgrid=False
        ),
        template="plotly_dark",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def plot_interactive_heatmap(matrix_df, title="Year-by-Year Monthly Return Matrix (%)"):
    """
    Renders an institutional dark heatmap with exact return annotations.
    """
    if matrix_df.empty:
        return go.Figure()

    z_vals = matrix_df.values
    x_labels = list(matrix_df.columns)
    y_labels = [str(yr) for yr in matrix_df.index]

    # Text formatting
    text_vals = []
    for row in z_vals:
        text_row = []
        for v in row:
            if pd.isna(v):
                text_row.append("-")
            else:
                text_row.append(f"{v:+.1f}%")
        text_row.append("") if False else None
        text_vals.append(text_row)

    # Custom Divergent Colorscale: Red -> Dark Navy -> Green
    colorscale = [
        [0.0, '#ef4444'],    # Large drop (-15%)
        [0.35, '#7f1d1d'],   # Moderate drop (-5%)
        [0.48, '#1e293b'],   # Flat (-0.5%)
        [0.50, '#0f172a'],   # Zero (0%)
        [0.52, '#1e293b'],   # Flat (+0.5%)
        [0.65, '#065f46'],   # Moderate gain (+5%)
        [1.0, '#00d26a']     # Large gain (+15%)
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=x_labels,
        y=y_labels,
        text=text_vals,
        texttemplate="%{text}",
        textfont=dict(size=11, color="#ffffff"),
        colorscale=colorscale,
        zmin=-12.0,
        zmax=12.0,
        colorbar=dict(title="Return %", ticksuffix="%")
    ))

    fig.update_layout(
        title=f"<b>{title}</b>",
        xaxis=dict(title="Month", side="top"),
        yaxis=dict(title="Year", dtick=1, autorange="reversed"),
        template="plotly_dark",
        height=max(450, len(matrix_df) * 22 + 100),
        margin=dict(l=50, r=40, t=80, b=40)
    )
    return fig


def plot_sector_matrix_heatmap(sector_matrix_df, title="Sector Month-Wise Performance Heatmap (%)"):
    """
    Renders 20 Sectors x 12 Months Heatmap.
    """
    if sector_matrix_df.empty:
        return go.Figure()

    z_vals = sector_matrix_df.values
    x_labels = list(sector_matrix_df.columns)
    y_labels = list(sector_matrix_df.index)

    text_vals = []
    for row in z_vals:
        text_row = []
        for v in row:
            if pd.isna(v):
                text_row.append("-")
            else:
                text_row.append(f"{v:+.1f}%")
        text_vals.append(text_row)

    colorscale = [
        [0.0, '#ef4444'],
        [0.4, '#7f1d1d'],
        [0.5, '#0f172a'],
        [0.6, '#065f46'],
        [1.0, '#00d26a']
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=x_labels,
        y=y_labels,
        text=text_vals,
        texttemplate="%{text}",
        textfont=dict(size=11, color="#ffffff"),
        colorscale=colorscale,
        zmin=-8.0,
        zmax=8.0,
        colorbar=dict(title="Avg Move %", ticksuffix="%")
    ))

    fig.update_layout(
        title=f"<b>{title}</b>",
        xaxis=dict(title="Calendar Month", side="top"),
        yaxis=dict(title="Sector / Industry", dtick=1, autorange="reversed"),
        template="plotly_dark",
        height=max(480, len(sector_matrix_df) * 26 + 100),
        margin=dict(l=150, r=40, t=80, b=40)
    )
    return fig


# =============================================================
# MAIN UI SECTION RENDERER
# =============================================================
def render_seasonality_section():
    """
    Renders Section 4: Monthly Seasonality Analytics for Nifty 50, Sectors, and Stocks.
    """
    st.header("🗓️ Section 4: Monthly Seasonality Analytics (Nifty, Sectors & Stocks)")
    st.markdown("""
    Explore empirical calendar tendencies, historical win rates, and directional probabilities across **NIFTY 50 Index**,
    **20 Market Sectors**, and all **500 Nifty Equities** across three decades of market history.
    """)

    # Load Data
    with st.spinner("Loading high-speed monthly dataset..."):
        df_nifty_all = load_nifty_monthly_seasonality_df()
        stocks_pl_all = load_all_monthly_stocks_df()
        df_csv, sym_to_ind, sym_to_comp = load_metadata()

    if df_nifty_all.empty:
        st.error("⚠️ Nifty historical data not available.")
        return

    # Global Lookback Filter
    st.markdown("---")
    min_avail_yr = int(df_nifty_all['year'].min())
    max_avail_yr = int(df_nifty_all['year'].max())

    col_filter1, col_filter2 = st.columns([2.5, 3.5])
    with col_filter1:
        lookback_preset = st.radio(
            "⏳ Historical Lookback Scope:",
            [
                f"Full Available History ({min_avail_yr} – {max_avail_yr})",
                "Last 20 Years (2006 – Present)",
                "Last 10 Years (2016 – Present)",
                "Last 5 Years (2021 – Present)",
                "Custom Year Range"
            ],
            index=0,
            horizontal=False,
            key="sec4_lookback_preset"
        )

    with col_filter2:
        if "Last 5 Years" in lookback_preset:
            filter_min_yr = max(min_avail_yr, max_avail_yr - 4)
            filter_max_yr = max_avail_yr
        elif "Last 10 Years" in lookback_preset:
            filter_min_yr = max(min_avail_yr, max_avail_yr - 9)
            filter_max_yr = max_avail_yr
        elif "Last 20 Years" in lookback_preset:
            filter_min_yr = max(min_avail_yr, max_avail_yr - 19)
            filter_max_yr = max_avail_yr
        elif "Custom" in lookback_preset:
            yr_range = st.slider(
                "Select Year Range:",
                min_value=min_avail_yr,
                max_value=max_avail_yr,
                value=(max(min_avail_yr, 2010), max_avail_yr),
                key="sec4_custom_yr_slider"
            )
            filter_min_yr, filter_max_yr = yr_range
        else:
            filter_min_yr = min_avail_yr
            filter_max_yr = max_avail_yr

        span_years = filter_max_yr - filter_min_yr + 1
        st.markdown(f"""
        <div style="background: rgba(56, 189, 248, 0.08); border-left: 3px solid #38bdf8; padding: 10px 14px; border-radius: 6px; font-size: 13px; color: #cbd5e1; margin-top: 4px;">
            <strong>🎯 Active Analysis Horizon:</strong> <span style="color:#38bdf8; font-weight:700;">{filter_min_yr} – {filter_max_yr}</span> ({span_years} Calendar Years)<br>
            • <strong>Move Formula:</strong> <code>(Month Close - Month Open) / Month Open &times; 100</code><br>
            • <strong>Win Rate (%):</strong> Percentage of historical years where the candle closed GREEN (> 0.0%).
        </div>
        """, unsafe_allow_html=True)

    # Filter Datasets
    df_nifty = df_nifty_all[(df_nifty_all['year'] >= filter_min_yr) & (df_nifty_all['year'] <= filter_max_yr)].copy()
    if not stocks_pl_all.is_empty():
        stocks_pl = stocks_pl_all.filter((pl.col("year") >= filter_min_yr) & (pl.col("year") <= filter_max_yr))
    else:
        stocks_pl = pl.DataFrame()

    st.markdown("---")

    # 4 Main Feature Tabs
    s_tab1, s_tab2, s_tab3, s_tab4 = st.tabs([
        "📈 1. NIFTY 50 Index Seasonality",
        "🏢 2. Sectoral Seasonality (20 Sectors)",
        "🎯 3. Nifty 50 Stocks Seasonality",
        "🔍 4. Individual Stock Deep-Dive (500 Stocks)"
    ])

    # =============================================================
    # TAB 1: NIFTY 50 INDEX SEASONALITY
    # =============================================================
    with s_tab1:
        st.subheader("📈 NIFTY 50 Index Monthly Performance & Seasonality")
        st.markdown(f"Statistical track record of the benchmark **Nifty 50 Index** from **{filter_min_yr} to {filter_max_yr}** ({span_years} Years).")

        nifty_stats = compute_monthly_summary(df_nifty)

        # 12 Month KPI Grid (4 columns x 3 rows)
        st.markdown("#### 📅 Jan – Dec Seasonality Performance Cards")
        for row_idx in range(3):
            k_cols = st.columns(4)
            for col_idx in range(4):
                m_num = row_idx * 4 + col_idx  # 0..11
                m_row = nifty_stats.iloc[m_num]
                m_name = m_row['Month']
                wr = m_row['Win_Rate']
                avg_m = m_row['Avg_Move']
                med_m = m_row['Median_Move']
                pos_c = m_row['Pos_Count']
                neg_c = m_row['Neg_Count']
                tot_y = m_row['Total_Years']
                best_yr = m_row['Best_Year']
                best_v = m_row['Best_Move']
                worst_yr = m_row['Worst_Year']
                worst_v = m_row['Worst_Move']

                # Badge colors
                wr_color = "#00d26a" if wr >= 60.0 else ("#38bdf8" if wr >= 50.0 else ("#fbbf24" if wr >= 40.0 else "#f43f5e"))
                move_color = "#00d26a" if avg_m >= 0 else "#f43f5e"

                with k_cols[col_idx]:
                    card_html = f"""<div class="metric-card" style="margin-bottom: 8px;">
<div style="display:flex; justify-content:space-between; align-items:center;">
    <strong style="color: #f1f5f9; font-size: 15px;">{m_name} ({m_row['Month_Full']})</strong>
    <span style="background:rgba(255,255,255,0.06); padding:2px 6px; border-radius:4px; font-size:11px; color:#94a3b8;">{tot_y} Yrs</span>
</div>
<hr style="margin: 6px 0; border-color: #232f48;">
<div style="display:flex; justify-content:space-between; margin-bottom: 3px;">
    <span style="font-size:12px; color:#94a3b8;">Win Rate:</span>
    <strong style="color: {wr_color}; font-size:13px;">{wr:.1f}% ({pos_c}🟢 / {neg_c}🔴)</strong>
</div>
<div style="display:flex; justify-content:space-between; margin-bottom: 3px;">
    <span style="font-size:12px; color:#94a3b8;">Average Move:</span>
    <strong style="color: {move_color}; font-size:13px;">{avg_m:+.2f}%</strong>
</div>
<div style="display:flex; justify-content:space-between; margin-bottom: 3px;">
    <span style="font-size:12px; color:#94a3b8;">Median Move:</span>
    <span style="color: #cbd5e1; font-size:12px;">{med_m:+.2f}%</span>
</div>
<div style="display:flex; justify-content:space-between; font-size:11px; color:#94a3b8; margin-top: 4px; padding-top: 4px; border-top: 1px dashed #1e293b;">
    <span>Best: <strong style="color:#00d26a;">{best_v:+.1f}%</strong> ({best_yr})</span>
    <span>Worst: <strong style="color:#f43f5e;">{worst_v:+.1f}%</strong> ({worst_yr})</span>
</div>
</div>"""
                    st.markdown(card_html, unsafe_allow_html=True)

        # Plotly Bar Chart
        st.markdown("---")
        st.markdown("#### 📊 Visual Seasonality Distribution (Win Rate % & Average Return %)")
        fig_nifty_bars = plot_seasonality_profile_chart(
            nifty_stats,
            title=f"NIFTY 50 Monthly Seasonality ({filter_min_yr} – {filter_max_yr})"
        )
        st.plotly_chart(fig_nifty_bars, width="stretch")

        # Key Observations
        sorted_by_wr = nifty_stats.sort_values('Win_Rate', ascending=False)
        top3_months = sorted_by_wr.head(3)
        bot3_months = sorted_by_wr.tail(3)

        o_col1, o_col2 = st.columns(2)
        with o_col1:
            st.markdown(f"""
            <div style="background: rgba(0, 210, 106, 0.08); border: 1px solid rgba(0, 210, 106, 0.3); border-radius: 8px; padding: 12px 16px;">
                <strong style="color: #00d26a; font-size: 14px;">🟢 Historically Strongest Months:</strong>
                <ul style="font-size: 13px; color: #cbd5e1; margin-top: 6px; margin-bottom: 0;">
                    <li><strong>{top3_months.iloc[0]['Month_Full']}:</strong> {top3_months.iloc[0]['Win_Rate']:.1f}% Win Rate | Avg Move: <strong style="color:#00d26a;">{top3_months.iloc[0]['Avg_Move']:+.2f}%</strong></li>
                    <li><strong>{top3_months.iloc[1]['Month_Full']}:</strong> {top3_months.iloc[1]['Win_Rate']:.1f}% Win Rate | Avg Move: <strong style="color:#00d26a;">{top3_months.iloc[1]['Avg_Move']:+.2f}%</strong></li>
                    <li><strong>{top3_months.iloc[2]['Month_Full']}:</strong> {top3_months.iloc[2]['Win_Rate']:.1f}% Win Rate | Avg Move: <strong style="color:#00d26a;">{top3_months.iloc[2]['Avg_Move']:+.2f}%</strong></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with o_col2:
            st.markdown(f"""
            <div style="background: rgba(244, 63, 94, 0.08); border: 1px solid rgba(244, 63, 94, 0.3); border-radius: 8px; padding: 12px 16px;">
                <strong style="color: #f43f5e; font-size: 14px;">🔴 Historically Challenging Months:</strong>
                <ul style="font-size: 13px; color: #cbd5e1; margin-top: 6px; margin-bottom: 0;">
                    <li><strong>{bot3_months.iloc[0]['Month_Full']}:</strong> {bot3_months.iloc[0]['Win_Rate']:.1f}% Win Rate | Avg Move: <strong style="color:#f43f5e;">{bot3_months.iloc[0]['Avg_Move']:+.2f}%</strong></li>
                    <li><strong>{bot3_months.iloc[1]['Month_Full']}:</strong> {bot3_months.iloc[1]['Win_Rate']:.1f}% Win Rate | Avg Move: <strong style="color:#f43f5e;">{bot3_months.iloc[1]['Avg_Move']:+.2f}%</strong></li>
                    <li><strong>{bot3_months.iloc[2]['Month_Full']}:</strong> {bot3_months.iloc[2]['Win_Rate']:.1f}% Win Rate | Avg Move: <strong style="color:#f43f5e;">{bot3_months.iloc[2]['Avg_Move']:+.2f}%</strong></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # Full Year-by-Year Historical Return Heatmap Matrix
        st.markdown("---")
        st.markdown("#### 🗺️ Year-by-Year Monthly Return Matrix (1996 – 2026)")
        nifty_matrix = build_year_matrix(df_nifty)
        if not nifty_matrix.empty:
            fig_nifty_matrix = plot_interactive_heatmap(
                nifty_matrix,
                title=f"NIFTY 50 Monthly Returns Matrix ({filter_min_yr} – {filter_max_yr})"
            )
            st.plotly_chart(fig_nifty_matrix, width="stretch")

            # Data Table expander
            with st.expander("📋 View Raw Numerical Table (Years & Returns)", expanded=False):
                styled_df = nifty_matrix.copy()
                for c in styled_df.columns:
                    styled_df[c] = styled_df[c].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "-")
                st.dataframe(styled_df, width="stretch")

    # =============================================================
    # TAB 2: SECTORAL SEASONALITY (20 SECTORS)
    # =============================================================
    with s_tab2:
        st.subheader("🏢 Sectoral Seasonality & Industry Monthly Patterns")
        st.markdown("Month-by-month comparative performance across all **20 NSE Industry Sectors** from `ind_nifty500list.csv`.")

        if stocks_pl.is_empty():
            st.warning("Stock monthly parquets not found. Please sync stocks in Section 3.")
        else:
            # Month Selection for Sector Breakdown
            sec_col1, sec_col2, sec_col3 = st.columns([1.8, 1.8, 2.4])
            with sec_col1:
                cur_month_idx = pd.Timestamp.now().month
                chosen_sec_month_name = st.selectbox(
                    "📅 Select Calendar Month:",
                    MONTH_FULL_NAMES,
                    index=cur_month_idx - 1,
                    key="sec_chosen_month"
                )
                chosen_sec_month = MONTH_FULL_NAMES.index(chosen_sec_month_name) + 1

            with sec_col2:
                sort_sec_by = st.selectbox(
                    "Rank Sectors By:",
                    ["Average Move (%)", "Win Rate (%)", "Median Move (%)"],
                    index=0,
                    key="sec_sort_by"
                )

            with sec_col3:
                st.markdown(f"""
                <div style="font-size:12px; color:#94a3b8; margin-top:24px;">
                    Evaluating performance of <strong>500 stocks</strong> across <strong>{span_years} years</strong> for <strong>{chosen_sec_month_name}</strong>.
                </div>
                """, unsafe_allow_html=True)

            # Compute Sector stats for chosen month
            stocks_m = stocks_pl.filter(pl.col("month") == chosen_sec_month)
            sec_stats = stocks_m.group_by("industry").agg([
                pl.len().alias("sample_candles"),
                pl.col("symbol").n_unique().alias("stocks_in_sector"),
                pl.col("is_positive").mean().mul(100.0).alias("win_rate"),
                pl.col("move_pct").mean().alias("avg_move"),
                pl.col("move_pct").median().alias("median_move"),
                pl.col("move_pct").max().alias("max_gain"),
                pl.col("move_pct").min().alias("max_drop")
            ]).to_pandas()

            # Find top performing stock in each sector for this month
            stock_m_agg = stocks_m.group_by(["industry", "symbol"]).agg([
                pl.col("move_pct").mean().alias("stock_avg_move")
            ]).to_pandas()

            top_stocks_per_sec = {}
            for ind, group in stock_m_agg.groupby("industry"):
                best_s = group.loc[group['stock_avg_move'].idxmax()]
                top_stocks_per_sec[ind] = f"{best_s['symbol']} ({best_s['stock_avg_move']:+.1f}%)"

            sec_stats['Top Performer Stock'] = sec_stats['industry'].map(top_stocks_per_sec)

            # Sort
            sort_key = "avg_move" if "Average" in sort_sec_by else ("win_rate" if "Win" in sort_sec_by else "median_move")
            sec_stats = sec_stats.sort_values(sort_key, ascending=False).reset_index(drop=True)

            # Leaderboard Display
            st.markdown(f"#### 🏆 Sector Performance Leaderboard in {chosen_sec_month_name}")
            show_sec_df = pd.DataFrame({
                "Sector / Industry": sec_stats['industry'],
                "Stocks": sec_stats['stocks_in_sector'],
                "Win Rate (%)": sec_stats['win_rate'].apply(lambda x: f"{x:.1f}%"),
                "Avg Move (%)": sec_stats['avg_move'].apply(lambda x: f"{x:+.2f}%"),
                "Median Move (%)": sec_stats['median_move'].apply(lambda x: f"{x:+.2f}%"),
                "Best Historical Gain": sec_stats['max_gain'].apply(lambda x: f"{x:+.1f}%"),
                "Max Drawdown": sec_stats['max_drop'].apply(lambda x: f"{x:+.1f}%"),
                "Top Performer Stock (Avg)": sec_stats['Top Performer Stock']
            })
            st.dataframe(show_sec_df, width="stretch", hide_index=True)

            # Full 20 Sectors x 12 Months Seasonality Heatmap
            st.markdown("---")
            st.markdown("#### 🗺️ Full 12-Month Sector Seasonality Matrix (Heatmap)")
            sec_all_m = stocks_pl.group_by(["industry", "month"]).agg([
                pl.col("move_pct").mean().alias("avg_move"),
                pl.col("is_positive").mean().mul(100.0).alias("win_rate")
            ]).to_pandas()

            matrix_metric = st.radio(
                "Heatmap Metric:",
                ["Average Monthly Return (%)", "Positive Win Rate (%)"],
                horizontal=True,
                key="sec_heatmap_metric"
            )
            val_col = "avg_move" if "Average" in matrix_metric else "win_rate"
            sec_matrix = sec_all_m.pivot(index="industry", columns="month", values=val_col)
            sec_matrix.columns = [MONTH_NAMES[m - 1] for m in sec_matrix.columns]
            for m_name in MONTH_NAMES:
                if m_name not in sec_matrix.columns:
                    sec_matrix[m_name] = np.nan
            sec_matrix = sec_matrix[MONTH_NAMES]

            fig_sec_heat = plot_sector_matrix_heatmap(
                sec_matrix,
                title=f"All Sectors Month-Wise {matrix_metric} ({filter_min_yr} – {filter_max_yr})"
            )
            st.plotly_chart(fig_sec_heat, width="stretch")

            # Sector Drill-down
            st.markdown("---")
            st.markdown("#### 🔍 Sector Deep-Dive (Inspect Constituent Stocks)")
            all_inds = sorted(sec_stats['industry'].unique().tolist())
            chosen_ind_drill = st.selectbox("Select Sector to Drill Down:", all_inds, key="sec_drill_pick")

            sec_stocks_pl = stocks_pl.filter(pl.col("industry") == chosen_ind_drill)
            sec_stocks_m = sec_stocks_pl.filter(pl.col("month") == chosen_sec_month)
            drill_stock_stats = sec_stocks_m.group_by(["symbol", "company_name"]).agg([
                pl.len().alias("years"),
                pl.col("is_positive").mean().mul(100.0).alias("win_rate"),
                pl.col("move_pct").mean().alias("avg_move"),
                pl.col("move_pct").median().alias("median_move"),
                pl.col("move_pct").max().alias("max_gain"),
                pl.col("move_pct").min().alias("max_drop")
            ]).sort("win_rate", descending=True).to_pandas()

            show_drill = pd.DataFrame({
                "Symbol": drill_stock_stats['symbol'],
                "Company Name": drill_stock_stats['company_name'],
                "Sample Years": drill_stock_stats['years'],
                "Win Rate (%)": drill_stock_stats['win_rate'].apply(lambda x: f"{x:.1f}%"),
                "Avg Move (%)": drill_stock_stats['avg_move'].apply(lambda x: f"{x:+.2f}%"),
                "Median Move (%)": drill_stock_stats['median_move'].apply(lambda x: f"{x:+.2f}%"),
                "Best Gain": drill_stock_stats['max_gain'].apply(lambda x: f"{x:+.1f}%"),
                "Max Drop": drill_stock_stats['max_drop'].apply(lambda x: f"{x:+.1f}%")
            })
            st.markdown(f"**Constituent Stocks of `{chosen_ind_drill}` in {chosen_sec_month_name} ({len(show_drill)} Stocks):**")
            st.dataframe(show_drill, width="stretch", hide_index=True)

    # =============================================================
    # TAB 3: NIFTY 50 STOCKS SEASONALITY
    # =============================================================
    with s_tab3:
        st.subheader("🎯 Nifty 50 Constituents Monthly Performance")
        st.markdown(f"Seasonality audit of all **50 Nifty 50 constituent stocks** across **{filter_min_yr} to {filter_max_yr}**.")

        if stocks_pl.is_empty():
            st.warning("Stock monthly parquets not found.")
        else:
            nifty50_pl = stocks_pl.filter(pl.col("is_nifty50"))

            # Controls
            n50_c1, n50_c2, n50_c3 = st.columns([1.8, 1.8, 2.4])
            with n50_c1:
                chosen_n50_month_name = st.selectbox(
                    "📅 Scout Calendar Month:",
                    MONTH_FULL_NAMES,
                    index=cur_month_idx - 1,
                    key="n50_chosen_month"
                )
                chosen_n50_m = MONTH_FULL_NAMES.index(chosen_n50_month_name) + 1

            with n50_c2:
                n50_sort_by = st.selectbox(
                    "Sort Nifty 50 By:",
                    ["Win Rate (%)", "Average Move (%)", "Median Move (%)"],
                    index=0,
                    key="n50_sort_by"
                )

            with n50_c3:
                n50_filter_tag = st.selectbox(
                    "Quick Reliability Filter:",
                    [
                        "All 50 Stocks",
                        "🟢 Highly Consistent (Win Rate >= 65%)",
                        "🚀 Highest Movers (Avg Return >= +3.0%)",
                        "🔴 Vulnerable / Bearish (Win Rate <= 40%)"
                    ],
                    index=0,
                    key="n50_filter_tag"
                )

            # Aggregate Nifty 50 for chosen month
            n50_month_sub = nifty50_pl.filter(pl.col("month") == chosen_n50_m)
            n50_agg = n50_month_sub.group_by(["symbol", "company_name", "industry"]).agg([
                pl.len().alias("sample_years"),
                pl.col("is_positive").mean().mul(100.0).alias("win_rate"),
                pl.col("move_pct").mean().alias("avg_move"),
                pl.col("move_pct").median().alias("median_move"),
                pl.col("move_pct").max().alias("max_gain"),
                pl.col("move_pct").min().alias("max_drop")
            ]).to_pandas()

            # Filter tags
            if "Highly Consistent" in n50_filter_tag:
                n50_agg = n50_agg[n50_agg['win_rate'] >= 65.0]
            elif "Highest Movers" in n50_filter_tag:
                n50_agg = n50_agg[n50_agg['avg_move'] >= 3.0]
            elif "Vulnerable" in n50_filter_tag:
                n50_agg = n50_agg[n50_agg['win_rate'] <= 40.0]

            sort_col = "win_rate" if "Win" in n50_sort_by else ("avg_move" if "Average" in n50_sort_by else "median_move")
            n50_agg = n50_agg.sort_values(sort_col, ascending=False).reset_index(drop=True)

            st.markdown(f"#### 🏆 Nifty 50 Seasonality Leaderboard in {chosen_n50_month_name} ({len(n50_agg)} Stocks)")
            show_n50 = pd.DataFrame({
                "Symbol": n50_agg['symbol'],
                "Company Name": n50_agg['company_name'],
                "Sector": n50_agg['industry'],
                "Sample Years": n50_agg['sample_years'],
                "Win Rate (%)": n50_agg['win_rate'].apply(lambda x: f"{x:.1f}%"),
                "Avg Move (%)": n50_agg['avg_move'].apply(lambda x: f"{x:+.2f}%"),
                "Median Move (%)": n50_agg['median_move'].apply(lambda x: f"{x:+.2f}%"),
                "Best Gain": n50_agg['max_gain'].apply(lambda x: f"{x:+.1f}%"),
                "Max Drop": n50_agg['max_drop'].apply(lambda x: f"{x:+.1f}%")
            })
            st.dataframe(show_n50, width="stretch", hide_index=True)

            # Full 50 Stocks x 12 Months Heatmap
            st.markdown("---")
            st.markdown("#### 🗺️ Nifty 50 Full Seasonality Matrix Heatmap (All 12 Months)")
            n50_all_m = nifty50_pl.group_by(["symbol", "month"]).agg([
                pl.col("move_pct").mean().alias("avg_move"),
                pl.col("is_positive").mean().mul(100.0).alias("win_rate")
            ]).to_pandas()

            n50_mat_metric = st.radio(
                "Nifty 50 Heatmap Metric:",
                ["Average Monthly Return (%)", "Positive Win Rate (%)"],
                horizontal=True,
                key="n50_heat_metric"
            )
            v_col = "avg_move" if "Average" in n50_mat_metric else "win_rate"
            n50_matrix = n50_all_m.pivot(index="symbol", columns="month", values=v_col)
            n50_matrix.columns = [MONTH_NAMES[m - 1] for m in n50_matrix.columns]
            for m_name in MONTH_NAMES:
                if m_name not in n50_matrix.columns:
                    n50_matrix[m_name] = np.nan
            n50_matrix = n50_matrix[MONTH_NAMES]

            fig_n50_heat = plot_sector_matrix_heatmap(
                n50_matrix,
                title=f"Nifty 50 Stocks Month-Wise {n50_mat_metric} ({filter_min_yr} – {filter_max_yr})"
            )
            st.plotly_chart(fig_n50_heat, width="stretch")

    # =============================================================
    # TAB 4: INDIVIDUAL STOCK DEEP-DIVE (500 STOCKS)
    # =============================================================
    with s_tab4:
        st.subheader("🔍 Individual Stock Seasonality Deep-Dive (Nifty 500)")
        st.markdown("Search or select **any of the 500 Nifty Equities** for complete 12-month historical breakdown, Win Rate statistics, and Year-by-Year heatmaps.")

        if stocks_pl.is_empty():
            st.warning("Stock monthly parquets not found.")
        else:
            all_syms = sorted(stocks_pl["symbol"].unique().to_list())
            # Format with company name for search
            sym_display_list = [f"{s} - {sym_to_comp.get(s, s)} ({sym_to_ind.get(s, 'Stock')})" for s in all_syms]

            default_idx = 0
            for idx, item in enumerate(all_syms):
                if item == "RELIANCE":
                    default_idx = idx
                    break

            picked_display = st.selectbox(
                "🔎 Pick or Type Stock Symbol:",
                sym_display_list,
                index=default_idx,
                key="sec4_picked_stock_deep"
            )
            picked_sym = picked_display.split(" - ")[0].strip()

            # Filter for picked stock
            stock_pl_sub = stocks_pl.filter(pl.col("symbol") == picked_sym).to_pandas()

            if stock_pl_sub.empty:
                st.warning(f"No monthly data found for {picked_sym}.")
            else:
                comp_name = sym_to_comp.get(picked_sym, picked_sym)
                ind_name = sym_to_ind.get(picked_sym, "Equity")
                is_n50_flag = picked_sym in set(NIFTY50_CONSTITUENTS)
                n50_badge = "<span class='badge badge-success' style='margin-left:8px;'>NIFTY 50</span>" if is_n50_flag else ""

                st.markdown(f"""
                <div style="font-size: 18px; font-weight: 700; color: #38bdf8; margin: 10px 0 4px 0;">
                    {picked_sym} — {comp_name} {n50_badge}
                </div>
                <div style="font-size: 13px; color: #94a3b8; margin-bottom: 14px;">
                    Sector: <strong style="color: #cbd5e1;">{ind_name}</strong> |
                    Sample Years: <strong style="color: #f1f5f9;">{len(stock_pl_sub['year'].unique())} Years</strong> |
                    Total Monthly Candles: <strong style="color: #f1f5f9;">{len(stock_pl_sub)} Months</strong>
                </div>
                """, unsafe_allow_html=True)

                stock_stats = compute_monthly_summary(stock_pl_sub)

                # Best Month & Worst Month Highlights
                s_by_wr = stock_stats.sort_values('Win_Rate', ascending=False)
                best_m = s_by_wr.iloc[0]
                worst_m = s_by_wr.iloc[-1]

                hc1, hc2, hc3, hc4 = st.columns(4)
                with hc1:
                    st.markdown(f"""<div class='metric-card'><span class='metric-label'>Strongest Month</span><div class='metric-value' style='color:#00d26a;'>{best_m['Month']} ({best_m['Win_Rate']:.1f}%)</div><span style='font-size:11px;color:#94a3b8;'>Avg Return: {best_m['Avg_Move']:+.2f}%</span></div>""", unsafe_allow_html=True)
                with hc2:
                    st.markdown(f"""<div class='metric-card'><span class='metric-label'>Weakest Month</span><div class='metric-value' style='color:#f43f5e;'>{worst_m['Month']} ({worst_m['Win_Rate']:.1f}%)</div><span style='font-size:11px;color:#94a3b8;'>Avg Return: {worst_m['Avg_Move']:+.2f}%</span></div>""", unsafe_allow_html=True)
                with hc3:
                    overall_wr = (stock_pl_sub['is_positive'].mean()) * 100.0
                    st.markdown(f"""<div class='metric-card'><span class='metric-label'>Overall Green Months</span><div class='metric-value' style='color:#38bdf8;'>{overall_wr:.1f}%</div><span style='font-size:11px;color:#94a3b8;'>Across all historical months</span></div>""", unsafe_allow_html=True)
                with hc4:
                    overall_avg = stock_pl_sub['move_pct'].mean()
                    st.markdown(f"""<div class='metric-card'><span class='metric-label'>Overall Monthly Avg</span><div class='metric-value' style='color:#fbbf24;'>{overall_avg:+.2f}%</div><span style='font-size:11px;color:#94a3b8;'>Average monthly move</span></div>""", unsafe_allow_html=True)

                # 12 Month Table
                st.markdown("---")
                st.markdown(f"#### 📅 12-Month Performance Seasonality Summary for **{picked_sym}**")
                show_stk_df = pd.DataFrame({
                    "Month": stock_stats['Month_Full'],
                    "Sample Years": stock_stats['Total_Years'],
                    "Win Rate (%)": stock_stats['Win_Rate'].apply(lambda x: f"{x:.1f}%"),
                    "Avg Move (%)": stock_stats['Avg_Move'].apply(lambda x: f"{x:+.2f}%"),
                    "Median Move (%)": stock_stats['Median_Move'].apply(lambda x: f"{x:+.2f}%"),
                    "Green / Red Count": stock_stats.apply(lambda r: f"{r['Pos_Count']}🟢 / {r['Neg_Count']}🔴", axis=1),
                    "Best Year Gain": stock_stats.apply(lambda r: f"{r['Best_Move']:+.1f}% ({r['Best_Year']})", axis=1),
                    "Max Drawdown": stock_stats.apply(lambda r: f"{r['Worst_Move']:+.1f}% ({r['Worst_Year']})", axis=1)
                })
                st.dataframe(show_stk_df, width="stretch", hide_index=True)

                # Plotly Visual Bar Chart
                fig_stk_bars = plot_seasonality_profile_chart(
                    stock_stats,
                    title=f"{picked_sym} Monthly Seasonality Profile ({filter_min_yr} – {filter_max_yr})"
                )
                st.plotly_chart(fig_stk_bars, width="stretch")

                # Year-by-Year Historical Return Heatmap Matrix
                st.markdown("---")
                st.markdown(f"#### 🗺️ {picked_sym} Year-by-Year Monthly Return Matrix (%)")
                stk_matrix = build_year_matrix(stock_pl_sub)
                if not stk_matrix.empty:
                    fig_stk_matrix = plot_interactive_heatmap(
                        stk_matrix,
                        title=f"{picked_sym} Monthly Returns Matrix ({filter_min_yr} – {filter_max_yr})"
                    )
                    st.plotly_chart(fig_stk_matrix, width="stretch")

                    with st.expander(f"📋 View Numerical Return Matrix for {picked_sym}", expanded=False):
                        s_df = stk_matrix.copy()
                        for c in s_df.columns:
                            s_df[c] = s_df[c].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "-")
                        st.dataframe(s_df, width="stretch")
