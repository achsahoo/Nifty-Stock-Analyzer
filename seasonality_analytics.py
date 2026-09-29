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
    Returns a DataFrame with 12 rows (Jan..Dec) containing positive frequency, avg/median move,
    avg up move, avg down move, best, worst.
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
                'Pos_Pct': 0.0,
                'Neg_Pct': 0.0,
                'Win_Rate': 0.0,  # alias for backward safety
                'Avg_Move': 0.0,
                'Median_Move': 0.0,
                'Pos_Count': 0,
                'Neg_Count': 0,
                'Avg_Up_Move': 0.0,
                'Avg_Down_Move': 0.0,
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
        pos_pct = (pos_cnt / tot) * 100.0
        neg_pct = (neg_cnt / tot) * 100.0
        avg_move = float(m_sub[move_col].mean())
        med_move = float(m_sub[move_col].median())

        up_moves = m_sub.loc[pos_mask, move_col]
        dn_moves = m_sub.loc[~pos_mask, move_col]
        avg_up = float(up_moves.mean()) if len(up_moves) > 0 else 0.0
        avg_dn = float(dn_moves.mean()) if len(dn_moves) > 0 else 0.0

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
            'Pos_Pct': pos_pct,
            'Neg_Pct': neg_pct,
            'Win_Rate': pos_pct,  # backward compatibility
            'Avg_Move': avg_move,
            'Median_Move': med_move,
            'Pos_Count': pos_cnt,
            'Neg_Count': neg_cnt,
            'Avg_Up_Move': avg_up,
            'Avg_Down_Move': avg_dn,
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
    Creates a dual-metric chart showing % Positive Months bars and Average Move (%) overlay.
    """
    fig = go.Figure()

    col_to_use = 'Pos_Pct' if 'Pos_Pct' in stats_df.columns else 'Win_Rate'
    colors = [
        '#00d26a' if p >= 60.0 else ('#38bdf8' if p >= 50.0 else ('#fbbf24' if p >= 40.0 else '#f43f5e'))
        for p in stats_df[col_to_use]
    ]

    fig.add_trace(go.Bar(
        x=stats_df['Month'],
        y=stats_df[col_to_use],
        name='% Positive Months (Green)',
        marker_color=colors,
        text=stats_df[col_to_use].apply(lambda x: f"{x:.1f}%"),
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
                  annotation_text="50% Even Split Baseline", annotation_position="top left")

    fig.update_layout(
        title=f"<b>{title}</b>",
        xaxis=dict(title="Calendar Month"),
        yaxis=dict(
            title="% Positive Months (Green)",
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

    # Modern Styling for Seasonality DataTables and Visual Cards
    st.markdown("""
    <style>
        /* Sleek Modern Dataframe styling */
        [data-testid="stDataFrame"] {
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 4px 18px -2px rgba(0, 0, 0, 0.45);
        }
        /* Modern Info Pill Banner */
        .modern-matrix-banner {
            display: flex;
            align-items: center;
            justify-content: flex-start;
            flex-wrap: wrap;
            gap: 10px;
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.75) 100%);
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 8px;
            padding: 8px 14px;
            margin: 6px 0 12px 0;
        }
        .modern-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 3px 10px;
            border-radius: 9999px;
            font-size: 11.5px;
            color: #94a3b8;
        }
        .modern-pill strong {
            color: #38bdf8;
        }
    </style>
    """, unsafe_allow_html=True)

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
            • <strong>Directional Probability:</strong> Historical frequency of positive green months vs. negative red months and their respective average move magnitudes.
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
        "🏢 2. Sector Macro Overview (20 Sectors)",
        "🎯 3. Stock Seasonality (Nifty 50, Sectoral & All Stocks)",
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
                wr = m_row['Pos_Pct']
                avg_m = m_row['Avg_Move']
                med_m = m_row['Median_Move']
                pos_c = m_row['Pos_Count']
                neg_c = m_row['Neg_Count']
                tot_y = m_row['Total_Years']
                best_yr = m_row['Best_Year']
                best_v = m_row['Best_Move']
                worst_yr = m_row['Worst_Year']
                worst_v = m_row['Worst_Move']
                avg_up = m_row['Avg_Up_Move']
                avg_dn = m_row['Avg_Down_Move']

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
    <span style="font-size:12px; color:#94a3b8;">Positive Years:</span>
    <strong style="color: {wr_color}; font-size:13px;">{pos_c}🟢 / {tot_y} Yrs ({wr:.1f}%)</strong>
</div>
<div style="display:flex; justify-content:space-between; margin-bottom: 3px;">
    <span style="font-size:12px; color:#94a3b8;">Average Move:</span>
    <strong style="color: {move_color}; font-size:13px;">{avg_m:+.2f}%</strong>
</div>
<div style="display:flex; justify-content:space-between; margin-bottom: 3px;">
    <span style="font-size:12px; color:#94a3b8;">Median Move:</span>
    <span style="color: #cbd5e1; font-size:12px;">{med_m:+.2f}%</span>
</div>
<div style="display:flex; justify-content:space-between; margin-bottom: 3px;">
    <span style="font-size:12px; color:#94a3b8;">Avg Up / Down:</span>
    <span style="font-size:11.5px;"><strong style="color:#00d26a;">{avg_up:+.2f}%</strong> / <strong style="color:#f43f5e;">{avg_dn:+.2f}%</strong></span>
</div>
<div style="display:flex; justify-content:space-between; font-size:11px; color:#94a3b8; margin-top: 4px; padding-top: 4px; border-top: 1px dashed #1e293b;">
    <span>Best: <strong style="color:#00d26a;">{best_v:+.1f}%</strong> ({best_yr})</span>
    <span>Worst: <strong style="color:#f43f5e;">{worst_v:+.1f}%</strong> ({worst_yr})</span>
</div>
</div>"""
                    st.markdown(card_html, unsafe_allow_html=True)

        # Plotly Bar Chart
        st.markdown("---")
        st.markdown("#### 📊 Visual Seasonality Distribution (% Positive Months & Average Return %)")
        fig_nifty_bars = plot_seasonality_profile_chart(
            nifty_stats,
            title=f"NIFTY 50 Monthly Seasonality ({filter_min_yr} – {filter_max_yr})"
        )
        st.plotly_chart(fig_nifty_bars, width="stretch")

        # Key Observations
        sorted_by_wr = nifty_stats.sort_values('Pos_Pct', ascending=False)
        top3_months = sorted_by_wr.head(3)
        bot3_months = sorted_by_wr.tail(3)

        o_col1, o_col2 = st.columns(2)
        with o_col1:
            st.markdown(f"""
            <div style="background: rgba(0, 210, 106, 0.08); border: 1px solid rgba(0, 210, 106, 0.3); border-radius: 8px; padding: 12px 16px;">
                <strong style="color: #00d26a; font-size: 14px;">🟢 Historically Positive Tendency Months:</strong>
                <ul style="font-size: 13px; color: #cbd5e1; margin-top: 6px; margin-bottom: 0;">
                    <li><strong>{top3_months.iloc[0]['Month_Full']}:</strong> {top3_months.iloc[0]['Pos_Pct']:.1f}% Positive Months | Avg Move: <strong style="color:#00d26a;">{top3_months.iloc[0]['Avg_Move']:+.2f}%</strong> (Up Avg: {top3_months.iloc[0]['Avg_Up_Move']:+.2f}%)</li>
                    <li><strong>{top3_months.iloc[1]['Month_Full']}:</strong> {top3_months.iloc[1]['Pos_Pct']:.1f}% Positive Months | Avg Move: <strong style="color:#00d26a;">{top3_months.iloc[1]['Avg_Move']:+.2f}%</strong> (Up Avg: {top3_months.iloc[1]['Avg_Up_Move']:+.2f}%)</li>
                    <li><strong>{top3_months.iloc[2]['Month_Full']}:</strong> {top3_months.iloc[2]['Pos_Pct']:.1f}% Positive Months | Avg Move: <strong style="color:#00d26a;">{top3_months.iloc[2]['Avg_Move']:+.2f}%</strong> (Up Avg: {top3_months.iloc[2]['Avg_Up_Move']:+.2f}%)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with o_col2:
            st.markdown(f"""
            <div style="background: rgba(244, 63, 94, 0.08); border: 1px solid rgba(244, 63, 94, 0.3); border-radius: 8px; padding: 12px 16px;">
                <strong style="color: #f43f5e; font-size: 14px;">🔴 Historically Vulnerable Months:</strong>
                <ul style="font-size: 13px; color: #cbd5e1; margin-top: 6px; margin-bottom: 0;">
                    <li><strong>{bot3_months.iloc[0]['Month_Full']}:</strong> {bot3_months.iloc[0]['Pos_Pct']:.1f}% Positive Months | Avg Move: <strong style="color:#f43f5e;">{bot3_months.iloc[0]['Avg_Move']:+.2f}%</strong> (Down Avg: {bot3_months.iloc[0]['Avg_Down_Move']:+.2f}%)</li>
                    <li><strong>{bot3_months.iloc[1]['Month_Full']}:</strong> {bot3_months.iloc[1]['Pos_Pct']:.1f}% Positive Months | Avg Move: <strong style="color:#f43f5e;">{bot3_months.iloc[1]['Avg_Move']:+.2f}%</strong> (Down Avg: {bot3_months.iloc[1]['Avg_Down_Move']:+.2f}%)</li>
                    <li><strong>{bot3_months.iloc[2]['Month_Full']}:</strong> {bot3_months.iloc[2]['Pos_Pct']:.1f}% Positive Months | Avg Move: <strong style="color:#f43f5e;">{bot3_months.iloc[2]['Avg_Move']:+.2f}%</strong> (Down Avg: {bot3_months.iloc[2]['Avg_Down_Move']:+.2f}%)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # Full Year-by-Year Historical Return Table
        st.markdown("---")
        st.markdown(f"#### 📅 Year-by-Year Monthly Return Table ({filter_min_yr} – {filter_max_yr})")
        nifty_matrix = build_year_matrix(df_nifty)
        if not nifty_matrix.empty:
            styled_df = nifty_matrix.reset_index()
            styled_df['year'] = styled_df['year'].astype(str)
            for c in MONTH_NAMES + ['Full Year']:
                if c in styled_df.columns:
                    styled_df[c] = styled_df[c].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "-")
            year_cfg = {"year": st.column_config.TextColumn("📌 Year", pinned=True)}
            for m in MONTH_NAMES:
                year_cfg[m] = st.column_config.TextColumn(m)
            year_cfg["Full Year"] = st.column_config.TextColumn("Full Year")
            st.dataframe(styled_df, column_config=year_cfg, width="stretch", hide_index=True)

    # =============================================================
    # TAB 2: SECTORAL SEASONALITY (20 SECTORS)
    # =============================================================
    with s_tab2:
        st.subheader("🏢 Sectoral Seasonality & Industry Monthly Patterns")
        st.markdown("Month-by-month comparative performance across all **20 NSE Industry Sectors** from `ind_nifty500list.csv`.")

        if stocks_pl.is_empty():
            st.warning("Stock monthly parquets not found. Please sync stocks in Section 3.")
        else:
            sec_view_mode = st.radio(
                "📊 Choose Display View:",
                [
                    "📅 Month-by-Month Full Matrix (Jan – Dec)",
                    "🎯 Single Month Sector Detail"
                ],
                index=0,
                horizontal=True,
                key="sec_tab_view_mode"
            )

            # =========================================================
            # VIEW 1: 12-MONTH FULL MATRIX (JAN – DEC)
            # =========================================================
            if "Month-by-Month Full Matrix" in sec_view_mode:
                st.markdown(f"#### 📅 20 Sectors Month-by-Month Seasonality Table ({filter_min_yr} – {filter_max_yr})")
                st.caption("Shows monthly breakdown for each sector: exact positive vs. negative counts out of total evaluated months.")

                # 1. 20 Sectors Matrix
                sec_yearly = stocks_pl.group_by(["industry", "year", "month"]).agg(pl.col("move_pct").mean().alias("sec_m_move"))
                sec_m_stats = sec_yearly.group_by(["industry", "month"]).agg([
                    pl.len().alias("m_tot"),
                    (pl.col("sec_m_move") > 0).sum().alias("m_pos"),
                    (pl.col("sec_m_move") <= 0).sum().alias("m_neg"),
                    ((pl.col("sec_m_move") > 0).mean() * 100.0).alias("m_pos_pct")
                ]).to_pandas()

                sec_overall = sec_yearly.group_by("industry").agg([
                    pl.len().alias("total_months"),
                    (pl.col("sec_m_move") > 0).sum().alias("pos_months_cnt"),
                    (pl.col("sec_m_move") <= 0).sum().alias("neg_months_cnt"),
                    ((pl.col("sec_m_move") > 0).mean() * 100.0).alias("overall_pos_pct")
                ]).to_pandas()

                sec_pos = sec_m_stats.pivot(index="industry", columns="month", values="m_pos").fillna(0)
                sec_neg = sec_m_stats.pivot(index="industry", columns="month", values="m_neg").fillna(0)
                sec_tot = sec_m_stats.pivot(index="industry", columns="month", values="m_tot").fillna(0)
                sec_pct = sec_m_stats.pivot(index="industry", columns="month", values="m_pos_pct").fillna(0)

                sec_matrix = sec_overall.set_index("industry")
                sec_matrix["Sector"] = sec_matrix.index
                for m_idx in range(1, 13):
                    m_name = MONTH_NAMES[m_idx - 1]
                    sec_matrix[f"{m_name}_pos"] = sec_pos[m_idx].astype(int) if m_idx in sec_pos.columns else 0
                    sec_matrix[f"{m_name}_neg"] = sec_neg[m_idx].astype(int) if m_idx in sec_neg.columns else 0
                    sec_matrix[f"{m_name}_tot"] = sec_tot[m_idx].astype(int) if m_idx in sec_tot.columns else 0
                    sec_matrix[f"{m_name}_pos_pct"] = sec_pct[m_idx] if m_idx in sec_pct.columns else 0.0

                sec_merged = sec_matrix.reset_index(drop=True)

                # Controls above sector table
                c_s1, c_s2, c_s3, c_s4 = st.columns([1.5, 1.4, 2.0, 1.6])
                with c_s1:
                    sort_sec_col = st.selectbox(
                        "Sort Sectors By:",
                        ["Overall Positive %", "Total Months"] + MONTH_NAMES,
                        index=0,
                        key="sec2_sort_col"
                    )
                with c_s2:
                    sort_sec_dir = st.radio("Sort Order:", ["Highest to Lowest", "Lowest to Highest"], horizontal=True, key="sec2_sort_dir")
                with c_s3:
                    sec_fmt_choice = st.selectbox(
                        "Month Cell Display Format:",
                        [
                            "total (pos🟢 %, neg🔴 %)  [e.g. 25 (16🟢 64%, 9🔴 36%)]",
                            "total (pos +ve %, neg -ve %)  [e.g. 25 (16 +ve 64%, 9 -ve 36%)]",
                            "total (pos🟢 %.1f%%, neg🔴 %.1f%%)  [e.g. 25 (16🟢 64.0%, 9🔴 36.0%)]",
                            "total (pos +ve, neg -ve)  [e.g. 25 (16 +ve, 9 -ve)]"
                        ],
                        index=0,
                        key="sec2_fmt_choice"
                    )
                with c_s4:
                    search_sec_q = st.text_input("🔍 Search Sector:", "", key="sec2_search_q")

                if search_sec_q:
                    q_lower = search_sec_q.strip().lower()
                    sec_merged = sec_merged[sec_merged['Sector'].str.lower().str.contains(q_lower)]

                asc_sec = (sort_sec_dir == "Lowest to Highest")
                if not sec_merged.empty:
                    if sort_sec_col == "Overall Positive %":
                        sec_merged = sec_merged.sort_values("overall_pos_pct", ascending=asc_sec)
                    elif sort_sec_col == "Total Months":
                        sec_merged = sec_merged.sort_values("total_months", ascending=asc_sec)
                    elif sort_sec_col in MONTH_NAMES:
                        sec_merged = sec_merged.sort_values(f"{sort_sec_col}_pos_pct", ascending=asc_sec)

                # Format sector table
                display_sec_table = pd.DataFrame()
                if not sec_merged.empty:
                    display_sec_table["Sector"] = sec_merged["Sector"]
                    for m_idx in range(1, 13):
                        m_name = MONTH_NAMES[m_idx - 1]
                        p_s = sec_merged[f"{m_name}_pos"]
                        n_s = sec_merged[f"{m_name}_neg"]
                        t_s = sec_merged[f"{m_name}_tot"]
                        col_vals = []
                        for p, n, t in zip(p_s, n_s, t_s):
                            if t == 0:
                                col_vals.append("-")
                            else:
                                p_pct = (p / t) * 100.0
                                n_pct = (n / t) * 100.0
                                if "pos🟢 %.1f%%" in sec_fmt_choice:
                                    col_vals.append(f"{t} ({p}🟢 {p_pct:.1f}%, {n}🔴 {n_pct:.1f}%)")
                                elif "pos +ve %" in sec_fmt_choice:
                                    col_vals.append(f"{t} ({p} +ve {p_pct:.0f}%, {n} -ve {n_pct:.0f}%)")
                                elif "pos +ve, neg -ve" in sec_fmt_choice:
                                    col_vals.append(f"{t} ({p} +ve, {n} -ve)")
                                else:
                                    col_vals.append(f"{t} ({p}🟢 {p_pct:.0f}%, {n}🔴 {n_pct:.0f}%)")
                        display_sec_table[m_name] = col_vals

                    display_sec_table["Total Months"] = sec_merged["total_months"]
                    display_sec_table["Positive Months"] = sec_merged.apply(lambda r: f"{r['pos_months_cnt']}🟢 ({r['overall_pos_pct']:.1f}%)", axis=1)
                    display_sec_table["Negative Months"] = sec_merged.apply(lambda r: f"{r['neg_months_cnt']}🔴 ({100.0 - r['overall_pos_pct']:.1f}%)", axis=1)
                    display_sec_table["Positive %"] = sec_merged["overall_pos_pct"].round(1)

                sec_col_cfg = {
                    "Sector": st.column_config.TextColumn("📌 Sector", pinned=True, help="NSE Industry Sector"),
                    "Total Months": st.column_config.NumberColumn("Total Mths", format="%d", help="Total sector months evaluated"),
                    "Positive Months": st.column_config.TextColumn("🟢 Positive", help="Total positive months count and percentage"),
                    "Negative Months": st.column_config.TextColumn("🔴 Negative", help="Total negative months count and percentage"),
                    "Positive %": st.column_config.ProgressColumn("Positive %", min_value=0, max_value=100, format="%.1f%%", help="Overall positive frequency progress bar")
                }
                for m_n in MONTH_NAMES:
                    sec_col_cfg[m_n] = st.column_config.TextColumn(m_n, help=f"{m_n} Seasonality: Total ({m_n} Positive Count & %, Negative Count & %)")

                st.markdown("""
                <div class="modern-matrix-banner">
                    <div class="modern-pill">📌 <strong>Pinned Column:</strong> Sector stays frozen on horizontal scroll</div>
                    <div class="modern-pill">💡 <strong>Cell Format:</strong> Total Months (Positive Count & %, Negative Count & %)</div>
                    <div class="modern-pill">📊 <strong>Progress Column:</strong> Visual Positive Frequency</div>
                </div>
                """, unsafe_allow_html=True)

                st.dataframe(display_sec_table, column_config=sec_col_cfg, width="stretch", hide_index=True)

                # 2. Sector Deep-Dive: Constituent Stocks Month-by-Month Matrix
                st.markdown("---")
                st.markdown("#### 🔍 Sector Deep-Dive: Constituent Stocks Month-by-Month Matrix")
                all_inds = sorted(stocks_pl['industry'].unique().to_list())
                chosen_drill_sec = st.selectbox("🏢 Select Sector to Inspect Stocks:", all_inds, index=0, key="sec2_drill_sec")

                # Filter stocks for chosen sector
                stk_sec_pl = stocks_pl.filter(pl.col("industry") == chosen_drill_sec)

                stk_sec_overall = stk_sec_pl.group_by(["symbol", "company_name"]).agg([
                    pl.len().alias("total_months"),
                    (pl.col("move_pct") > 0).sum().alias("pos_months_cnt"),
                    (pl.col("move_pct") <= 0).sum().alias("neg_months_cnt"),
                    ((pl.col("move_pct") > 0).mean() * 100.0).alias("overall_pos_pct")
                ]).to_pandas()

                stk_sec_m = stk_sec_pl.group_by(["symbol", "month"]).agg([
                    pl.len().alias("m_tot"),
                    (pl.col("move_pct") > 0).sum().alias("m_pos"),
                    (pl.col("move_pct") <= 0).sum().alias("m_neg"),
                    ((pl.col("move_pct") > 0).mean() * 100.0).alias("m_pos_pct")
                ]).to_pandas()

                stk_pos = stk_sec_m.pivot(index="symbol", columns="month", values="m_pos").fillna(0)
                stk_neg = stk_sec_m.pivot(index="symbol", columns="month", values="m_neg").fillna(0)
                stk_tot = stk_sec_m.pivot(index="symbol", columns="month", values="m_tot").fillna(0)
                stk_pct = stk_sec_m.pivot(index="symbol", columns="month", values="m_pos_pct").fillna(0)

                stk_sec_matrix = stk_sec_overall.set_index("symbol")
                stk_sec_matrix["Stock"] = stk_sec_matrix.index
                for m_idx in range(1, 13):
                    m_name = MONTH_NAMES[m_idx - 1]
                    stk_sec_matrix[f"{m_name}_pos"] = stk_pos[m_idx].astype(int) if m_idx in stk_pos.columns else 0
                    stk_sec_matrix[f"{m_name}_neg"] = stk_neg[m_idx].astype(int) if m_idx in stk_neg.columns else 0
                    stk_sec_matrix[f"{m_name}_tot"] = stk_tot[m_idx].astype(int) if m_idx in stk_tot.columns else 0
                    stk_sec_matrix[f"{m_name}_pos_pct"] = stk_pct[m_idx] if m_idx in stk_pct.columns else 0.0

                stk_sec_merged = stk_sec_matrix.reset_index(drop=True)

                # Controls above drill-down table
                cd_s1, cd_s2, cd_s3 = st.columns([1.8, 1.8, 2.4])
                with cd_s1:
                    sort_stk_col = st.selectbox(
                        "Sort Stocks By:",
                        ["Overall Positive %", "Total Months"] + MONTH_NAMES,
                        index=0,
                        key="sec2_stk_sort_col"
                    )
                with cd_s2:
                    sort_stk_dir = st.radio("Sort Order:", ["Highest to Lowest", "Lowest to Highest"], horizontal=True, key="sec2_stk_sort_dir")
                with cd_s3:
                    search_stk_q = st.text_input("🔍 Search Stock in Sector:", "", key="sec2_stk_search_q")

                if search_stk_q and not stk_sec_merged.empty:
                    q_lower = search_stk_q.strip().lower()
                    stk_sec_merged = stk_sec_merged[
                        stk_sec_merged['Stock'].str.lower().str.contains(q_lower) |
                        stk_sec_merged['company_name'].str.lower().str.contains(q_lower)
                    ]

                asc_stk = (sort_stk_dir == "Lowest to Highest")
                if not stk_sec_merged.empty:
                    if sort_stk_col == "Overall Positive %":
                        stk_sec_merged = stk_sec_merged.sort_values("overall_pos_pct", ascending=asc_stk)
                    elif sort_stk_col == "Total Months":
                        stk_sec_merged = stk_sec_merged.sort_values("total_months", ascending=asc_stk)
                    elif sort_stk_col in MONTH_NAMES:
                        stk_sec_merged = stk_sec_merged.sort_values(f"{sort_stk_col}_pos_pct", ascending=asc_stk)

                display_stk_table = pd.DataFrame()
                if not stk_sec_merged.empty:
                    display_stk_table["Stock"] = stk_sec_merged["Stock"]
                    display_stk_table["Company Name"] = stk_sec_merged["company_name"]
                    for m_idx in range(1, 13):
                        m_name = MONTH_NAMES[m_idx - 1]
                        p_s = stk_sec_merged[f"{m_name}_pos"]
                        n_s = stk_sec_merged[f"{m_name}_neg"]
                        t_s = stk_sec_merged[f"{m_name}_tot"]
                        col_vals = []
                        for p, n, t in zip(p_s, n_s, t_s):
                            if t == 0:
                                col_vals.append("-")
                            else:
                                p_pct = (p / t) * 100.0
                                n_pct = (n / t) * 100.0
                                if "pos🟢 %.1f%%" in sec_fmt_choice:
                                    col_vals.append(f"{t} ({p}🟢 {p_pct:.1f}%, {n}🔴 {n_pct:.1f}%)")
                                elif "pos +ve %" in sec_fmt_choice:
                                    col_vals.append(f"{t} ({p} +ve {p_pct:.0f}%, {n} -ve {n_pct:.0f}%)")
                                elif "pos +ve, neg -ve" in sec_fmt_choice:
                                    col_vals.append(f"{t} ({p} +ve, {n} -ve)")
                                else:
                                    col_vals.append(f"{t} ({p}🟢 {p_pct:.0f}%, {n}🔴 {n_pct:.0f}%)")
                        display_stk_table[m_name] = col_vals
                    display_stk_table["Total Months"] = stk_sec_merged["total_months"]
                    display_stk_table["Positive Months"] = stk_sec_merged.apply(lambda r: f"{r['pos_months_cnt']}🟢 ({r['overall_pos_pct']:.1f}%)", axis=1)
                    display_stk_table["Negative Months"] = stk_sec_merged.apply(lambda r: f"{r['neg_months_cnt']}🔴 ({100.0 - r['overall_pos_pct']:.1f}%)", axis=1)
                    display_stk_table["Positive %"] = stk_sec_merged["overall_pos_pct"].round(1)

                stk_drill_col_cfg = {
                    "Stock": st.column_config.TextColumn("📌 Stock", pinned=True, help="Stock ticker symbol"),
                    "Company Name": st.column_config.TextColumn("Company Name", help="Company Name"),
                    "Total Months": st.column_config.NumberColumn("Total Mths", format="%d"),
                    "Positive Months": st.column_config.TextColumn("🟢 Positive"),
                    "Negative Months": st.column_config.TextColumn("🔴 Negative"),
                    "Positive %": st.column_config.ProgressColumn("Positive %", min_value=0, max_value=100, format="%.1f%%")
                }
                for m_n in MONTH_NAMES:
                    stk_drill_col_cfg[m_n] = st.column_config.TextColumn(m_n, help=f"{m_n} Seasonality: Total ({m_n} Positive Count & %, Negative Count & %)")

                st.markdown(f"**Constituent Stocks of `{chosen_drill_sec}` ({len(display_stk_table)} Stocks):**")
                st.dataframe(display_stk_table, column_config=stk_drill_col_cfg, width="stretch", hide_index=True)

            # =========================================================
            # VIEW 2: SINGLE MONTH SECTOR DETAIL
            # =========================================================
            else:
                sec_c1, sec_c2 = st.columns([2.0, 4.0])
                with sec_c1:
                    cur_month_idx = pd.Timestamp.now().month
                    chosen_sec_month_name = st.selectbox(
                        "📅 Select Calendar Month:",
                        MONTH_FULL_NAMES,
                        index=cur_month_idx - 1,
                        key="sec_chosen_single_month"
                    )
                    chosen_sec_month = MONTH_FULL_NAMES.index(chosen_sec_month_name) + 1
                with sec_c2:
                    st.markdown(f"""
                    <div style="font-size:12px; color:#94a3b8; margin-top:28px;">
                        Evaluating performance of <strong>500 stocks</strong> across <strong>{span_years} years</strong> for <strong>{chosen_sec_month_name}</strong>.
                    </div>
                    """, unsafe_allow_html=True)

                # Compute Sector stats for chosen month
                stocks_m = stocks_pl.filter(pl.col("month") == chosen_sec_month)
                sec_stats = stocks_m.group_by("industry").agg([
                    pl.len().alias("sample_candles"),
                    pl.col("symbol").n_unique().alias("stocks_in_sector"),
                    (pl.col("move_pct") > 0).sum().alias("pos_candles"),
                    (pl.col("move_pct") <= 0).sum().alias("neg_candles"),
                    pl.col("is_positive").mean().mul(100.0).alias("pos_pct")
                ]).sort("pos_pct", descending=True).to_pandas()

                st.markdown(f"#### 🏆 Sector Performance in {chosen_sec_month_name}")
                show_sec_single_df = pd.DataFrame({
                    "Sector / Industry": sec_stats['industry'],
                    "Stocks in Sector": sec_stats['stocks_in_sector'],
                    "Total Evaluated Months": sec_stats['sample_candles'],
                    "Positive Months": sec_stats.apply(lambda r: f"{r['pos_candles']}🟢 ({r['pos_pct']:.1f}%)", axis=1),
                    "Negative Months": sec_stats.apply(lambda r: f"{r['neg_candles']}🔴 ({100.0 - r['pos_pct']:.1f}%)", axis=1),
                    "Positive %": sec_stats['pos_pct'].round(1)
                })
                sec_single_cfg = {
                    "Sector / Industry": st.column_config.TextColumn("📌 Sector", pinned=True),
                    "Stocks in Sector": st.column_config.NumberColumn("Stocks in Sector", format="%d"),
                    "Total Evaluated Months": st.column_config.NumberColumn("Total Mths", format="%d"),
                    "Positive Months": st.column_config.TextColumn("🟢 Positive"),
                    "Negative Months": st.column_config.TextColumn("🔴 Negative"),
                    "Positive %": st.column_config.ProgressColumn("Positive %", min_value=0, max_value=100, format="%.1f%%")
                }
                st.dataframe(show_sec_single_df, column_config=sec_single_cfg, width="stretch", hide_index=True)

                # Sector Drill-down in Single Month
                st.markdown("---")
                all_inds = sorted(sec_stats['industry'].unique().tolist())
                chosen_ind_drill = st.selectbox("Select Sector to Drill Down:", all_inds, key="sec_single_drill_pick")

                sec_stocks_pl = stocks_pl.filter(pl.col("industry") == chosen_ind_drill)
                sec_stocks_m = sec_stocks_pl.filter(pl.col("month") == chosen_sec_month)
                drill_stock_stats = sec_stocks_m.group_by(["symbol", "company_name"]).agg([
                    pl.len().alias("years"),
                    (pl.col("move_pct") > 0).sum().alias("pos_count"),
                    (pl.col("move_pct") <= 0).sum().alias("neg_count"),
                    pl.col("is_positive").mean().mul(100.0).alias("pos_pct")
                ]).sort("pos_pct", descending=True).to_pandas()

                show_drill = pd.DataFrame({
                    "Symbol": drill_stock_stats['symbol'],
                    "Company Name": drill_stock_stats['company_name'],
                    "Total Years": drill_stock_stats['years'].apply(lambda x: f"{x} Yrs"),
                    "Positive Months": drill_stock_stats.apply(lambda r: f"{r['pos_count']}🟢 ({r['pos_pct']:.1f}%)", axis=1),
                    "Negative Months": drill_stock_stats.apply(lambda r: f"{r['neg_count']}🔴 ({100.0 - r['pos_pct']:.1f}%)", axis=1),
                    "Positive %": drill_stock_stats['pos_pct'].round(1)
                })
                drill_single_cfg = {
                    "Symbol": st.column_config.TextColumn("📌 Symbol", pinned=True),
                    "Company Name": st.column_config.TextColumn("Company Name"),
                    "Total Years": st.column_config.TextColumn("Total Years"),
                    "Positive Months": st.column_config.TextColumn("🟢 Positive"),
                    "Negative Months": st.column_config.TextColumn("🔴 Negative"),
                    "Positive %": st.column_config.ProgressColumn("Positive %", min_value=0, max_value=100, format="%.1f%%")
                }
                st.markdown(f"**Constituent Stocks of `{chosen_ind_drill}` in {chosen_sec_month_name} ({len(show_drill)} Stocks):**")
                st.dataframe(show_drill, column_config=drill_single_cfg, width="stretch", hide_index=True)

    # =============================================================
    # TAB 3: STOCK SEASONALITY PERFORMANCE (NIFTY 50, SECTORAL & ALL STOCKS)
    # =============================================================
    with s_tab3:
        st.subheader("🎯 Stock Seasonality Performance (Nifty 50, Sectoral & All Stocks)")
        st.markdown(f"Detailed month-wise moves, directional tendency, and historical up/down magnitude across **{filter_min_yr} to {filter_max_yr}**.")

        if stocks_pl.is_empty():
            st.warning("Stock monthly parquets not found.")
        else:
            # Universe Selection Bar
            u_col1, u_col2, u_col3 = st.columns([2.2, 1.8, 2.0])
            with u_col1:
                scope_choice = st.radio(
                    "🎯 Select Asset Universe:",
                    [
                        "🌟 Combined (Index + Sectors + Nifty 50 Stocks)",
                        "🎯 NIFTY 50 Constituents (50 Stocks)",
                        "📈 NIFTY 50 Benchmark Index",
                        "🏢 20 Market Sectors",
                        "🌐 All 500 Nifty Equities",
                        "📂 Specific Sector Constituents"
                    ],
                    index=0,
                    key="sec4_stock_scope"
                )

            with u_col2:
                if "Specific Sector" in scope_choice:
                    all_sectors_list = sorted(stocks_pl["industry"].unique().to_list())
                    chosen_sec_filter = st.selectbox(
                        "🏢 Pick Industry / Sector:",
                        all_sectors_list,
                        index=0,
                        key="sec4_stock_chosen_sec"
                    )
                else:
                    chosen_sec_filter = None
                    st.markdown("""
                    <div style="font-size:12px; color:#94a3b8; margin-top:28px;">
                        Quick Universe: <strong>Combined</strong>, <strong>Nifty 50</strong>, <strong>Index</strong>, <strong>Sectors</strong>, or <strong>All Equities</strong>.
                    </div>
                    """, unsafe_allow_html=True)

            with u_col3:
                view_mode = st.radio(
                    "📊 Choose Display View:",
                    [
                        "📅 Month-by-Month Full Matrix (Jan – Dec)",
                        "🎯 Single Month Detail (Deep Breakdown)"
                    ],
                    index=0,
                    key="sec4_stock_view_mode"
                )

            # Filter stocks_pl for Single Month Detail or specific scoped logic
            if "Combined" in scope_choice:
                scoped_stocks_pl = stocks_pl.filter(pl.col("is_nifty50"))
                scope_label = "Combined Master Universe (Index, Sectors & Stocks)"
            elif "NIFTY 50 Constituents" in scope_choice:
                scoped_stocks_pl = stocks_pl.filter(pl.col("is_nifty50"))
                scope_label = "Nifty 50 Constituents"
            elif "Benchmark Index" in scope_choice:
                scoped_stocks_pl = stocks_pl.filter(pl.col("is_nifty50"))
                scope_label = "NIFTY 50 Benchmark Index"
            elif "20 Market Sectors" in scope_choice:
                scoped_stocks_pl = stocks_pl
                scope_label = "20 Market Sectors"
            elif "Specific Sector" in scope_choice and chosen_sec_filter:
                scoped_stocks_pl = stocks_pl.filter(pl.col("industry") == chosen_sec_filter)
                scope_label = f"Sector: {chosen_sec_filter}"
            else:
                scoped_stocks_pl = stocks_pl
                scope_label = "All 500 Nifty Equities"

            # -------------------------------------------------------------
            # VIEW 1: MONTH-BY-MONTH FULL MATRIX (JAN – DEC)
            # -------------------------------------------------------------
            if "Month-by-Month Full Matrix" in view_mode:
                st.markdown(f"#### 📅 12-Month Performance Matrix for **{scope_label}**")
                st.markdown("Shows monthly breakdown for each calendar month (Jan–Dec): exact positive vs. negative counts out of total evaluated months, and overall full-year performance statistics.")

                matrix_dfs = []

                # Part 1: NIFTY 50 Benchmark Index
                if ("Combined" in scope_choice or "Benchmark Index" in scope_choice) and not df_nifty.empty:
                    n_agg = df_nifty.groupby('month').agg(
                        m_tot=('move_pct', 'count'),
                        m_pos=('move_pct', lambda s: (s > 0).sum()),
                        m_neg=('move_pct', lambda s: (s <= 0).sum()),
                        m_avg=('move_pct', 'mean'),
                        m_pos_pct=('move_pct', lambda s: (s > 0).mean() * 100.0)
                    ).reset_index()

                    tot_all = len(df_nifty)
                    pos_all = int((df_nifty['move_pct'] > 0).sum())
                    neg_all = int((df_nifty['move_pct'] <= 0).sum())
                    pos_pct_all = (pos_all / tot_all * 100.0) if tot_all > 0 else 0.0
                    avg_all = df_nifty['move_pct'].mean()

                    n_row = {
                        'Name': 'NIFTY 50',
                        'company_name': 'NIFTY 50 Benchmark Index',
                        'Type / Sector': 'Benchmark Index',
                        'total_months': tot_all,
                        'pos_months_cnt': pos_all,
                        'neg_months_cnt': neg_all,
                        'overall_pos_pct': pos_pct_all,
                        'overall_avg_move': avg_all
                    }
                    for m_idx in range(1, 13):
                        m_name = MONTH_NAMES[m_idx - 1]
                        sub = n_agg[n_agg['month'] == m_idx]
                        if not sub.empty:
                            n_row[f'{m_name}_tot'] = int(sub['m_tot'].iloc[0])
                            n_row[f'{m_name}_pos'] = int(sub['m_pos'].iloc[0])
                            n_row[f'{m_name}_neg'] = int(sub['m_neg'].iloc[0])
                            n_row[f'{m_name}_avg'] = float(sub['m_avg'].iloc[0])
                            n_row[f'{m_name}_pos_pct'] = float(sub['m_pos_pct'].iloc[0])
                        else:
                            n_row[f'{m_name}_tot'] = 0
                            n_row[f'{m_name}_pos'] = 0
                            n_row[f'{m_name}_neg'] = 0
                            n_row[f'{m_name}_avg'] = np.nan
                            n_row[f'{m_name}_pos_pct'] = np.nan
                    matrix_dfs.append(pd.DataFrame([n_row]))

                # Part 2: Market Sectors (20 Sectors)
                if ("Combined" in scope_choice or "20 Market Sectors" in scope_choice) and not stocks_pl.is_empty():
                    sec_yearly = stocks_pl.group_by(["industry", "year", "month"]).agg(pl.col("move_pct").mean().alias("sec_m_move"))
                    sec_m_stats = sec_yearly.group_by(["industry", "month"]).agg([
                        pl.len().alias("m_tot"),
                        (pl.col("sec_m_move") > 0).sum().alias("m_pos"),
                        (pl.col("sec_m_move") <= 0).sum().alias("m_neg"),
                        pl.col("sec_m_move").mean().alias("m_avg"),
                        ((pl.col("sec_m_move") > 0).mean() * 100.0).alias("m_pos_pct")
                    ]).to_pandas()

                    sec_overall = sec_yearly.group_by("industry").agg([
                        pl.len().alias("total_months"),
                        (pl.col("sec_m_move") > 0).sum().alias("pos_months_cnt"),
                        (pl.col("sec_m_move") <= 0).sum().alias("neg_months_cnt"),
                        ((pl.col("sec_m_move") > 0).mean() * 100.0).alias("overall_pos_pct"),
                        pl.col("sec_m_move").mean().alias("overall_avg_move")
                    ]).to_pandas()

                    sec_pos = sec_m_stats.pivot(index="industry", columns="month", values="m_pos").fillna(0)
                    sec_neg = sec_m_stats.pivot(index="industry", columns="month", values="m_neg").fillna(0)
                    sec_tot = sec_m_stats.pivot(index="industry", columns="month", values="m_tot").fillna(0)
                    sec_avg = sec_m_stats.pivot(index="industry", columns="month", values="m_avg")
                    sec_pct = sec_m_stats.pivot(index="industry", columns="month", values="m_pos_pct")

                    sec_matrix = sec_overall.set_index("industry")
                    sec_matrix["Name"] = sec_matrix.index
                    sec_matrix["company_name"] = sec_matrix.index
                    sec_matrix["Type / Sector"] = "Market Sector"
                    for m_idx in range(1, 13):
                        m_name = MONTH_NAMES[m_idx - 1]
                        sec_matrix[f"{m_name}_pos"] = sec_pos[m_idx].astype(int) if m_idx in sec_pos.columns else 0
                        sec_matrix[f"{m_name}_neg"] = sec_neg[m_idx].astype(int) if m_idx in sec_neg.columns else 0
                        sec_matrix[f"{m_name}_tot"] = sec_tot[m_idx].astype(int) if m_idx in sec_tot.columns else 0
                        sec_matrix[f"{m_name}_avg"] = sec_avg[m_idx] if m_idx in sec_avg.columns else np.nan
                        sec_matrix[f"{m_name}_pos_pct"] = sec_pct[m_idx] if m_idx in sec_pct.columns else np.nan
                    matrix_dfs.append(sec_matrix.reset_index(drop=True))

                # Part 3: Individual Stocks
                if ("Combined" in scope_choice or "Constituents" in scope_choice or "All 500" in scope_choice or "Specific Sector" in scope_choice) and not stocks_pl.is_empty():
                    if "Combined" in scope_choice or "NIFTY 50 Constituents" in scope_choice:
                        stk_pl = stocks_pl.filter(pl.col("is_nifty50"))
                    elif "Specific Sector" in scope_choice and chosen_sec_filter:
                        stk_pl = stocks_pl.filter(pl.col("industry") == chosen_sec_filter)
                    else:
                        stk_pl = stocks_pl

                    overall_stats = stk_pl.group_by(["symbol", "company_name", "industry"]).agg([
                        pl.len().alias("total_months"),
                        (pl.col("move_pct") > 0).sum().alias("pos_months_cnt"),
                        (pl.col("move_pct") <= 0).sum().alias("neg_months_cnt"),
                        ((pl.col("move_pct") > 0).mean() * 100.0).alias("overall_pos_pct"),
                        pl.col("move_pct").mean().alias("overall_avg_move")
                    ]).to_pandas()

                    m_stats_pl = stk_pl.group_by(["symbol", "month"]).agg([
                        pl.len().alias("m_tot"),
                        (pl.col("move_pct") > 0).sum().alias("m_pos"),
                        (pl.col("move_pct") <= 0).sum().alias("m_neg"),
                        pl.col("move_pct").mean().alias("m_avg"),
                        ((pl.col("move_pct") > 0).mean() * 100.0).alias("m_pos_pct")
                    ]).to_pandas()

                    pivot_pos = m_stats_pl.pivot(index="symbol", columns="month", values="m_pos").fillna(0)
                    pivot_neg = m_stats_pl.pivot(index="symbol", columns="month", values="m_neg").fillna(0)
                    pivot_tot = m_stats_pl.pivot(index="symbol", columns="month", values="m_tot").fillna(0)
                    pivot_avg = m_stats_pl.pivot(index="symbol", columns="month", values="m_avg")
                    pivot_pct = m_stats_pl.pivot(index="symbol", columns="month", values="m_pos_pct")

                    stk_matrix = overall_stats.set_index("symbol")
                    stk_matrix["Name"] = stk_matrix.index
                    stk_matrix["Type / Sector"] = stk_matrix["industry"]
                    for m_idx in range(1, 13):
                        m_name = MONTH_NAMES[m_idx - 1]
                        stk_matrix[f"{m_name}_pos"] = pivot_pos[m_idx].astype(int) if m_idx in pivot_pos.columns else 0
                        stk_matrix[f"{m_name}_neg"] = pivot_neg[m_idx].astype(int) if m_idx in pivot_neg.columns else 0
                        stk_matrix[f"{m_name}_tot"] = pivot_tot[m_idx].astype(int) if m_idx in pivot_tot.columns else 0
                        stk_matrix[f"{m_name}_avg"] = pivot_avg[m_idx] if m_idx in pivot_avg.columns else np.nan
                        stk_matrix[f"{m_name}_pos_pct"] = pivot_pct[m_idx] if m_idx in pivot_pct.columns else np.nan
                    matrix_dfs.append(stk_matrix.reset_index(drop=True))

                merged_matrix = pd.concat(matrix_dfs, ignore_index=True) if matrix_dfs else pd.DataFrame()

                # Controls above table
                c_sort1, c_sort2, c_fmt, c_search = st.columns([1.5, 1.4, 2.2, 1.6])
                with c_sort1:
                    sort_m_col = st.selectbox(
                        "Sort Matrix By:",
                        ["Overall Positive %", "Overall Avg Move %", "Total Months"] + MONTH_NAMES,
                        index=0,
                        key="sec4_matrix_sort_col"
                    )
                with c_sort2:
                    sort_m_dir = st.radio("Sort Order:", ["Highest to Lowest", "Lowest to Highest"], horizontal=True, key="sec4_matrix_sort_dir")
                with c_fmt:
                    fmt_choice = st.selectbox(
                        "Month Cell Display Format:",
                        [
                            "total (pos🟢 %, neg🔴 %)  [e.g. 25 (16🟢 64%, 9🔴 36%)]",
                            "total (pos +ve %, neg -ve %)  [e.g. 25 (16 +ve 64%, 9 -ve 36%)]",
                            "total (pos🟢 %, neg🔴 %) + Move %  [e.g. 25 (16🟢 64%, 9🔴 36%) • +2.1%]",
                            "total (pos🟢 %.1f%%, neg🔴 %.1f%%)  [e.g. 25 (16🟢 64.0%, 9🔴 36.0%)]",
                            "total (pos +ve, neg -ve)  [e.g. 25 (16 +ve, 9 -ve)]",
                            "Move % Only  [e.g. +2.1%]"
                        ],
                        index=0,
                        key="sec4_matrix_fmt_choice"
                    )
                with c_search:
                    search_q = st.text_input("🔍 Search Name (Stock / Sector / Nifty):", "", key="sec4_matrix_search_q")

                if search_q and not merged_matrix.empty:
                    q_lower = search_q.strip().lower()
                    name_s = merged_matrix['Name'].astype(str).str.lower()
                    type_s = merged_matrix['Type / Sector'].astype(str).str.lower()
                    comp_s = merged_matrix.get('company_name', pd.Series(dtype=str)).fillna('').astype(str).str.lower()
                    merged_matrix = merged_matrix[
                        name_s.str.contains(q_lower) |
                        type_s.str.contains(q_lower) |
                        comp_s.str.contains(q_lower)
                    ]

                # Sorting logic
                asc = (sort_m_dir == "Lowest to Highest")
                if not merged_matrix.empty:
                    if sort_m_col == "Overall Positive %":
                        merged_matrix = merged_matrix.sort_values("overall_pos_pct", ascending=asc)
                    elif sort_m_col == "Overall Avg Move %":
                        merged_matrix = merged_matrix.sort_values("overall_avg_move", ascending=asc)
                    elif sort_m_col == "Total Months":
                        merged_matrix = merged_matrix.sort_values("total_months", ascending=asc)
                    elif sort_m_col in MONTH_NAMES:
                        merged_matrix = merged_matrix.sort_values(
                            by=[f"{sort_m_col}_pos_pct", f"{sort_m_col}_avg"],
                            ascending=[asc, asc]
                        )

                # Format DataFrame for display
                display_matrix = pd.DataFrame()
                if not merged_matrix.empty:
                    display_matrix["Name"] = merged_matrix["Name"]

                    # Populate Month columns based on user format choice
                    for m_idx in range(1, 13):
                        m_name = MONTH_NAMES[m_idx - 1]
                        pos_s = merged_matrix[f"{m_name}_pos"]
                        neg_s = merged_matrix[f"{m_name}_neg"]
                        tot_s = merged_matrix[f"{m_name}_tot"]
                        avg_s = merged_matrix[f"{m_name}_avg"]

                        col_vals = []
                        for p, n, t, a in zip(pos_s, neg_s, tot_s, avg_s):
                            if t == 0:
                                col_vals.append("-")
                            else:
                                p_pct = (p / t) * 100.0
                                n_pct = (n / t) * 100.0
                                if "+ Move %" in fmt_choice:
                                    a_str = f" • {a:+.1f}%" if pd.notna(a) else ""
                                    col_vals.append(f"{t} ({p}🟢 {p_pct:.0f}%, {n}🔴 {n_pct:.0f}%){a_str}")
                                elif "pos +ve %" in fmt_choice:
                                    col_vals.append(f"{t} ({p} +ve {p_pct:.0f}%, {n} -ve {n_pct:.0f}%)")
                                elif "pos🟢 %.1f%%" in fmt_choice:
                                    col_vals.append(f"{t} ({p}🟢 {p_pct:.1f}%, {n}🔴 {n_pct:.1f}%)")
                                elif "total (pos +ve, neg -ve)" in fmt_choice:
                                    col_vals.append(f"{t} ({p} +ve, {n} -ve)")
                                elif "Move % Only" in fmt_choice:
                                    col_vals.append(f"{a:+.2f}%" if pd.notna(a) else "-")
                                else:
                                    # Default: 25 (16🟢 64%, 9🔴 36%)
                                    col_vals.append(f"{t} ({p}🟢 {p_pct:.0f}%, {n}🔴 {n_pct:.0f}%)")
                        display_matrix[m_name] = col_vals

                    display_matrix["Total Months"] = merged_matrix["total_months"]
                    display_matrix["Positive Months"] = merged_matrix.apply(lambda r: f"{r['pos_months_cnt']}🟢 ({r['overall_pos_pct']:.1f}%)", axis=1)
                    display_matrix["Negative Months"] = merged_matrix.apply(lambda r: f"{r['neg_months_cnt']}🔴 ({100.0 - r['overall_pos_pct']:.1f}%)", axis=1)
                    display_matrix["Positive %"] = merged_matrix["overall_pos_pct"].round(1)
                    display_matrix["Overall Avg Move"] = merged_matrix["overall_avg_move"].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "-")
                    display_matrix["Type / Sector"] = merged_matrix["Type / Sector"]

                matrix_col_cfg = {
                    "Name": st.column_config.TextColumn("📌 Name", pinned=True, help="Asset Name (Benchmark / Sector / Stock) - Pinned for horizontal scroll"),
                    "Total Months": st.column_config.NumberColumn("Total Mths", format="%d", help="Total historical evaluated months"),
                    "Positive Months": st.column_config.TextColumn("🟢 Positive", help="Historical positive count and win percentage"),
                    "Negative Months": st.column_config.TextColumn("🔴 Negative", help="Historical negative count and loss percentage"),
                    "Positive %": st.column_config.ProgressColumn("Positive %", min_value=0, max_value=100, format="%.1f%%", help="Overall positive frequency progress bar"),
                    "Overall Avg Move": st.column_config.TextColumn("Overall Avg Move", help="Average monthly percentage return across history"),
                    "Type / Sector": st.column_config.TextColumn("Category / Sector", help="Asset type or market sector")
                }
                for m_n in MONTH_NAMES:
                    matrix_col_cfg[m_n] = st.column_config.TextColumn(m_n, help=f"{m_n} Seasonality: Total ({m_n} Positive Count & %, Negative Count & %)")

                st.markdown("""
                <div class="modern-matrix-banner">
                    <div class="modern-pill">📌 <strong>Pinned Name Column:</strong> Stays fixed on the left while scrolling across months Jan–Dec</div>
                    <div class="modern-pill">💡 <strong>Cell Format:</strong> Total Months (Positive Count & %, Negative Count & %)</div>
                    <div class="modern-pill">📊 <strong>Progress Bar:</strong> Visual Positive Frequency Indicator</div>
                </div>
                """, unsafe_allow_html=True)

                st.dataframe(display_matrix, column_config=matrix_col_cfg, width="stretch", hide_index=True)

            # -------------------------------------------------------------
            # VIEW 2: SINGLE MONTH DETAIL (DEEP BREAKDOWN)
            # -------------------------------------------------------------
            else:
                f_c1, f_c2, f_c3, f_c4 = st.columns([1.8, 1.8, 1.8, 2.0])
                with f_c1:
                    chosen_m_name = st.selectbox(
                        "📅 Calendar Month:",
                        MONTH_FULL_NAMES,
                        index=cur_month_idx - 1,
                        key="sec4_single_m_name"
                    )
                    chosen_m_num = MONTH_FULL_NAMES.index(chosen_m_name) + 1
                with f_c2:
                    min_years_req = st.selectbox(
                        "Minimum History:",
                        ["All Available Years", "At least 3 Years", "At least 5 Years", "At least 10 Years"],
                        index=0,
                        key="sec4_min_years"
                    )
                with f_c3:
                    tendency_filter = st.selectbox(
                        "Tendency Filter:",
                        [
                            "All Stocks in Universe",
                            "🟢 Positive Avg Move (> 0.0%)",
                            "🔴 Negative Avg Move (< 0.0%)",
                            "🚀 Strong Movers (Avg >= +3.0%)",
                            "🛡️ Mostly Positive (>= 65% Positive)"
                        ],
                        index=0,
                        key="sec4_tendency_filter"
                    )
                with f_c4:
                    sort_single_by = st.selectbox(
                        "Sort Stocks By:",
                        ["Average Move (%)", "% Positive Months", "Median Move (%)", "Total Years"],
                        index=0,
                        key="sec4_sort_single_by"
                    )

                # Filter by month
                month_sub_pl = scoped_stocks_pl.filter(pl.col("month") == chosen_m_num)
                month_agg = month_sub_pl.group_by(["symbol", "company_name", "industry"]).agg([
                    pl.len().alias("sample_years"),
                    (pl.col("move_pct") > 0).sum().alias("pos_count"),
                    (pl.col("move_pct") <= 0).sum().alias("neg_count"),
                    ((pl.col("move_pct") > 0).mean() * 100.0).alias("pos_pct"),
                    pl.col("move_pct").mean().alias("avg_move"),
                    pl.col("move_pct").median().alias("median_move"),
                    pl.col("move_pct").filter(pl.col("move_pct") > 0).mean().alias("avg_up_move"),
                    pl.col("move_pct").filter(pl.col("move_pct") < 0).mean().alias("avg_down_move"),
                    pl.col("move_pct").max().alias("max_gain"),
                    pl.col("move_pct").min().alias("max_drop")
                ]).to_pandas()

                # Filter by min years
                if "3 Years" in min_years_req:
                    month_agg = month_agg[month_agg['sample_years'] >= 3]
                elif "5 Years" in min_years_req:
                    month_agg = month_agg[month_agg['sample_years'] >= 5]
                elif "10 Years" in min_years_req:
                    month_agg = month_agg[month_agg['sample_years'] >= 10]

                # Filter by tendency
                if "Positive Avg Move" in tendency_filter:
                    month_agg = month_agg[month_agg['avg_move'] > 0.0]
                elif "Negative Avg Move" in tendency_filter:
                    month_agg = month_agg[month_agg['avg_move'] < 0.0]
                elif "Strong Movers" in tendency_filter:
                    month_agg = month_agg[month_agg['avg_move'] >= 3.0]
                elif "Mostly Positive" in tendency_filter:
                    month_agg = month_agg[month_agg['pos_pct'] >= 65.0]

                # Sort
                if "% Positive" in sort_single_by:
                    month_agg = month_agg.sort_values("pos_pct", ascending=False)
                elif "Average" in sort_single_by:
                    month_agg = month_agg.sort_values("avg_move", ascending=False)
                elif "Median" in sort_single_by:
                    month_agg = month_agg.sort_values("median_move", ascending=False)
                else:
                    month_agg = month_agg.sort_values("sample_years", ascending=False)

                st.markdown(f"#### 🏆 {scope_label} Seasonality Leaderboard in {chosen_m_name} ({len(month_agg)} Stocks)")
                show_single_df = pd.DataFrame({
                    "Symbol": month_agg['symbol'],
                    "Company Name": month_agg['company_name'],
                    "Sector": month_agg['industry'],
                    "Average Move (%)": month_agg['avg_move'].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "-"),
                    "Median Move (%)": month_agg['median_move'].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "-"),
                    "Total Years": month_agg['sample_years'].apply(lambda x: f"{x} Yrs"),
                    "Positive Months": month_agg.apply(lambda r: f"{r['pos_count']}🟢 ({r['pos_pct']:.1f}%)", axis=1),
                    "Negative Months": month_agg.apply(lambda r: f"{r['neg_count']}🔴 ({100.0 - r['pos_pct']:.1f}%)", axis=1),
                    "Positive %": month_agg['pos_pct'].round(1),
                    "Avg Up Move (%)": month_agg['avg_up_move'].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "-"),
                    "Avg Down Move (%)": month_agg['avg_down_move'].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "-"),
                    "Best Historical Gain": month_agg['max_gain'].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "-"),
                    "Max Drawdown": month_agg['max_drop'].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "-")
                })
                single_col_cfg = {
                    "Symbol": st.column_config.TextColumn("📌 Symbol", pinned=True),
                    "Company Name": st.column_config.TextColumn("Company Name"),
                    "Sector": st.column_config.TextColumn("Sector"),
                    "Positive %": st.column_config.ProgressColumn("Positive %", min_value=0, max_value=100, format="%.1f%%")
                }
                st.dataframe(show_single_df, column_config=single_col_cfg, width="stretch", hide_index=True)

    # =============================================================
    # TAB 4: INDIVIDUAL STOCK DEEP-DIVE (500 STOCKS)
    # =============================================================
    with s_tab4:
        st.subheader("🔍 Individual Stock Seasonality Deep-Dive (Nifty 500)")
        st.markdown("Search or select **any of the 500 Nifty Equities** for complete 12-month historical breakdown, directional probabilities, and Year-by-Year heatmaps.")

        if stocks_pl.is_empty():
            st.warning("Stock monthly parquets not found.")
        else:
            all_syms = sorted(stocks_pl["symbol"].unique().to_list())
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
                s_by_pos = stock_stats.sort_values('Pos_Pct', ascending=False)
                best_m = s_by_pos.iloc[0]
                worst_m = s_by_pos.iloc[-1]

                hc1, hc2, hc3, hc4 = st.columns(4)
                with hc1:
                    st.markdown(f"""<div class='metric-card'><span class='metric-label'>Strongest Positive Month</span><div class='metric-value' style='color:#00d26a;'>{best_m['Month']} ({best_m['Pos_Pct']:.1f}%)</div><span style='font-size:11px;color:#94a3b8;'>Avg Return: {best_m['Avg_Move']:+.2f}%</span></div>""", unsafe_allow_html=True)
                with hc2:
                    st.markdown(f"""<div class='metric-card'><span class='metric-label'>Most Vulnerable Month</span><div class='metric-value' style='color:#f43f5e;'>{worst_m['Month']} ({worst_m['Pos_Pct']:.1f}%)</div><span style='font-size:11px;color:#94a3b8;'>Avg Return: {worst_m['Avg_Move']:+.2f}%</span></div>""", unsafe_allow_html=True)
                with hc3:
                    overall_wr = (stock_pl_sub['is_positive'].mean()) * 100.0
                    pos_tot = int(stock_pl_sub['is_positive'].sum())
                    tot_all = len(stock_pl_sub)
                    st.markdown(f"""<div class='metric-card'><span class='metric-label'>Overall Positive Months</span><div class='metric-value' style='color:#38bdf8;'>{overall_wr:.1f}%</div><span style='font-size:11px;color:#94a3b8;'>{pos_tot}🟢 / {tot_all} Months</span></div>""", unsafe_allow_html=True)
                with hc4:
                    overall_avg = stock_pl_sub['move_pct'].mean()
                    st.markdown(f"""<div class='metric-card'><span class='metric-label'>Overall Monthly Avg</span><div class='metric-value' style='color:#fbbf24;'>{overall_avg:+.2f}%</div><span style='font-size:11px;color:#94a3b8;'>Average monthly move</span></div>""", unsafe_allow_html=True)

                # 12 Month Table
                st.markdown("---")
                st.markdown(f"#### 📅 12-Month Performance Seasonality Summary for **{picked_sym}**")
                show_stk_df = pd.DataFrame({
                    "Month": stock_stats['Month_Full'],
                    "Sample Years": stock_stats['Total_Years'],
                    "Positive Months": stock_stats.apply(lambda r: f"{r['Pos_Count']}🟢 ({r['Pos_Pct']:.1f}%)", axis=1),
                    "Negative Months": stock_stats.apply(lambda r: f"{r['Neg_Count']}🔴 ({r['Neg_Pct']:.1f}%)", axis=1),
                    "Positive %": stock_stats['Pos_Pct'].round(1),
                    "Average Move (%)": stock_stats['Avg_Move'].apply(lambda x: f"{x:+.2f}%"),
                    "Median Move (%)": stock_stats['Median_Move'].apply(lambda x: f"{x:+.2f}%"),
                    "Avg Up Move (%)": stock_stats['Avg_Up_Move'].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "-"),
                    "Avg Down Move (%)": stock_stats['Avg_Down_Move'].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "-"),
                    "Best Year Gain": stock_stats.apply(lambda r: f"{r['Best_Move']:+.1f}% ({r['Best_Year']})", axis=1),
                    "Max Drawdown": stock_stats.apply(lambda r: f"{r['Worst_Move']:+.1f}% ({r['Worst_Year']})", axis=1)
                })
                stk_summary_cfg = {
                    "Month": st.column_config.TextColumn("📌 Month", pinned=True),
                    "Positive %": st.column_config.ProgressColumn("Positive %", min_value=0, max_value=100, format="%.1f%%")
                }
                st.dataframe(show_stk_df, column_config=stk_summary_cfg, width="stretch", hide_index=True)

                # Plotly Visual Bar Chart
                fig_stk_bars = plot_seasonality_profile_chart(
                    stock_stats,
                    title=f"{picked_sym} Monthly Seasonality Profile ({filter_min_yr} – {filter_max_yr})"
                )
                st.plotly_chart(fig_stk_bars, width="stretch")

                # Year-by-Year Historical Return Table
                st.markdown("---")
                st.markdown(f"#### 📅 {picked_sym} Year-by-Year Monthly Return Table (%)")
                stk_matrix = build_year_matrix(stock_pl_sub)
                if not stk_matrix.empty:
                    s_df = stk_matrix.reset_index()
                    s_df['year'] = s_df['year'].astype(str)
                    for c in MONTH_NAMES + ['Full Year']:
                        if c in s_df.columns:
                            s_df[c] = s_df[c].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "-")
                    stk_yr_cfg = {"year": st.column_config.TextColumn("📌 Year", pinned=True)}
                    for m in MONTH_NAMES:
                        stk_yr_cfg[m] = st.column_config.TextColumn(m)
                    stk_yr_cfg["Full Year"] = st.column_config.TextColumn("Full Year")
                    st.dataframe(s_df, column_config=stk_yr_cfg, width="stretch", hide_index=True)
