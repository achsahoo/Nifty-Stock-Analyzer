import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

from rsi_divergence_ui import (
    load_asset_ohlc,
    get_available_stock_symbols,
    compute_backtest_stats
)

# =============================================================
# EMA CALCULATION
# =============================================================
def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculates Exponential Moving Average using pandas ewm."""
    return series.ewm(span=period, adjust=False).mean()

# =============================================================
# DUAL EMA CROSSOVER DETECTION
# =============================================================
def detect_ema_crossovers(df: pd.DataFrame, fast_period: int = 9, slow_period: int = 21):
    """
    Detects Golden Crosses and Death Crosses on candle CLOSE:
    - Golden Cross (BUY): Fast EMA crosses Slow EMA from bottom to top.
    - Death Cross (SELL): Fast EMA crosses Slow EMA from top to bottom.
    Returns:
    - df with 'fast_ema' and 'slow_ema' columns added
    - df_signals: DataFrame of all crossover events with signal details
    """
    df = df.copy().reset_index(drop=True)
    
    # Ensure fast is strictly smaller than slow
    fast_p = min(fast_period, slow_period)
    slow_p = max(fast_period, slow_period)
    
    df['fast_ema'] = calculate_ema(df['close'], fast_p)
    df['slow_ema'] = calculate_ema(df['close'], slow_p)
    
    n = len(df)
    fast_vals = df['fast_ema'].values
    slow_vals = df['slow_ema'].values
    dates = df['date'].values
    lows = df['low'].values
    highs = df['high'].values
    closes = df['close'].values
    opens = df['open'].values
    
    signals = []
    
    # Start after slow_p to ensure EMA stability
    start_bar = max(slow_p, 2)
    for i in range(start_bar, n):
        prev_fast = fast_vals[i - 1]
        prev_slow = slow_vals[i - 1]
        curr_fast = fast_vals[i]
        curr_slow = slow_vals[i]
        
        # Golden Cross (BUY)
        if prev_fast <= prev_slow and curr_fast > curr_slow:
            signals.append({
                'signal_idx': i,
                'signal_date': pd.to_datetime(dates[i]),
                'signal': 'BUY',
                'reason': f"Fast EMA ({fast_p}) crossed above Slow EMA ({slow_p})",
                'crossover_price': curr_slow,
                'signal_open': opens[i],
                'signal_high': highs[i],
                'signal_low': lows[i],
                'signal_close': closes[i],
                'fast_ema': curr_fast,
                'slow_ema': curr_slow
            })
            
        # Death Cross (SELL)
        elif prev_fast >= prev_slow and curr_fast < curr_slow:
            signals.append({
                'signal_idx': i,
                'signal_date': pd.to_datetime(dates[i]),
                'signal': 'SELL',
                'reason': f"Fast EMA ({fast_p}) crossed below Slow EMA ({slow_p})",
                'crossover_price': curr_slow,
                'signal_open': opens[i],
                'signal_high': highs[i],
                'signal_low': lows[i],
                'signal_close': closes[i],
                'fast_ema': curr_fast,
                'slow_ema': curr_slow
            })
            
    df_signals = pd.DataFrame(signals)
    return df, df_signals

# =============================================================
# BACKTEST SIMULATION ENGINE FOR EMA CROSSOVER
# =============================================================
def run_ema_crossover_backtest(
    df: pd.DataFrame,
    sig_df: pd.DataFrame,
    sl_pct: float = 0.0,
    tgt_pct: float = 5.0,
    exit_on_opposite: bool = True
):
    """
    Backtest simulation matching the exact user specifications:
    1. Confirmation: Cross strictly confirmed on candle CLOSE.
    2. Entry: Next day / candle OPEN.
    3. Stop Loss:
       - If seekbar is 0.0%: Dynamic SL is the exact crossover point price on the signal candle.
       - If seekbar > 0.0%: Fixed percentage stop loss from entry price.
    4. Target:
       - If seekbar > 0.0%: Fixed percentage target from entry price.
       - If seekbar is 0.0%: Open-ended trailing ride until SL or opposite crossover.
    5. Opposite Crossover:
       - If exit_on_opposite is True, an open position will exit when the reverse cross fires.
    """
    df = df.copy().reset_index(drop=True)
    if sig_df.empty:
        return pd.DataFrame(), {}
        
    n_bars = len(df)
    
    # Pre-map opposite signals for quick lookup
    opposite_signal_dates = {}
    for idx, row in sig_df.iterrows():
        opposite_signal_dates[row['signal_idx']] = row['signal']
        
    trades = []
    
    for _, sig in sig_df.iterrows():
        sig_bar = sig['signal_idx']
        e_bar = sig_bar + 1
        if e_bar >= n_bars:
            continue
            
        action = sig['signal']
        entry_price = df['open'].iloc[e_bar]
        entry_date = df['date'].iloc[e_bar]
        cross_price = sig['crossover_price']
        
        # Stop Loss Calculation
        if sl_pct == 0.0 or sl_pct is None:
            # Dynamic SL is the point of crossing as requested
            sl_price = cross_price
            sl_desc = f"Crossover Point (₹{cross_price:,.1f})"
        else:
            if action == 'BUY':
                sl_price = entry_price * (1.0 - sl_pct / 100.0)
            else:
                sl_price = entry_price * (1.0 + sl_pct / 100.0)
            sl_desc = f"Fixed {sl_pct:.1f}%"
            
        # Target Calculation
        if tgt_pct > 0.0:
            if action == 'BUY':
                tgt_price = entry_price * (1.0 + tgt_pct / 100.0)
            else:
                tgt_price = entry_price * (1.0 - tgt_pct / 100.0)
            tgt_desc = f"+{tgt_pct:.2f}%"
        else:
            tgt_price = None
            tgt_desc = "Open-Ended (Ride Trend)"
            
        exit_bar = None
        exit_price = None
        exit_date = None
        exit_reason = None
        
        for bar in range(e_bar, n_bars):
            b_high = df['high'].iloc[bar]
            b_low = df['low'].iloc[bar]
            b_date = df['date'].iloc[bar]
            
            # Check opposite crossover exit
            if exit_on_opposite and bar in opposite_signal_dates:
                opp_signal = opposite_signal_dates[bar]
                if (action == 'BUY' and opp_signal == 'SELL') or (action == 'SELL' and opp_signal == 'BUY'):
                    exit_bar = bar
                    exit_date = b_date
                    exit_price = df['close'].iloc[bar]
                    exit_reason = f"Opposite Cross ({opp_signal})"
                    break
            
            if action == 'BUY':
                # Check Stop Loss breach
                if b_low <= sl_price:
                    exit_bar = bar
                    exit_date = b_date
                    exit_price = sl_price
                    exit_reason = 'Stop Loss'
                    break
                # Check Target reach
                elif tgt_price is not None and b_high >= tgt_price:
                    exit_bar = bar
                    exit_date = b_date
                    exit_price = tgt_price
                    exit_reason = 'Target'
                    break
            else: # SELL
                # Check Stop Loss breach
                if b_high >= sl_price:
                    exit_bar = bar
                    exit_date = b_date
                    exit_price = sl_price
                    exit_reason = 'Stop Loss'
                    break
                # Check Target reach
                elif tgt_price is not None and b_low <= tgt_price:
                    exit_bar = bar
                    exit_date = b_date
                    exit_price = tgt_price
                    exit_reason = 'Target'
                    break
                    
        if exit_bar is None:
            exit_bar = n_bars - 1
            exit_date = df['date'].iloc[-1]
            exit_price = df['close'].iloc[-1]
            exit_reason = 'Open Position'
            
        pnl_pts = (exit_price - entry_price) if action == 'BUY' else (entry_price - exit_price)
        pnl_pct = (pnl_pts / entry_price) * 100.0 if entry_price > 0 else 0.0
        
        trades.append({
            'signal_date': sig['signal_date'],
            'entry_date': entry_date,
            'exit_date': exit_date,
            'action': action,
            'reason': sig['reason'],
            'entry_price': entry_price,
            'cross_price': cross_price,
            'sl_price': sl_price,
            'tgt_price': tgt_price if tgt_price else np.nan,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'holding_bars': exit_bar - e_bar + 1,
            'pnl_pts': pnl_pts,
            'pnl_pct': pnl_pct,
            'win': pnl_pts > 0,
            'sl_desc': sl_desc,
            'tgt_desc': tgt_desc
        })
        
    trades_df = pd.DataFrame(trades)
    stats = compute_backtest_stats(trades_df)
    return trades_df, stats

# =============================================================
# STREAMLIT UI FOR SECTION 7
# =============================================================
def render_ema_crossover_section(analytics_engine=None):
    st.header("📈 Section 7: Dual EMA Crossover Backtest Engine")
    st.markdown("""
    Quantitative backtest engine for **Dual Exponential Moving Average (EMA) Crossover** strategies across **NIFTY 50** and **200+ F&O Equities** on **Daily, Weekly, and Monthly** timeframes.
    Customize the **Fast EMA** and **Slow EMA** periods, adjust the **Stop Loss** and **Target** seekbars, and review real-time performance statistics, equity curves, candlestick signals, and full trade logs.
    """)
    
    # -------------------------------------------------------------
    # 1. ASSET & TIMEFRAME SELECTION
    # -------------------------------------------------------------
    st.markdown("#### 🎯 Asset & Timeframe Selection")
    ac1, ac2, ac3 = st.columns([2.0, 2.5, 1.8])
    
    with ac1:
        asset_type = st.radio(
            "Asset Universe:",
            ["🇮🇳 NIFTY 50 (Index)", "📈 Individual Stocks (F&O Universe)"],
            horizontal=True,
            key="sec7_asset_type"
        )
        
    with ac2:
        if asset_type == "🇮🇳 NIFTY 50 (Index)":
            selected_asset = "NIFTY 50"
            st.markdown("<span style='font-size: 12px; color: #38bdf8;'>Data Source: <code>NIFTY 50 (Daily/Weekly Parquet 1990–2026)</code></span>", unsafe_allow_html=True)
        else:
            stocks = get_available_stock_symbols()
            if "selected_stock_override" in st.session_state and st.session_state["selected_stock_override"] in stocks:
                def_stock_idx = stocks.index(st.session_state["selected_stock_override"])
            else:
                def_stock_idx = stocks.index("INFY") if "INFY" in stocks else 0
            selected_asset = st.selectbox("Select Stock:", stocks, index=def_stock_idx, key="sec7_stock_pick")
            st.markdown(f"<span style='font-size: 12px; color: #38bdf8;'>Data Source: <code>{selected_asset} Stock Parquet</code></span>", unsafe_allow_html=True)
            
    with ac3:
        timeframe = st.selectbox("📅 Timeframe:", ["Daily", "Weekly", "Monthly"], index=0, key="sec7_tf_pick")
        
    # Load raw asset data
    raw_df = load_asset_ohlc(selected_asset, timeframe)
    if raw_df.empty:
        st.warning(f"No data available for {selected_asset} on {timeframe} timeframe.")
        return
        
    min_data_date = raw_df['date'].min().date()
    max_data_date = raw_df['date'].max().date()
    
    # -------------------------------------------------------------
    # 2. EMA PARAMETERS & SEEKBAR CONTROLS
    # -------------------------------------------------------------
    st.markdown("#### ⚙️ Dual EMA Parameters & Seekbar Controls")
    
    e1, e2, e3, e4 = st.columns([1.5, 1.5, 2.5, 2.5])
    with e1:
        fast_ema_val = st.number_input(
            "⚡ Fast EMA Period:",
            min_value=2,
            max_value=200,
            value=9,
            step=1,
            help="Smaller EMA period (e.g. 9). When it crosses Slow EMA from bottom -> BUY; from top -> SELL."
        )
    with e2:
        slow_ema_val = st.number_input(
            "🐢 Slow EMA Period:",
            min_value=3,
            max_value=500,
            value=21,
            step=1,
            help="Larger EMA period (e.g. 21)."
        )
        
    if fast_ema_val >= slow_ema_val:
        st.error("⚠️ **Fast EMA** must be strictly smaller than **Slow EMA** for valid crossover detection.")
        return
        
    with e3:
        sl_val = st.slider(
            "🛑 Stop Loss Seekbar (%):",
            min_value=0.0,
            max_value=15.0,
            value=0.0,
            step=0.1,
            format="%.1f%%",
            help="Set to 0.0% to dynamically use the Crossover Point Price at signal candle. Any value > 0 sets a fixed percentage stop loss from next-day Open entry."
        )
        if sl_val == 0.0:
            st.markdown("<span style='color:#38bdf8; font-size:12px;'>🛡️ <strong>SL Mode:</strong> Exact Crossover Point Price</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color:#fbbf24; font-size:12px;'>🛡️ <strong>SL Mode:</strong> Fixed {sl_val:.1f}% from Entry</span>", unsafe_allow_html=True)
            
    with e4:
        def_tgt = 3.0 if timeframe == "Daily" else (6.0 if timeframe == "Weekly" else 10.0)
        tgt_val = st.slider(
            "🎯 Target Seekbar (%):",
            min_value=0.0,
            max_value=30.0,
            value=float(def_tgt),
            step=0.25,
            format="%.2f%%",
            help="Set to 0.0% for open-ended trend ride until SL or opposite crossover. Any value > 0 sets a fixed percentage target from entry."
        )
        if tgt_val == 0.0:
            st.markdown("<span style='color:#94a3b8; font-size:12px;'>🎯 <strong>Target Mode:</strong> Open-Ended (Ride Trend)</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color:#00d26a; font-size:12px;'>🎯 <strong>Target Mode:</strong> Fixed +{tgt_val:.2f}% from Entry</span>", unsafe_allow_html=True)

    # Date Range & Advanced Settings
    with st.expander("🛠️ Date Range & Execution Rules", expanded=False):
        d1, d2, d3 = st.columns([1.5, 1.5, 2.0])
        with d1:
            date_filter_mode = st.selectbox(
                "Filter Date History:",
                [
                    "Full Available History (Inception - Present)",
                    "Last 10 Years (2016 - Present)",
                    "Post-Covid (2020 - Present)",
                    "Custom Range"
                ],
                index=0,
                key="sec7_date_filter"
            )
        with d2:
            if date_filter_mode == "Custom Range":
                start_f = st.date_input("Filter Start Date:", value=min_data_date, key="sec7_sd")
                end_f = st.date_input("Filter End Date:", value=max_data_date, key="sec7_ed")
            elif date_filter_mode.startswith("Last 10 Years"):
                start_f = pd.to_datetime("2016-01-01").date()
                end_f = max_data_date
            elif date_filter_mode.startswith("Post-Covid"):
                start_f = pd.to_datetime("2020-01-01").date()
                end_f = max_data_date
            else:
                start_f = min_data_date
                end_f = max_data_date
        with d3:
            st.markdown(f"<span style='font-size:12px; color:#94a3b8;'>Active Range: <strong>{start_f}</strong> to <strong>{end_f}</strong></span>", unsafe_allow_html=True)
            opp_exit = st.checkbox("🔄 Exit trade on Opposite Crossover (Reverse Signal)", value=True, help="Automatically closes trade if reverse crossover fires before SL or Target.")

    # -------------------------------------------------------------
    # 3. DETECT CROSSOVERS & RUN BACKTEST
    # -------------------------------------------------------------
    df_calc, sig_df_all = detect_ema_crossovers(raw_df, fast_period=fast_ema_val, slow_period=slow_ema_val)
    
    # Apply date filter
    df_filtered = df_calc[(df_calc['date'].dt.date >= start_f) & (df_calc['date'].dt.date <= end_f)].copy().reset_index(drop=True)
    if df_filtered.empty:
        st.warning("No data found for the selected date filter range.")
        return
        
    sig_df = sig_df_all[(sig_df_all['signal_date'].dt.date >= start_f) & (sig_df_all['signal_date'].dt.date <= end_f)].copy().reset_index(drop=True)
    
    trades_df, stats = run_ema_crossover_backtest(
        df_filtered,
        sig_df,
        sl_pct=sl_val,
        tgt_pct=tgt_val,
        exit_on_opposite=opp_exit
    )
    
    if trades_df.empty:
        st.info(f"No EMA crossover signals generated for {selected_asset} with {fast_ema_val}/{slow_ema_val} EMA in the selected timeframe.")
        return

    # -------------------------------------------------------------
    # 4. EXECUTIVE KPI DASHBOARD
    # -------------------------------------------------------------
    st.markdown("---")
    st.markdown(f"### 📊 Performance Summary: {selected_asset} ({fast_ema_val} & {slow_ema_val} EMA Crossover)")
    
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    
    total_trades = stats.get('total_trades', 0)
    win_rate = stats.get('win_rate', 0.0)
    total_pnl = stats.get('total_pnl_pts', 0.0)
    pf = stats.get('profit_factor', np.nan)
    max_dd = stats.get('max_dd_pts', 0.0)
    avg_trade_pct = stats.get('avg_trade_pct', 0.0)
    
    pf_str = f"{pf:.2f}" if not np.isnan(pf) else "N/A"
    pnl_color = "#00d26a" if total_pnl >= 0 else "#ef4444"
    pts_label = "Points" if asset_type == "🇮🇳 NIFTY 50 (Index)" else "₹ / Share"
    
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-label">Total Trades</span>
            <div class="metric-value" style="color: #38bdf8;">{total_trades}</div>
            <span style="font-size: 11px; color: #94a3b8;">Executions</span>
        </div>
        """, unsafe_allow_html=True)
        
    with m2:
        wr_color = "#00d26a" if win_rate >= 50 else ("#fbbf24" if win_rate >= 40 else "#ef4444")
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-label">Win Rate</span>
            <div class="metric-value" style="color: {wr_color};">{win_rate:.1f}%</div>
            <span style="font-size: 11px; color: #94a3b8;">Profitability</span>
        </div>
        """, unsafe_allow_html=True)
        
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-label">Net PnL</span>
            <div class="metric-value" style="color: {pnl_color};">{total_pnl:+,.1f}</div>
            <span style="font-size: 11px; color: #94a3b8;">{pts_label}</span>
        </div>
        """, unsafe_allow_html=True)
        
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-label">Profit Factor</span>
            <div class="metric-value" style="color: #38bdf8;">{pf_str}</div>
            <span style="font-size: 11px; color: #94a3b8;">Gross Win / Loss</span>
        </div>
        """, unsafe_allow_html=True)
        
    with m5:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-label">Max Drawdown</span>
            <div class="metric-value" style="color: #ef4444;">{max_dd:,.1f}</div>
            <span style="font-size: 11px; color: #94a3b8;">Peak to Trough</span>
        </div>
        """, unsafe_allow_html=True)
        
    with m6:
        avg_color = "#00d26a" if avg_trade_pct >= 0 else "#ef4444"
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-label">Avg Trade Return</span>
            <div class="metric-value" style="color: {avg_color};">{avg_trade_pct:+.2f}%</div>
            <span style="font-size: 11px; color: #94a3b8;">Per Trade Edge</span>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 5. CUMULATIVE PNL & DRAWDOWN CHARTS
    # -------------------------------------------------------------
    trades_df['cum_pnl_pts'] = trades_df['pnl_pts'].cumsum()
    trades_df['peak_pnl_pts'] = trades_df['cum_pnl_pts'].cummax()
    trades_df['drawdown_pts'] = trades_df['cum_pnl_pts'] - trades_df['peak_pnl_pts']
    
    st.markdown("### 📈 Cumulative Equity Curve & Underwater Drawdown")
    fig_eq = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3],
        subplot_titles=(f"Cumulative Net PnL ({pts_label})", "Drawdown Depth (Points)")
    )
    
    fig_eq.add_trace(go.Scatter(
        x=trades_df['exit_date'],
        y=trades_df['cum_pnl_pts'],
        mode='lines+markers',
        name='Cumulative PnL',
        line=dict(color='#00d26a', width=2.5),
        marker=dict(size=4)
    ), row=1, col=1)
    
    fig_eq.add_trace(go.Scatter(
        x=trades_df['exit_date'],
        y=trades_df['peak_pnl_pts'],
        mode='lines',
        name='High Water Mark',
        line=dict(color='#38bdf8', width=1, dash='dot')
    ), row=1, col=1)
    
    fig_eq.add_trace(go.Scatter(
        x=trades_df['exit_date'],
        y=trades_df['drawdown_pts'],
        mode='lines',
        name='Drawdown Underwater',
        fill='tozeroy',
        line=dict(color='#ef4444', width=1.5),
        fillcolor='rgba(239, 68, 68, 0.25)'
    ), row=2, col=1)
    
    fig_eq.update_layout(
        template="plotly_dark",
        height=460,
        margin=dict(l=40, r=40, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_eq, width="stretch")

    # -------------------------------------------------------------
    # 6. INTERACTIVE CANDLESTICK CHART WITH EMA LINES & SIGNALS
    # -------------------------------------------------------------
    st.markdown(f"### 🕯️ Candlestick Price Action & EMA Crossover Signals ({selected_asset})")
    with st.expander("🔍 View Interactive Chart with Candlesticks, EMAs & Crossover Markers", expanded=True):
        fig_cand = go.Figure()
        
        # Price Candlestick
        fig_cand.add_trace(go.Candlestick(
            x=df_filtered['date'],
            open=df_filtered['open'],
            high=df_filtered['high'],
            low=df_filtered['low'],
            close=df_filtered['close'],
            name=selected_asset,
            increasing_line_color='#00d26a',
            decreasing_line_color='#ef4444'
        ))
        
        # Fast EMA Line
        fig_cand.add_trace(go.Scatter(
            x=df_filtered['date'],
            y=df_filtered['fast_ema'],
            mode='lines',
            name=f'Fast EMA ({fast_ema_val})',
            line=dict(color='#38bdf8', width=1.8)
        ))
        
        # Slow EMA Line
        fig_cand.add_trace(go.Scatter(
            x=df_filtered['date'],
            y=df_filtered['slow_ema'],
            mode='lines',
            name=f'Slow EMA ({slow_ema_val})',
            line=dict(color='#fbbf24', width=2.0)
        ))
        
        # BUY Markers (Golden Cross)
        buys = sig_df[sig_df['signal'] == 'BUY']
        if not buys.empty:
            fig_cand.add_trace(go.Scatter(
                x=buys['signal_date'],
                y=buys['crossover_price'],
                mode='markers+text',
                marker=dict(symbol='triangle-up', size=13, color='#00d26a'),
                text=["BUY" for _ in range(len(buys))],
                textposition='bottom center',
                name='Golden Cross (BUY)'
            ))
            
        # SELL Markers (Death Cross)
        sells = sig_df[sig_df['signal'] == 'SELL']
        if not sells.empty:
            fig_cand.add_trace(go.Scatter(
                x=sells['signal_date'],
                y=sells['crossover_price'],
                mode='markers+text',
                marker=dict(symbol='triangle-down', size=13, color='#ef4444'),
                text=["SELL" for _ in range(len(sells))],
                textposition='top center',
                name='Death Cross (SELL)'
            ))
            
        fig_cand.update_layout(
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=580,
            margin=dict(l=40, r=40, t=30, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_cand, width="stretch")

    # -------------------------------------------------------------
    # 7. CALENDAR YEAR BREAKDOWN
    # -------------------------------------------------------------
    st.markdown("### 📅 Calendar Year Performance Breakdown")
    trades_df['year'] = pd.to_datetime(trades_df['entry_date']).dt.year
    
    y_stats = []
    for yr, y_df in trades_df.groupby('year'):
        n_t = len(y_df)
        w_t = len(y_df[y_df['win'] == True])
        wr = (w_t / n_t) * 100.0 if n_t > 0 else 0
        pnl_pts = y_df['pnl_pts'].sum()
        avg_pct = y_df['pnl_pct'].mean()
        
        # Separate BUY and SELL
        b_df = y_df[y_df['action'] == 'BUY']
        n_b = len(b_df)
        w_b = len(b_df[b_df['win'] == True])
        wr_b = f"{(w_b / n_b) * 100.0:.1f}% ({w_b}/{n_b})" if n_b > 0 else "-"
        
        s_df = y_df[y_df['action'] == 'SELL']
        n_s = len(s_df)
        w_s = len(s_df[s_df['win'] == True])
        wr_s = f"{(w_s / n_s) * 100.0:.1f}% ({w_s}/{n_s})" if n_s > 0 else "-"
        
        y_stats.append({
            'Year': yr,
            'Trades': n_t,
            'Wins': w_t,
            'Losses': n_t - w_t,
            'Win Rate (%)': f"{wr:.1f}%",
            '🟢 BUY Win Rate': wr_b,
            '🔴 SELL Win Rate': wr_s,
            'Net Points': f"{pnl_pts:+,.1f}",
            'Avg Trade (%)': f"{avg_pct:+.2f}%",
            'raw_pnl': pnl_pts
        })
        
    y_df_display = pd.DataFrame(y_stats)
    if not y_df_display.empty:
        st.dataframe(y_df_display.drop(columns=['raw_pnl']), width="stretch", hide_index=True)

    # -------------------------------------------------------------
    # 8. COMPLETE TRADE AUDIT LOG
    # -------------------------------------------------------------
    st.markdown("### 📋 Complete Trade Audit Log")
    with st.expander("🔍 View Itemized Trade Records & Export CSV", expanded=False):
        audit_cols = [
            'signal_date', 'entry_date', 'exit_date', 'action', 'entry_price',
            'cross_price', 'sl_price', 'tgt_price', 'exit_price', 'exit_reason',
            'holding_bars', 'pnl_pts', 'pnl_pct', 'win'
        ]
        audit_df = trades_df[audit_cols].copy()
        audit_df['signal_date'] = audit_df['signal_date'].dt.strftime('%Y-%m-%d')
        audit_df['entry_date'] = audit_df['entry_date'].dt.strftime('%Y-%m-%d')
        audit_df['exit_date'] = audit_df['exit_date'].dt.strftime('%Y-%m-%d')
        audit_df['entry_price'] = audit_df['entry_price'].apply(lambda x: f"₹{x:,.2f}")
        audit_df['cross_price'] = audit_df['cross_price'].apply(lambda x: f"₹{x:,.2f}")
        audit_df['sl_price'] = audit_df['sl_price'].apply(lambda x: f"₹{x:,.2f}")
        audit_df['tgt_price'] = audit_df['tgt_price'].apply(lambda x: f"₹{x:,.2f}" if not pd.isna(x) else "Open-Ended")
        audit_df['exit_price'] = audit_df['exit_price'].apply(lambda x: f"₹{x:,.2f}")
        audit_df['pnl_pts'] = audit_df['pnl_pts'].apply(lambda x: f"{x:+,.1f}")
        audit_df['pnl_pct'] = audit_df['pnl_pct'].apply(lambda x: f"{x:+.2f}%")
        audit_df['win'] = audit_df['win'].apply(lambda x: "✅ Win" if x else "❌ Loss")
        audit_df.rename(columns={
            'signal_date': 'Signal Date',
            'entry_date': 'Entry Date',
            'exit_date': 'Exit Date',
            'action': 'Action',
            'entry_price': 'Entry Price',
            'cross_price': 'Crossover Point',
            'sl_price': 'Stop Loss',
            'tgt_price': 'Target',
            'exit_price': 'Exit Price',
            'exit_reason': 'Exit Reason',
            'holding_bars': 'Bars Held',
            'pnl_pts': 'PnL Points',
            'pnl_pct': 'PnL %',
            'win': 'Outcome'
        }, inplace=True)
        
        st.dataframe(audit_df, width="stretch", hide_index=True)
        
        csv_data = trades_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Backtest Trade Log (CSV)",
            data=csv_data,
            file_name=f"{selected_asset}_{timeframe}_EMA_{fast_ema_val}_{slow_ema_val}_Backtest.csv",
            mime="text/csv",
            key="sec7_download_csv"
        )
