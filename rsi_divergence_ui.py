import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# =============================================================
# DATA PATH CONFIGURATIONS (PORTABLE FOR LOCAL & STREAMLIT CLOUD)
# =============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

NIFTY_PARQUET_DIR = os.path.join(DATA_DIR, "nifty_daily_weekly_parquet") if os.path.exists(os.path.join(DATA_DIR, "nifty_daily_weekly_parquet")) else r"D:\NIFTY_MODEL\nifty_daily_weekly_parquet"
STOCK_DAILY_DIR = os.path.join(DATA_DIR, "stock_daily_parquet") if os.path.exists(os.path.join(DATA_DIR, "stock_daily_parquet")) else r"D:\NIFTY_MODEL\stock_daily_parquet"
STOCK_WEEKLY_DIR = os.path.join(DATA_DIR, "stock_weekly_parquet") if os.path.exists(os.path.join(DATA_DIR, "stock_weekly_parquet")) else r"D:\NIFTY_MODEL\stock_weekly_parquet"
STOCK_MONTHLY_DIR = os.path.join(DATA_DIR, "stock_monthly_parquet") if os.path.exists(os.path.join(DATA_DIR, "stock_monthly_parquet")) else r"D:\NIFTY_MODEL\stock_monthly_parquet"

@st.cache_data(show_spinner=False)
def get_available_stock_symbols():
    if not os.path.exists(STOCK_WEEKLY_DIR):
        return ["INFY", "RELIANCE", "TCS", "HDFCBANK"]
    symbols = [
        f.replace("_weekly_data_raw.parquet", "")
        for f in os.listdir(STOCK_WEEKLY_DIR)
        if f.endswith("_weekly_data_raw.parquet")
    ]
    return sorted(symbols)

@st.cache_data(show_spinner=False)
def load_asset_ohlc(asset_name: str, timeframe: str) -> pd.DataFrame:
    """
    Loads raw daily and weekly parquet data for NIFTY 50 or individual stocks:
    - NIFTY 50: from D:\\NIFTY_MODEL\\nifty_daily_weekly_parquet
    - Stocks: from D:\\NIFTY_MODEL\\stock_daily_parquet and stock_weekly_parquet
    - Monthly: aggregated from daily
    """
    if asset_name == "NIFTY 50":
        if timeframe == "Daily":
            p = os.path.join(NIFTY_PARQUET_DIR, "nifty_daily_data_raw.parquet")
            df = pd.read_parquet(p)
        elif timeframe == "Weekly":
            p = os.path.join(NIFTY_PARQUET_DIR, "nifty_weekly_data_raw.parquet")
            df = pd.read_parquet(p)
        else: # Monthly
            p = os.path.join(NIFTY_PARQUET_DIR, "nifty_daily_data_raw.parquet")
            df_d = pd.read_parquet(p)
            df_d['datetime'] = pd.to_datetime(df_d['datetime'])
            df = df_d.set_index('datetime').resample('ME').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna().reset_index()
    else: # Stock
        if timeframe == "Daily":
            p = os.path.join(STOCK_DAILY_DIR, f"{asset_name}_daily_data_raw.parquet")
            df = pd.read_parquet(p)
        elif timeframe == "Weekly":
            p = os.path.join(STOCK_WEEKLY_DIR, f"{asset_name}_weekly_data_raw.parquet")
            df = pd.read_parquet(p)
        else: # Monthly
            p = os.path.join(STOCK_DAILY_DIR, f"{asset_name}_daily_data_raw.parquet")
            df_d = pd.read_parquet(p)
            df_d['datetime'] = pd.to_datetime(df_d['datetime'])
            df = df_d.set_index('datetime').resample('ME').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna().reset_index()

    df['date'] = pd.to_datetime(df['datetime'])
    return df.sort_values('date').reset_index(drop=True)

# =============================================================
# RSI CALCULATION (MATCHING PINESCRIPT ta.rsi WITH WILDER RMA)
# =============================================================
def calculate_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).copy()
    loss = (-delta.where(delta < 0, 0)).copy()
    
    avg_gain = gain.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

# =============================================================
# PINESCRIPT PIVOT DETECTION & DIVERGENCE SIGNALS
# =============================================================
def detect_pivots_and_signals(df: pd.DataFrame, rsi_len: int = 14, lb_left: int = 5, lb_right: int = 5, tol: float = 0.002, price_basis: str = 'close'):
    """
    PineScript RSI Custom Divergence Indicator with configurable Price Basis:
    - price_basis='close': Standard confirmation on Candle Close (ta.rsi(close, len))
    - price_basis='open': Early signal evaluation on Candle Open (ta.rsi(open, len))
    - pl = ta.pivotlow(low, lbL, lbR)
    - ph = ta.pivothigh(high, lbL, lbR)
    """
    df = df.copy().reset_index(drop=True)
    basis_col = 'open' if price_basis.lower() == 'open' else 'close'
    df['rsi'] = calculate_rsi(df[basis_col], rsi_len)
    
    n = len(df)
    lows = df['low'].values
    highs = df['high'].values
    rsis = df['rsi'].values
    
    is_pivot_low = np.zeros(n, dtype=bool)
    is_pivot_high = np.zeros(n, dtype=bool)
    
    for p in range(lb_left, n - lb_right):
        val_low = lows[p]
        if all(val_low < lows[p - k] for k in range(1, lb_left + 1)) and \
           all(val_low <= lows[p + k] for k in range(1, lb_right + 1)):
            is_pivot_low[p] = True
            
        val_high = highs[p]
        if all(val_high > highs[p - k] for k in range(1, lb_left + 1)) and \
           all(val_high >= highs[p + k] for k in range(1, lb_right + 1)):
            is_pivot_high[p] = True
            
    last_low_price = None
    last_rsi_low = None
    last_high_price = None
    last_rsi_high = None
    
    signals = []
    
    for i in range(lb_left + lb_right, n):
        p = i - lb_right
        
        # Check Low Pivot
        if is_pivot_low[p]:
            curr_low_price = lows[p]
            curr_rsi_low = rsis[p]
            
            if last_low_price is not None and not np.isnan(curr_rsi_low) and not np.isnan(last_rsi_low):
                is_equal_low = abs(curr_low_price - last_low_price) / last_low_price <= tol
                if is_equal_low and curr_rsi_low < last_rsi_low:
                    signals.append({
                        'pivot_idx': p,
                        'confirm_idx': i,
                        'signal_date': df['date'].iloc[p],
                        'confirm_date': df['date'].iloc[i],
                        'signal': 'SELL',
                        'reason': 'Equal Lows (Support Retest) + Lower RSI Low',
                        'pivot_type': 'LOW',
                        'price': curr_low_price,
                        'signal_low': lows[p],
                        'signal_high': highs[p],
                        'signal_close': df['close'].iloc[p],
                        'rsi': curr_rsi_low,
                        'prev_price': last_low_price,
                        'prev_rsi': last_rsi_low
                    })
                elif curr_low_price <= last_low_price and curr_rsi_low > last_rsi_low:
                    signals.append({
                        'pivot_idx': p,
                        'confirm_idx': i,
                        'signal_date': df['date'].iloc[p],
                        'confirm_date': df['date'].iloc[i],
                        'signal': 'BUY',
                        'reason': 'Lower/Equal Price Low + Higher RSI Low (Bullish Div)',
                        'pivot_type': 'LOW',
                        'price': curr_low_price,
                        'signal_low': lows[p],
                        'signal_high': highs[p],
                        'signal_close': df['close'].iloc[p],
                        'rsi': curr_rsi_low,
                        'prev_price': last_low_price,
                        'prev_rsi': last_rsi_low
                    })
            last_low_price = curr_low_price
            last_rsi_low = curr_rsi_low
            
        # Check High Pivot
        if is_pivot_high[p]:
            curr_high_price = highs[p]
            curr_rsi_high = rsis[p]
            
            if last_high_price is not None and not np.isnan(curr_rsi_high) and not np.isnan(last_rsi_high):
                is_equal_high = abs(curr_high_price - last_high_price) / last_high_price <= tol
                if is_equal_high and curr_rsi_high > last_rsi_high:
                    signals.append({
                        'pivot_idx': p,
                        'confirm_idx': i,
                        'signal_date': df['date'].iloc[p],
                        'confirm_date': df['date'].iloc[i],
                        'signal': 'BUY',
                        'reason': 'Equal Highs (Resistance Retest) + Higher RSI High',
                        'pivot_type': 'HIGH',
                        'price': curr_high_price,
                        'signal_low': lows[p],
                        'signal_high': highs[p],
                        'signal_close': df['close'].iloc[p],
                        'rsi': curr_rsi_high,
                        'prev_price': last_high_price,
                        'prev_rsi': last_rsi_high
                    })
                elif curr_high_price >= last_high_price and curr_rsi_high < last_rsi_high:
                    signals.append({
                        'pivot_idx': p,
                        'confirm_idx': i,
                        'signal_date': df['date'].iloc[p],
                        'confirm_date': df['date'].iloc[i],
                        'signal': 'SELL',
                        'reason': 'Higher/Equal Price High + Lower RSI High (Bearish Div)',
                        'pivot_type': 'HIGH',
                        'price': curr_high_price,
                        'signal_low': lows[p],
                        'signal_high': highs[p],
                        'signal_close': df['close'].iloc[p],
                        'rsi': curr_rsi_high,
                        'prev_price': last_high_price,
                        'prev_rsi': last_rsi_high
                    })
            last_high_price = curr_high_price
            last_rsi_high = curr_rsi_high
            
    df_signals = pd.DataFrame(signals)
    return df, df_signals

# =============================================================
# BACKTEST METRIC CALCULATOR
# =============================================================
def compute_backtest_stats(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {}
        
    total_trades = len(trades_df)
    wins = trades_df[trades_df['win'] == True]
    losses = trades_df[trades_df['win'] == False]
    
    win_rate = (len(wins) / total_trades) * 100.0 if total_trades > 0 else 0.0
    total_pnl_pts = trades_df['pnl_pts'].sum()
    gross_profit = wins['pnl_pts'].sum() if not wins.empty else 0.0
    gross_loss = abs(losses['pnl_pts'].sum()) if not losses.empty else 1e-6
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.nan
    
    cum_pts = trades_df['pnl_pts'].cumsum()
    max_dd_pts = (cum_pts - cum_pts.cummax()).min()
    
    avg_win_pts = wins['pnl_pts'].mean() if not wins.empty else 0.0
    avg_loss_pts = losses['pnl_pts'].mean() if not losses.empty else 0.0
    reward_risk_ratio = abs(avg_win_pts / avg_loss_pts) if abs(avg_loss_pts) > 0 else np.nan
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_pnl_pts': total_pnl_pts,
        'profit_factor': profit_factor,
        'max_dd_pts': max_dd_pts,
        'avg_trade_pct': trades_df['pnl_pct'].mean(),
        'avg_win_pts': avg_win_pts,
        'avg_loss_pts': avg_loss_pts,
        'reward_risk_ratio': reward_risk_ratio,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss
    }

# =============================================================
# BACKTEST SIMULATION ENGINE
# =============================================================
def run_rsi_divergence_backtest(df: pd.DataFrame, sig_df: pd.DataFrame, sl_pct: float = 0.0, tgt_pct: float = 3.0, entry_mode: str = 'signal_next'):
    """
    Backtest simulation matching the exact user specifications:
    1. Entry: Next candle Open on which signal shows (Weekly, Daily, or Monthly).
    2. Exit:
       - Stop Loss from seekbar (if > 0%)
       - If seekbar is set to zero (0.0%):
         * For Long (BUY) entry -> Stoploss is Signal Candle Low
         * For Short (SELL) entry -> Stoploss is Signal Candle High
       - Target from seekbar (if > 0%)
       - If Target is 0.0% -> Open-ended ride until SL
    """
    df = df.copy().reset_index(drop=True)
    if sig_df.empty:
        return pd.DataFrame(), {}
        
    n_bars = len(df)
    date_to_idx = {d: i for i, d in enumerate(df['date'])}
    
    trades = []
    
    for _, sig in sig_df.iterrows():
        ref_date = sig['signal_date'] if entry_mode == 'signal_next' else sig['confirm_date']
        if ref_date not in date_to_idx:
            continue
        e_bar = date_to_idx[ref_date] + 1
        if e_bar >= n_bars:
            continue
            
        action = sig['signal']
        entry_price = df['open'].iloc[e_bar]
        entry_date = df['date'].iloc[e_bar]
        
        # Stop Loss Calculation
        if sl_pct == 0.0 or sl_pct is None:
            if action == 'BUY':
                sl_price = sig['signal_low']
            else:
                sl_price = sig['signal_high']
            sl_desc = "Signal Low/High"
        else:
            if action == 'BUY':
                sl_price = entry_price * (1.0 - sl_pct / 100.0)
            else:
                sl_price = entry_price * (1.0 + sl_pct / 100.0)
            sl_desc = f"{sl_pct:.1f}% Fixed"
            
        # Target Calculation
        if tgt_pct > 0.0:
            if action == 'BUY':
                tgt_price = entry_price * (1.0 + tgt_pct / 100.0)
            else:
                tgt_price = entry_price * (1.0 - tgt_pct / 100.0)
            tgt_desc = f"+{tgt_pct:.2f}%"
        else:
            tgt_price = None
            tgt_desc = "Open-Ended"
            
        exit_bar = None
        exit_price = None
        exit_date = None
        exit_reason = None
        
        for bar in range(e_bar, n_bars):
            b_high = df['high'].iloc[bar]
            b_low = df['low'].iloc[bar]
            b_date = df['date'].iloc[bar]
            
            if action == 'BUY':
                if b_low <= sl_price:
                    exit_bar = bar
                    exit_date = b_date
                    exit_price = sl_price
                    exit_reason = 'Stop Loss'
                    break
                elif tgt_price is not None and b_high >= tgt_price:
                    exit_bar = bar
                    exit_date = b_date
                    exit_price = tgt_price
                    exit_reason = 'Target'
                    break
            else: # SELL
                if b_high >= sl_price:
                    exit_bar = bar
                    exit_date = b_date
                    exit_price = sl_price
                    exit_reason = 'Stop Loss'
                    break
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
        pnl_pct = (pnl_pts / entry_price) * 100.0
        
        trades.append({
            'signal_date': sig['signal_date'],
            'entry_date': entry_date,
            'exit_date': exit_date,
            'action': action,
            'reason': sig['reason'],
            'pivot_type': sig['pivot_type'],
            'entry_price': entry_price,
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
# STREAMLIT UI COMPONENT FOR SECTION 4
# =============================================================
def render_rsi_divergence_section(analytics_engine):
    st.header("📐 Section 3: RSI Custom Divergence Backtest Engine")
    st.markdown("""
    Interactive quantitative backtest for your **RSI Custom Divergence Indicator** across **NIFTY 50** and **200+ F&O Stocks** on **Daily, Weekly, and Monthly** timeframes.
    Adjust the **Stop-Loss** and **Target** seekbars to immediately see recalculated performance statistics, equity growth curves, candlestick signals, and complete trade audits.
    """)
    
    # Asset Selection Panel
    st.markdown("#### 🎯 Asset & Timeframe Selection")
    ac1, ac2, ac3 = st.columns([2.0, 2.5, 1.8])
    
    with ac1:
        asset_type = st.radio(
            "Asset Universe:",
            ["🇮🇳 NIFTY 50 (Index)", "📈 Individual Stocks (F&O Universe)"],
            horizontal=True
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
            selected_asset = st.selectbox("Select Stock:", stocks, index=def_stock_idx)
            st.markdown(f"<span style='font-size: 12px; color: #38bdf8;'>Data Source: <code>{selected_asset} Stock Parquet</code></span>", unsafe_allow_html=True)
            
    with ac3:
        timeframe = st.selectbox("📅 Timeframe:", ["Daily", "Weekly", "Monthly"], index=1)


        
    # Load raw asset data
    raw_df = load_asset_ohlc(selected_asset, timeframe)
    min_data_date = raw_df['date'].min().date()
    max_data_date = raw_df['date'].max().date()
    
    # Controls & Seekbars
    st.markdown("#### ⚙️ Backtest Parameters & Seekbar Controls")
    c1, c2, c3, c4 = st.columns([1.8, 1.8, 2.2, 2.2])
    
    with c1:
        sl_val = st.slider(
            "🛑 Stop Loss Seekbar (%):",
            min_value=0.0,
            max_value=15.0,
            value=0.0,
            step=0.1,
            format="%.1f%%",
            help="Set to 0.0% to dynamically use the Signal Candle Low (for Long/BUY) or Signal Candle High (for Short/SELL). Any value > 0 sets a fixed percentage stop loss from entry."
        )
        if sl_val == 0.0:
            st.markdown("<span style='color:#38bdf8; font-size:12px;'>🛡️ <strong>SL Mode:</strong> Signal Low (Long) / High (Short)</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color:#fbbf24; font-size:12px;'>🛡️ <strong>SL Mode:</strong> Fixed {sl_val:.1f}% from Entry</span>", unsafe_allow_html=True)
            
    with c2:
        def_tgt = 3.0 if timeframe == "Daily" else (5.0 if timeframe == "Weekly" else 8.0)
        tgt_val = st.slider(
            "🎯 Target Seekbar (%):",
            min_value=0.0,
            max_value=30.0,
            value=float(def_tgt),
            step=0.25,
            format="%.2f%%",
            help="Set to 0.0% for open-ended trailing until Stop Loss is hit. Any value > 0 sets a fixed percentage target from entry."
        )
        if tgt_val == 0.0:
            st.markdown("<span style='color:#94a3b8; font-size:12px;'>🎯 <strong>Target Mode:</strong> Open-Ended (Ride Trend)</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color:#00d26a; font-size:12px;'>🎯 <strong>Target Mode:</strong> Fixed +{tgt_val:.2f}% from Entry</span>", unsafe_allow_html=True)
            
    with c3:
        entry_timing = st.radio(
            "🚀 Entry Execution Rule:",
            [
                "Next Open after Signal (p + 1)",
                "Next Open after Confirmation (p + lbR + 1)"
            ],
            index=0,
            help="TradingView displays labels on signal candle p. Real-time confirmation occurs after lbR right bars close."
        )
        entry_mode_key = 'signal_next' if "Signal" in entry_timing else 'confirm_next'

    with c4:
        price_basis_choice = st.radio(
            "📍 Signal Timing / Basis:",
            [
                "📊 Candle Close (Standard EOD)",
                "🌅 Candle Open (Early Signal)"
            ],
            index=0,
            help="Select whether RSI is calculated and divergence is triggered on candle Close or early on candle Open."
        )
        price_basis_arg = "open" if "Open" in price_basis_choice else "close"
        
    # Date Range Filter & Advanced Parameters
    with st.expander("🛠️ Date Range Filter & PineScript Indicator Settings", expanded=False):
        f1, f2, f3 = st.columns([1.5, 1.5, 2.0])
        with f1:
            date_filter_mode = st.selectbox(
                "Filter Date History:",
                [
                    "Full Available History (Inception - Present)",
                    "Last 10 Years (2016 - Present - Matches TradingView View)",
                    "Post-Covid (2020 - Present)",
                    "Custom Range"
                ],
                index=0
            )
        with f2:
            if date_filter_mode == "Custom Range":
                start_f = st.date_input("Filter Start Date:", value=min_data_date)
                end_f = st.date_input("Filter End Date:", value=max_data_date)
            elif date_filter_mode.startswith("Last 10 Years"):
                start_f = pd.to_datetime("2016-01-01").date()
                end_f = max_data_date
            elif date_filter_mode.startswith("Post-Covid"):
                start_f = pd.to_datetime("2020-01-01").date()
                end_f = max_data_date
            else:
                start_f = min_data_date
                end_f = max_data_date
                
        with f3:
            st.markdown(f"<span style='font-size:12px; color:#94a3b8;'>Active Period: <strong>{start_f}</strong> to <strong>{end_f}</strong></span>", unsafe_allow_html=True)
            
        st.markdown("---")
        pi1, pi2, pi3, pi4 = st.columns(4)
        with pi1:
            rsi_len = st.number_input("RSI Length:", min_value=5, max_value=50, value=14, step=1)
        with pi2:
            lb_left = st.number_input("Pivot Left Bars (lbL):", min_value=2, max_value=20, value=5, step=1)
        with pi3:
            lb_right = st.number_input("Pivot Right Bars (lbR):", min_value=2, max_value=20, value=5, step=1)
        with pi4:
            tol_pct = st.number_input("Equal Level Tolerance (%):", min_value=0.05, max_value=2.0, value=0.20, step=0.05, format="%.2f")
            tol_val = tol_pct / 100.0

    # Run Signal Detection on Complete Asset Data (so RSI warmup & history are seamless)
    df_calc, sig_df_full = detect_pivots_and_signals(
        raw_df,
        rsi_len=int(rsi_len),
        lb_left=int(lb_left),
        lb_right=int(lb_right),
        tol=tol_val,
        price_basis=price_basis_arg
    )
    
    # Filter for active date range
    mask_range = (df_calc['date'].dt.date >= start_f) & (df_calc['date'].dt.date <= end_f)
    df_filtered = df_calc[mask_range].copy().reset_index(drop=True)
    
    if not sig_df_full.empty:
        sig_df = sig_df_full[(sig_df_full['signal_date'].dt.date >= start_f) & (sig_df_full['signal_date'].dt.date <= end_f)].copy().reset_index(drop=True)
    else:
        sig_df = pd.DataFrame()
        
    if sig_df.empty:
        st.warning(f"No divergence signals found on {selected_asset} ({timeframe}) within the selected date window ({start_f} to {end_f}).")
        return

    # Run backtest on full continuous data series so all trades complete their full lifecycle
    trades_df_full, _ = run_rsi_divergence_backtest(df_calc, sig_df_full, sl_pct=sl_val, tgt_pct=tgt_val, entry_mode=entry_mode_key)
    
    if not trades_df_full.empty:
        trades_df = trades_df_full[
            (pd.to_datetime(trades_df_full['signal_date']).dt.date >= start_f) &
            (pd.to_datetime(trades_df_full['signal_date']).dt.date <= end_f)
        ].copy().reset_index(drop=True)
        stats = compute_backtest_stats(trades_df)
    else:
        trades_df = pd.DataFrame()
        stats = {}
        
    if trades_df.empty or not stats:
        st.warning(f"No executed trades found on {selected_asset} ({timeframe}) within the selected date window ({start_f} to {end_f}).")
        return

    # =============================================================
    # TOP PERFORMANCE METRIC CARDS
    # =============================================================
    st.markdown("---")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    
    total_trades = stats['total_trades']
    win_rate = stats['win_rate']
    total_pnl = stats['total_pnl_pts']
    pf = stats['profit_factor']
    max_dd = stats['max_dd_pts']
    avg_trade_pct = stats['avg_trade_pct']
    
    pnl_class = "metric-value" if total_pnl >= 0 else "metric-value-neg"
    pts_label = "Index Points" if selected_asset == "NIFTY 50" else "Stock Price Points"
    
    with m1:
        st.markdown(f"""<div class='metric-card'><span class='metric-label'>Total Trades</span><div class='metric-value' style='color:#38bdf8;'>{total_trades:,}</div><span style='font-size:11px;color:#94a3b8;'>{timeframe} Signals</span></div>""", unsafe_allow_html=True)
    with m2:
        wr_color = "#00d26a" if win_rate >= 50 else "#f8312f"
        st.markdown(f"""<div class='metric-card'><span class='metric-label'>Win Rate</span><div class='metric-value' style='color:{wr_color};'>{win_rate:.1f}%</div><span style='font-size:11px;color:#94a3b8;'>{(trades_df['win']).sum()} Wins / {(~trades_df['win']).sum()} Losses</span></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class='metric-card'><span class='metric-label'>Net Points PnL</span><div class='{pnl_class}'>{total_pnl:+,.1f}</div><span style='font-size:11px;color:#94a3b8;'>{pts_label}</span></div>""", unsafe_allow_html=True)
    with m4:
        pf_str = f"{pf:.2f}" if not np.isnan(pf) else "N/A"
        pf_color = "#00d26a" if pf >= 1.5 else ("#38bdf8" if pf >= 1.0 else "#f8312f")
        st.markdown(f"""<div class='metric-card'><span class='metric-label'>Profit Factor</span><div class='metric-value' style='color:{pf_color};'>{pf_str}</div><span style='font-size:11px;color:#94a3b8;'>Gross Win / Loss</span></div>""", unsafe_allow_html=True)
    with m5:
        st.markdown(f"""<div class='metric-card'><span class='metric-label'>Max Drawdown</span><div class='metric-value-neg'>{max_dd:,.1f}</div><span style='font-size:11px;color:#94a3b8;'>Peak-to-Trough Points</span></div>""", unsafe_allow_html=True)
    with m6:
        avg_col = "#00d26a" if avg_trade_pct >= 0 else "#f8312f"
        st.markdown(f"""<div class='metric-card'><span class='metric-label'>Avg Trade %</span><div class='metric-value' style='color:{avg_col};'>{avg_trade_pct:+.2f}%</div><span style='font-size:11px;color:#94a3b8;'>Per Executed Trade</span></div>""", unsafe_allow_html=True)

    # =============================================================
    # EQUITY CURVE & CUMULATIVE PNL CHART
    # =============================================================
    st.markdown(f"### 📈 Cumulative Equity Growth & Drawdown Analysis ({selected_asset})")
    trades_df['cum_pnl_pts'] = trades_df['pnl_pts'].cumsum()
    trades_df['peak_pnl_pts'] = trades_df['cum_pnl_pts'].cummax()
    trades_df['drawdown_pts'] = trades_df['cum_pnl_pts'] - trades_df['peak_pnl_pts']
    
    fig_eq = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3],
        subplot_titles=(f"Cumulative Net PnL ({pts_label})", "Drawdown Depth (Points)")
    )
                           
    fig_eq.add_trace(go.Scatter(
        x=trades_df['exit_date'],
        y=trades_df['cum_pnl_pts'],
        mode='lines+markers',
        name='Cumulative PnL (Points)',
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
        height=480,
        margin=dict(l=40, r=40, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_eq, width="stretch")

    # =============================================================
    # INTERACTIVE CANDLESTICK & RSI INDICATOR CHART
    # =============================================================
    st.markdown(f"### 🕯️ Candlestick Price Action & RSI Divergence Signals ({selected_asset})")
    with st.expander("🔍 View Interactive Chart with Signal Markers & RSI Pane", expanded=True):
        fig_cand = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
            row_heights=[0.75, 0.25],
            subplot_titles=(f"{selected_asset} ({timeframe}) with Divergence Labels", "RSI (14)")
        )
                                 
        # Candlestick
        fig_cand.add_trace(go.Candlestick(
            x=df_filtered['date'],
            open=df_filtered['open'],
            high=df_filtered['high'],
            low=df_filtered['low'],
            close=df_filtered['close'],
            name=selected_asset,
            increasing_line_color='#00d26a',
            decreasing_line_color='#ef4444'
        ), row=1, col=1)
        
        # BUY Markers
        buys = sig_df[sig_df['signal'] == 'BUY']
        if not buys.empty:
            fig_cand.add_trace(go.Scatter(
                x=buys['signal_date'],
                y=buys['price'],
                mode='markers+text',
                marker=dict(symbol='triangle-up', size=13, color='#00d26a'),
                text=["BUY" for _ in range(len(buys))],
                textposition='bottom center',
                name='BUY Signal'
            ), row=1, col=1)
            
        # SELL Markers
        sells = sig_df[sig_df['signal'] == 'SELL']
        if not sells.empty:
            fig_cand.add_trace(go.Scatter(
                x=sells['signal_date'],
                y=sells['price'],
                mode='markers+text',
                marker=dict(symbol='triangle-down', size=13, color='#ef4444'),
                text=["SELL" for _ in range(len(sells))],
                textposition='top center',
                name='SELL Signal'
            ), row=1, col=1)
            
        # RSI Pane
        fig_cand.add_trace(go.Scatter(
            x=df_filtered['date'],
            y=df_filtered['rsi'],
            mode='lines',
            name='RSI',
            line=dict(color='#38bdf8', width=1.5)
        ), row=2, col=1)
        
        fig_cand.add_hline(y=70, line_dash="dash", line_color="#ef4444", line_width=1, row=2, col=1)
        fig_cand.add_hline(y=30, line_dash="dash", line_color="#00d26a", line_width=1, row=2, col=1)
        fig_cand.add_hline(y=50, line_dash="dot", line_color="#94a3b8", line_width=0.8, row=2, col=1)
        
        fig_cand.update_layout(
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=580,
            margin=dict(l=40, r=40, t=30, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_cand, width="stretch")

    # =============================================================
    # YEAR-BY-YEAR PERFORMANCE AUDIT
    # =============================================================
    st.markdown("### 📅 Calendar Year Performance Breakdown")
    trades_df['year'] = pd.to_datetime(trades_df['entry_date']).dt.year
    
    y_stats = []
    for yr, y_df in trades_df.groupby('year'):
        n_t = len(y_df)
        w_t = len(y_df[y_df['win'] == True])
        wr = (w_t / n_t) * 100.0 if n_t > 0 else 0
        pnl_pts = y_df['pnl_pts'].sum()
        avg_pct = y_df['pnl_pct'].mean()
        
        # Separate Buy and Sell win rates
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
    
    yc1, yc2 = st.columns([1.5, 1.5])
    with yc1:
        st.dataframe(y_df_display.drop(columns=['raw_pnl']), width="stretch", hide_index=True)
    with yc2:
        fig_y = go.Figure()
        colors = ['#00d26a' if p >= 0 else '#ef4444' for p in y_df_display['raw_pnl']]
        fig_y.add_trace(go.Bar(
            x=y_df_display['Year'],
            y=y_df_display['raw_pnl'],
            marker_color=colors,
            text=[f"{p:+,.0f}" for p in y_df_display['raw_pnl']],
            textposition='auto'
        ))
        fig_y.update_layout(
            title=f"Annual Points PnL Distribution ({selected_asset})",
            xaxis_title="Calendar Year",
            yaxis_title=f"Net Points ({pts_label})",
            template="plotly_dark",
            height=320,
            margin=dict(l=30, r=30, t=40, b=30)
        )
        st.plotly_chart(fig_y, width="stretch")

    # =============================================================
    # COMPLETE INTERACTIVE TRADE LOG
    # =============================================================
    st.markdown("### 📋 Complete Executed Trade Log")
    
    disp_cols = ['signal_date', 'entry_date', 'exit_date', 'action', 'reason', 'entry_price', 'sl_price', 'tgt_price', 'exit_price', 'exit_reason', 'holding_bars', 'pnl_pts', 'pnl_pct']
    
    trade_table = trades_df[disp_cols].copy()
    trade_table['signal_date'] = pd.to_datetime(trade_table['signal_date']).dt.strftime('%Y-%m-%d')
    trade_table['entry_date'] = pd.to_datetime(trade_table['entry_date']).dt.strftime('%Y-%m-%d')
    trade_table['exit_date'] = pd.to_datetime(trade_table['exit_date']).dt.strftime('%Y-%m-%d')
    trade_table['entry_price'] = trade_table['entry_price'].map(lambda x: f"₹{x:,.2f}")
    trade_table['sl_price'] = trade_table['sl_price'].map(lambda x: f"₹{x:,.2f}")
    trade_table['tgt_price'] = trade_table['tgt_price'].map(lambda x: f"₹{x:,.2f}" if not np.isnan(x) else "Open")
    trade_table['exit_price'] = trade_table['exit_price'].map(lambda x: f"₹{x:,.2f}")
    trade_table['pnl_pts'] = trade_table['pnl_pts'].map(lambda x: f"{x:+,.1f}")
    trade_table['pnl_pct'] = trade_table['pnl_pct'].map(lambda x: f"{x:+.2f}%")
    
    st.dataframe(trade_table, width="stretch", hide_index=True)
    
    # Download Trade Log CSV
    csv_bytes = trades_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📥 Download Full {selected_asset} ({timeframe}) Trade Log (CSV)",
        data=csv_bytes,
        file_name=f"{selected_asset.lower()}_rsi_divergence_{timeframe.lower()}_trades.csv",
        mime="text/csv"
    )


