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

# =============================================================
# BENCHMARK BLUE-CHIP CONSTITUENTS & PORTFOLIO RESOLVER
# =============================================================
NIFTY_50_BENCHMARK_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "BAJFINANCE", "ASIANPAINT", "MARUTI", "HCLTECH", "SUNPHARMA", "TITAN", "ULTRACEMCO", "TATAMOTORS",
    "NTPC", "POWERGRID", "ONGC", "JSWSTEEL", "ADANIENT", "ADANIPORTS", "COALINDIA", "TATASTEEL", "BAJAJFINSV", "M&M",
    "GRASIM", "TECHM", "NESTLEIND", "CIPLA", "WIPRO", "HINDALCO", "DRREDDY", "EICHERMOT", "BPCL", "APOLLOHOSP",
    "BRITANNIA", "TATACONSUM", "SBILIFE", "DIVISLAB", "HEROMOTOCO", "BAJAJ-AUTO", "LTIM", "HDFCLIFE", "SHRIRAMFIN", "BEL"
]

def get_portfolio_universe_symbols(scope: str, available_stocks: list) -> list:
    """Returns the subset of stock symbols matching the selected scope."""
    if "Top 50" in scope:
        matched = [s for s in NIFTY_50_BENCHMARK_SYMBOLS if s in available_stocks]
        if len(matched) < 50:
            remaining = [s for s in available_stocks if s not in matched]
            matched.extend(remaining[:50 - len(matched)])
        return matched[:50]
    elif "Top 100" in scope:
        top50 = [s for s in NIFTY_50_BENCHMARK_SYMBOLS if s in available_stocks]
        remaining = [s for s in available_stocks if s not in top50]
        return (top50 + remaining)[:100]
    elif "All Nifty 500" in scope:
        return available_stocks
    return available_stocks[:50]


@st.cache_data(show_spinner=False)
def load_asset_ohlc(asset_name: str, timeframe: str) -> pd.DataFrame:
    """
    Loads raw daily and weekly parquet data for NIFTY 50 or individual stocks:
    - NIFTY 50: from D:\\NIFTY_MODEL\\nifty_daily_weekly_parquet
    - Stocks: from D:\\NIFTY_MODEL\\stock_daily_parquet and stock_weekly_parquet
    - Monthly: aggregated from daily
    """
    try:
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
    except Exception:
        return pd.DataFrame()


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
    
    avg_holding_days = trades_df['holding_days'].mean() if 'holding_days' in trades_df.columns and not trades_df.empty else 0.0
    min_holding_days = int(trades_df['holding_days'].min()) if 'holding_days' in trades_df.columns and not trades_df.empty else 0
    max_holding_days = int(trades_df['holding_days'].max()) if 'holding_days' in trades_df.columns and not trades_df.empty else 0

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
        'gross_loss': gross_loss,
        'avg_holding_days': avg_holding_days,
        'min_holding_days': min_holding_days,
        'max_holding_days': max_holding_days
    }


# =============================================================
# BACKTEST SIMULATION ENGINE
# =============================================================
def run_rsi_divergence_backtest(
    df: pd.DataFrame,
    sig_df: pd.DataFrame,
    sl_pct: float = 0.0,
    tgt_pct: float = 3.0,
    entry_mode: str = 'signal_next',
    enable_ema_filter: bool = False,
    ema_period: int = 21
):
    """
    Backtest simulation matching user specifications with optional Anti-Chasing EMA filter:
    1. Entry: Next candle Open on which signal shows (Weekly, Daily, or Monthly).
    2. Exit:
       - Stop Loss from seekbar (if > 0%)
       - If seekbar is set to zero (0.0%):
         * For Long (BUY) entry -> Stoploss is Signal Candle Low
         * For Short (SELL) entry -> Stoploss is Signal Candle High
       - Target from seekbar (if > 0%)
       - If Target is 0.0% -> Open-ended ride until SL
    3. Anti-Chasing EMA Filter (when enable_ema_filter=True):
       - If Candle Close < EMA on SELL -> Skip signal (avoid selling into the bottom)
       - If Candle Close < EMA on BUY  -> Take signal (dip buying)
       - If Candle Close > EMA on BUY  -> Skip signal (avoid buying at peak)
       - If Candle Close > EMA on SELL -> Take signal (top selling)
       - Evaluates skipped signals to audit whether skipping was Correct (saved a loss) or Wrong (missed a win).
    """
    df = df.copy().reset_index(drop=True)
    if sig_df.empty:
        return pd.DataFrame(), {}, pd.DataFrame(), {}
        
    n_bars = len(df)
    date_to_idx = {d: i for i, d in enumerate(df['date'])}
    
    if enable_ema_filter and 'ema' not in df.columns:
        df['ema'] = df['close'].ewm(span=int(ema_period), adjust=False).mean()
        
    active_trades = []
    skipped_trades = []
    all_trades = []
    
    for _, sig in sig_df.iterrows():
        ref_date = sig['signal_date'] if entry_mode == 'signal_next' else sig['confirm_date']
        if ref_date not in date_to_idx:
            continue
        ref_bar = date_to_idx[ref_date]
        e_bar = ref_bar + 1
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
        is_win = pnl_pts > 0
        
        e_dt = pd.to_datetime(entry_date)
        x_dt = pd.to_datetime(exit_date)
        cal_days = int((x_dt - e_dt).days)
        bar_interval = int((df['date'].iloc[1] - df['date'].iloc[0]).days) if len(df) > 1 else 1
        if cal_days == 0:
            holding_days = 7 if bar_interval > 4 else (30 if bar_interval > 20 else 1)
        else:
            holding_days = cal_days

        trade_data = {
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
            'holding_days': holding_days,
            'pnl_pts': pnl_pts,
            'pnl_pct': pnl_pct,
            'win': is_win,
            'sl_desc': sl_desc,
            'tgt_desc': tgt_desc
        }
        all_trades.append(trade_data)

        
        if enable_ema_filter:
            close_price = df['close'].iloc[ref_bar]
            ema_val = df['ema'].iloc[ref_bar]
            is_skipped = False
            skip_reason = ''
            
            if close_price < ema_val and action == 'SELL':
                is_skipped = True
                skip_reason = f"Close (₹{close_price:,.2f}) < {ema_period} EMA (₹{ema_val:,.2f}) on SELL"
            elif close_price > ema_val and action == 'BUY':
                is_skipped = True
                skip_reason = f"Close (₹{close_price:,.2f}) > {ema_period} EMA (₹{ema_val:,.2f}) on BUY"
                
            if is_skipped:
                is_correct = not is_win
                verdict_str = "🟢 Correct Skip (Saved Loss)" if is_correct else "🔴 Wrong Skip (Missed Win)"
                skipped_item = dict(trade_data)
                skipped_item.update({
                    'signal_close': close_price,
                    'ema_val': ema_val,
                    'skip_reason': skip_reason,
                    'is_correct_skip': is_correct,
                    'verdict': verdict_str
                })
                skipped_trades.append(skipped_item)
            else:
                active_trades.append(trade_data)
        else:
            active_trades.append(trade_data)
            
    df_active = pd.DataFrame(active_trades)
    df_skipped = pd.DataFrame(skipped_trades)
    df_all = pd.DataFrame(all_trades)
    
    stats_active = compute_backtest_stats(df_active)
    stats_all = compute_backtest_stats(df_all)
    
    return df_active, stats_active, df_skipped, stats_all

# =============================================================
# PORTFOLIO AGGREGATE BACKTEST ENGINE (ALL STOCKS COMBINED)
# =============================================================
@st.cache_data(show_spinner=False)
def run_portfolio_combined_backtest(
    symbols_tuple: tuple,
    timeframe: str,
    sl_pct: float,
    tgt_pct: float,
    entry_mode: str,
    price_basis: str,
    rsi_len: int,
    lb_left: int,
    lb_right: int,
    tol_val: float,
    enable_ema_filter: bool,
    ema_period: int,
    start_date_str: str,
    end_date_str: str
):
    """
    Executes RSI Divergence backtesting across an entire universe of stocks.
    Aggregates all individual trades into a portfolio-wide dataset.
    """
    start_dt = pd.to_datetime(start_date_str).date() if start_date_str else None
    end_dt = pd.to_datetime(end_date_str).date() if end_date_str else None
    
    all_trades = []
    all_skips = []
    stock_stats_list = []
    
    for sym in symbols_tuple:
        try:
            raw_df = load_asset_ohlc(sym, timeframe)
            if raw_df.empty or len(raw_df) < (rsi_len + lb_left + lb_right + 10):
                continue
            
            df_calc, sig_df_full = detect_pivots_and_signals(
                raw_df, rsi_len=rsi_len, lb_left=lb_left, lb_right=lb_right, tol=tol_val, price_basis=price_basis
            )
            
            if enable_ema_filter:
                df_calc['ema'] = df_calc['close'].ewm(span=int(ema_period), adjust=False).mean()
                
            if not sig_df_full.empty and start_dt and end_dt:
                sig_df = sig_df_full[(sig_df_full['signal_date'].dt.date >= start_dt) & (sig_df_full['signal_date'].dt.date <= end_dt)].copy().reset_index(drop=True)
            else:
                sig_df = sig_df_full
                
            if sig_df.empty:
                continue
                
            t_df, _, sk_df, _ = run_rsi_divergence_backtest(
                df_calc, sig_df, sl_pct=sl_pct, tgt_pct=tgt_pct, entry_mode=entry_mode,
                enable_ema_filter=enable_ema_filter, ema_period=int(ema_period)
            )
            
            if not t_df.empty:
                t_df['symbol'] = sym
                all_trades.append(t_df)
                
                n_t = len(t_df)
                n_w = int((t_df['win'] == True).sum())
                wr = (n_w / n_t) * 100.0 if n_t > 0 else 0.0
                pos_ret = t_df[t_df['pnl_pct'] > 0]['pnl_pct'].sum()
                neg_ret = abs(t_df[t_df['pnl_pct'] < 0]['pnl_pct'].sum())
                pf = (pos_ret / neg_ret) if neg_ret > 0 else (99.9 if pos_ret > 0 else 0.0)
                
                stock_stats_list.append({
                    'Symbol': sym,
                    'Trades': n_t,
                    'Win Rate (%)': wr,
                    'Wins': n_w,
                    'Losses': n_t - n_w,
                    'Profit Factor': pf,
                    'Avg Return (%)': t_df['pnl_pct'].mean(),
                    'Total Return (%)': t_df['pnl_pct'].sum(),
                    'Best Trade (%)': t_df['pnl_pct'].max(),
                    'Worst Trade (%)': t_df['pnl_pct'].min(),
                    'Avg Holding (Days)': t_df['holding_days'].mean(),
                    'Min Holding (Days)': int(t_df['holding_days'].min()),
                    'Max Holding (Days)': int(t_df['holding_days'].max()),
                    'Avg Holding (Bars)': t_df['holding_bars'].mean()
                })
                
            if not sk_df.empty:
                sk_df['symbol'] = sym
                all_skips.append(sk_df)
                
        except Exception:
            continue
            
    df_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    df_skips = pd.concat(all_skips, ignore_index=True) if all_skips else pd.DataFrame()
    df_stocks = pd.DataFrame(stock_stats_list)
    
    if not df_trades.empty:
        df_trades['entry_date'] = pd.to_datetime(df_trades['entry_date'])
        df_trades['exit_date'] = pd.to_datetime(df_trades['exit_date'])
        df_trades['year'] = df_trades['entry_date'].dt.year
        df_trades = df_trades.sort_values('exit_date').reset_index(drop=True)
        
        total_trades = len(df_trades)
        wins = int((df_trades['win'] == True).sum())
        losses = total_trades - wins
        win_rate = (wins / total_trades) * 100.0 if total_trades > 0 else 0.0
        
        years = sorted(df_trades['year'].unique())
        num_years = max(len(years), 1)
        avg_yearly_signals = total_trades / num_years
        
        unique_stocks = df_trades['symbol'].nunique()
        avg_signals_per_stock_yr = avg_yearly_signals / max(unique_stocks, 1)
        
        buy_t = df_trades[df_trades['action'] == 'BUY']
        buy_wins = int((buy_t['win'] == True).sum())
        buy_wr = (buy_wins / len(buy_t) * 100.0) if len(buy_t) > 0 else 0.0
        
        sell_t = df_trades[df_trades['action'] == 'SELL']
        sell_wins = int((sell_t['win'] == True).sum())
        sell_wr = (sell_wins / len(sell_t) * 100.0) if len(sell_t) > 0 else 0.0
        
        pos_rets = df_trades[df_trades['pnl_pct'] > 0]['pnl_pct'].sum()
        neg_rets = abs(df_trades[df_trades['pnl_pct'] < 0]['pnl_pct'].sum())
        pf = (pos_rets / neg_rets) if neg_rets > 0 else (99.9 if pos_rets > 0 else 0.0)
        
        port_stats = {
            'total_trades': total_trades,
            'total_wins': wins,
            'total_losses': losses,
            'win_rate': win_rate,
            'active_years': years,
            'num_years': num_years,
            'avg_yearly_signals': avg_yearly_signals,
            'unique_stocks': unique_stocks,
            'avg_signals_per_stock_yr': avg_signals_per_stock_yr,
            'buy_trades': len(buy_t),
            'buy_wins': buy_wins,
            'buy_win_rate': buy_wr,
            'sell_trades': len(sell_t),
            'sell_wins': sell_wins,
            'sell_win_rate': sell_wr,
            'avg_trade_pct': df_trades['pnl_pct'].mean(),
            'total_return_pct': df_trades['pnl_pct'].sum(),
            'profit_factor': pf,
            'best_trade_pct': df_trades['pnl_pct'].max(),
            'worst_trade_pct': df_trades['pnl_pct'].min(),
            'avg_holding_bars': df_trades['holding_bars'].mean(),
            'avg_holding_days': df_trades['holding_days'].mean(),
            'min_holding_days': int(df_trades['holding_days'].min()),
            'max_holding_days': int(df_trades['holding_days'].max())
        }
    else:
        port_stats = {}

        
    return df_trades, df_skips, df_stocks, port_stats


def render_portfolio_combined_view(
    symbols_to_run: list,
    timeframe: str,
    sl_val: float,
    tgt_val: float,
    entry_mode_key: str,
    price_basis_arg: str,
    rsi_len: int,
    lb_left: int,
    lb_right: int,
    tol_val: float,
    enable_ema_filter: bool,
    ema_period: int,
    start_f,
    end_f,
    scope_label: str
):
    """
    Renders the comprehensive All Stocks Combined portfolio view for Section 3.
    """
    st.markdown("---")
    st.markdown("### 🌐 All Stocks Combined Portfolio Backtest Engine")
    basis_str = "🌅 Candle Open (Early)" if price_basis_arg == "open" else "📊 Candle Close (Standard)"
    sl_str = f"{sl_val:.1f}% Fixed" if sl_val > 0 else "Dynamic Signal Low/High"
    tgt_str = f"+{tgt_val:.2f}% Fixed" if tgt_val > 0 else "Open-Ended (Ride Trend)"
    ema_str = f"Active ({ema_period} EMA)" if enable_ema_filter else "Disabled"
    
    st.markdown(f"""
    <div style='background:rgba(15, 23, 42, 0.6); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 8px; padding: 12px 18px; margin-bottom: 20px; font-size: 13px; color: #cbd5e1;'>
        <strong>Universe Target:</strong> <span style='color:#38bdf8;'>{len(symbols_to_run)} Equities ({scope_label})</span> &nbsp;|&nbsp;
        <strong>Timeframe:</strong> <span style='color:#38bdf8;'>{timeframe}</span> &nbsp;|&nbsp;
        <strong>Date Window:</strong> <span style='color:#38bdf8;'>{start_f} to {end_f}</span> &nbsp;|&nbsp;
        <strong>Signal Basis:</strong> <span style='color:#38bdf8;'>{basis_str}</span><br>
        <strong>Stop Loss:</strong> <span style='color:#fbbf24;'>{sl_str}</span> &nbsp;|&nbsp;
        <strong>Target:</strong> <span style='color:#00d26a;'>{tgt_str}</span> &nbsp;|&nbsp;
        <strong>Anti-Chasing EMA Filter:</strong> <span style='color:#38bdf8;'>{ema_str}</span>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner(f"⚡ Backtesting RSI Divergence across {len(symbols_to_run)} equities ({timeframe})..."):
        trades_df, skipped_df, stock_summary_df, port_stats = run_portfolio_combined_backtest(
            symbols_tuple=tuple(symbols_to_run),
            timeframe=timeframe,
            sl_pct=sl_val,
            tgt_pct=tgt_val,
            entry_mode=entry_mode_key,
            price_basis=price_basis_arg,
            rsi_len=rsi_len,
            lb_left=lb_left,
            lb_right=lb_right,
            tol_val=tol_val,
            enable_ema_filter=enable_ema_filter,
            ema_period=ema_period,
            start_date_str=str(start_f),
            end_date_str=str(end_f)
        )
        
    if trades_df.empty or not port_stats:
        st.warning(f"No executed trades found across the selected {len(symbols_to_run)} stocks within the active date window ({start_f} to {end_f}).")
        if enable_ema_filter and not skipped_df.empty:
            st.info(f"💡 Notice: The {ema_period} EMA filter skipped all {len(skipped_df)} divergence signals found across this universe.")
        return
        
    # =============================================================
    # 1. EXECUTIVE PORTFOLIO METRIC CARDS
    # =============================================================
    total_trades = port_stats['total_trades']
    total_wins = port_stats['total_wins']
    total_losses = port_stats['total_losses']
    win_rate = port_stats['win_rate']
    avg_yearly_signals = port_stats['avg_yearly_signals']
    unique_stocks = port_stats['unique_stocks']
    avg_per_stock_yr = port_stats['avg_signals_per_stock_yr']
    num_years = port_stats['num_years']
    years_range = f"{port_stats['active_years'][0]}–{port_stats['active_years'][-1]}" if port_stats['active_years'] else ""
    
    buy_trades = port_stats['buy_trades']
    buy_wins = port_stats['buy_wins']
    buy_wr = port_stats['buy_win_rate']
    
    sell_trades = port_stats['sell_trades']
    sell_wins = port_stats['sell_wins']
    sell_wr = port_stats['sell_win_rate']
    
    avg_trade_pct = port_stats['avg_trade_pct']
    pf = port_stats['profit_factor']
    pf_str = f"{pf:.2f}" if not np.isnan(pf) else "N/A"
    
    avg_h_days = port_stats.get('avg_holding_days', 0.0)
    min_h_days = port_stats.get('min_holding_days', 0)
    max_h_days = port_stats.get('max_holding_days', 0)
    
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    
    wr_color = "#00d26a" if win_rate >= 50 else "#f8312f"
    with m1:
        st.markdown(f"""
        <div class='metric-card'>
            <span class='metric-label'>Total Success Rate</span>
            <div class='metric-value' style='color:{wr_color};'>{win_rate:.1f}%</div>
            <span style='font-size:11px;color:#94a3b8;'>{total_wins:,} Wins / {total_losses:,} Losses</span>
        </div>
        """, unsafe_allow_html=True)
        
    with m2:
        st.markdown(f"""
        <div class='metric-card'>
            <span class='metric-label'>Total Signals</span>
            <div class='metric-value' style='color:#38bdf8;'>{total_trades:,}</div>
            <span style='font-size:11px;color:#94a3b8;'>Across {unique_stocks} Equities ({timeframe})</span>
        </div>
        """, unsafe_allow_html=True)
        
    with m3:
        st.markdown(f"""
        <div class='metric-card'>
            <span class='metric-label'>Avg Yearly Signals</span>
            <div class='metric-value' style='color:#fbbf24;'>{avg_yearly_signals:.1f} / yr</div>
            <span style='font-size:11px;color:#94a3b8;'>~{avg_per_stock_yr:.1f}/stock/yr ({years_range})</span>
        </div>
        """, unsafe_allow_html=True)
        
    with m4:
        b_col = "#00d26a" if buy_wr >= 50 else "#f8312f"
        st.markdown(f"""
        <div class='metric-card'>
            <span class='metric-label'>🟢 BUY Success Rate</span>
            <div class='metric-value' style='color:{b_col};'>{buy_wr:.1f}%</div>
            <span style='font-size:11px;color:#94a3b8;'>{buy_wins:,} Wins / {buy_trades:,} Signals</span>
        </div>
        """, unsafe_allow_html=True)
        
    with m5:
        s_col = "#00d26a" if sell_wr >= 50 else "#f8312f"
        st.markdown(f"""
        <div class='metric-card'>
            <span class='metric-label'>🔴 SELL Success Rate</span>
            <div class='metric-value' style='color:{s_col};'>{sell_wr:.1f}%</div>
            <span style='font-size:11px;color:#94a3b8;'>{sell_wins:,} Wins / {sell_trades:,} Signals</span>
        </div>
        """, unsafe_allow_html=True)
        
    with m6:
        avg_col = "#00d26a" if avg_trade_pct >= 0 else "#f8312f"
        st.markdown(f"""
        <div class='metric-card'>
            <span class='metric-label'>Avg Trade Return</span>
            <div class='metric-value' style='color:{avg_col};'>{avg_trade_pct:+.2f}%</div>
            <span style='font-size:11px;color:#94a3b8;'>Profit Factor: {pf_str}</span>
        </div>
        """, unsafe_allow_html=True)

    with m7:
        st.markdown(f"""
        <div class='metric-card'>
            <span class='metric-label'>Avg Holding Days</span>
            <div class='metric-value' style='color:#a78bfa;'>{avg_h_days:.1f}d</div>
            <span style='font-size:11px;color:#94a3b8;'>Min: {min_h_days}d | Max: {max_h_days}d</span>
        </div>
        """, unsafe_allow_html=True)
        
    # =============================================================
    # 2. ANTI-CHASING EMA FILTER COLLECTIVE AUDIT (IF ACTIVE)
    # =============================================================
    if enable_ema_filter and not skipped_df.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"### 🛡️ Anti-Chasing {ema_period} EMA Filter Impact (Portfolio Collective Audit)")
        
        num_skips = len(skipped_df)
        all_combined = pd.concat([trades_df, skipped_df], ignore_index=True)
        total_raw_signals = len(all_combined)
        num_correct = int((skipped_df['is_correct_skip'] == True).sum())
        num_wrong = int((skipped_df['is_correct_skip'] == False).sum())
        pct_correct = (num_correct / num_skips * 100.0) if num_skips > 0 else 0.0
        pct_wrong = (num_wrong / num_skips * 100.0) if num_skips > 0 else 0.0
        net_saved_pct = -float(skipped_df['pnl_pct'].sum())
        
        # Directional Breakdown
        buy_skips = skipped_df[skipped_df['action'] == 'BUY']
        num_buy_skips = len(buy_skips)
        buy_correct = int((buy_skips['is_correct_skip'] == True).sum()) if num_buy_skips > 0 else 0
        buy_wrong = int((buy_skips['is_correct_skip'] == False).sum()) if num_buy_skips > 0 else 0
        buy_corr_pct = (buy_correct / num_buy_skips * 100.0) if num_buy_skips > 0 else 0.0
        buy_wrng_pct = (buy_wrong / num_buy_skips * 100.0) if num_buy_skips > 0 else 0.0
        net_saved_buy = -float(buy_skips['pnl_pct'].sum()) if num_buy_skips > 0 else 0.0

        sell_skips = skipped_df[skipped_df['action'] == 'SELL']
        num_sell_skips = len(sell_skips)
        sell_correct = int((sell_skips['is_correct_skip'] == True).sum()) if num_sell_skips > 0 else 0
        sell_wrong = int((sell_skips['is_correct_skip'] == False).sum()) if num_sell_skips > 0 else 0
        sell_corr_pct = (sell_correct / num_sell_skips * 100.0) if num_sell_skips > 0 else 0.0
        sell_wrng_pct = (sell_wrong / num_sell_skips * 100.0) if num_sell_skips > 0 else 0.0
        net_saved_sell = -float(sell_skips['pnl_pct'].sum()) if num_sell_skips > 0 else 0.0
        
        raw_wins = int((all_combined['win'] == True).sum())
        orig_wr = (raw_wins / total_raw_signals * 100.0) if total_raw_signals > 0 else 0.0
        wr_delta = win_rate - orig_wr
        
        sk1, sk2, sk3, sk4, sk5, sk6 = st.columns(6)
        with sk1:
            st.markdown(f"""<div class='metric-card'><span class='metric-label'>Total Skipped</span><div class='metric-value' style='color:#fbbf24;'>{num_skips:,}</div><span style='font-size:11px;color:#94a3b8;'>{num_skips/total_raw_signals*100:.1f}% of {total_raw_signals:,} Signals</span></div>""", unsafe_allow_html=True)
        with sk2:
            st.markdown(f"""<div class='metric-card'><span class='metric-label'>Correct Skips</span><div class='metric-value' style='color:#00d26a;'>{num_correct:,}</div><span style='font-size:11px;color:#94a3b8;'>🟢 BUY: {buy_correct:,} | 🔴 SELL: {sell_correct:,}</span></div>""", unsafe_allow_html=True)
        with sk3:
            st.markdown(f"""<div class='metric-card'><span class='metric-label'>Wrong Skips</span><div class='metric-value' style='color:#ef4444;'>{num_wrong:,}</div><span style='font-size:11px;color:#94a3b8;'>🟢 BUY: {buy_wrong:,} | 🔴 SELL: {sell_wrong:,}</span></div>""", unsafe_allow_html=True)
        with sk4:
            acc_col = "#00d26a" if pct_correct >= 50 else "#f8312f"
            st.markdown(f"""<div class='metric-card'><span class='metric-label'>Filter Accuracy</span><div class='metric-value' style='color:{acc_col};'>{pct_correct:.1f}%</div><span style='font-size:11px;color:#94a3b8;'>Loss Prevention Rate</span></div>""", unsafe_allow_html=True)
        with sk5:
            saved_col = "#00d26a" if net_saved_pct >= 0 else "#ef4444"
            st.markdown(f"""<div class='metric-card'><span class='metric-label'>Net Return Saved</span><div class='metric-value' style='color:{saved_col};'>{net_saved_pct:+.1f}%</div><span style='font-size:11px;color:#94a3b8;'>Avoided Loss vs Missed Win</span></div>""", unsafe_allow_html=True)
        with sk6:
            wr_d_col = "#00d26a" if wr_delta >= 0 else "#ef4444"
            st.markdown(f"""<div class='metric-card'><span class='metric-label'>Portfolio WR Uplift</span><div class='metric-value' style='color:{wr_d_col};'>{wr_delta:+.1f}%</div><span style='font-size:11px;color:#94a3b8;'>{orig_wr:.1f}% ➔ {win_rate:.1f}%</span></div>""", unsafe_allow_html=True)

        st.markdown("##### 📊 BUY vs. SELL Skipped Signals Accuracy & Distribution")
        dir_breakdown_df = pd.DataFrame([
            {
                'Signal Direction': f'🟢 BUY Divergences (Close > {ema_period} EMA)',
                'Total Skipped': f"{num_buy_skips:,}",
                '🟢 Correct Skips (Saved Loss)': f"{buy_correct:,} ({buy_corr_pct:.1f}%)",
                '🔴 Wrong Skips (Missed Win)': f"{buy_wrong:,} ({buy_wrng_pct:.1f}%)",
                'Filter Accuracy': f"{buy_corr_pct:.1f}%",
                'Net % Saved': f"{net_saved_buy:+.2f}%"
            },
            {
                'Signal Direction': f'🔴 SELL Divergences (Close < {ema_period} EMA)',
                'Total Skipped': f"{num_sell_skips:,}",
                '🟢 Correct Skips (Saved Loss)': f"{sell_correct:,} ({sell_corr_pct:.1f}%)",
                '🔴 Wrong Skips (Missed Win)': f"{sell_wrong:,} ({sell_wrng_pct:.1f}%)",
                'Filter Accuracy': f"{sell_corr_pct:.1f}%",
                'Net % Saved': f"{net_saved_sell:+.2f}%"
            },
            {
                'Signal Direction': '🔥 Portfolio Combined Skipped',
                'Total Skipped': f"{num_skips:,}",
                '🟢 Correct Skips (Saved Loss)': f"{num_correct:,} ({pct_correct:.1f}%)",
                '🔴 Wrong Skips (Missed Win)': f"{num_wrong:,} ({pct_wrong:.1f}%)",
                'Filter Accuracy': f"{pct_correct:.1f}%",
                'Net % Saved': f"{net_saved_pct:+.2f}%"
            }
        ])
        st.dataframe(dir_breakdown_df, width="stretch", hide_index=True)
        
    # =============================================================
    # 3. CALENDAR YEAR COMBINED PERFORMANCE BREAKDOWN
    # =============================================================
    st.markdown("### 📅 Calendar Year Combined Performance Breakdown")
    trades_df['year'] = pd.to_datetime(trades_df['entry_date']).dt.year
    
    y_stats = []
    for yr, y_df in trades_df.groupby('year'):
        n_t = len(y_df)
        w_t = int((y_df['win'] == True).sum())
        wr = (w_t / n_t) * 100.0 if n_t > 0 else 0
        avg_pct = y_df['pnl_pct'].mean()
        cum_pct = y_df['pnl_pct'].sum()
        
        b_df = y_df[y_df['action'] == 'BUY']
        n_b = len(b_df)
        w_b = int((b_df['win'] == True).sum())
        wr_b = (w_b / n_b * 100.0) if n_b > 0 else np.nan
        
        s_df = y_df[y_df['action'] == 'SELL']
        n_s = len(s_df)
        w_s = int((s_df['win'] == True).sum())
        wr_s = (w_s / n_s * 100.0) if n_s > 0 else np.nan
        
        y_stats.append({
            'Year': int(yr),
            'Active Stocks': int(y_df['symbol'].nunique()),
            'Total Signals': int(n_t),
            'Wins': int(w_t),
            'Losses': int(n_t - w_t),
            'Combined Win Rate (%)': float(wr),
            '🟢 BUY Win Rate (%)': float(wr_b) if not np.isnan(wr_b) else np.nan,
            '🔴 SELL Win Rate (%)': float(wr_s) if not np.isnan(wr_s) else np.nan,
            'Avg Return (%)': float(avg_pct),
            'Cumulative Return (%)': float(cum_pct),
            'Avg Holding (Days)': float(y_df['holding_days'].mean()) if 'holding_days' in y_df.columns and not y_df.empty else 0.0,
            'raw_signals': n_t,
            'raw_wr': wr
        })
        
    y_df_disp = pd.DataFrame(y_stats)
    
    year_port_cfg = {
        'Year': st.column_config.NumberColumn('Year', format='%d'),
        'Active Stocks': st.column_config.NumberColumn('Active Stocks', format='%d'),
        'Total Signals': st.column_config.NumberColumn('Total Signals', format='%d'),
        'Wins': st.column_config.NumberColumn('Wins', format='%d'),
        'Losses': st.column_config.NumberColumn('Losses', format='%d'),
        'Combined Win Rate (%)': st.column_config.NumberColumn('Combined Win Rate (%)', format='%.1f%%'),
        '🟢 BUY Win Rate (%)': st.column_config.NumberColumn('🟢 BUY Win Rate (%)', format='%.1f%%'),
        '🔴 SELL Win Rate (%)': st.column_config.NumberColumn('🔴 SELL Win Rate (%)', format='%.1f%%'),
        'Avg Return (%)': st.column_config.NumberColumn('Avg Return (%)', format='%+.2f%%'),
        'Cumulative Return (%)': st.column_config.NumberColumn('Cumulative Return (%)', format='%+.1f%%'),
        'Avg Holding (Days)': st.column_config.NumberColumn('Avg Holding (Days)', format='%.1f')
    }
    
    yc1, yc2 = st.columns([1.5, 1.5])
    with yc1:
        st.dataframe(y_df_disp.drop(columns=['raw_signals', 'raw_wr']), column_config=year_port_cfg, width="stretch", hide_index=True)

        
    with yc2:
        fig_y = make_subplots(specs=[[{"secondary_y": True}]])
        fig_y.add_trace(go.Bar(
            x=y_df_disp['Year'],
            y=y_df_disp['raw_signals'],
            name="Signals Volume",
            marker_color="#38bdf8",
            opacity=0.75
        ), secondary_y=False)
        
        fig_y.add_trace(go.Scatter(
            x=y_df_disp['Year'],
            y=y_df_disp['raw_wr'],
            name="Combined Win Rate (%)",
            mode="lines+markers",
            line=dict(color="#00d26a", width=2.5),
            marker=dict(size=6)
        ), secondary_y=True)
        
        fig_y.add_hline(y=50, line_dash="dash", line_color="#ef4444", line_width=1, secondary_y=True)
        fig_y.update_layout(
            title="Yearly Signals Volume & Combined Success Rate",
            template="plotly_dark",
            height=340,
            margin=dict(l=30, r=30, t=40, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_y, width="stretch")
        
    # =============================================================
    # 4. STOCK-BY-STOCK PERFORMANCE MATRIX & LEADERBOARD
    # =============================================================
    st.markdown("### 🏆 Stock-by-Stock Performance Matrix & Leaderboard")
    if not stock_summary_df.empty:
        top_wr = stock_summary_df.sort_values(by=['Win Rate (%)', 'Trades'], ascending=False).head(3)
        top_ret = stock_summary_df.sort_values(by='Total Return (%)', ascending=False).head(3)
        
        hl1, hl2 = st.columns(2)
        with hl1:
            st.markdown("**🥇 Top 3 Highest Win Rate Equities:**")
            badges_wr = " &nbsp; ".join([f"<span style='background:rgba(0,210,106,0.15); border:1px solid #00d26a; padding:3px 8px; border-radius:4px; font-size:12px; color:#00d26a;'><strong>{row['Symbol']}</strong>: {row['Win Rate (%)']:.1f}% ({row['Trades']} signals)</span>" for _, row in top_wr.iterrows()])
            st.markdown(badges_wr, unsafe_allow_html=True)
        with hl2:
            st.markdown("**🚀 Top 3 Highest Cumulative Return Equities:**")
            badges_ret = " &nbsp; ".join([f"<span style='background:rgba(56,189,248,0.15); border:1px solid #38bdf8; padding:3px 8px; border-radius:4px; font-size:12px; color:#38bdf8;'><strong>{row['Symbol']}</strong>: {row['Total Return (%)']:+.1f}%</span>" for _, row in top_ret.iterrows()])
            st.markdown(badges_ret, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        disp_stocks = stock_summary_df.copy()
        stock_cfg = {
            'Trades': st.column_config.NumberColumn('Trades', format='%d'),
            'Win Rate (%)': st.column_config.NumberColumn('Win Rate (%)', format='%.1f%%'),
            'Wins': st.column_config.NumberColumn('Wins', format='%d'),
            'Losses': st.column_config.NumberColumn('Losses', format='%d'),
            'Profit Factor': st.column_config.NumberColumn('Profit Factor', format='%.2f'),
            'Avg Return (%)': st.column_config.NumberColumn('Avg Return (%)', format='%+.2f%%'),
            'Total Return (%)': st.column_config.NumberColumn('Total Return (%)', format='%+.1f%%'),
            'Best Trade (%)': st.column_config.NumberColumn('Best Trade (%)', format='%+.1f%%'),
            'Worst Trade (%)': st.column_config.NumberColumn('Worst Trade (%)', format='%+.1f%%'),
            'Avg Holding (Days)': st.column_config.NumberColumn('Avg Holding (Days)', format='%.1f'),
            'Min Holding (Days)': st.column_config.NumberColumn('Min Holding (Days)', format='%d'),
            'Max Holding (Days)': st.column_config.NumberColumn('Max Holding (Days)', format='%d'),
            'Avg Holding (Bars)': st.column_config.NumberColumn('Avg Holding (Bars)', format='%.1f')
        }
        st.dataframe(disp_stocks, column_config=stock_cfg, width="stretch", hide_index=True)

        
    # =============================================================
    # 5. CUMULATIVE PORTFOLIO GROWTH CURVE
    # =============================================================
    st.markdown("### 📈 Cumulative Portfolio Return & Drawdown Curve")
    trades_df_sorted = trades_df.sort_values('exit_date').reset_index(drop=True)
    trades_df_sorted['cum_pnl_pct'] = trades_df_sorted['pnl_pct'].cumsum()
    trades_df_sorted['peak_pnl_pct'] = trades_df_sorted['cum_pnl_pct'].cummax()
    trades_df_sorted['drawdown_pct'] = trades_df_sorted['cum_pnl_pct'] - trades_df_sorted['peak_pnl_pct']
    
    fig_peq = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3],
        subplot_titles=("Cumulative Portfolio % Return Across All Executed Signals", "Portfolio Drawdown (%)")
    )
    
    fig_peq.add_trace(go.Scatter(
        x=trades_df_sorted['exit_date'],
        y=trades_df_sorted['cum_pnl_pct'],
        mode='lines',
        name='Cumulative Return (%)',
        line=dict(color='#00d26a', width=2.0)
    ), row=1, col=1)
    
    fig_peq.add_trace(go.Scatter(
        x=trades_df_sorted['exit_date'],
        y=trades_df_sorted['peak_pnl_pct'],
        mode='lines',
        name='High Water Mark',
        line=dict(color='#38bdf8', width=1, dash='dot')
    ), row=1, col=1)
    
    fig_peq.add_trace(go.Scatter(
        x=trades_df_sorted['exit_date'],
        y=trades_df_sorted['drawdown_pct'],
        mode='lines',
        name='Drawdown Underwater',
        fill='tozeroy',
        line=dict(color='#ef4444', width=1.5),
        fillcolor='rgba(239, 68, 68, 0.25)'
    ), row=2, col=1)
    
    fig_peq.update_layout(
        template="plotly_dark",
        height=460,
        margin=dict(l=40, r=40, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_peq, width="stretch")
    
    # =============================================================
    # 6. COMPLETE FILTERABLE EXECUTED TRADE LOG
    # =============================================================
    st.markdown("### 📋 Complete Portfolio Executed Trade Log")
    fl1, fl2, fl3 = st.columns([1.5, 1.2, 1.2])
    with fl1:
        sym_options = ["All Equities"] + sorted(trades_df['symbol'].unique().tolist())
        sel_sym = st.selectbox("Filter by Symbol:", sym_options, key="sec3_port_filt_sym")
    with fl2:
        sel_act = st.selectbox("Filter by Action:", ["All Actions", "BUY Only", "SELL Only"], key="sec3_port_filt_act")
    with fl3:
        sel_out = st.selectbox("Filter by Outcome:", ["All Outcomes", "Wins Only", "Losses Only"], key="sec3_port_filt_out")
        
    filt_df = trades_df.copy()
    if sel_sym != "All Equities":
        filt_df = filt_df[filt_df['symbol'] == sel_sym]
    if sel_act == "BUY Only":
        filt_df = filt_df[filt_df['action'] == 'BUY']
    elif sel_act == "SELL Only":
        filt_df = filt_df[filt_df['action'] == 'SELL']
    if sel_out == "Wins Only":
        filt_df = filt_df[filt_df['win'] == True]
    elif sel_out == "Losses Only":
        filt_df = filt_df[filt_df['win'] == False]
        
    disp_cols = ['symbol', 'signal_date', 'entry_date', 'exit_date', 'action', 'reason', 'entry_price', 'exit_price', 'exit_reason', 'holding_bars', 'holding_days', 'pnl_pct']
    
    disp_t = filt_df[disp_cols].copy()
    disp_t['signal_date'] = pd.to_datetime(disp_t['signal_date']).dt.strftime('%Y-%m-%d')
    disp_t['entry_date'] = pd.to_datetime(disp_t['entry_date']).dt.strftime('%Y-%m-%d')
    disp_t['exit_date'] = pd.to_datetime(disp_t['exit_date']).dt.strftime('%Y-%m-%d')
    disp_t = disp_t.rename(columns={
        'symbol': 'Symbol',
        'signal_date': 'Signal Date',
        'entry_date': 'Entry Date',
        'exit_date': 'Exit Date',
        'action': 'Action',
        'reason': 'Divergence Reason',
        'entry_price': 'Entry Price',
        'exit_price': 'Exit Price',
        'exit_reason': 'Exit Reason',
        'holding_bars': 'Holding Bars',
        'holding_days': 'Holding Days',
        'pnl_pct': 'Return (%)'
    })
    
    trade_port_cfg = {
        'Entry Price': st.column_config.NumberColumn('Entry Price', format='₹%.2f'),
        'Exit Price': st.column_config.NumberColumn('Exit Price', format='₹%.2f'),
        'Holding Bars': st.column_config.NumberColumn('Holding Bars', format='%d'),
        'Holding Days': st.column_config.NumberColumn('Holding Days', format='%d'),
        'Return (%)': st.column_config.NumberColumn('Return (%)', format='%+.2f%%')
    }
    st.dataframe(disp_t, column_config=trade_port_cfg, width="stretch", hide_index=True)

    
    csv_bytes = trades_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📥 Download Full Portfolio ({len(trades_df):,} Trades) Log CSV",
        data=csv_bytes,
        file_name=f"portfolio_rsi_divergence_{timeframe.lower()}_trades.csv",
        mime="text/csv",
        key="btn_download_portfolio_trades_csv"
    )

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
            ["🇮🇳 NIFTY 50 (Index)", "📈 Individual Stock (Single Ticker)", "🌐 All Stocks Combined (Portfolio Aggregate)"],
            horizontal=True
        )
        
    with ac2:
        if asset_type == "🇮🇳 NIFTY 50 (Index)":
            selected_asset = "NIFTY 50"
            st.markdown("<span style='font-size: 12px; color: #38bdf8;'>Data Source: <code>NIFTY 50 (Daily/Weekly Parquet 1990–2026)</code></span>", unsafe_allow_html=True)
        elif asset_type == "📈 Individual Stock (Single Ticker)":
            stocks = get_available_stock_symbols()
            if "selected_stock_override" in st.session_state and st.session_state["selected_stock_override"] in stocks:
                def_stock_idx = stocks.index(st.session_state["selected_stock_override"])
            else:
                def_stock_idx = stocks.index("INFY") if "INFY" in stocks else 0
            selected_asset = st.selectbox("Select Stock:", stocks, index=def_stock_idx)
            st.markdown(f"<span style='font-size: 12px; color: #38bdf8;'>Data Source: <code>{selected_asset} Stock Parquet</code></span>", unsafe_allow_html=True)
        else: # "🌐 All Stocks Combined (Portfolio Aggregate)"
            all_avail_stocks = get_available_stock_symbols()
            portfolio_scope = st.selectbox(
                "Universe Scope:",
                [
                    "🏆 Top 50 Benchmark Equities (Nifty 50 - Fast ~1.5s)",
                    "⭐ Top 100 Large & Mid Cap Equities (~3s)",
                    "🔥 All Nifty 500 Equities (501 Stocks - Full Universe)",
                    "🎯 Custom Stock Selection (Multiselect)"
                ],
                index=0
            )
            if "Custom" in portfolio_scope:
                portfolio_symbols = st.multiselect(
                    "Select Custom Equities:",
                    all_avail_stocks,
                    default=["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "ITC", "BHARTIARTL", "LT", "AXISBANK"]
                )
            else:
                portfolio_symbols = get_portfolio_universe_symbols(portfolio_scope, all_avail_stocks)
            selected_asset = "PORTFOLIO_COMBINED"
            st.markdown(f"<span style='font-size: 12px; color: #38bdf8;'>Portfolio Target: <strong>{len(portfolio_symbols)} Stocks</strong> selected for aggregate analysis</span>", unsafe_allow_html=True)
            
    with ac3:
        timeframe = st.selectbox("📅 Timeframe:", ["Daily", "Weekly", "Monthly"], index=1)

    # Load raw asset data
    if asset_type == "🌐 All Stocks Combined (Portfolio Aggregate)":
        min_data_date = pd.to_datetime("2000-01-01").date()
        max_data_date = pd.to_datetime("2026-09-19").date()
    else:
        raw_df = load_asset_ohlc(selected_asset, timeframe)
        min_data_date = raw_df['date'].min().date() if not raw_df.empty else pd.to_datetime("2000-01-01").date()
        max_data_date = raw_df['date'].max().date() if not raw_df.empty else pd.to_datetime("2026-09-19").date()

    
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
        
    # Anti-Chasing EMA Filter Toggle & Input Controls
    st.markdown("---")
    st.markdown("#### 🛡️ Anti-Chasing EMA Filter (Mean-Reversion Gate)")
    ef1, ef2, ef3 = st.columns([1.8, 1.4, 3.2])
    with ef1:
        enable_ema_filter = st.toggle(
            "🛡️ Enable Anti-Chasing EMA Filter",
            value=False,
            key="sec3_enable_ema_filter",
            help="When enabled: Skips SELL when candle close is below EMA (avoids selling into the low) and skips BUY when candle close is above EMA (avoids buying into the high)."
        )
    with ef2:
        if enable_ema_filter:
            ema_period = st.number_input(
                "⚡ EMA Period:",
                min_value=2,
                max_value=200,
                value=21,
                step=1,
                key="sec3_ema_period",
                help="Variable EMA period (e.g. 9, 21, 50, etc.)."
            )
        else:
            ema_period = 21
    with ef3:
        if enable_ema_filter:
            st.markdown(f"""
            <div style='background:rgba(56, 189, 248, 0.08); border-left: 3px solid #38bdf8; padding: 6px 12px; border-radius: 4px; font-size: 12px; color: #cbd5e1;'>
                <strong>Anti-Chasing Logic Active ({ema_period} EMA):</strong><br>
                • <strong>Close &lt; {ema_period} EMA:</strong> 🟢 Take BUY | ❌ Skip SELL<br>
                • <strong>Close &gt; {ema_period} EMA:</strong> 🔴 Take SELL | ❌ Skip BUY
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:#94a3b8; font-size:12px;'>Filter is disabled. Backtest is taking all divergence signals without EMA restrictions.</span>", unsafe_allow_html=True)
        
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

    # If All Stocks Combined is active, branch to the portfolio aggregate renderer
    if asset_type == "🌐 All Stocks Combined (Portfolio Aggregate)":
        render_portfolio_combined_view(
            symbols_to_run=portfolio_symbols,
            timeframe=timeframe,
            sl_val=sl_val,
            tgt_val=tgt_val,
            entry_mode_key=entry_mode_key,
            price_basis_arg=price_basis_arg,
            rsi_len=int(rsi_len),
            lb_left=int(lb_left),
            lb_right=int(lb_right),
            tol_val=tol_val,
            enable_ema_filter=enable_ema_filter,
            ema_period=int(ema_period),
            start_f=start_f,
            end_f=end_f,
            scope_label=portfolio_scope
        )
        return

    # Run Signal Detection on Complete Asset Data (so RSI warmup & history are seamless)

    df_calc, sig_df_full = detect_pivots_and_signals(
        raw_df,
        rsi_len=int(rsi_len),
        lb_left=int(lb_left),
        lb_right=int(lb_right),
        tol=tol_val,
        price_basis=price_basis_arg
    )
    
    # Calculate EMA on full data if filter enabled
    if enable_ema_filter:
        df_calc['ema'] = df_calc['close'].ewm(span=int(ema_period), adjust=False).mean()
    
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
    trades_df_full, _, skipped_df_full, _ = run_rsi_divergence_backtest(
        df_calc,
        sig_df_full,
        sl_pct=sl_val,
        tgt_pct=tgt_val,
        entry_mode=entry_mode_key,
        enable_ema_filter=enable_ema_filter,
        ema_period=int(ema_period)
    )
    
    if not trades_df_full.empty:
        trades_df = trades_df_full[
            (pd.to_datetime(trades_df_full['signal_date']).dt.date >= start_f) &
            (pd.to_datetime(trades_df_full['signal_date']).dt.date <= end_f)
        ].copy().reset_index(drop=True)
        stats = compute_backtest_stats(trades_df)
    else:
        trades_df = pd.DataFrame()
        stats = {}
        
    if not skipped_df_full.empty:
        skipped_df = skipped_df_full[
            (pd.to_datetime(skipped_df_full['signal_date']).dt.date >= start_f) &
            (pd.to_datetime(skipped_df_full['signal_date']).dt.date <= end_f)
        ].copy().reset_index(drop=True)
    else:
        skipped_df = pd.DataFrame()
        
    all_trades_in_range = pd.concat([trades_df, skipped_df], ignore_index=True) if not skipped_df.empty else trades_df
    all_stats = compute_backtest_stats(all_trades_in_range)
        
    if trades_df.empty or not stats:
        st.warning(f"No executed trades found on {selected_asset} ({timeframe}) within the selected date window ({start_f} to {end_f}).")
        if enable_ema_filter and not skipped_df.empty:
            st.info(f"💡 Notice: The {ema_period} EMA filter skipped all {len(skipped_df)} divergence signals in this window.")
        return

    # =============================================================
    # TOP PERFORMANCE METRIC CARDS
    # =============================================================
    st.markdown("---")
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    
    total_trades = stats['total_trades']
    win_rate = stats['win_rate']
    total_pnl = stats['total_pnl_pts']
    pf = stats['profit_factor']
    max_dd = stats['max_dd_pts']
    avg_trade_pct = stats['avg_trade_pct']
    avg_h_days = stats.get('avg_holding_days', 0.0)
    min_h_days = stats.get('min_holding_days', 0)
    max_h_days = stats.get('max_holding_days', 0)
    
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
    with m7:
        st.markdown(f"""<div class='metric-card'><span class='metric-label'>Avg Holding Days</span><div class='metric-value' style='color:#a78bfa;'>{avg_h_days:.1f}d</div><span style='font-size:11px;color:#94a3b8;'>Min: {min_h_days}d | Max: {max_h_days}d</span></div>""", unsafe_allow_html=True)

    # =============================================================
    # EXTRA INFORMATION FOR SKIPPED SIGNALS (EMA FILTER AUDIT)
    # =============================================================
    if enable_ema_filter and not skipped_df.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"### 🛡️ Anti-Chasing {ema_period} EMA Filter Impact & Skipped Signals Audit")
        
        num_skips = len(skipped_df)
        total_raw_signals = len(all_trades_in_range)
        num_correct = int((skipped_df['is_correct_skip'] == True).sum())
        num_wrong = int((skipped_df['is_correct_skip'] == False).sum())
        pct_correct = (num_correct / num_skips * 100.0) if num_skips > 0 else 0.0
        pct_wrong = (num_wrong / num_skips * 100.0) if num_skips > 0 else 0.0
        net_pts_saved = -float(skipped_df['pnl_pts'].sum())
        
        # Directional Breakdown of Skipped Signals
        buy_skips = skipped_df[skipped_df['action'] == 'BUY']
        num_buy_skips = len(buy_skips)
        buy_correct = int((buy_skips['is_correct_skip'] == True).sum()) if num_buy_skips > 0 else 0
        buy_wrong = int((buy_skips['is_correct_skip'] == False).sum()) if num_buy_skips > 0 else 0
        buy_corr_pct = (buy_correct / num_buy_skips * 100.0) if num_buy_skips > 0 else 0.0
        buy_wrng_pct = (buy_wrong / num_buy_skips * 100.0) if num_buy_skips > 0 else 0.0
        net_pts_buy = -float(buy_skips['pnl_pts'].sum()) if num_buy_skips > 0 else 0.0

        sell_skips = skipped_df[skipped_df['action'] == 'SELL']
        num_sell_skips = len(sell_skips)
        sell_correct = int((sell_skips['is_correct_skip'] == True).sum()) if num_sell_skips > 0 else 0
        sell_wrong = int((sell_skips['is_correct_skip'] == False).sum()) if num_sell_skips > 0 else 0
        sell_corr_pct = (sell_correct / num_sell_skips * 100.0) if num_sell_skips > 0 else 0.0
        sell_wrng_pct = (sell_wrong / num_sell_skips * 100.0) if num_sell_skips > 0 else 0.0
        net_pts_sell = -float(sell_skips['pnl_pts'].sum()) if num_sell_skips > 0 else 0.0
        
        orig_wr = all_stats.get('win_rate', 0.0)
        wr_delta = win_rate - orig_wr
        orig_pnl = all_stats.get('total_pnl_pts', 0.0)
        pnl_delta = total_pnl - orig_pnl
        
        sk1, sk2, sk3, sk4, sk5, sk6 = st.columns(6)
        with sk1:
            st.markdown(f"""<div class='metric-card'><span class='metric-label'>Total Skipped</span><div class='metric-value' style='color:#fbbf24;'>{num_skips}</div><span style='font-size:11px;color:#94a3b8;'>{num_skips/total_raw_signals*100:.1f}% of {total_raw_signals} Signals</span></div>""", unsafe_allow_html=True)
        with sk2:
            st.markdown(f"""<div class='metric-card'><span class='metric-label'>Correct Skips</span><div class='metric-value' style='color:#00d26a;'>{num_correct}</div><span style='font-size:11px;color:#94a3b8;'>🟢 BUY: {buy_correct} | 🔴 SELL: {sell_correct}</span></div>""", unsafe_allow_html=True)
        with sk3:
            st.markdown(f"""<div class='metric-card'><span class='metric-label'>Wrong Skips</span><div class='metric-value' style='color:#ef4444;'>{num_wrong}</div><span style='font-size:11px;color:#94a3b8;'>🟢 BUY: {buy_wrong} | 🔴 SELL: {sell_wrong}</span></div>""", unsafe_allow_html=True)
        with sk4:
            acc_col = "#00d26a" if pct_correct >= 50 else "#f8312f"
            st.markdown(f"""<div class='metric-card'><span class='metric-label'>Filter Accuracy</span><div class='metric-value' style='color:{acc_col};'>{pct_correct:.1f}%</div><span style='font-size:11px;color:#94a3b8;'>Loss Prevention Rate</span></div>""", unsafe_allow_html=True)
        with sk5:
            saved_col = "#00d26a" if net_pts_saved >= 0 else "#ef4444"
            st.markdown(f"""<div class='metric-card'><span class='metric-label'>Net Points Saved</span><div class='metric-value' style='color:{saved_col};'>{net_pts_saved:+,.1f}</div><span style='font-size:11px;color:#94a3b8;'>Avoided Loss vs Missed Win</span></div>""", unsafe_allow_html=True)
        with sk6:
            wr_d_col = "#00d26a" if wr_delta >= 0 else "#ef4444"
            st.markdown(f"""<div class='metric-card'><span class='metric-label'>Win Rate Delta</span><div class='metric-value' style='color:{wr_d_col};'>{wr_delta:+.1f}%</div><span style='font-size:11px;color:#94a3b8;'>{orig_wr:.1f}% ➔ {win_rate:.1f}%</span></div>""", unsafe_allow_html=True)

        # Directional Accuracy Breakdown (BUY vs SELL)
        st.markdown("##### 📊 BUY vs. SELL Skipped Signals Accuracy & Distribution")
        dir_breakdown_df = pd.DataFrame([
            {
                'Signal Direction': f'🟢 BUY Divergences (Close > {ema_period} EMA)',
                'Total Skipped': f"{num_buy_skips}",
                '🟢 Correct Skips (Saved Loss)': f"{buy_correct} ({buy_corr_pct:.1f}%)",
                '🔴 Wrong Skips (Missed Win)': f"{buy_wrong} ({buy_wrng_pct:.1f}%)",
                'Accuracy (%)': f"{buy_corr_pct:.1f}%",
                f'Net Points Saved ({pts_label})': f"{net_pts_buy:+,.1f}"
            },
            {
                'Signal Direction': f'🔴 SELL Divergences (Close < {ema_period} EMA)',
                'Total Skipped': f"{num_sell_skips}",
                '🟢 Correct Skips (Saved Loss)': f"{sell_correct} ({sell_corr_pct:.1f}%)",
                '🔴 Wrong Skips (Missed Win)': f"{sell_wrong} ({sell_wrng_pct:.1f}%)",
                'Accuracy (%)': f"{sell_corr_pct:.1f}%",
                f'Net Points Saved ({pts_label})': f"{net_pts_sell:+,.1f}"
            },
            {
                'Signal Direction': '🔥 Combined Total Skipped',
                'Total Skipped': f"{num_skips}",
                '🟢 Correct Skips (Saved Loss)': f"{num_correct} ({pct_correct:.1f}%)",
                '🔴 Wrong Skips (Missed Win)': f"{num_wrong} ({pct_wrong:.1f}%)",
                'Accuracy (%)': f"{pct_correct:.1f}%",
                f'Net Points Saved ({pts_label})': f"{net_pts_saved:+,.1f}"
            }
        ])
        st.dataframe(dir_breakdown_df, width="stretch", hide_index=True)

        # Comparative Side-by-Side Summary
        comp_df = pd.DataFrame([
            {
                'Metric Dimension': 'Total Executed Trades',
                'Unfiltered (All Signals)': f"{all_stats.get('total_trades', 0):,}",
                f'With {ema_period} EMA Filter': f"{total_trades:,}",
                'Filter Impact / Gain': f"-{num_skips} skipped (-{num_skips/total_raw_signals*100:.1f}%)"
            },
            {
                'Metric Dimension': 'Win Rate (%)',
                'Unfiltered (All Signals)': f"{orig_wr:.1f}%",
                f'With {ema_period} EMA Filter': f"{win_rate:.1f}%",
                'Filter Impact / Gain': f"{wr_delta:+.1f}%"
            },
            {
                'Metric Dimension': f'Total Net PnL ({pts_label})',
                'Unfiltered (All Signals)': f"{orig_pnl:+,.1f}",
                f'With {ema_period} EMA Filter': f"{total_pnl:+,.1f}",
                'Filter Impact / Gain': f"{pnl_delta:+,.1f}"
            },
            {
                'Metric Dimension': 'Profit Factor',
                'Unfiltered (All Signals)': f"{all_stats.get('profit_factor', np.nan):.2f}" if not np.isnan(all_stats.get('profit_factor', np.nan)) else 'N/A',
                f'With {ema_period} EMA Filter': f"{pf_str}",
                'Filter Impact / Gain': f"{pf - all_stats.get('profit_factor', 0):+.2f}" if not np.isnan(pf) and not np.isnan(all_stats.get('profit_factor', np.nan)) else '-'
            },
            {
                'Metric Dimension': 'Max Drawdown (Points)',
                'Unfiltered (All Signals)': f"{all_stats.get('max_dd_pts', 0):,.1f}",
                f'With {ema_period} EMA Filter': f"{max_dd:,.1f}",
                'Filter Impact / Gain': f"{max_dd - all_stats.get('max_dd_pts', 0):+,.1f} pts saved"
            },
            {
                'Metric Dimension': 'Average Trade Return (%)',
                'Unfiltered (All Signals)': f"{all_stats.get('avg_trade_pct', 0):+.2f}%",
                f'With {ema_period} EMA Filter': f"{avg_trade_pct:+.2f}%",
                'Filter Impact / Gain': f"{avg_trade_pct - all_stats.get('avg_trade_pct', 0):+.2f}%"
            },
            {
                'Metric Dimension': 'Avg Holding Days',
                'Unfiltered (All Signals)': f"{all_stats.get('avg_holding_days', 0.0):.1f}d",
                f'With {ema_period} EMA Filter': f"{avg_h_days:.1f}d",
                'Filter Impact / Gain': f"{avg_h_days - all_stats.get('avg_holding_days', 0.0):+.1f}d"
            }
        ])
        st.dataframe(comp_df, width="stretch", hide_index=True)

        # Expandable Skipped Signals Audit Table
        with st.expander(f"🔍 Inspect All {num_skips} Skipped Signals & Their Simulated Outcomes", expanded=False):
            st.markdown(f"""
            Itemized breakdown of every divergence signal skipped by the **{ema_period} EMA rule**:
            - 🟢 **Correct Skip (Saved Loss)**: Signal would have hit Stop Loss or closed in loss. Skipping protected your capital.
            - 🔴 **Wrong Skip (Missed Win)**: Signal would have hit Target or closed in profit. Skipping missed a winning trade.
            """)
            sk_disp_cols = [
                'signal_date', 'action', 'signal_close', 'ema_val', 'skip_reason',
                'entry_price', 'exit_price', 'exit_reason', 'pnl_pts', 'pnl_pct', 'verdict'
            ]
            sk_table = skipped_df[sk_disp_cols].copy()
            sk_table['signal_date'] = pd.to_datetime(sk_table['signal_date']).dt.strftime('%Y-%m-%d')
            
            sk_table = sk_table.rename(columns={
                'signal_date': 'Signal Date',
                'action': 'Signal',
                'signal_close': 'Signal Close',
                'ema_val': f'{ema_period} EMA Price',
                'skip_reason': 'Filter Condition Triggered',
                'entry_price': 'Hypothetical Entry',
                'exit_price': 'Hypothetical Exit',
                'exit_reason': 'Simulated Exit Reason',
                'pnl_pts': 'Hypothetical PnL (Pts)',
                'pnl_pct': 'Hypothetical PnL (%)',
                'verdict': 'Filter Outcome / Verdict'
            })
            
            sk_cfg = {
                'Signal Close': st.column_config.NumberColumn('Signal Close', format='₹%.2f'),
                f'{ema_period} EMA Price': st.column_config.NumberColumn(f'{ema_period} EMA Price', format='₹%.2f'),
                'Hypothetical Entry': st.column_config.NumberColumn('Hypothetical Entry', format='₹%.2f'),
                'Hypothetical Exit': st.column_config.NumberColumn('Hypothetical Exit', format='₹%.2f'),
                'Hypothetical PnL (Pts)': st.column_config.NumberColumn('Hypothetical PnL (Pts)', format='%+,.1f'),
                'Hypothetical PnL (%)': st.column_config.NumberColumn('Hypothetical PnL (%)', format='%+.2f%%')
            }
            st.dataframe(sk_table, column_config=sk_cfg, width="stretch", hide_index=True)

            
            sk_csv = skipped_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Download Skipped Signals Audit CSV",
                data=sk_csv,
                file_name=f"{selected_asset.lower()}_skipped_signals_ema{ema_period}.csv",
                mime="text/csv",
                key="btn_download_skipped_csv"
            )

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
            
        # EMA Curve & Skipped Signals (when EMA filter enabled)
        if enable_ema_filter and 'ema' in df_filtered.columns:
            fig_cand.add_trace(go.Scatter(
                x=df_filtered['date'],
                y=df_filtered['ema'],
                mode='lines',
                name=f'{ema_period} EMA',
                line=dict(color='#f59e0b', width=2)
            ), row=1, col=1)
            
            if not skipped_df.empty:
                sk_buys = skipped_df[skipped_df['action'] == 'BUY']
                if not sk_buys.empty:
                    fig_cand.add_trace(go.Scatter(
                        x=sk_buys['signal_date'],
                        y=sk_buys['entry_price'],
                        mode='markers',
                        marker=dict(symbol='x', size=11, color='#94a3b8'),
                        name=f'Skipped BUY (Close > {ema_period} EMA)',
                        hovertext=[f"Skipped BUY: {r['skip_reason']} | {r['verdict']}" for _, r in sk_buys.iterrows()]
                    ), row=1, col=1)
                sk_sells = skipped_df[skipped_df['action'] == 'SELL']
                if not sk_sells.empty:
                    fig_cand.add_trace(go.Scatter(
                        x=sk_sells['signal_date'],
                        y=sk_sells['entry_price'],
                        mode='markers',
                        marker=dict(symbol='x', size=11, color='#fca5a5'),
                        name=f'Skipped SELL (Close < {ema_period} EMA)',
                        hovertext=[f"Skipped SELL: {r['skip_reason']} | {r['verdict']}" for _, r in sk_sells.iterrows()]
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
        wr_b = (w_b / n_b * 100.0) if n_b > 0 else np.nan
        
        s_df = y_df[y_df['action'] == 'SELL']
        n_s = len(s_df)
        w_s = len(s_df[s_df['win'] == True])
        wr_s = (w_s / n_s * 100.0) if n_s > 0 else np.nan
        
        y_stats.append({
            'Year': int(yr),
            'Trades': int(n_t),
            'Wins': int(w_t),
            'Losses': int(n_t - w_t),
            'Win Rate (%)': float(wr),
            '🟢 BUY Win Rate (%)': float(wr_b) if not np.isnan(wr_b) else np.nan,
            '🔴 SELL Win Rate (%)': float(wr_s) if not np.isnan(wr_s) else np.nan,
            'Net Points': float(pnl_pts),
            'Avg Trade (%)': float(avg_pct),
            'Avg Holding (Days)': float(y_df['holding_days'].mean()) if 'holding_days' in y_df.columns and not y_df.empty else 0.0,
            'raw_pnl': pnl_pts
        })
        
    y_df_display = pd.DataFrame(y_stats)
    
    single_year_cfg = {
        'Year': st.column_config.NumberColumn('Year', format='%d'),
        'Trades': st.column_config.NumberColumn('Trades', format='%d'),
        'Wins': st.column_config.NumberColumn('Wins', format='%d'),
        'Losses': st.column_config.NumberColumn('Losses', format='%d'),
        'Win Rate (%)': st.column_config.NumberColumn('Win Rate (%)', format='%.1f%%'),
        '🟢 BUY Win Rate (%)': st.column_config.NumberColumn('🟢 BUY Win Rate (%)', format='%.1f%%'),
        '🔴 SELL Win Rate (%)': st.column_config.NumberColumn('🔴 SELL Win Rate (%)', format='%.1f%%'),
        'Net Points': st.column_config.NumberColumn('Net Points', format='%+,.1f'),
        'Avg Trade (%)': st.column_config.NumberColumn('Avg Trade (%)', format='%+.2f%%'),
        'Avg Holding (Days)': st.column_config.NumberColumn('Avg Holding (Days)', format='%.1f')
    }
    
    yc1, yc2 = st.columns([1.5, 1.5])
    with yc1:
        st.dataframe(y_df_display.drop(columns=['raw_pnl']), column_config=single_year_cfg, width="stretch", hide_index=True)
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
    
    disp_cols = ['signal_date', 'entry_date', 'exit_date', 'action', 'reason', 'entry_price', 'sl_price', 'tgt_price', 'exit_price', 'exit_reason', 'holding_bars', 'holding_days', 'pnl_pts', 'pnl_pct']
    
    trade_table = trades_df[disp_cols].copy()
    trade_table['signal_date'] = pd.to_datetime(trade_table['signal_date']).dt.strftime('%Y-%m-%d')
    trade_table['entry_date'] = pd.to_datetime(trade_table['entry_date']).dt.strftime('%Y-%m-%d')
    trade_table['exit_date'] = pd.to_datetime(trade_table['exit_date']).dt.strftime('%Y-%m-%d')
    
    trade_table = trade_table.rename(columns={
        'signal_date': 'Signal Date',
        'entry_date': 'Entry Date',
        'exit_date': 'Exit Date',
        'action': 'Action',
        'reason': 'Reason',
        'entry_price': 'Entry Price',
        'sl_price': 'Stop Loss',
        'tgt_price': 'Target',
        'exit_price': 'Exit Price',
        'exit_reason': 'Exit Reason',
        'holding_bars': 'Holding Bars',
        'holding_days': 'Holding Days',
        'pnl_pts': 'PnL (Points)',
        'pnl_pct': 'PnL (%)'
    })
    
    single_trade_cfg = {
        'Entry Price': st.column_config.NumberColumn('Entry Price', format='₹%.2f'),
        'Stop Loss': st.column_config.NumberColumn('Stop Loss', format='₹%.2f'),
        'Target': st.column_config.NumberColumn('Target', format='₹%.2f'),
        'Exit Price': st.column_config.NumberColumn('Exit Price', format='₹%.2f'),
        'Holding Bars': st.column_config.NumberColumn('Holding Bars', format='%d'),
        'Holding Days': st.column_config.NumberColumn('Holding Days', format='%d'),
        'PnL (Points)': st.column_config.NumberColumn('PnL (Points)', format='%+,.1f'),
        'PnL (%)': st.column_config.NumberColumn('PnL (%)', format='%+.2f%%')
    }
    st.dataframe(trade_table, column_config=single_trade_cfg, width="stretch", hide_index=True)

    
    # Download Trade Log CSV
    csv_bytes = trades_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📥 Download Full {selected_asset} ({timeframe}) Trade Log (CSV)",
        data=csv_bytes,
        file_name=f"{selected_asset.lower()}_rsi_divergence_{timeframe.lower()}_trades.csv",
        mime="text/csv"
    )


