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
    cum_pts = trades_df['pnl_pts'].cumsum() if 'pnl_pts' in trades_df.columns else pd.Series(dtype=float)
    
    # 1. Calculate drawdown episodes and peak-to-trough dates (POINTS - for NIFTY 50)
    start_dt = pd.to_datetime(trades_df['entry_date'].iloc[0]) if 'entry_date' in trades_df.columns and not trades_df.empty else pd.Timestamp.now()
    hwm = 0.0
    hwm_dt = start_dt
    cum = 0.0
    in_dd = False
    curr_peak_pts = 0.0
    curr_peak_dt = start_dt
    curr_trough_pts = 0.0
    curr_trough_dt = start_dt
    
    dd_episodes = []
    
    for _, row in trades_df.iterrows():
        pnl = float(row['pnl_pts']) if 'pnl_pts' in row and pd.notna(row['pnl_pts']) else 0.0
        cum += pnl
        exit_dt = pd.to_datetime(row['exit_date']) if 'exit_date' in row and pd.notna(row['exit_date']) else (
            pd.to_datetime(row['entry_date']) if 'entry_date' in row and pd.notna(row['entry_date']) else start_dt
        )
        
        if cum >= hwm:
            if in_dd:
                dd_episodes.append({
                    'peak_dt': curr_peak_dt,
                    'trough_dt': curr_trough_dt,
                    'rec_dt': exit_dt,
                    'peak_pts': curr_peak_pts,
                    'trough_pts': curr_trough_pts,
                    'dd_pts': curr_peak_pts - curr_trough_pts,
                    'recovered': True,
                    'days': (exit_dt - curr_peak_dt).days,
                    'peak_to_trough_days': (curr_trough_dt - curr_peak_dt).days
                })
                in_dd = False
            hwm = cum
            hwm_dt = exit_dt
        else:
            if not in_dd:
                in_dd = True
                curr_peak_pts = hwm
                curr_peak_dt = hwm_dt
                curr_trough_pts = cum
                curr_trough_dt = exit_dt
            else:
                if cum < curr_trough_pts:
                    curr_trough_pts = cum
                    curr_trough_dt = exit_dt
                    
    if in_dd:
        last_exit_dt = pd.to_datetime(trades_df['exit_date'].iloc[-1]) if 'exit_date' in trades_df.columns and pd.notna(trades_df['exit_date'].iloc[-1]) else curr_trough_dt
        dd_episodes.append({
            'peak_dt': curr_peak_dt,
            'trough_dt': curr_trough_dt,
            'rec_dt': None,
            'peak_pts': curr_peak_pts,
            'trough_pts': curr_trough_pts,
            'dd_pts': curr_peak_pts - curr_trough_pts,
            'recovered': False,
            'days': (last_exit_dt - curr_peak_dt).days,
            'peak_to_trough_days': (curr_trough_dt - curr_peak_dt).days
        })
        
    if dd_episodes:
        max_ep = max(dd_episodes, key=lambda x: x['dd_pts'])
        p_str = max_ep['peak_dt'].strftime('%Y-%m-%d')
        t_str = max_ep['trough_dt'].strftime('%Y-%m-%d')
        r_str = max_ep['rec_dt'].strftime('%Y-%m-%d') if max_ep['rec_dt'] is not None else 'Ongoing'
        max_dd_pts = -float(max_ep['dd_pts'])
        max_dd_peak_date = p_str
        max_dd_trough_date = t_str
        max_dd_rec_date = r_str
        max_dd_recovered = max_ep['recovered']
        max_dd_days = int(max_ep['days'])
        max_dd_pt_days = int(max_ep['peak_to_trough_days'])
        max_dd_period_str = f"{p_str} ➔ {t_str}"
    else:
        max_dd_pts = 0.0
        max_dd_peak_date = '-'
        max_dd_trough_date = '-'
        max_dd_rec_date = '-'
        max_dd_recovered = True
        max_dd_days = 0
        max_dd_pt_days = 0
        max_dd_period_str = 'No Drawdown'

    # 2. Calculate drawdown episodes and peak-to-trough dates (PERCENTAGE - for Equities & Portfolio Leaderboard)
    hwm_pct = 0.0
    hwm_dt_pct = start_dt
    cum_pct = 0.0
    in_dd_pct = False
    curr_peak_pct = 0.0
    curr_peak_dt_pct = start_dt
    curr_trough_pct = 0.0
    curr_trough_dt_pct = start_dt
    
    dd_episodes_pct = []
    
    for _, row in trades_df.iterrows():
        pnl_pct_val = float(row['pnl_pct']) if 'pnl_pct' in row and pd.notna(row['pnl_pct']) else 0.0
        cum_pct += pnl_pct_val
        exit_dt = pd.to_datetime(row['exit_date']) if 'exit_date' in row and pd.notna(row['exit_date']) else (
            pd.to_datetime(row['entry_date']) if 'entry_date' in row and pd.notna(row['entry_date']) else start_dt
        )
        
        if cum_pct >= hwm_pct:
            if in_dd_pct:
                dd_episodes_pct.append({
                    'peak_dt': curr_peak_dt_pct,
                    'trough_dt': curr_trough_dt_pct,
                    'rec_dt': exit_dt,
                    'peak_pct': curr_peak_pct,
                    'trough_pct': curr_trough_pct,
                    'dd_pct': curr_trough_pct - curr_peak_pct,
                    'recovered': True,
                    'days': (exit_dt - curr_peak_dt_pct).days,
                    'peak_to_trough_days': (curr_trough_dt_pct - curr_peak_dt_pct).days
                })
                in_dd_pct = False
            hwm_pct = cum_pct
            hwm_dt_pct = exit_dt
        else:
            if not in_dd_pct:
                in_dd_pct = True
                curr_peak_pct = hwm_pct
                curr_peak_dt_pct = hwm_dt_pct
                curr_trough_pct = cum_pct
                curr_trough_dt_pct = exit_dt
            else:
                if cum_pct < curr_trough_pct:
                    curr_trough_pct = cum_pct
                    curr_trough_dt_pct = exit_dt
                    
    if in_dd_pct:
        last_exit_dt = pd.to_datetime(trades_df['exit_date'].iloc[-1]) if 'exit_date' in trades_df.columns and pd.notna(trades_df['exit_date'].iloc[-1]) else curr_trough_dt_pct
        dd_episodes_pct.append({
            'peak_dt': curr_peak_dt_pct,
            'trough_dt': curr_trough_dt_pct,
            'rec_dt': None,
            'peak_pct': curr_peak_pct,
            'trough_pct': curr_trough_pct,
            'dd_pct': curr_trough_pct - curr_peak_pct,
            'recovered': False,
            'days': (last_exit_dt - curr_peak_dt_pct).days,
            'peak_to_trough_days': (curr_trough_dt_pct - curr_peak_dt_pct).days
        })
        
    if dd_episodes_pct:
        worst_ep_pct = min(dd_episodes_pct, key=lambda x: x['dd_pct'])
        p_pct_str = worst_ep_pct['peak_dt'].strftime('%Y-%m-%d')
        t_pct_str = worst_ep_pct['trough_dt'].strftime('%Y-%m-%d')
        r_pct_str = worst_ep_pct['rec_dt'].strftime('%Y-%m-%d') if worst_ep_pct['rec_dt'] is not None else 'Ongoing'
        max_dd_pct = float(worst_ep_pct['dd_pct'])
        max_dd_pct_peak_date = p_pct_str
        max_dd_pct_trough_date = t_pct_str
        max_dd_pct_rec_date = r_pct_str
        max_dd_pct_recovered = worst_ep_pct['recovered']
        max_dd_pct_days = int(worst_ep_pct['days'])
        max_dd_pct_pt_days = int(worst_ep_pct['peak_to_trough_days'])
        max_dd_pct_period_str = f"{p_pct_str} ➔ {t_pct_str}"
    else:
        max_dd_pct = 0.0
        max_dd_pct_peak_date = '-'
        max_dd_pct_trough_date = '-'
        max_dd_pct_rec_date = '-'
        max_dd_pct_recovered = True
        max_dd_pct_days = 0
        max_dd_pct_pt_days = 0
        max_dd_pct_period_str = 'No Drawdown'
    
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
        'total_pnl_pct': trades_df['pnl_pct'].sum() if 'pnl_pct' in trades_df.columns and not trades_df.empty else 0.0,
        'profit_factor': profit_factor,
        # Points Drawdown (NIFTY 50)
        'max_dd_pts': max_dd_pts,
        'max_dd_peak_date': max_dd_peak_date,
        'max_dd_trough_date': max_dd_trough_date,
        'max_dd_rec_date': max_dd_rec_date,
        'max_dd_recovered': max_dd_recovered,
        'max_dd_days': max_dd_days,
        'max_dd_pt_days': max_dd_pt_days,
        'max_dd_period_str': max_dd_period_str,
        # Percentage Drawdown (Equities & Portfolio Matrix)
        'max_dd_pct': max_dd_pct,
        'max_dd_pct_peak_date': max_dd_pct_peak_date,
        'max_dd_pct_trough_date': max_dd_pct_trough_date,
        'max_dd_pct_rec_date': max_dd_pct_rec_date,
        'max_dd_pct_recovered': max_dd_pct_recovered,
        'max_dd_pct_days': max_dd_pct_days,
        'max_dd_pct_pt_days': max_dd_pct_pt_days,
        'max_dd_pct_period_str': max_dd_pct_period_str,
        # Averages & Holding
        'avg_trade_pct': trades_df['pnl_pct'].mean() if 'pnl_pct' in trades_df.columns and not trades_df.empty else 0.0,
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
                
            if sig_df_full.empty:
                continue
                
            t_df_full, _, sk_df_full, _ = run_rsi_divergence_backtest(
                df_calc, sig_df_full, sl_pct=sl_pct, tgt_pct=tgt_pct, entry_mode=entry_mode,
                enable_ema_filter=enable_ema_filter, ema_period=int(ema_period)
            )
            
            if not t_df_full.empty and start_dt and end_dt:
                t_df = t_df_full[
                    (pd.to_datetime(t_df_full['signal_date']).dt.date >= start_dt) &
                    (pd.to_datetime(t_df_full['signal_date']).dt.date <= end_dt)
                ].copy().reset_index(drop=True)
            else:
                t_df = t_df_full
                
            if not sk_df_full.empty and start_dt and end_dt:
                sk_df = sk_df_full[
                    (pd.to_datetime(sk_df_full['signal_date']).dt.date >= start_dt) &
                    (pd.to_datetime(sk_df_full['signal_date']).dt.date <= end_dt)
                ].copy().reset_index(drop=True)
            else:
                sk_df = sk_df_full
                
            if not t_df.empty:
                t_df['symbol'] = sym
                all_trades.append(t_df)
                
                s_stats = compute_backtest_stats(t_df)
                n_t = s_stats['total_trades']
                n_w = int((t_df['win'] == True).sum())
                
                stock_stats_list.append({
                    'Symbol': sym,
                    'Trades': n_t,
                    'Win Rate (%)': s_stats['win_rate'],
                    'Wins': n_w,
                    'Losses': n_t - n_w,
                    'Profit Factor': s_stats['profit_factor'],
                    'Avg Return (%)': s_stats['avg_trade_pct'],
                    'Total Return (%)': s_stats['total_pnl_pct'],
                    'Best Trade (%)': t_df['pnl_pct'].max(),
                    'Worst Trade (%)': t_df['pnl_pct'].min(),
                    'Max DD (%)': s_stats['max_dd_pct'],
                    'Max DD Holding (Days)': s_stats['max_dd_pct_pt_days'],
                    'Avg Holding (Days)': s_stats['avg_holding_days'],
                    'Min Holding (Days)': s_stats['min_holding_days'],
                    'Max Holding (Days)': s_stats['max_holding_days'],
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


# =============================================================
# CAPITAL GROWTH & INDIAN REGULATORY CHARGES SIMULATION ENGINE
# =============================================================

def simulate_capital_and_charges(
    trades_df: pd.DataFrame,
    initial_capital: float = 100000.0,
    allocation_pct: float = 70.0,
    compounding: bool = True,
    stt_buy_pct: float = 0.100,
    stt_sell_pct: float = 0.100,
    stamp_duty_pct: float = 0.015,
    exchange_pct: float = 0.00297,
    sebi_pct: float = 0.00010,
    ipft_pct: float = 0.00010,
    brokerage_flat: float = 0.0,
    brokerage_pct: float = 0.0,
    gst_pct: float = 18.0
) -> tuple[pd.DataFrame, dict]:
    """
    Simulates real-world capital growth and net monetary PnL on an initial investment (default ₹1,00,000)
    with custom per-trade capital allocation (default 70%) and Indian regulatory charges (STT, Stamp Duty,
    NSE Exchange, SEBI, IPFT, Brokerage, and GST).
    """
    if trades_df.empty:
        return pd.DataFrame(), {}
        
    trades = trades_df.sort_values(by=['entry_date', 'exit_date']).copy().reset_index(drop=True)
    
    current_balance = float(initial_capital)
    gross_balance = float(initial_capital)
    
    sim_rows = []
    
    total_stt = 0.0
    total_stamp_duty = 0.0
    total_exchange = 0.0
    total_sebi_ipft = 0.0
    total_brokerage_gst = 0.0
    total_charges = 0.0
    total_gross_pnl = 0.0
    
    peak_balance = float(initial_capital)
    max_dd_inr = 0.0
    max_dd_pct = 0.0
    
    for idx, row in trades.iterrows():
        entry_price = float(row.get('entry_price', 0.0))
        exit_price = float(row.get('exit_price', 0.0))
        action = str(row.get('action', 'BUY')).upper()
        
        if entry_price <= 0 or exit_price <= 0:
            continue
            
        # Determine capital to deploy
        if compounding:
            allocated_capital = current_balance * (allocation_pct / 100.0)
        else:
            allocated_capital = initial_capital * (allocation_pct / 100.0)
            
        # Constrain to available current balance
        allocated_capital = min(allocated_capital, max(0.0, current_balance))
        
        # Calculate share quantity
        if current_balance >= entry_price and allocated_capital >= entry_price:
            qty = max(1, min(1_000_000_000, int(allocated_capital // entry_price)))
            if qty * entry_price > current_balance:
                qty = max(1, min(1_000_000_000, int(current_balance // entry_price)))
        elif current_balance >= entry_price:
            qty = 1
        else:
            qty = 0
            
        if qty == 0:
            continue
            
        buy_turnover = qty * entry_price
        sell_turnover = qty * exit_price
        
        # Buy-side charges
        stt_buy = buy_turnover * (stt_buy_pct / 100.0)
        stamp_duty = buy_turnover * (stamp_duty_pct / 100.0)
        exchange_buy = buy_turnover * (exchange_pct / 100.0)
        sebi_buy = buy_turnover * (sebi_pct / 100.0)
        ipft_buy = buy_turnover * (ipft_pct / 100.0)
        brokerage_buy = brokerage_flat + (buy_turnover * (brokerage_pct / 100.0))
        gst_buy = (exchange_buy + sebi_buy + brokerage_buy) * (gst_pct / 100.0)
        buy_charges = stt_buy + stamp_duty + exchange_buy + sebi_buy + ipft_buy + brokerage_buy + gst_buy
        
        # Sell-side charges
        stt_sell = sell_turnover * (stt_sell_pct / 100.0)
        exchange_sell = sell_turnover * (exchange_pct / 100.0)
        sebi_sell = sell_turnover * (sebi_pct / 100.0)
        ipft_sell = sell_turnover * (ipft_pct / 100.0)
        brokerage_sell = brokerage_flat + (sell_turnover * (brokerage_pct / 100.0))
        gst_sell = (exchange_sell + sebi_sell + brokerage_sell) * (gst_pct / 100.0)
        sell_charges = stt_sell + exchange_sell + sebi_sell + ipft_sell + brokerage_sell + gst_sell
        
        # Aggregated charges for this trade
        trade_stt = stt_buy + stt_sell
        trade_stamp_duty = stamp_duty
        trade_exchange = exchange_buy + exchange_sell
        trade_sebi_ipft = sebi_buy + sebi_sell + ipft_buy + ipft_sell
        trade_brokerage_gst = brokerage_buy + brokerage_sell + gst_buy + gst_sell
        trade_charges = buy_charges + sell_charges
        
        # Gross and Net PnL
        if action == 'BUY':
            gross_pnl = (exit_price - entry_price) * qty
        else:
            gross_pnl = (entry_price - exit_price) * qty
            
        net_pnl = gross_pnl - trade_charges
        net_trade_ret_pct = (net_pnl / buy_turnover) * 100.0 if buy_turnover > 0 else 0.0
        
        current_balance += net_pnl
        gross_balance += gross_pnl
        
        # Track peak and drawdowns
        if current_balance > peak_balance:
            peak_balance = current_balance
        dd_inr = peak_balance - current_balance
        dd_pct = (dd_inr / peak_balance * 100.0) if peak_balance > 0 else 0.0
        if dd_inr > max_dd_inr:
            max_dd_inr = dd_inr
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
            
        total_stt += trade_stt
        total_stamp_duty += trade_stamp_duty
        total_exchange += trade_exchange
        total_sebi_ipft += trade_sebi_ipft
        total_brokerage_gst += trade_brokerage_gst
        total_charges += trade_charges
        total_gross_pnl += gross_pnl
        
        trade_dict = dict(row)
        trade_dict.update({
            'deployed_capital': buy_turnover,
            'qty': qty,
            'gross_pnl_inr': gross_pnl,
            'stt_inr': trade_stt,
            'stamp_duty_inr': trade_stamp_duty,
            'exchange_fee_inr': trade_exchange,
            'sebi_ipft_inr': trade_sebi_ipft,
            'brokerage_gst_inr': trade_brokerage_gst,
            'total_charges_inr': trade_charges,
            'net_pnl_inr': net_pnl,
            'net_roi_pct': net_trade_ret_pct,
            'ending_balance': current_balance,
            'gross_balance': gross_balance,
            'drawdown_inr': dd_inr,
            'drawdown_pct': dd_pct
        })
        sim_rows.append(trade_dict)
        
    sim_df = pd.DataFrame(sim_rows)
    if sim_df.empty:
        return pd.DataFrame(), {}
        
    net_profit_inr = current_balance - initial_capital
    net_roi_pct = (net_profit_inr / initial_capital) * 100.0 if initial_capital > 0 else 0.0
    profit_retention = (net_profit_inr / total_gross_pnl * 100.0) if total_gross_pnl > 0 else 0.0
    charges_drag = (total_charges / total_gross_pnl * 100.0) if total_gross_pnl > 0 else 0.0

    # 1. Streaks calculation (Winning & Losing based on Net PnL after charges)
    win_streak = 0
    max_win_streak = 0
    loss_streak = 0
    max_loss_streak = 0
    for pnl in sim_df['net_pnl_inr']:
        if pnl > 0:
            win_streak += 1
            loss_streak = 0
            if win_streak > max_win_streak:
                max_win_streak = win_streak
        elif pnl < 0:
            loss_streak += 1
            win_streak = 0
            if loss_streak > max_loss_streak:
                max_loss_streak = loss_streak
        else:
            win_streak = 0
            loss_streak = 0

    # 2. Best and Worst Trades (Net PnL after charges)
    best_idx = sim_df['net_pnl_inr'].idxmax()
    worst_idx = sim_df['net_pnl_inr'].idxmin()
    best_row = sim_df.loc[best_idx]
    worst_row = sim_df.loc[worst_idx]

    best_trade_inr = float(best_row['net_pnl_inr'])
    best_trade_pct = float(best_row.get('net_roi_pct', 0.0))
    best_trade_date = pd.to_datetime(best_row['exit_date']).strftime('%Y-%m-%d')
    best_trade_sym = str(best_row.get('symbol', ''))

    worst_trade_inr = float(worst_row['net_pnl_inr'])
    worst_trade_pct = float(worst_row.get('net_roi_pct', 0.0))
    worst_trade_date = pd.to_datetime(worst_row['exit_date']).strftime('%Y-%m-%d')
    worst_trade_sym = str(worst_row.get('symbol', ''))

    # 3. Drawdown Episodes & Recovery Days Calculation
    hwm = float(initial_capital)
    hwm_date = pd.to_datetime(sim_df['entry_date'].iloc[0])
    
    dd_episodes = []
    in_dd = False
    curr_peak_bal = hwm
    curr_peak_date = hwm_date
    curr_trough_bal = hwm
    curr_trough_date = hwm_date
    
    for idx, row in sim_df.iterrows():
        bal = float(row['ending_balance'])
        exit_dt = pd.to_datetime(row['exit_date'])
        
        if bal >= hwm:
            if in_dd:
                ep_dd_inr = curr_peak_bal - curr_trough_bal
                ep_dd_pct = (ep_dd_inr / curr_peak_bal) * 100.0 if curr_peak_bal > 0 else 0.0
                dd_episodes.append({
                    'peak_date': curr_peak_date,
                    'trough_date': curr_trough_date,
                    'recovery_date': exit_dt,
                    'peak_bal': curr_peak_bal,
                    'trough_bal': curr_trough_bal,
                    'dd_inr': ep_dd_inr,
                    'dd_pct': ep_dd_pct,
                    'days_peak_to_rec': (exit_dt - curr_peak_date).days,
                    'days_trough_to_rec': (exit_dt - curr_trough_date).days,
                    'recovered': True
                })
                in_dd = False
            hwm = bal
            hwm_date = exit_dt
        else:
            if not in_dd:
                in_dd = True
                curr_peak_bal = hwm
                curr_peak_date = hwm_date
                curr_trough_bal = bal
                curr_trough_date = exit_dt
            else:
                if bal < curr_trough_bal:
                    curr_trough_bal = bal
                    curr_trough_date = exit_dt

    if in_dd:
        last_exit_dt = pd.to_datetime(sim_df['exit_date'].iloc[-1])
        ep_dd_inr = curr_peak_bal - curr_trough_bal
        ep_dd_pct = (ep_dd_inr / curr_peak_bal) * 100.0 if curr_peak_bal > 0 else 0.0
        dd_episodes.append({
            'peak_date': curr_peak_date,
            'trough_date': curr_trough_date,
            'recovery_date': None,
            'peak_bal': curr_peak_bal,
            'trough_bal': curr_trough_bal,
            'dd_inr': ep_dd_inr,
            'dd_pct': ep_dd_pct,
            'days_peak_to_rec': (last_exit_dt - curr_peak_date).days,
            'days_trough_to_rec': (last_exit_dt - curr_trough_date).days,
            'recovered': False
        })
        
    if dd_episodes:
        max_ep = max(dd_episodes, key=lambda x: x['dd_inr'])
        max_dd_inr = max_ep['dd_inr']
        max_dd_pct = max_ep['dd_pct']
        max_dd_recovered = max_ep['recovered']
        max_dd_recovery_days = max_ep['days_trough_to_rec']
        max_dd_peak_to_rec_days = max_ep['days_peak_to_rec']
        max_dd_peak_date = max_ep['peak_date'].strftime('%Y-%m-%d')
        max_dd_trough_date = max_ep['trough_date'].strftime('%Y-%m-%d')
        max_dd_rec_date = max_ep['recovery_date'].strftime('%Y-%m-%d') if max_ep['recovery_date'] is not None else 'Ongoing'
        longest_recovery_days = max(ep['days_trough_to_rec'] for ep in dd_episodes)
        longest_peak_to_rec_days = max(ep['days_peak_to_rec'] for ep in dd_episodes)
    else:
        max_dd_inr = 0.0
        max_dd_pct = 0.0
        max_dd_recovered = True
        max_dd_recovery_days = 0
        max_dd_peak_to_rec_days = 0
        max_dd_peak_date = '-'
        max_dd_trough_date = '-'
        max_dd_rec_date = '-'
        longest_recovery_days = 0
        longest_peak_to_rec_days = 0
    
    summary = {
        'initial_capital': initial_capital,
        'final_capital': current_balance,
        'gross_final_capital': gross_balance,
        'net_profit_inr': net_profit_inr,
        'net_roi_pct': net_roi_pct,
        'gross_profit_inr': total_gross_pnl,
        'total_charges_inr': total_charges,
        'stt_total': total_stt,
        'stamp_duty_total': total_stamp_duty,
        'exchange_total': total_exchange,
        'sebi_ipft_total': total_sebi_ipft,
        'brokerage_gst_total': total_brokerage_gst,
        'profit_retention_pct': profit_retention,
        'charges_drag_pct': charges_drag,
        'peak_balance': peak_balance,
        'max_dd_inr': max_dd_inr,
        'max_dd_pct': max_dd_pct,
        'max_win_streak': max_win_streak,
        'max_loss_streak': max_loss_streak,
        'best_trade_inr': best_trade_inr,
        'best_trade_pct': best_trade_pct,
        'best_trade_date': best_trade_date,
        'best_trade_sym': best_trade_sym,
        'worst_trade_inr': worst_trade_inr,
        'worst_trade_pct': worst_trade_pct,
        'worst_trade_date': worst_trade_date,
        'worst_trade_sym': worst_trade_sym,
        'max_dd_recovery_days': max_dd_recovery_days,
        'max_dd_peak_to_rec_days': max_dd_peak_to_rec_days,
        'max_dd_recovered': max_dd_recovered,
        'max_dd_peak_date': max_dd_peak_date,
        'max_dd_trough_date': max_dd_trough_date,
        'max_dd_rec_date': max_dd_rec_date,
        'longest_recovery_days': longest_recovery_days,
        'longest_peak_to_rec_days': longest_peak_to_rec_days,
        'total_executed_trades': len(sim_df)
    }
    return sim_df, summary


def compute_annual_capital_breakdown(sim_df: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    """
    Computes annual financial ledger showing year-by-year starting balance, gross return,
    taxes & charges deducted, net profit, ending balance, and annual net ROI %.
    """
    if sim_df.empty:
        return pd.DataFrame()
        
    df = sim_df.copy()
    df['year'] = pd.to_datetime(df['entry_date']).dt.year
    df = df.sort_values(by=['entry_date', 'exit_date']).reset_index(drop=True)
    
    annual_records = []
    years = sorted(df['year'].unique())
    curr_start_cap = initial_capital
    
    for yr in years:
        yr_df = df[df['year'] == yr]
        n_trades = len(yr_df)
        yr_gross_pnl = yr_df['gross_pnl_inr'].sum()
        yr_charges = yr_df['total_charges_inr'].sum()
        yr_net_pnl = yr_df['net_pnl_inr'].sum()
        ending_cap = yr_df['ending_balance'].iloc[-1]
        annual_roi = (yr_net_pnl / curr_start_cap * 100.0) if curr_start_cap > 0 else 0.0
        
        annual_records.append({
            'Year': int(yr),
            'Starting Capital': curr_start_cap,
            'Trades': n_trades,
            'Gross Return (₹)': yr_gross_pnl,
            'Taxes & Charges (₹)': yr_charges,
            'Net Return (₹)': yr_net_pnl,
            'Ending Capital': ending_cap,
            'Net ROI (%)': annual_roi
        })
        curr_start_cap = ending_cap
        
    return pd.DataFrame(annual_records)


def render_capital_charges_simulator(
    trades_df: pd.DataFrame,
    asset_label: str,
    key_prefix: str = "cap_sim"
):
    """
    Renders an institutional, interactive Capital & Indian Regulatory Charges Simulation dashboard
    showing actual net monetary returns after all statutory taxes & broker fees.
    """
    if trades_df.empty:
        return

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"### 💰 Net Capital Growth & Indian Regulatory Charges Simulator ({asset_label})")
    st.markdown("""
    Interactive monetary investment simulation calculating actual **net portfolio balance** after deducting
    all statutory Indian market taxes & regulatory fees (**STT, Stamp Duty, NSE Exchange, SEBI, IPFT, Brokerage, and GST**).
    Adjust starting capital, allocation percentage per trade, and statutory fee schedules in real time.
    """)
    
    # 1. User Interactive Controls
    c1, c2, c3 = st.columns([1.5, 1.8, 1.7])
    with c1:
        init_cap = st.number_input(
            "💵 Starting Capital (₹):",
            min_value=10000.0,
            max_value=100000000.0,
            value=100000.0,
            step=10000.0,
            format="%.0f",
            key=f"{key_prefix}_init_cap",
            help="Initial starting portfolio investment amount in INR. Default: ₹1,00,000."
        )
    with c2:
        alloc_pct = st.slider(
            "🎯 Trade Capital Allocation (%):",
            min_value=10,
            max_value=100,
            value=70,
            step=5,
            key=f"{key_prefix}_alloc_pct",
            help="Percentage of total portfolio balance deployed into each executed divergence trade. Default: 70%."
        )
    with c3:
        comp_mode = st.selectbox(
            "🔄 Allocation Sizing Method:",
            ["Dynamic Compounding (Reinvest Profits)", "Static Allocation (Fixed % of Initial)"],
            index=0,
            key=f"{key_prefix}_comp_mode",
            help="Dynamic Compounding deploys 70% of current equity (reinvesting gains). Static deploys 70% of initial ₹1,00,000."
        )
        is_compounding = "Compounding" in comp_mode

    # 2. Expandable Regulatory Charges & Brokerage Schedule
    with st.expander("⚙️ View & Customize Indian Regulatory Charges & Brokerage Schedule", expanded=False):
        st.markdown("""
        Regulatory schedules per SEBI, NSE, and Ministry of Finance mandates:
        - **STT**: 0.1% on Buy & Sell (Equity Delivery) / 0.025% on Sell (Intraday)
        - **Stamp Duty**: 0.015% on Buy (Delivery) / 0.003% on Buy (Intraday)
        - **Exchange Turnover Fee (NSE)**: 0.00297% on Buy & Sell
        - **SEBI Turnover Fee**: 0.0001% (₹10 / crore) on Buy & Sell
        - **IPFT Fee**: 0.0001% on Buy & Sell
        - **Brokerage**: Default ₹0 (free delivery) or customizable
        - **GST**: 18.0% on (Brokerage + Exchange Fees + SEBI Fees)
        """)
        preset_key = f"{key_prefix}_tax_preset"
        
        def _apply_charges_preset(k_prefix):
            chosen = st.session_state.get(f"{k_prefix}_tax_preset", "📋 Standard Equity Delivery")
            if chosen.startswith("📋 Standard Equity Delivery"):
                st.session_state[f"{k_prefix}_stt_b"] = 0.100
                st.session_state[f"{k_prefix}_stt_s"] = 0.100
                st.session_state[f"{k_prefix}_stamp"] = 0.015
                st.session_state[f"{k_prefix}_nse"] = 0.00297
                st.session_state[f"{k_prefix}_sebi"] = 0.00010
                st.session_state[f"{k_prefix}_ipft"] = 0.00010
                st.session_state[f"{k_prefix}_brok"] = 0.0
                st.session_state[f"{k_prefix}_gst"] = 18.0
            elif chosen.startswith("⚡ Equity Intraday"):
                st.session_state[f"{k_prefix}_stt_b"] = 0.000
                st.session_state[f"{k_prefix}_stt_s"] = 0.025
                st.session_state[f"{k_prefix}_stamp"] = 0.003
                st.session_state[f"{k_prefix}_nse"] = 0.00297
                st.session_state[f"{k_prefix}_sebi"] = 0.00010
                st.session_state[f"{k_prefix}_ipft"] = 0.00010
                st.session_state[f"{k_prefix}_brok"] = 20.0
                st.session_state[f"{k_prefix}_gst"] = 18.0
            else:
                st.session_state[f"{k_prefix}_stt_b"] = 0.0
                st.session_state[f"{k_prefix}_stt_s"] = 0.0
                st.session_state[f"{k_prefix}_stamp"] = 0.0
                st.session_state[f"{k_prefix}_nse"] = 0.0
                st.session_state[f"{k_prefix}_sebi"] = 0.0
                st.session_state[f"{k_prefix}_ipft"] = 0.0
                st.session_state[f"{k_prefix}_brok"] = 0.0
                st.session_state[f"{k_prefix}_gst"] = 0.0

        if f"{key_prefix}_stt_b" not in st.session_state:
            _apply_charges_preset(key_prefix)

        curr_preset = st.radio(
            "Charges Preset:",
            ["📋 Standard Equity Delivery (Statutory Schedule)", "⚡ Equity Intraday", "🛠️ Custom / Zero Charges"],
            horizontal=True,
            key=preset_key,
            on_change=_apply_charges_preset,
            args=(key_prefix,)
        )
            
        tc1, tc2, tc3, tc4 = st.columns(4)
        with tc1:
            stt_b = st.number_input("STT Buy (%):", min_value=0.0, max_value=2.0, step=0.01, format="%.3f", key=f"{key_prefix}_stt_b")
            stt_s = st.number_input("STT Sell (%):", min_value=0.0, max_value=2.0, step=0.01, format="%.3f", key=f"{key_prefix}_stt_s")
        with tc2:
            stamp = st.number_input("Stamp Duty Buy (%):", min_value=0.0, max_value=1.0, step=0.005, format="%.3f", key=f"{key_prefix}_stamp")
            nse_fee = st.number_input("NSE Exchange Fee (%):", min_value=0.0, max_value=1.0, step=0.0001, format="%.5f", key=f"{key_prefix}_nse")
        with tc3:
            sebi_fee = st.number_input("SEBI Turnover Fee (%):", min_value=0.0, max_value=0.1, step=0.00005, format="%.5f", key=f"{key_prefix}_sebi")
            ipft_fee = st.number_input("IPFT Fee (%):", min_value=0.0, max_value=0.1, step=0.00005, format="%.5f", key=f"{key_prefix}_ipft")
        with tc4:
            brok_val = st.number_input("Flat Brokerage / Order (₹):", min_value=0.0, max_value=500.0, step=5.0, format="%.1f", key=f"{key_prefix}_brok")
            gst_val = st.number_input("GST Rate (%):", min_value=0.0, max_value=30.0, step=1.0, format="%.1f", key=f"{key_prefix}_gst")
            
        est_statutory_rt = stt_b + stt_s + stamp + (nse_fee * 2) + (sebi_fee * 2) + (ipft_fee * 2)
        st.markdown(f"<span style='font-size:12px; color:#38bdf8;'>Total Round-Trip Statutory Rate: <strong>{est_statutory_rt:.5f}%</strong> of trade turnover (+ Flat Brokerage ₹{brok_val:.1f} & GST {gst_val:.1f}%)</span>", unsafe_allow_html=True)

    # 3. Execute Simulation Engine
    sim_df, summary = simulate_capital_and_charges(
        trades_df=trades_df,
        initial_capital=init_cap,
        allocation_pct=float(alloc_pct),
        compounding=is_compounding,
        stt_buy_pct=stt_b,
        stt_sell_pct=stt_s,
        stamp_duty_pct=stamp,
        exchange_pct=nse_fee,
        sebi_pct=sebi_fee,
        ipft_pct=ipft_fee,
        brokerage_flat=brok_val,
        brokerage_pct=0.0,
        gst_pct=gst_val
    )
    
    if sim_df.empty or not summary:
        st.info("Insufficient trade records to simulate capital performance.")
        return

    # 4. Executive Capital Metric Cards
    final_cap = summary['final_capital']
    net_pnl = summary['net_profit_inr']
    net_roi = summary['net_roi_pct']
    gross_pnl = summary['gross_profit_inr']
    tot_charges = summary['total_charges_inr']
    retention = summary['profit_retention_pct']
    drag = summary['charges_drag_pct']
    peak = summary['peak_balance']
    
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    
    with m1:
        st.markdown(f"""
        <div class='metric-card'>
            <span class='metric-label'>Starting Capital</span>
            <div class='metric-value' style='color:#38bdf8;'>₹{init_cap:,.0f}</div>
            <span style='font-size:11px;color:#94a3b8;'>{alloc_pct}% Allocated / Trade</span>
        </div>
        """, unsafe_allow_html=True)
        
    with m2:
        cap_col = "#00d26a" if final_cap >= init_cap else "#f8312f"
        st.markdown(f"""
        <div class='metric-card'>
            <span class='metric-label'>Final Net Capital</span>
            <div class='metric-value' style='color:{cap_col};'>₹{final_cap:,.0f}</div>
            <span style='font-size:11px;color:#94a3b8;'>Peak: ₹{peak:,.0f}</span>
        </div>
        """, unsafe_allow_html=True)
        
    with m3:
        pnl_col = "#00d26a" if net_pnl >= 0 else "#f8312f"
        st.markdown(f"""
        <div class='metric-card'>
            <span class='metric-label'>Net Profit (₹)</span>
            <div class='metric-value' style='color:{pnl_col};'>{net_pnl:+,.0f}</div>
            <span style='font-size:11px;color:#94a3b8;'>After All Taxes & Fees</span>
        </div>
        """, unsafe_allow_html=True)
        
    with m4:
        roi_col = "#00d26a" if net_roi >= 0 else "#f8312f"
        st.markdown(f"""
        <div class='metric-card'>
            <span class='metric-label'>Net ROI (%)</span>
            <div class='metric-value' style='color:{roi_col};'>{net_roi:+.1f}%</div>
            <span style='font-size:11px;color:#94a3b8;'>Overall Portfolio Gain</span>
        </div>
        """, unsafe_allow_html=True)
        
    with m5:
        gp_col = "#00d26a" if gross_pnl >= 0 else "#f8312f"
        st.markdown(f"""
        <div class='metric-card'>
            <span class='metric-label'>Gross Profit (₹)</span>
            <div class='metric-value' style='color:{gp_col};'>{gross_pnl:+,.0f}</div>
            <span style='font-size:11px;color:#94a3b8;'>Pre-Tax Signal Return</span>
        </div>
        """, unsafe_allow_html=True)
        
    with m6:
        st.markdown(f"""
        <div class='metric-card'>
            <span class='metric-label'>Total Taxes & Fees</span>
            <div class='metric-value' style='color:#fbbf24;'>-₹{tot_charges:,.0f}</div>
            <span style='font-size:11px;color:#94a3b8;'>STT, Stamp, NSE, SEBI</span>
        </div>
        """, unsafe_allow_html=True)
        
    with m7:
        ret_col = "#00d26a" if retention >= 70 else ("#fbbf24" if retention >= 40 else "#f8312f")
        st.markdown(f"""
        <div class='metric-card'>
            <span class='metric-label'>Profit Retained</span>
            <div class='metric-value' style='color:{ret_col};'>{retention:.1f}%</div>
            <span style='font-size:11px;color:#94a3b8;'>Fee Drag: {drag:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)
        
    # Row 2: Risk, Streaks & Drawdown Analytics (6 Cards)
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    r1, r2, r3, r4, r5, r6 = st.columns(6)
    
    with r1:
        st.markdown(f"""
        <div class='metric-card'>
            <span class='metric-label'>Max Drawdown</span>
            <div class='metric-value-neg'>-₹{summary['max_dd_inr']:,.0f}</div>
            <span style='font-size:11px;color:#f8312f;'>-{summary['max_dd_pct']:.1f}% Peak-to-Trough</span>
        </div>
        """, unsafe_allow_html=True)
        
    with r2:
        if summary['max_dd_inr'] == 0:
            rec_val = "0 Days"
            rec_sub = "No Drawdowns"
            rec_col = "#00d26a"
            dd_tooltip = "No Drawdowns Detected"
        elif summary['max_dd_recovered']:
            rec_val = f"{summary['max_dd_peak_to_rec_days']:,} Days"
            rec_sub = f"{summary['max_dd_peak_date']} ➔ {summary['max_dd_rec_date']}"
            rec_col = "#38bdf8"
            dd_tooltip = f"Peak: {summary['max_dd_peak_date']} | Bottom (Trough): {summary['max_dd_trough_date']} | Recovery: {summary['max_dd_rec_date']}"
        else:
            rec_val = f"{summary['max_dd_peak_to_rec_days']:,}d (Ongoing)"
            rec_sub = f"{summary['max_dd_peak_date']} ➔ Ongoing"
            rec_col = "#fbbf24"
            dd_tooltip = f"Peak: {summary['max_dd_peak_date']} | Bottom (Trough): {summary['max_dd_trough_date']} | Status: Ongoing / Unrecovered"
            
        st.markdown(f"""
        <div class='metric-card'>
            <span class='metric-label'>Days to Recover DD</span>
            <div class='metric-value' style='color:{rec_col};'>{rec_val}</div>
            <span style='font-size:11px;color:#94a3b8;' title='{dd_tooltip}'>{rec_sub}</span>
        </div>
        """, unsafe_allow_html=True)
        
    with r3:
        st.markdown(f"""
        <div class='metric-card'>
            <span class='metric-label'>Winning Streak</span>
            <div class='metric-value' style='color:#00d26a;'>{summary['max_win_streak']} Trades</div>
            <span style='font-size:11px;color:#94a3b8;'>Max Consecutive Wins</span>
        </div>
        """, unsafe_allow_html=True)
        
    with r4:
        st.markdown(f"""
        <div class='metric-card'>
            <span class='metric-label'>Losing Streak</span>
            <div class='metric-value' style='color:#f8312f;'>{summary['max_loss_streak']} Trades</div>
            <span style='font-size:11px;color:#94a3b8;'>Max Consecutive Losses</span>
        </div>
        """, unsafe_allow_html=True)
        
    with r5:
        best_sym_txt = f"{summary['best_trade_sym']} • " if summary.get('best_trade_sym') else ""
        st.markdown(f"""
        <div class='metric-card'>
            <span class='metric-label'>Best Trade</span>
            <div class='metric-value' style='color:#00d26a;'>+₹{summary['best_trade_inr']:,.0f}</div>
            <span style='font-size:11px;color:#94a3b8;'>{best_sym_txt}+{summary['best_trade_pct']:.1f}% ({summary['best_trade_date']})</span>
        </div>
        """, unsafe_allow_html=True)
        
    with r6:
        worst_sym_txt = f"{summary['worst_trade_sym']} • " if summary.get('worst_trade_sym') else ""
        worst_val = summary['worst_trade_inr']
        worst_col = "#f8312f" if worst_val < 0 else "#94a3b8"
        worst_prefix = "-" if worst_val < 0 else ""
        st.markdown(f"""
        <div class='metric-card'>
            <span class='metric-label'>Worst Trade</span>
            <div class='metric-value' style='color:{worst_col};'>{worst_prefix}₹{abs(worst_val):,.0f}</div>
            <span style='font-size:11px;color:#94a3b8;'>{worst_sym_txt}{summary['worst_trade_pct']:+.1f}% ({summary['worst_trade_date']})</span>
        </div>
        """, unsafe_allow_html=True)
        
    # Itemized Taxes Summary Ribbon
    st.markdown(f"""
    <div style='background:rgba(15, 23, 42, 0.6); border:1px solid #1e293b; border-radius:6px; padding:8px 14px; margin-top:10px; font-size:12px; color:#cbd5e1;'>
        <strong>🧾 Statutory Taxes Breakdown:</strong> &nbsp;
        <span style='color:#38bdf8;'>STT:</span> ₹{summary['stt_total']:,.2f} &nbsp;|&nbsp;
        <span style='color:#38bdf8;'>Stamp Duty:</span> ₹{summary['stamp_duty_total']:,.2f} &nbsp;|&nbsp;
        <span style='color:#38bdf8;'>NSE Charges:</span> ₹{summary['exchange_total']:,.2f} &nbsp;|&nbsp;
        <span style='color:#38bdf8;'>SEBI & IPFT:</span> ₹{summary['sebi_ipft_total']:,.2f} &nbsp;|&nbsp;
        <span style='color:#38bdf8;'>Brokerage & GST:</span> ₹{summary['brokerage_gst_total']:,.2f} &nbsp;|&nbsp;
        <span style='color:#fbbf24;'>Total Deducted:</span> <strong>₹{tot_charges:,.2f}</strong> &nbsp;|&nbsp;
        <span style='color:#a78bfa;'>Max DD Episode:</span> Peak {summary['max_dd_peak_date']} ➔ Trough {summary['max_dd_trough_date']} ➔ Recovery: {summary['max_dd_rec_date']}
    </div>
    """, unsafe_allow_html=True)
    
    # 5. Plotly Interactive Capital Growth Chart
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📈 Compounded Net Portfolio Balance & Charges Drag Curve")
    
    sim_df_sorted = sim_df.sort_values(by='exit_date').reset_index(drop=True)
    sim_df_sorted['cum_charges'] = sim_df_sorted['total_charges_inr'].cumsum()
    
    fig_cap = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.72, 0.28],
        subplot_titles=(
            f"Net Capital Growth After Indian Regulatory Charges (₹) — {asset_label}",
            "Cumulative Regulatory Taxes & Brokerage Deducted (₹)"
        )
    )
    
    fig_cap.add_trace(go.Scatter(
        x=sim_df_sorted['exit_date'],
        y=sim_df_sorted['ending_balance'],
        mode='lines',
        name='Net Portfolio Balance (After Charges)',
        line=dict(color='#00d26a', width=2.5)
    ), row=1, col=1)
    
    fig_cap.add_trace(go.Scatter(
        x=sim_df_sorted['exit_date'],
        y=sim_df_sorted['gross_balance'],
        mode='lines',
        name='Gross Portfolio Balance (Pre-Tax)',
        line=dict(color='#38bdf8', width=1.5, dash='dot')
    ), row=1, col=1)
    
    fig_cap.add_trace(go.Scatter(
        x=[sim_df_sorted['exit_date'].iloc[0], sim_df_sorted['exit_date'].iloc[-1]],
        y=[init_cap, init_cap],
        mode='lines',
        name=f'Initial Capital Baseline (₹{init_cap:,.0f})',
        line=dict(color='#94a3b8', width=1.5, dash='dash')
    ), row=1, col=1)
    
    fig_cap.add_trace(go.Scatter(
        x=sim_df_sorted['exit_date'],
        y=sim_df_sorted['cum_charges'],
        mode='lines',
        name='Cumulative Taxes & Fees (₹)',
        fill='tozeroy',
        line=dict(color='#fbbf24', width=1.5),
        fillcolor='rgba(251, 191, 36, 0.20)'
    ), row=2, col=1)
    
    fig_cap.update_layout(
        template="plotly_dark",
        height=500,
        margin=dict(l=40, r=40, t=65, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1)
    )
    st.plotly_chart(fig_cap, width="stretch")
    
    # 6. Annual Capital Growth Breakdown Table
    st.markdown("#### 📅 Calendar Year Capital Growth & Taxes Audit")
    annual_df = compute_annual_capital_breakdown(sim_df, initial_capital=init_cap)
    
    if not annual_df.empty:
        ann_cfg = {
            'Year': st.column_config.NumberColumn('Year', format='%d'),
            'Starting Capital': st.column_config.NumberColumn('Starting Capital (₹)', format='₹%.0f'),
            'Trades': st.column_config.NumberColumn('Trades', format='%d'),
            'Gross Return (₹)': st.column_config.NumberColumn('Gross Return (₹)', format='₹%+.0f'),
            'Taxes & Charges (₹)': st.column_config.NumberColumn('Taxes & Charges (₹)', format='₹%.0f'),
            'Net Return (₹)': st.column_config.NumberColumn('Net Return (₹)', format='₹%+.0f'),
            'Ending Capital': st.column_config.NumberColumn('Ending Capital (₹)', format='₹%.0f'),
            'Net ROI (%)': st.column_config.NumberColumn('Net ROI (%)', format='%+.1f%%')
        }
        st.dataframe(annual_df, column_config=ann_cfg, width="stretch", hide_index=True)
        
    # 7. Itemized Trade Capital & Charges Audit Ledger
    st.markdown("#### 📋 Itemized Trade Capital & Charges Audit Ledger")
    
    ledger_cols = [
        'signal_date', 'entry_date', 'exit_date', 'action',
        'entry_price', 'exit_price', 'qty', 'deployed_capital',
        'gross_pnl_inr', 'stt_inr', 'stamp_duty_inr', 'exchange_fee_inr',
        'total_charges_inr', 'net_pnl_inr', 'net_roi_pct', 'ending_balance',
        'drawdown_pct'
    ]
    if 'symbol' in sim_df.columns:
        ledger_cols.insert(0, 'symbol')
        
    disp_ledger = sim_df[ledger_cols].copy()
    disp_ledger['signal_date'] = pd.to_datetime(disp_ledger['signal_date']).dt.strftime('%Y-%m-%d')
    disp_ledger['entry_date'] = pd.to_datetime(disp_ledger['entry_date']).dt.strftime('%Y-%m-%d')
    disp_ledger['exit_date'] = pd.to_datetime(disp_ledger['exit_date']).dt.strftime('%Y-%m-%d')
    
    disp_ledger = disp_ledger.rename(columns={
        'symbol': 'Symbol',
        'signal_date': 'Signal Date',
        'entry_date': 'Entry Date',
        'exit_date': 'Exit Date',
        'action': 'Action',
        'entry_price': 'Entry Price',
        'exit_price': 'Exit Price',
        'qty': 'Qty',
        'deployed_capital': 'Capital Deployed',
        'gross_pnl_inr': 'Gross PnL (₹)',
        'stt_inr': 'STT (₹)',
        'stamp_duty_inr': 'Stamp Duty (₹)',
        'exchange_fee_inr': 'Exchange & SEBI (₹)',
        'total_charges_inr': 'Total Charges (₹)',
        'net_pnl_inr': 'Net PnL (₹)',
        'net_roi_pct': 'Net ROI (%)',
        'ending_balance': 'Account Balance (₹)',
        'drawdown_pct': 'Drawdown (%)'
    })
    
    ledger_cfg = {
        'Entry Price': st.column_config.NumberColumn('Entry Price', format='₹%.2f'),
        'Exit Price': st.column_config.NumberColumn('Exit Price', format='₹%.2f'),
        'Qty': st.column_config.NumberColumn('Qty', format='%d'),
        'Capital Deployed': st.column_config.NumberColumn('Capital Deployed', format='₹%.0f'),
        'Gross PnL (₹)': st.column_config.NumberColumn('Gross PnL (₹)', format='₹%+.2f'),
        'STT (₹)': st.column_config.NumberColumn('STT (₹)', format='₹%.2f'),
        'Stamp Duty (₹)': st.column_config.NumberColumn('Stamp Duty (₹)', format='₹%.2f'),
        'Exchange & SEBI (₹)': st.column_config.NumberColumn('Exchange & SEBI (₹)', format='₹%.2f'),
        'Total Charges (₹)': st.column_config.NumberColumn('Total Charges (₹)', format='₹%.2f'),
        'Net PnL (₹)': st.column_config.NumberColumn('Net PnL (₹)', format='₹%+.2f'),
        'Net ROI (%)': st.column_config.NumberColumn('Net ROI (%)', format='%+.2f%%'),
        'Account Balance (₹)': st.column_config.NumberColumn('Account Balance (₹)', format='₹%.2f'),
        'Drawdown (%)': st.column_config.NumberColumn('Drawdown (%)', format='-%.1f%%')
    }
    st.dataframe(disp_ledger, column_config=ledger_cfg, width="stretch", hide_index=True)
    
    csv_ledger = sim_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📥 Download Capital & Charges Ledger CSV ({len(sim_df):,} Trades)",
        data=csv_ledger,
        file_name=f"{asset_label.lower().replace(' ', '_')}_capital_charges_ledger.csv",
        mime="text/csv",
        key=f"{key_prefix}_btn_download_ledger"
    )


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
            'Max DD (%)': st.column_config.NumberColumn('Max DD (%)', format='%.2f%%', help='Maximum peak-to-trough equity drawdown percentage'),
            'Max DD Holding (Days)': st.column_config.NumberColumn('Max DD Holding (Days)', format='%d', help='Duration in calendar days of the maximum drawdown episode (Peak to Trough)'),
            'Avg Holding (Days)': st.column_config.NumberColumn('Avg Holding (Days)', format='%.1f'),
            'Min Holding (Days)': st.column_config.NumberColumn('Min Holding (Days)', format='%d'),
            'Max Holding (Days)': st.column_config.NumberColumn('Max Holding (Days)', format='%d'),
            'Avg Holding (Bars)': st.column_config.NumberColumn('Avg Holding (Bars)', format='%.1f')
        }
        st.dataframe(disp_stocks, column_config=stock_cfg, width="stretch", hide_index=True)
        
        csv_stocks = disp_stocks.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Stock Performance Matrix CSV",
            data=csv_stocks,
            file_name=f"stock_performance_matrix_{timeframe.lower()}.csv",
            mime="text/csv",
            key="btn_download_stock_matrix_csv"
        )

        
    # =============================================================
    # 5. CUMULATIVE PORTFOLIO GROWTH CURVE
    # =============================================================
    st.markdown("### 📈 Cumulative Portfolio Return & Drawdown Curve")
    trades_df_sorted = trades_df.sort_values('exit_date').reset_index(drop=True)
    trades_df_sorted['cum_pnl_pct'] = trades_df_sorted['pnl_pct'].cumsum()
    trades_df_sorted['peak_pnl_pct'] = trades_df_sorted['cum_pnl_pct'].cummax()
    trades_df_sorted['drawdown_pct'] = trades_df_sorted['cum_pnl_pct'] - trades_df_sorted['peak_pnl_pct']
    
    # Calculate portfolio drawdown episodes and days underwater
    peak_val = 0.0
    peak_date = pd.to_datetime(trades_df_sorted['entry_date'].iloc[0])
    dd_days_col = []
    port_dd_episodes = []
    in_port_dd = False
    ep_trough_val = 0.0
    ep_trough_date = peak_date
    ep_peak_date = peak_date

    for idx, row in trades_df_sorted.iterrows():
        cum = float(row['cum_pnl_pct'])
        exit_dt = pd.to_datetime(row['exit_date'])
        if cum >= peak_val:
            if in_port_dd:
                ep_depth = ep_trough_val - peak_val
                port_dd_episodes.append({
                    'peak_date': ep_peak_date,
                    'trough_date': ep_trough_date,
                    'rec_date': exit_dt,
                    'dd_pct': ep_depth,
                    'recovery_days': (exit_dt - ep_peak_date).days,
                    'recovered': True
                })
                in_port_dd = False
            peak_val = cum
            peak_date = exit_dt
            dd_days_col.append(0)
        else:
            if not in_port_dd:
                in_port_dd = True
                ep_peak_date = peak_date
                ep_trough_val = cum
                ep_trough_date = exit_dt
            else:
                if cum < ep_trough_val:
                    ep_trough_val = cum
                    ep_trough_date = exit_dt
            days_under = (exit_dt - peak_date).days
            dd_days_col.append(days_under)

    if in_port_dd:
        last_exit_dt = pd.to_datetime(trades_df_sorted['exit_date'].iloc[-1])
        port_dd_episodes.append({
            'peak_date': ep_peak_date,
            'trough_date': ep_trough_date,
            'rec_date': None,
            'dd_pct': ep_trough_val - peak_val,
            'recovery_days': (last_exit_dt - ep_peak_date).days,
            'recovered': False
        })

    trades_df_sorted['dd_days'] = dd_days_col
    port_max_dd_pct = float(trades_df_sorted['drawdown_pct'].min())
    if port_dd_episodes:
        max_ep = min(port_dd_episodes, key=lambda x: x['dd_pct'])
        port_max_dd_days = max_ep['recovery_days']
        port_max_dd_peak = max_ep['peak_date'].strftime('%Y-%m-%d')
        port_max_dd_rec = max_ep['rec_date'].strftime('%Y-%m-%d') if max_ep['rec_date'] is not None else 'Ongoing'
    else:
        port_max_dd_days = 0
        port_max_dd_peak = '-'
        port_max_dd_rec = '-'

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

    # Drawdown KPI Summary Strip
    pk1, pk2, pk3 = st.columns([1.5, 2.0, 1.5])
    with pk1:
        st.markdown(f"""
        <div class='metric-card'>
            <span class='metric-label'>Portfolio Max Drawdown</span>
            <div class='metric-value-neg'>{port_max_dd_pct:.2f}%</div>
            <span style='font-size:11px;color:#f8312f;'>Peak-to-Trough Decline</span>
        </div>
        """, unsafe_allow_html=True)
    with pk2:
        rec_period_str = f"{port_max_dd_peak} ➔ {port_max_dd_rec}"
        st.markdown(f"""
        <div class='metric-card'>
            <span class='metric-label'>Max DD Days</span>
            <div class='metric-value' style='color:#38bdf8;'>{port_max_dd_days:,} Days</div>
            <span style='font-size:11px;color:#94a3b8;'>Period: {rec_period_str}</span>
        </div>
        """, unsafe_allow_html=True)
    with pk3:
        st.markdown(f"""
        <div class='metric-card'>
            <span class='metric-label'>Total Executed Trades</span>
            <div class='metric-value' style='color:#00d26a;'>{len(trades_df):,}</div>
            <span style='font-size:11px;color:#94a3b8;'>Across {unique_stocks} Equities</span>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

    fl1, fl2, fl3 = st.columns([1.5, 1.2, 1.2])
    with fl1:
        sym_options = ["All Equities"] + sorted(trades_df_sorted['symbol'].unique().tolist())
        sel_sym = st.selectbox("Filter by Symbol:", sym_options, key="sec3_port_filt_sym")
    with fl2:
        sel_act = st.selectbox("Filter by Action:", ["All Actions", "BUY Only", "SELL Only"], key="sec3_port_filt_act")
    with fl3:
        sel_out = st.selectbox("Filter by Outcome:", ["All Outcomes", "Wins Only", "Losses Only"], key="sec3_port_filt_out")
        
    filt_df = trades_df_sorted.copy()
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
        
    disp_cols = [
        'symbol', 'signal_date', 'entry_date', 'exit_date', 'action', 'reason',
        'entry_price', 'exit_price', 'exit_reason', 'holding_bars', 'holding_days',
        'pnl_pct', 'drawdown_pct', 'dd_days'
    ]
    
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
        'pnl_pct': 'Return (%)',
        'drawdown_pct': 'Drawdown (%)',
        'dd_days': 'DD Days'
    })
    
    trade_port_cfg = {
        'Entry Price': st.column_config.NumberColumn('Entry Price', format='₹%.2f'),
        'Exit Price': st.column_config.NumberColumn('Exit Price', format='₹%.2f'),
        'Holding Bars': st.column_config.NumberColumn('Holding Bars', format='%d'),
        'Holding Days': st.column_config.NumberColumn('Holding Days', format='%d'),
        'Return (%)': st.column_config.NumberColumn('Return (%)', format='%+.2f%%'),
        'Drawdown (%)': st.column_config.NumberColumn('Drawdown (%)', format='%.2f%%'),
        'DD Days': st.column_config.NumberColumn('DD Days', format='%d')
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
    <style>
    /* Responsive Mobile & Tablet Adjustments for Section 3 */
    @media (max-width: 900px) {
        div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 12px !important;
        }
    }
    
    /* Touch-friendly Toggle Styles across Mobile, Tablet, and Desktop */
    div[data-testid="stToggle"] {
        cursor: pointer !important;
        touch-action: manipulation !important;
        user-select: none !important;
        -webkit-tap-highlight-color: rgba(56, 189, 248, 0.2) !important;
        padding: 4px 0 !important;
        width: 100% !important;
    }
    
    div[data-testid="stToggle"] label {
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
        width: 100% !important;
    }
    
    div[data-testid="stToggle"] label p {
        font-size: 14.5px !important;
        font-weight: 600 !important;
        color: #e2e8f0 !important;
        cursor: pointer !important;
        margin: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
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
        min_data_date = pd.to_datetime("1990-01-01").date()
        max_data_date = pd.to_datetime("2026-09-19").date()
    else:
        raw_df = load_asset_ohlc(selected_asset, timeframe)
        min_data_date = raw_df['date'].min().date() if not raw_df.empty else pd.to_datetime("1990-01-01").date()
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
    
    # Toggle placed directly on its own row to provide 100% full-width touch area on mobile, tablet, and desktop
    enable_ema_filter = st.toggle(
        "🛡️ Enable Anti-Chasing EMA Filter",
        value=False,
        key="sec3_enable_ema_filter",
        help="When enabled: Skips SELL when candle close is below EMA (avoids selling into the low) and skips BUY when candle close is above EMA (avoids buying into the high)."
    )
    
    if enable_ema_filter:
        c1, c2 = st.columns([1.5, 3.5])
        with c1:
            ema_period = st.number_input(
                "⚡ EMA Period:",
                min_value=2,
                max_value=200,
                value=21,
                step=1,
                key="sec3_ema_period",
                help="Variable EMA period (e.g. 9, 21, 50, etc.)."
            )
        with c2:
            st.markdown(f"""
            <div style='background:rgba(56, 189, 248, 0.08); border-left: 3px solid #38bdf8; padding: 8px 14px; border-radius: 6px; font-size: 13px; color: #cbd5e1; margin-top: 4px;'>
                <strong>Anti-Chasing Logic Active ({ema_period} EMA):</strong><br>
                • <strong>Close &lt; {ema_period} EMA:</strong> 🟢 Take BUY | ❌ Skip SELL (avoids selling bottom)<br>
                • <strong>Close &gt; {ema_period} EMA:</strong> 🔴 Take SELL | ❌ Skip BUY (avoids buying top)
            </div>
            """, unsafe_allow_html=True)
    else:
        ema_period = 21
        st.caption("ℹ️ Anti-chasing filter is disabled. The backtest engine executes all divergence signals without EMA restrictions.")
        
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
            min_lim = min(min_data_date, max_data_date)
            max_lim = max(min_data_date, max_data_date)
            if date_filter_mode == "Custom Range":
                start_f = st.date_input(
                    "Filter Start Date:",
                    value=min_lim,
                    min_value=min_lim,
                    max_value=max_lim,
                    key="sec3_custom_start_date"
                )
                end_f = st.date_input(
                    "Filter End Date:",
                    value=max_lim,
                    min_value=min_lim,
                    max_value=max_lim,
                    key="sec3_custom_end_date"
                )
            elif date_filter_mode.startswith("Last 10 Years"):
                start_f = max(min_lim, pd.to_datetime("2016-01-01").date())
                end_f = max_data_date
            elif date_filter_mode.startswith("Post-Covid"):
                start_f = max(min_lim, pd.to_datetime("2020-01-01").date())
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
    
    is_nifty = (selected_asset == "NIFTY 50")
    total_trades = stats['total_trades']
    win_rate = stats['win_rate']
    total_pnl = stats['total_pnl_pts']
    pf = stats['profit_factor']
    
    if is_nifty:
        max_dd = stats['max_dd_pts']
        max_dd_val_str = f"{max_dd:,.1f}"
        max_dd_period_str = stats.get('max_dd_period_str', '')
        max_dd_peak_date = stats.get('max_dd_peak_date', '-')
        max_dd_trough_date = stats.get('max_dd_trough_date', '-')
        max_dd_rec_date = stats.get('max_dd_rec_date', '-')
        max_dd_recovered = stats.get('max_dd_recovered', True)
        max_dd_days = stats.get('max_dd_days', 0)
        max_dd_pt_days = stats.get('max_dd_pt_days', 0)
    else:
        max_dd_pct_val = stats.get('max_dd_pct', 0.0)
        max_dd = max_dd_pct_val
        max_dd_val_str = f"{max_dd_pct_val:.2f}%"
        max_dd_period_str = stats.get('max_dd_pct_period_str', '')
        max_dd_peak_date = stats.get('max_dd_pct_peak_date', '-')
        max_dd_trough_date = stats.get('max_dd_pct_trough_date', '-')
        max_dd_rec_date = stats.get('max_dd_pct_rec_date', '-')
        max_dd_recovered = stats.get('max_dd_pct_recovered', True)
        max_dd_days = stats.get('max_dd_pct_days', 0)
        max_dd_pt_days = stats.get('max_dd_pct_pt_days', 0)

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
        if max_dd_period_str and max_dd_period_str != "No Drawdown":
            rec_status = f"Recovery: {max_dd_rec_date} ({max_dd_days} Days)" if max_dd_recovered else f"Status: Ongoing / Unrecovered ({max_dd_days} Days)"
            pts_extra = f" | Points DD: {stats['max_dd_pts']:,.1f} pts" if not is_nifty else ""
            dd_tooltip = f"Peak: {max_dd_peak_date} | Bottom (Trough): {max_dd_trough_date} ({max_dd_pt_days} days) | {rec_status}{pts_extra}"
            dd_sub = f"<span style='color:#94a3b8;'>{max_dd_pt_days} days </span><span style='color:#fca5a5;'>({max_dd_period_str})</span>"
        else:
            dd_tooltip = "No Drawdowns Detected"
            dd_sub = "<span style='color:#94a3b8;'>0 days (No Drawdown)</span>"
        st.markdown(f"""<div class='metric-card'><span class='metric-label'>Max Drawdown</span><div class='metric-value-neg'>{max_dd_val_str}</div><span style='font-size:10.5px;color:#94a3b8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:block;' title='{dd_tooltip}'>{dd_sub}</span></div>""", unsafe_allow_html=True)
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
                'Metric Dimension': 'Max Drawdown (Points)' if is_nifty else 'Max Drawdown (%)',
                'Unfiltered (All Signals)': f"{all_stats.get('max_dd_pts', 0):,.1f}" if is_nifty else f"{all_stats.get('max_dd_pct', 0.0):.2f}%",
                f'With {ema_period} EMA Filter': f"{max_dd:,.1f}" if is_nifty else f"{stats.get('max_dd_pct', 0.0):.2f}%",
                'Filter Impact / Gain': f"{max_dd - all_stats.get('max_dd_pts', 0):+,.1f} pts saved" if is_nifty else f"{stats.get('max_dd_pct', 0.0) - all_stats.get('max_dd_pct', 0.0):+.2f}%"
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

    # 7. CAPITAL GROWTH & INDIAN REGULATORY CHARGES SIMULATION (FOR STOCKS ONLY, EXCLUDED FOR NIFTY 50)
    if selected_asset != "NIFTY 50":
        render_capital_charges_simulator(
            trades_df=trades_df,
            asset_label=selected_asset,
            key_prefix="single_stock"
        )


