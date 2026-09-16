# ⚡ Nifty & Stock Analyzer

A comprehensive, institutional-grade market analytics platform combining statistical probability models for the **NIFTY 50 Index** with automated multi-timeframe **RSI Divergence scanning and backtesting** across all **500 Nifty Equities**.

---

## 🌟 Platform Sections & Core Features

### 📊 Section 1: Range & Move Distribution
- Multi-timeframe (Daily, Weekly, Monthly) statistical profiling of Nifty historical moves from 2015 to 2026.
- Directional move probabilities, bar size percentiles, and volatility distributions.

### 🔄 Section 2: First-Touch Stretch & Reversals
- Analysis of intraday first-touch threshold behaviors, forward extensions, and mean-reversion tendencies.
- Conditional probability models for intraday and swing trend exhaustion.

### 📐 Section 3: RSI Custom Divergence Backtest Engine
- PineScript-exact RSI divergence algorithm matching user parameters (`rsi_len=14, lbL=5, lbR=5, tol=0.002`).
- Interactive candlestick charts with Bullish/Bearish divergence markers, stop loss lines, and target projections.
- Realistic execution modeling:
  - **Theoretical Peak Entry:** Next Open $(p+1)$
  - **Real-Time Safe Entry:** Confirmation Close $(p+lbR)$
- Comprehensive performance audits:
  - Total Trades, Net Return, Win Rate (%), Profit Factor, Max Drawdown
  - **Annual Breakdown:** Calendar Year Performance with separate **🟢 BUY Win Rate** and **🔴 SELL Win Rate**.

### 📥 Section 4: Nifty 500 Automated Data Downloader
- One-click batch downloader for all 501 symbols from `ind_nifty500list.csv` using TradingView (`tvDatafeed`).
- Direct in-memory conversion to Apache Parquet across Daily, Weekly, and Monthly timeframes.
- Real-time audit dashboard with separate Success vs. Failure tables and a 1-click **"🔄 Re-try Failed Stocks Only"** button.

### 🔍 Section 5: Nifty 500 Multi-Stock Divergence Scanner
- Ultra-fast scanner auditing all 500 stocks in seconds across Daily, Weekly, and Monthly candles.
- Flexible Signal Recency Filters:
  - **`🔥 Today (Signal on Latest Close)`**: Strictly checks if the pivot candle formed on the latest closed candle.
  - **`⚡ Today (Alert Confirmed at p+lbR)`**: Shows stocks where the 5-bar confirmation completed today (real-time alert trigger).
  - **`Last 4 Weeks / 20 Days`**, **`Recent (2026 - Present)`**, and **`All Confirmed Signals`**.
- Transparent columns: `Current / Entry Price` alongside `Pivot Price (p)`.
- 1-Click **"📊 Open in Section 3"** quick-loader for deep dive analysis.

---

## 🚀 Running Locally

```bash
# Clone the repository
git clone <repo-url>
cd "Nifty & Stock Analyzer"

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

---

## ☁️ Streamlit Cloud Deployment

1. Push this repository to GitHub.
2. In [Streamlit Cloud](https://share.streamlit.io):
   - **Repository:** `<your-username>/<repo-name>`
   - **Branch:** `main`
   - **Main file path:** `app.py`
3. Click **Deploy!** All parquet datasets in `data/` and dependencies in `requirements.txt` are self-contained.
