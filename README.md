# ⚡ Nifty & Stock Analyzer

A comprehensive, institutional-grade market analytics platform combining statistical probability models for the **NIFTY 50 Index** with automated data synchronization engines for **Nifty Equities** and the **Nifty 50 Index**.

---

## 🌟 Platform Sections & Core Features

### 📊 Section 1: Range & Move Distribution
- Multi-timeframe (Daily, Weekly, Monthly) statistical profiling of Nifty historical moves from **1996 to 2026** (30+ years, 7,605 authentic trading days).
- Directional move probabilities, bar size percentiles, high/low breach frequencies, and containment stats.
- Dual Close Modes: **TradingView Close** (Standard Calendar) and **Expiry Close** (Official NSE Thursday/Tuesday F&O contract settlement cycles with exchange holiday backward shifting).

### 🔄 Section 2: First-Touch Stretch & Reversals
- Analysis of intraday and intraweek first-touch threshold behaviors, forward breakout extensions, and mean-reversion tendencies.
- Evaluates first touch across Daily (2,896 1-minute sessions), Weekly (1,603 weeks), and Monthly (369 months).
- Measures 50% retracement probabilities, return to period open, trend continuation closes, and failed reversal settlements.

### 📥 Section 3: Automated Market Data Downloader & Parquet Sync (Nifty 50 & Stocks)
- Unified institutional data synchronization engine with sub-tabs for both **NIFTY 50** and **500 Nifty Equities**:
  - **⚡ Nifty 50 Spot & Index:** Maintains **1-Minute Precision Spot History** (`nifty_1min_2015_2026.parquet`) and **Daily / Weekly Higher Timeframes** (`nifty_daily_data_raw.parquet`, `nifty_weekly_data_raw.parquet`) with instant incremental partial syncs.
  - **🏢 Nifty 500 Equities:** One-click batch downloader for all 501 symbols from `ind_nifty500list.csv` converting in-memory directly to Apache Parquet (Daily, Weekly, Monthly) with automated retry for failed downloads.

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
