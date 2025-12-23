# 📊 NIFTY 50 & INDIA VIX Trading Dashboard

A comprehensive trading analysis dashboard for NIFTY 50 and INDIA VIX with advanced analytics, event analysis, and strategy backtesting.

## 🎯 Features

### 1. **NIFTY 50 Analysis**
- ✅ Daily / Weekly / Monthly returns (Simple & Log)
- ✅ Rolling mean of returns (5, 21, 50 days)
- ✅ Drawdowns & Maximum Drawdown tracking
- ✅ Trend labeling: Bullish, Bearish, Sideways

### 2. **INDIA VIX Analysis**
- ✅ Daily change & % change
- ✅ Rolling mean & standard deviation
- ✅ Regime classification:
  - **Low VIX**: < 12
  - **Normal**: 12–18
  - **High**: 18–25
  - **Panic**: > 25
- ✅ VIX trend detection (Rising/Falling/Flat)
- ✅ Quantile-based regime analysis

### 3. **Correlation & Lead-Lag Analysis**
- ✅ NIFTY returns vs VIX change correlation
- ✅ Rolling 30-day and 60-day correlation
- ✅ Cross-correlation lead-lag analysis
- ✅ Granger causality test (Does VIX predict NIFTY?)

### 4. **IV-RV Spread Analysis**
- ✅ Realized Volatility (5D, 10D, 21D, 30D)
- ✅ IV-RV Spread calculation
- ✅ Z-score of spread for trading signals
- ✅ Automated trading signals:
  - **IV >> RV** → Sell options
  - **IV << RV** → Buy options

### 5. **Event-Based Analysis**
- ✅ Expiry week identification & analysis
- ✅ VIX decay pattern after expiry
- ✅ Budget day & RBI policy week analysis
- ✅ Pre-event vs Post-event volatility
- ✅ Extreme move detection

### 6. **Risk Analysis**
- ✅ Kurtosis & Skewness of returns
- ✅ Tail risk metrics (95th, 99th percentile)
- ✅ Extreme move frequency (>2%, >3%)
- ✅ VIX jump probability by regime

### 7. **Strategy Backtesting**
Three popular options strategies with full backtests:
- 📈 **Short Strangle**: High VIX + Flat trend
- 📉 **Long Straddle**: Low VIX + Rising
- 💰 **Iron Condor**: Low VIX + Sideways market

Each strategy includes:
- Total trades
- Win rate %
- Average return per trade
- Sharpe ratio
- Equity curve visualization

### 8. **Trading Insights (Automated)**
The dashboard automatically generates actionable insights like:
- 📈 "BULLISH + LOW VIX → Buy calls / Put spreads"
- ↔️ "SIDEWAYS + LOW VIX → Iron condors"
- ⚠️ "HIGH VIX → Option selling with hedges"
- 📊 "RISING VIX → Long straddles / strangles"

## 🚀 Installation & Setup

### Step 1: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 2: Fetch Data from NSE
```powershell
python nse_data_fetcher.py
```

This will:
- Attempt to fetch real data from NSE website (last 10 years)
- If NSE fetch fails, generate realistic sample data for testing
- Save data to `nifty_history.csv` and `india_vix_history.csv`

### Step 3: Run the Dashboard
```powershell
python dashboard.py
```

The dashboard will open at: **http://localhost:8050**

## 📁 File Structure

```
trading/
│
├── nse_data_fetcher.py      # Fetches NIFTY & VIX data from NSE
├── analysis.py               # Core analysis module
├── event_analysis.py         # Event-based analysis & backtesting
├── dashboard.py              # Main interactive dashboard
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── nifty_history.csv         # Generated: NIFTY 50 data
└── india_vix_history.csv     # Generated: INDIA VIX data
```

## 📊 Dashboard Tabs

### 1. **Price & Returns**
- NIFTY 50 price chart with drawdown
- Returns distribution (histogram)
- Log vs Simple returns comparison
- Rolling returns analysis

### 2. **VIX Analysis**
- India VIX with regime zones (color-coded)
- VIX regime distribution (pie chart)
- VIX histogram with KDE

### 3. **Correlation Analysis**
- Rolling correlation chart
- Lead-lag cross-correlation
- NIFTY returns vs VIX change scatter plot

### 4. **IV-RV Analysis**
- Implied vs Realized volatility comparison
- IV-RV spread over time
- Z-score of spread with buy/sell signals

### 5. **Event Analysis**
- Expiry week vs non-expiry comparison
- VIX decay pattern after expiry
- Event day statistics

### 6. **Strategy Backtests**
- Select from 3 strategies
- View performance metrics
- Equity curve visualization
- Entry signal markers on price chart

### 7. **Risk Analysis**
- Drawdown chart
- Tail risk distribution
- Statistical risk metrics

## 🎯 Trading Insights by Market Regime

| VIX Regime | VIX Trend | Best Strategy |
|-----------|-----------|---------------|
| Low & Falling | - | Buy debit spreads |
| Low & Rising | - | Long straddle |
| High & Flat | - | Short strangle |
| Extreme | - | Hedge only |

## 📈 Understanding IV-RV Signals

| IV-RV Z-Score | Signal | Action |
|--------------|--------|--------|
| > +1.5 | Sell Options | IV overpriced, sell premium |
| -1.5 to +1.5 | Neutral | No clear edge |
| < -1.5 | Buy Options | IV underpriced, buy straddles |

## 🔍 Key Insights from Analysis

### Stylized Facts:
- **Negative correlation** between NIFTY returns and VIX change
- **VIX leads** large market drops by 1–3 days
- **Volatility clustering**: High VIX tends to persist
- **Mean reversion** in IV-RV spread

### Risk Management:
- Max drawdown typically occurs during panic VIX regime (>25)
- Extreme moves (>2%) happen more frequently when VIX > 18
- Post-expiry, VIX tends to decay by ~5-10% over 5 days

## 🛠️ Customization

### Modify VIX Regimes
Edit in `analysis.py`:
```python
conditions = [
    df['Close_vix'] < 12,      # Low
    (df['Close_vix'] >= 12) & (df['Close_vix'] < 18),  # Normal
    (df['Close_vix'] >= 18) & (df['Close_vix'] < 25),  # High
    df['Close_vix'] >= 25       # Panic
]
```

### Add Custom Strategy
Edit in `event_analysis.py`, add to `backtest_simple_strategy()`:
```python
elif strategy_type == 'my_custom_strategy':
    df['Signal'] = # your logic here
    df['Strategy_Return'] = # your P&L calculation
```

## 📚 Dependencies

- **pandas**: Data manipulation
- **numpy**: Numerical calculations
- **requests**: Fetch data from NSE
- **dash**: Interactive web dashboard
- **plotly**: Interactive charts
- **scipy**: Statistical tests
- **statsmodels**: Time series analysis (Granger causality)

## ⚠️ Important Notes

### Data Fetching
- NSE API may have rate limits or change over time
- If real data fetch fails, sample data is automatically generated
- Sample data is realistic but synthetic for testing purposes

### Strategy Backtests
- Simplified P&L models (not actual options pricing)
- Does not account for:
  - Slippage
  - Transaction costs
  - Bid-ask spreads
  - Options Greeks
- Use for directional insights, not exact P&L estimates

### Risk Disclaimer
⚠️ **This tool is for educational and research purposes only.**
- Not financial advice
- Past performance does not guarantee future results
- Always do your own due diligence
- Consider transaction costs and market impact

## 🎓 Learning Resources

### Key Concepts Covered:
1. **Returns Analysis**: Simple vs Log returns, why they differ
2. **Volatility Metrics**: Historical vs Implied volatility
3. **Statistical Tests**: Granger causality, correlation analysis
4. **Options Strategies**: When and why to use each strategy
5. **Risk Management**: Drawdowns, tail risk, position sizing

## 🔧 Troubleshooting

### Data fetch fails:
```powershell
# The script will automatically use sample data
# Or manually download CSVs from NSE and place in trading/ folder
```

### Dashboard won't start:
```powershell
# Check if port 8050 is already in use
# Try a different port:
python -c "from dashboard import *; dashboard = TradingDashboard(nifty_df, vix_df); dashboard.run(port=8051)"
```

### Import errors:
```powershell
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

## 📞 Support

For issues or questions:
1. Check the console output for error messages
2. Verify all CSV files are generated
3. Ensure all dependencies are installed

## 🎉 Quick Start Command Summary

```powershell
# 1. Install
pip install -r requirements.txt

# 2. Fetch data
python nse_data_fetcher.py

# 3. Run dashboard
python dashboard.py

# 4. Open browser to http://localhost:8050
```

## 📊 Example Output

After running, you'll see:
- 📈 Interactive charts with zoom/pan
- 📊 Real-time metric cards
- 🎯 Automated trading insights
- 💰 Strategy performance metrics
- ⚠️ Risk analysis dashboards

---

**Happy Trading! 📈💰**

*Remember: The best strategy is the one you understand completely.*
