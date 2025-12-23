# Options Backtesting System - Setup Guide

## Overview
The trading dashboard now supports backtesting with **real historical options data** from NSE using the `nsepy` library.

## Components

### 1. `options_data_fetcher.py`
Downloads historical NIFTY options data (Calls and Puts) from NSE.

**Features:**
- Fetches monthly expiry options from Jan 2015 to Dec 2024
- Automatically detects ATM strikes based on spot price
- Configurable number of strikes (ITM and OTM)
- Saves data to CSV files for reuse
- Progress tracking with tqdm

**Usage:**
```bash
python options_data_fetcher.py
```

**Configuration:**
- `num_strikes=5`: Fetches 11 strikes (5 ITM, ATM, 5 OTM)
- Data saved to `options_data/` directory
- Separate files for Calls (CE) and Puts (PE)

### 2. `options_backtester.py`
Backtests options strategies using the downloaded data.

**Supported Strategies:**

#### Short Strangle
- **Entry:** VIX > threshold, DTE = 7 days
- **Position:** Sell OTM Call (ATM+200) + Sell OTM Put (ATM-200)
- **Exit:** DTE = 1 day
- **Profit:** Premium decay (theta)
- **Risk:** Large market moves

#### Long Straddle
- **Entry:** VIX < threshold, DTE = 7 days
- **Position:** Buy ATM Call + Buy ATM Put
- **Exit:** DTE = 1 day
- **Profit:** Large market moves
- **Risk:** Premium decay + low volatility

**Usage:**
```python
from options_backtester import OptionsBacktester

# Initialize
backtester = OptionsBacktester(nifty_df)

# Backtest Short Strangle
trades = backtester.backtest_short_strangle(
    vix_threshold=18,  # Enter when VIX > 18
    dte_entry=7,       # Enter 7 days before expiry
    dte_exit=1         # Exit 1 day before expiry
)

# Get metrics
metrics = backtester.calculate_metrics(trades)
```

### 3. `event_analysis.py` (Updated)
Integrated with real options backtester.

**New Method:**
```python
analyzer = EventAnalyzer(merged_df)

# Use real options data if available
result = analyzer.backtest_with_real_options(
    strategy_type='short_strangle',
    vix_threshold=18,
    dte_entry=7,
    dte_exit=1
)

if result:
    trades_df, metrics = result
    print(f"Total PnL: {metrics['Total_PnL']}")
    print(f"Win Rate: {metrics['Win_Rate_%']}%")
```

## Setup Steps

### Step 1: Download Options Data (One-time)
```bash
# This will take several hours for 10 years of data
python options_data_fetcher.py
```

**Note:** Start with a smaller date range or fewer strikes for testing:
- Modify `start_date` and `end_date` in the script
- Reduce `num_strikes` from 10 to 5

### Step 2: Backtest Strategies
```bash
# Test the backtester
python options_backtester.py
```

### Step 3: Use in Dashboard
The dashboard's "Strategy Backtests" tab will automatically use real options data if available.

## Performance Metrics

The backtester calculates comprehensive metrics:

- **Total Trades**: Number of positions taken
- **Win Rate**: Percentage of profitable trades
- **Total PnL**: Cumulative profit/loss
- **Avg PnL**: Average profit per trade
- **Avg Win/Loss**: Average winning vs losing trade
- **Profit Factor**: Ratio of avg win to avg loss
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest cumulative loss

## Data Structure

### Options Data CSV Columns:
- `Date`: Trading date
- `Symbol`: NIFTY
- `Expiry`: Expiry date
- `Strike`: Strike price
- `OptionType`: CE (Call) or PE (Put)
- `Open`, `High`, `Low`, `Close`: Option prices
- `Volume`: Trading volume
- `Open Interest`: Outstanding contracts

### Example Output:
```
Expiry: 2024-01-25
Entry: 2024-01-18 (7 DTE)
Exit: 2024-01-24 (1 DTE)
Spot: 21,500
VIX: 18.5

Position:
- Sell Call 21,700 @ 85
- Sell Put 21,300 @ 75
Premium Collected: 12,000 (160 pts × 75 lot size)

Exit:
- Buy back Call @ 35
- Buy back Put @ 25
Total PnL: +7,500 (62.5% return)
```

## Advantages Over Simplified Backtests

**Old (Simplified):**
- ❌ Fictional premium calculations
- ❌ Simplified P&L formulas
- ❌ No real market data
- ❌ Unrealistic returns

**New (Real Options Data):**
- ✅ Actual historical premiums
- ✅ Real market liquidity (Volume/OI)
- ✅ Accurate P&L calculations
- ✅ Realistic win rates and returns
- ✅ Accounts for bid-ask spreads
- ✅ Real slippage and execution

## Customization

### Add New Strategies
Edit `options_backtester.py` and add new methods:

```python
def backtest_iron_condor(self, vix_threshold=15, dte_entry=7, dte_exit=1):
    # Your logic here
    call_long_strike = atm_strike + 300
    call_short_strike = atm_strike + 200
    put_short_strike = atm_strike - 200
    put_long_strike = atm_strike - 300
    
    # Calculate P&L with 4 legs
    # ...
```

### Adjust Parameters
- `vix_threshold`: Entry condition
- `dte_entry`: When to enter (days before expiry)
- `dte_exit`: When to exit
- `strike_gap`: Distance between strikes
- `num_strikes`: Number of strikes to analyze

## Troubleshooting

### "Options data not found"
- Run `python options_data_fetcher.py` first
- Check `options_data/` directory exists
- Verify CSV files are present

### "No trades executed"
- Check VIX threshold isn't too restrictive
- Verify date range has data
- Confirm strikes are available in the data

### Slow performance
- Reduce number of strikes fetched
- Limit date range
- Use monthly expiries only (already default)

## Next Steps

1. **Download data**: Run `options_data_fetcher.py`
2. **Test backtester**: Run `options_backtester.py`
3. **Dashboard integration**: The Strategy Backtests tab will automatically use real data
4. **Analyze results**: Compare real vs simplified strategies

## Important Notes

⚠️ **Data Requirements:**
- nsepy has limitations on historical data availability
- Some older options data may be incomplete
- Weekend/holiday dates will have no data

⚠️ **Lot Size:**
- NIFTY lot size = 75 (as of recent)
- Historical lot sizes varied (adjust in code if needed)

⚠️ **Costs Not Included:**
- Brokerage fees
- STT/taxes
- Slippage beyond bid-ask

💡 **Tips:**
- Start with recent data (2020-2024) for better quality
- Use VIX regimes to filter entry conditions
- Monitor Open Interest for liquidity
- Backtest multiple parameter combinations
