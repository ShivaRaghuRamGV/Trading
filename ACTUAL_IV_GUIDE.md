# Actual IV Integration Guide

## Overview
The Greek Regime Flip Model now supports **actual Implied Volatility (IV)** data from daily option chain CSV files, replacing the previous Newton-Raphson IV calculation.

---

## File Naming Convention

Daily option chain CSV files should follow this format:
```
option-chain-ED-NIFTY-{Expiry Date}_{Trading Date}.csv
```

### Examples:
- `option-chain-ED-NIFTY-30-Dec-2025_19 Dec 2025.csv`
- `option-chain-ED-NIFTY-27-Dec-2025_20 Dec 2025.csv`
- `option-chain-ED-NIFTY-26-Dec-2025_21 Dec 2025.csv`

### Date Formats:
- **Expiry Date**: `DD-MMM-YYYY` (e.g., `30-Dec-2025`)
- **Trading Date**: `DD MMM YYYY` (e.g., `19 Dec 2025`)

---

## File Structure

The CSV file must have this structure:

### Header Row 1: `CALLS,,PUTS`
### Header Row 2: Column names

### Columns (skip row 1, use row 2):
```
OI, CHNG IN OI, VOLUME, IV, LTP, CHNG, BID QTY, BID, ASK, ASK QTY, STRIKE,
BID QTY, BID, ASK, ASK QTY, CHNG, LTP, IV, VOLUME, CHNG IN OI, OI
```

### Key Columns Used:
- **STRIKE**: Strike price (with commas, e.g., "25,000.00")
- **IV** (Calls): Implied Volatility for Call options (percentage, e.g., "15.64")
- **IV.1** (Puts): Implied Volatility for Put options (percentage)
- **LTP** / **LTP.1**: Last Traded Price (Premium)
- **VOLUME** / **VOLUME.1**: Trading volume
- **OI** / **OI.1**: Open Interest
- **CHNG IN OI** / **CHNG IN OI.1**: Change in Open Interest

---

## How It Works

### 1. **Dashboard Input**
In the Greek Regime Flip dashboard, enter the **Trading Date** in the new input field:
```
Trading Date: 19 Dec 2025
```

### 2. **Automatic File Loading**
When you select an expiry (e.g., `2025-12-30`) and click **LOAD OPTION DATA**, the system:

1. Converts expiry `2025-12-30` → `30-Dec-2025`
2. Looks for: `option-chain-ED-NIFTY-30-Dec-2025_19 Dec 2025.csv`
3. Loads actual IV values for all strikes (Calls & Puts)
4. Calculates **IV Change %** (if previous day file exists)

### 3. **Fallback Mechanism**
If the CSV file is **not found**:
- System uses historical parquet data
- Calculates IV from premiums using **Newton-Raphson method**
- Sets `IV_Change_%` to `0.0`

---

## IV Change Calculation

The system can calculate **per-strike IV changes** by comparing:
- **Current trading date** CSV (e.g., `19 Dec 2025`)
- **Previous trading date** CSV (e.g., `18 Dec 2025`)

### Formula:
```
IV_Change_% = ((IV_current - IV_previous) / IV_previous) × 100
```

### Example:
- Strike 25000 CE: IV = 11.51% (today), IV = 12.20% (yesterday)
- IV_Change_% = ((11.51 - 12.20) / 12.20) × 100 = **-5.66%**

This shows option IV decreased by 5.66%, indicating volatility compression.

---

## Dashboard Display

### New Column in Full Chain Table:
| Strike | Type | Premium | IV (%) | **IV_Change_%** | Delta | Gamma | Vega | Theta |
|--------|------|---------|--------|-----------------|-------|-------|------|-------|
| 25000  | CE   | 1045.40 | 11.51  | **-5.66**      | 0.52  | 0.003 | 125  | -45   |
| 25000  | PE   | 920.35  | 12.08  | **-4.21**      | -0.48 | 0.003 | 122  | -42   |

### Key Metrics Updated:
- **Avg IV**: Now uses actual IV from CSV (not calculated)
- **IV Range**: Shows min/max IV from actual market data
- **Greeks**: Calculated using actual IV (more accurate)

---

## Benefits of Actual IV

### 1. **Accuracy**
- No approximation errors from Newton-Raphson
- Reflects real market-implied volatility
- Captures volatility smile/skew precisely

### 2. **IV Change Tracking**
- See which strikes are experiencing IV expansion/compression
- Identify options with volatility mispricing
- Track IV term structure evolution

### 3. **Better Greeks**
- Delta, Gamma, Vega, Theta calculated with real IV
- More reliable regime classification
- Improved position Greeks aggregation

### 4. **Historical Analysis**
- Compare IV levels across different expiries
- Analyze IV behavior around events
- Build IV percentile rankings

---

## Usage Workflow

### Daily Routine:
1. **Download option chain** from your broker/data provider
2. **Save as CSV** with naming convention: `option-chain-ED-NIFTY-{Expiry}_{Date}.csv`
3. **Place in folder**: `c:\Users\USER\trading\nifty_option_excel\`
4. **Open dashboard**: http://127.0.0.1:8055
5. **Enter trading date**: e.g., "19 Dec 2025"
6. **Select expiry** from dropdown
7. **Click LOAD OPTION DATA**
8. **Analyze** using actual IV and IV changes

---

## Technical Details

### Code Functions:

#### `load_daily_option_chain_csv(expiry_date, trading_date)`
- Loads CSV file with actual IV
- Parses Calls and Puts separately
- Returns: DataFrame with columns `[Strike, Type, IV, LTP, VOLUME, OI, CHNG_IN_OI]`

#### `calculate_iv_change_per_strike(df_current, expiry_date, current_date, previous_date)`
- Loads previous day's CSV
- Merges on Strike & Type
- Calculates IV_Change_% per strike
- Returns: DataFrame with added `IV_Change_%` column

#### `enrich_greeks(df)` - Modified
- Checks if IV already present in DataFrame (from CSV)
- If yes: Uses actual IV
- If no: Calculates IV from premium (fallback)
- Computes Greeks using appropriate IV

---

## Example Output

```
✓ Loaded 184 options from option-chain-ED-NIFTY-30-Dec-2025_19 Dec 2025.csv 
  (IV range: 6.4%-120.6%)
✓ Using actual IV from daily CSV for 19 Dec 2025
✓ IV changes calculated (range: -15.2% to +8.7%)

Avg IV: 18.52% (+64.5%)  ← 64.5% higher than previous VIX
```

---

## Troubleshooting

### ⚠️ "Daily chain file not found"
**Cause**: CSV file doesn't exist for the selected expiry/date combination.
**Solution**: 
- Check filename format exactly matches convention
- Verify file is in `nifty_option_excel/` folder
- Ensure dates are spelled correctly (Dec, not DEC)

### ⚠️ "Previous day data not available, IV change set to 0"
**Cause**: No CSV file for previous trading day.
**Solution**:
- IV values will still be accurate (from current day CSV)
- IV_Change_% will show as 0.0
- Download and save previous day's CSV for future reference

### ⚠️ "Error loading daily IV data"
**Cause**: CSV file format mismatch.
**Solution**:
- Ensure CSV has proper header structure
- Check columns match expected format
- Verify no extra commas or missing columns

---

## Integration with Regime Model

### How Actual IV Affects Regime Classification:

1. **Delta-Driven Regime**
   - More accurate when IV correctly reflects market expectations
   - Low IV → Higher Delta sensitivity → More likely Delta-driven

2. **Gamma-Driven Regime**
   - Gamma highest at-the-money
   - Actual IV helps identify precise ATM strikes
   - IV smile affects Gamma distribution

3. **Vega-Driven Regime**
   - **Direct impact**: Vega = ∂Price/∂IV
   - High actual IV → High Vega → More likely Vega-driven
   - IV changes show volatility risk magnitude

4. **Theta-Driven Regime**
   - Theta calculation uses IV
   - Higher IV → Higher premiums → Higher absolute Theta
   - Actual IV improves Theta estimates

---

## Future Enhancements

### Planned Features:
1. **Automatic date detection** - Parse multiple files, select latest
2. **IV percentile calculations** - Compare current IV to historical distribution
3. **IV term structure charts** - Visualize IV across expiries
4. **IV skew analysis** - Plot IV vs Strike (volatility smile)
5. **IV rank/percentile** - Show where current IV sits in 30/60/90-day range

---

## Summary

✅ **Use actual IV** from broker/exchange option chain data  
✅ **Track IV changes** per strike, per day  
✅ **More accurate Greeks** for better position analysis  
✅ **Flexible**: Falls back to calculated IV if CSV unavailable  
✅ **Easy integration**: Just save CSV files with proper naming  

**Result**: More reliable regime classification and Greeks-based trading signals.
