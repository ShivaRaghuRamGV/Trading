# Decision Dashboard Guide

## Overview
The **Decision Dashboard** is an integrated options trading strategy selector that combines signals from three specialized dashboards to provide actionable trading recommendations.

**Dashboard URL**: http://127.0.0.1:8060

---

## Integration Architecture

### Data Sources

#### 1. **Dashboard.py** (Port 8050)
**Inputs Used:**
- **Volatility Regime**: Low/Medium/High based on India VIX levels
- **NIFTY Return Distribution**: Monthly return statistics (mean, std, skew, kurtosis)

#### 2. **VIX Forecasting Dashboard** (Port 8051)
**Inputs Used:**
- **VIX Forecast**: 22-day ahead VIX prediction (% change)
- **NIFTY Forecast**: 22-day ahead NIFTY prediction (% change)

#### 3. **Greek Regime Flip Model** (Port 8055)
**Inputs Used:**
- **Greek Regime**: Delta-driven, Gamma-driven, Vega-driven, or Theta-driven
- **Dominant Greek**: Which Greek is currently dominating option pricing

---

## Strategy Matrix

The Decision Dashboard evaluates **10 option strategies** based on current market conditions:

### 1. **LONG CALL**
- **Best When**: 
  - VIX rising, NIFTY bullish forecast
  - Delta-driven or Gamma-driven regime
  - Low/Medium volatility environment
  - Positive return skew
- **Avoid When**: 
  - VIX falling, Vega-driven regime, High volatility
- **Strike Selection**: ATM to slightly ITM (spot ± 100)
- **Risk**: Limited to premium
- **Max Profit**: Unlimited

### 2. **LONG PUT**
- **Best When**: 
  - VIX rising, NIFTY bearish forecast
  - Delta-driven or Gamma-driven regime
  - Low/Medium volatility
  - Negative return skew
- **Avoid When**: 
  - VIX falling, Vega-driven regime, High volatility
- **Strike Selection**: ATM to slightly ITM (spot ± 100)
- **Risk**: Limited to premium
- **Max Profit**: Substantial

### 3. **BULL CALL SPREAD**
- **Best When**: 
  - Moderately bullish NIFTY forecast
  - Delta-driven or Theta-driven regime
  - Medium/High volatility
  - Stable VIX
- **Avoid When**: 
  - Bearish forecast, Gamma-driven regime, Low volatility
- **Strike Selection**: Buy ATM, Sell OTM (spot + 200)
- **Risk**: Limited
- **Max Profit**: Limited (spread width - premium paid)

### 4. **BEAR PUT SPREAD**
- **Best When**: 
  - Moderately bearish NIFTY forecast
  - Delta-driven or Theta-driven regime
  - Medium/High volatility
- **Avoid When**: 
  - Bullish forecast, Gamma-driven regime
- **Strike Selection**: Buy ATM, Sell OTM (spot - 200)
- **Risk**: Limited
- **Max Profit**: Limited

### 5. **LONG STRADDLE**
- **Best When**: 
  - VIX rising, uncertain direction
  - Gamma-driven or Vega-driven regime
  - Low volatility environment
- **Avoid When**: 
  - VIX falling, Theta-driven regime, High volatility
- **Strike Selection**: ATM (spot ± 50)
- **Risk**: Limited to total premium
- **Max Profit**: Unlimited

### 6. **SHORT STRADDLE**
- **Best When**: 
  - VIX falling, range-bound market
  - Theta-driven regime
  - High volatility (sell high premium)
- **Avoid When**: 
  - VIX rising, Gamma/Vega-driven regime, Low volatility
- **Strike Selection**: ATM (spot ± 50)
- **Risk**: **UNLIMITED**
- **Max Profit**: Limited to premium collected

### 7. **IRON CONDOR**
- **Best When**: 
  - Range-bound NIFTY forecast
  - Theta-driven regime
  - Medium/High volatility
  - Stable to falling VIX
- **Avoid When**: 
  - Trending market, Gamma/Vega-driven regime, Low volatility
- **Strike Selection**: 
  - Sell OTM Put (spot - 300)
  - Buy OTM Put (spot - 500)
  - Sell OTM Call (spot + 300)
  - Buy OTM Call (spot + 500)
- **Risk**: Limited (spread width - credit)
- **Max Profit**: Limited to net credit

### 8. **LONG STRANGLE**
- **Best When**: 
  - VIX rising, uncertain direction
  - Vega-driven or Gamma-driven regime
  - Low volatility
- **Avoid When**: 
  - VIX falling, Theta-driven regime, High volatility
- **Strike Selection**: OTM Put (spot - 200), OTM Call (spot + 200)
- **Risk**: Limited to premium
- **Max Profit**: Unlimited

### 9. **CALENDAR SPREAD**
- **Best When**: 
  - Vega-driven or Theta-driven regime
  - Medium volatility
  - Stable VIX, range-bound market
- **Avoid When**: 
  - Gamma-driven regime, strongly trending market
- **Strike Selection**: Sell near-term ATM, Buy far-term ATM
- **Risk**: Limited
- **Max Profit**: Limited

### 10. **BUTTERFLY SPREAD**
- **Best When**: 
  - Range-bound NIFTY forecast
  - Theta-driven or Delta-driven regime
  - Medium volatility
  - Stable VIX
- **Avoid When**: 
  - Trending market, Gamma-driven regime
- **Strike Selection**: 
  - Buy ITM (spot - 200)
  - Sell 2x ATM
  - Buy OTM (spot + 200)
- **Risk**: Limited
- **Max Profit**: Limited

---

## How to Use

### Step 1: Enter Current Market Data

1. **NIFTY Spot**: Current NIFTY index level (e.g., 24900)
2. **VIX Forecast %**: Enter the forecasted VIX change from the VIX Forecasting Dashboard
   - Example: +15% (VIX expected to rise 15%)
3. **NIFTY Forecast %**: Enter the forecasted NIFTY change
   - Example: +5.6% (NIFTY expected to rise 5.6%)
4. **Greek Regime**: Select from dropdown (from Greek Regime Flip Model dashboard)
   - Delta-driven, Gamma-driven, Vega-driven, or Theta-driven
5. **Volatility Regime**: Select current VIX level classification
   - Low (VIX < 12), Medium (VIX 12-18), High (VIX > 18)

### Step 2: Click ANALYZE

The dashboard will:
- Process all signals
- Score each strategy (0-100%)
- Categorize strategies into:
  - ✅ **Recommended** (Score ≥ 70%)
  - ⚠️ **Caution** (Score 40-69%)
  - ❌ **Avoid** (Score < 40% or has avoid conditions)

### Step 3: Review Recommendations

#### Signal Summary
Shows interpreted market conditions:
- VIX direction: rising/falling/stable
- NIFTY direction: bullish/moderately_bullish/range_bound/moderately_bearish/bearish
- Greek regime classification
- Volatility regime classification
- Return distribution skew

#### Strategy Cards
Quick overview:
- Number of recommended strategies
- Number of caution strategies
- Number to avoid

#### Detailed Tabs

**✅ Recommended Strategies Tab**:
- Shows high-confidence strategies (≥70% match)
- **Specific strike prices** calculated based on current spot
- Risk/reward profile
- Detailed analysis of why recommended

**⚠️ Caution Strategies Tab**:
- Medium-confidence strategies (40-69% match)
- Some conditions met, some not
- Can be traded with extra risk management

**❌ Avoid Strategies Tab**:
- Low-confidence or explicitly flagged strategies
- Explains why conditions don't match
- Should be avoided under current market conditions

**📈 Signal Analysis Tab**:
- Market condition matrix
- Interpretation of each signal
- How signals combine

---

## Example Workflow

### Scenario 1: Bullish Rally with Low VIX

**Inputs**:
- NIFTY Spot: 24900
- VIX Forecast: -10% (falling)
- NIFTY Forecast: +5.5% (bullish)
- Greek Regime: Delta-driven
- Volatility Regime: Low

**Expected Recommendations**:
1. ✅ **LONG CALL** - High score
   - Buy Call: 24900 (ATM)
   - Benefits from directional move without IV crush
2. ✅ **BULL CALL SPREAD** - Good score
   - Buy Call: 24900, Sell Call: 25100
   - Limited risk with falling IV
3. ⚠️ **LONG STRADDLE** - Caution
   - VIX falling hurts Vega
4. ❌ **SHORT STRADDLE** - Avoid
   - Unlimited risk in trending market

### Scenario 2: High Volatility, Uncertain Direction

**Inputs**:
- NIFTY Spot: 24900
- VIX Forecast: +15% (rising)
- NIFTY Forecast: +1.0% (range-bound)
- Greek Regime: Vega-driven
- Volatility Regime: High

**Expected Recommendations**:
1. ✅ **LONG STRADDLE** - High score
   - Call: 24900, Put: 24900
   - Profit from volatility expansion
2. ✅ **LONG STRANGLE** - Good score
   - Call: 25100, Put: 24700
   - Cheaper than straddle, still benefits from Vol
3. ❌ **SHORT STRADDLE** - Avoid
   - Selling volatility when it's rising = bad idea
4. ❌ **IRON CONDOR** - Avoid
   - VIX rising + Vega-driven = avoid selling premium

### Scenario 3: Range-Bound, Time Decay

**Inputs**:
- NIFTY Spot: 24900
- VIX Forecast: -5% (stable to falling)
- NIFTY Forecast: +0.5% (range-bound)
- Greek Regime: Theta-driven
- Volatility Regime: Medium

**Expected Recommendations**:
1. ✅ **IRON CONDOR** - High score
   - Sell Put: 24600, Buy Put: 24400
   - Sell Call: 25200, Buy Call: 25400
   - Collect premium in range-bound market
2. ✅ **BUTTERFLY SPREAD** - Good score
   - Buy 24700, Sell 2x 24900, Buy 25100
   - Profits from lack of movement
3. ✅ **CALENDAR SPREAD** - Good score
   - Sell near-term ATM, Buy far-term ATM
4. ❌ **LONG CALL/PUT** - Avoid
   - No directional edge, theta decay hurts

---

## Strike Selection Logic

All strikes are **rounded to nearest 50** for NIFTY options.

### ATM Strike
```
ATM = round(spot / 50) * 50
```
Example: Spot = 24923 → ATM = 24900

### ITM Strikes
- Calls: ATM - 100, ATM - 200
- Puts: ATM + 100, ATM + 200

### OTM Strikes
- Calls: ATM + 100, ATM + 200, ATM + 300, etc.
- Puts: ATM - 100, ATM - 200, ATM - 300, etc.

### Spread Widths
- **Narrow spreads** (100-200): Lower risk, lower profit
- **Medium spreads** (200-300): Balanced risk/reward
- **Wide spreads** (400-500): Higher risk, higher profit potential

---

## Scoring Methodology

Each strategy is scored based on:

1. **Positive Conditions** (+1 point each):
   - VIX forecast matches expected
   - NIFTY forecast matches expected
   - Greek regime in favorable list
   - Volatility regime in favorable list
   - Return skew matches expected

2. **Avoid Conditions** (Auto-disqualify):
   - VIX forecast opposite of needed
   - Greek regime in avoid list
   - Volatility regime in avoid list

3. **Final Score**:
```
Score = (Conditions Met / Total Conditions) × 100
```

4. **Classification**:
   - **Recommended**: Score ≥ 70% AND no avoid conditions
   - **Caution**: Score 40-69% AND no avoid conditions
   - **Avoid**: Score < 40% OR has avoid conditions

---

## Integration with Other Dashboards

### Getting VIX Forecast (from port 8051)
1. Open http://127.0.0.1:8051
2. Look at "VIX Forecast (22 Days)" section
3. Note the forecasted change % (e.g., +12.5%)
4. Enter this value in Decision Dashboard

### Getting NIFTY Forecast (from port 8051)
1. Same dashboard as VIX
2. Look at "Strategy Recommendation" section
3. Note expected NIFTY movement (e.g., "LONG_CALL: +5.62% UP")
4. Enter this % in Decision Dashboard

### Getting Greek Regime (from port 8055)
1. Open http://127.0.0.1:8055
2. Enter trading date and select expiry
3. Click "LOAD OPTION DATA"
4. Look at regime box (e.g., "Vega-driven Regime")
5. Select this in Decision Dashboard dropdown

### Getting Volatility Regime (from port 8050)
1. Open http://127.0.0.1:8050
2. Look at India VIX chart
3. Check current VIX value
4. Classify: < 12 = Low, 12-18 = Medium, > 18 = High
5. Select in Decision Dashboard dropdown

---

## Risk Management Guidelines

### For Recommended Strategies
- ✅ Execute with **70-80% of intended position size**
- ✅ Set stop-loss at **50% of premium paid** (long options)
- ✅ Set profit target at **100-200% gain**
- ✅ Monitor daily for signal changes

### For Caution Strategies
- ⚠️ Execute with **40-50% of intended position size**
- ⚠️ Tighter stop-loss: **30-40% of premium**
- ⚠️ Scale in gradually if conditions improve
- ⚠️ Monitor twice daily

### For Avoid Strategies
- ❌ **Do not trade** under current conditions
- ❌ Wait for signal improvement
- ❌ If already in position, consider exiting

---

## Advanced Features

### Return Skew Analysis
- **Positive Skew**: More frequent small losses, occasional big gains
  - Favors: Buying options (calls/puts)
- **Negative Skew**: More frequent small gains, occasional big losses
  - Favors: Selling premium (spreads, condors)

### Multi-Signal Confirmation
The dashboard requires **multiple signals to align**:
- VIX + NIFTY forecasts must complement
- Greek regime must support strategy mechanics
- Volatility regime must enable favorable entry pricing

### Dynamic Strike Calculation
Strikes auto-adjust to current spot price, ensuring:
- ATM options are truly at-the-money
- OTM strikes maintain proper distance
- Spreads maintain consistent risk/reward ratios

---

## Troubleshooting

### "No Recommended Strategies"
**Cause**: Conflicting signals (e.g., VIX rising + bearish NIFTY forecast)
**Solution**: 
- Review signal consistency
- Consider caution strategies
- Wait for clearer market direction

### "All Strategies in Avoid"
**Cause**: Extreme or contradictory market conditions
**Solution**:
- Stay in cash
- Wait for volatility to normalize
- Re-analyze when signals stabilize

### "Strike Selection Seems Wrong"
**Cause**: Spot price might be stale
**Solution**:
- Verify current NIFTY spot price
- Update spot input
- Re-click ANALYZE

---

## Dashboard Ports Summary

| Dashboard | Port | Purpose |
|-----------|------|---------|
| Main Dashboard | 8050 | VIX regime, NIFTY returns |
| VIX Forecasting | 8051 | VIX & NIFTY forecasts |
| Greek Regime Flip | 8055 | Greek regime, IV analysis |
| **Decision Dashboard** | **8060** | **Integrated strategy selector** |

---

## Example Strategy Cards

### High Confidence (✅ Recommended)

```
LONG CALL
Match Score: 85%

Strikes: Buy Call: 24900

Risk: Limited to premium
Max Profit: Unlimited

✓ vix_forecast: stable (expected: rising)
✓ nifty_forecast: bullish
✓ greek_regime: Delta-driven
✓ volatility_regime: Medium
⚠ return_skew: negative (expected: positive)
```

### Medium Confidence (⚠️ Caution)

```
IRON CONDOR
Match Score: 55%

Strikes: Sell Put: 24600, Buy Put: 24400, Sell Call: 25200, Buy Call: 25400

Risk: Limited
Max Profit: Limited to credit

✓ nifty_forecast: range_bound
✓ greek_regime: Theta-driven
⚠ volatility_regime: Low (expected: Medium/High)
⚠ vix_forecast: rising (expected: stable_to_falling)
```

### Avoid (❌)

```
SHORT STRADDLE
Match Score: 20%

⛔ Not recommended under current conditions

❌ vix_forecast: rising (avoid)
❌ greek_regime: Vega-driven (avoid)
✓ volatility_regime: High
⚠ nifty_forecast: bullish (expected: range_bound)
```

---

## Summary

The Decision Dashboard provides:
✅ **Automated strategy selection** based on multi-dashboard signals  
✅ **Specific strike recommendations** for each strategy  
✅ **Risk-adjusted categorization** (Recommended/Caution/Avoid)  
✅ **Transparent scoring** with detailed analysis  
✅ **Real-time integration** with existing analytics dashboards  

**Start analyzing**: http://127.0.0.1:8060
