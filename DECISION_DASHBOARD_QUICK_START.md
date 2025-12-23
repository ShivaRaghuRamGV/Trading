# Decision Dashboard - Quick Start Card

## 🚀 Quick Setup

### 1. Start All Dashboards
```powershell
# Terminal 1: Main Dashboard
python dashboard.py

# Terminal 2: VIX Forecasting
python forecast_dashboard.py

# Terminal 3: Greek Regime
python greek_regime_flip_live.py

# Terminal 4: Decision Dashboard
python decision_dashboard.py
```

### 2. Access URLs
- Main Dashboard: http://127.0.0.1:8050
- VIX Forecasting: http://127.0.0.1:8051  
- Greek Regime: http://127.0.0.1:8055
- **Decision Dashboard: http://127.0.0.1:8060** ⭐

---

## 📋 Data Collection Checklist

### From VIX Forecasting Dashboard (8051)
- [ ] VIX Forecast: ______% (e.g., +15%, -10%)
- [ ] NIFTY Forecast: ______% (e.g., +5.6%, -3.2%)

### From Greek Regime Dashboard (8055)
- [ ] Greek Regime: ☐ Delta ☐ Gamma ☐ Vega ☐ Theta

### From Main Dashboard (8050) / Manual
- [ ] Current VIX: ______ → Regime: ☐ Low (<12) ☐ Medium (12-18) ☐ High (>18)
- [ ] NIFTY Spot: ₹______

---

## 🎯 Signal Interpretation

### VIX Forecast Classification
- **Rising**: > +10%
- **Stable**: -10% to +10%
- **Falling**: < -10%

### NIFTY Forecast Classification
- **Bullish**: > +3%
- **Moderately Bullish**: +1% to +3%
- **Range-Bound**: -1% to +1%
- **Moderately Bearish**: -3% to -1%
- **Bearish**: < -3%

---

## 📊 Strategy Selection Cheat Sheet

### BULLISH SCENARIOS

#### Strong Bull (NIFTY +5%, VIX falling, Delta regime)
✅ **LONG CALL**
- Strike: ATM (24900)
- Best when: Low/Medium vol, VIX falling

✅ **BULL CALL SPREAD**  
- Strikes: Buy 24900, Sell 25100
- Best when: Moderate confidence

#### Moderate Bull (NIFTY +2%, VIX stable)
✅ **BULL CALL SPREAD**
⚠️ **LONG CALL** (if vol low)

---

### BEARISH SCENARIOS

#### Strong Bear (NIFTY -5%, VIX rising, Delta regime)
✅ **LONG PUT**
- Strike: ATM (24900)

✅ **BEAR PUT SPREAD**
- Strikes: Buy 24900, Sell 24700

---

### VOLATILE SCENARIOS

#### High Vol Expected (VIX +15%, Vega regime)
✅ **LONG STRADDLE**
- Strikes: Both 24900

✅ **LONG STRANGLE**
- Strikes: Put 24700, Call 25100

#### Already High Vol (VIX >18, Vega regime)
❌ Avoid buying premium
⚠️ Wait for vol to drop

---

### RANGE-BOUND SCENARIOS

#### Theta Decay (VIX falling, Theta regime, neutral NIFTY)
✅ **IRON CONDOR**
- Strikes: 24600/24400 puts, 25200/25400 calls

✅ **BUTTERFLY SPREAD**
- Strikes: 24700/24900/24900/25100

✅ **SHORT STRADDLE** (⚠️ **Unlimited risk!**)
- Strikes: Both 24900

---

## ⚠️ Red Flags - DO NOT TRADE

### Conflicting Signals
❌ VIX rising + NIFTY strongly bullish  
→ IV crush will hurt call buyers

❌ Vega regime + VIX falling  
→ Options losing value despite favorable Greeks

❌ Gamma regime + Range-bound forecast  
→ Low movement = gamma doesn't help

### Dangerous Combinations
❌ **SHORT STRADDLE** when:
- VIX forecasted to rise
- Gamma or Vega regime
- Trending market

❌ **LONG OPTIONS** when:
- Theta regime dominant
- High volatility environment
- VIX falling (for Vega-driven)

❌ **SPREADS** when:
- Gamma regime (need naked options)
- High transaction costs

---

## 💡 Pro Tips

### Position Sizing by Confidence

| Recommendation | Position Size | Stop Loss |
|----------------|---------------|-----------|
| ✅ Recommended (70%+) | 70-80% of full | 50% of premium |
| ⚠️ Caution (40-70%) | 40-50% of full | 30% of premium |
| ❌ Avoid (<40%) | **0%** - Don't trade | N/A |

### Strike Selection Rules
1. **ATM**: Round spot to nearest 50
   - 24923 → 24900
2. **ITM Calls**: ATM - 100 or ATM - 200
3. **OTM Calls**: ATM + 100, +200, +300
4. **ITM Puts**: ATM + 100 or ATM + 200
5. **OTM Puts**: ATM - 100, -200, -300

### Timing
- **Enter**: When score ≥ 70%
- **Exit**: When score drops below 50%
- **Re-evaluate**: Daily before market open

---

## 🔥 Example Scenarios

### Scenario 1: Bull Rally
```
Inputs:
- NIFTY: 24900
- VIX Forecast: -8% (falling)
- NIFTY Forecast: +5.2% (bullish)
- Greek: Delta-driven
- Vol Regime: Medium

Output:
✅ LONG CALL (88% score)
   Buy: 24900 CE
   
✅ BULL CALL SPREAD (75% score)
   Buy: 24900 CE, Sell: 25100 CE
```

### Scenario 2: Volatility Spike
```
Inputs:
- NIFTY: 24900
- VIX Forecast: +18% (rising)
- NIFTY Forecast: +0.8% (uncertain)
- Greek: Vega-driven
- Vol Regime: Low

Output:
✅ LONG STRADDLE (92% score)
   Buy: 24900 CE + 24900 PE
   
✅ LONG STRANGLE (85% score)
   Buy: 25100 CE + 24700 PE
   
❌ Avoid: SHORT STRADDLE
```

### Scenario 3: Sideways Grind
```
Inputs:
- NIFTY: 24900
- VIX Forecast: -5% (stable/falling)
- NIFTY Forecast: +0.3% (range-bound)
- Greek: Theta-driven
- Vol Regime: High

Output:
✅ IRON CONDOR (87% score)
   Sell: 24600 PE, Buy: 24400 PE
   Sell: 25200 CE, Buy: 25400 CE
   
✅ BUTTERFLY SPREAD (73% score)
   Buy: 24700, Sell 2x: 24900, Buy: 25100
   
❌ Avoid: LONG CALL, LONG PUT
```

---

## 📞 Workflow Summary

1. **Morning**: Check all 3 source dashboards (8050, 8051, 8055)
2. **Collect**: VIX forecast, NIFTY forecast, Greek regime, Vol regime
3. **Input**: Enter values in Decision Dashboard (8060)
4. **Analyze**: Click ANALYZE button
5. **Review**: Check Recommended tab
6. **Select**: Choose strategy with highest score
7. **Execute**: Trade the specific strikes shown
8. **Monitor**: Re-analyze if market changes significantly

---

## 📈 Dashboard Navigation

### Tabs in Decision Dashboard

1. **✅ Recommended Strategies**
   - High-confidence trades (≥70%)
   - Specific strikes calculated
   - Full risk/reward details

2. **⚠️ Caution Strategies**
   - Medium-confidence (40-69%)
   - Some conditions met
   - Trade with reduced size

3. **❌ Avoid Strategies**
   - Low-confidence (<40%)
   - Or has explicit avoid conditions
   - Don't trade these!

4. **📈 Signal Analysis**
   - Market condition matrix
   - How signals were interpreted
   - Educational view

---

## 🛡️ Risk Disclaimer

⚠️ **The Decision Dashboard is a TOOL, not financial advice**

- Always verify signals manually
- Use proper position sizing
- Set stop-losses
- Don't overtrade
- Past performance ≠ future results
- Options trading carries significant risk
- Can lose 100% of premium paid

---

## 🎓 Learning Path

### Beginner
1. Start with **BULL/BEAR SPREADS** (limited risk)
2. Use only ✅ Recommended strategies
3. Paper trade first

### Intermediate  
2. Add **STRADDLES/STRANGLES** (vol plays)
3. Use ⚠️ Caution strategies selectively
4. Understand Greek interactions

### Advanced
5. Trade **IRON CONDORS** (income generation)
6. Use **CALENDAR/BUTTERFLY** spreads
7. Combine multiple strategies
8. Adjust positions dynamically

---

**Dashboard Live**: http://127.0.0.1:8060

**Full Guide**: DECISION_DASHBOARD_GUIDE.md
