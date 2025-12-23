# 📊 Trading Signals Guide - VIX Forecasting Dashboard
## Entry & Exit Criteria for 5 Lacs Capital Deployment

---

## 🎯 **OVERVIEW**

Your forecasting dashboard uses a **3-Model ML System** combined with **GARCH(1,1)-X** forecasting to generate trading signals for NIFTY options. Here's how to use the signals for deploying ₹5,00,000 capital.

---

## 📈 **ENTRY SIGNALS**

### **Step 1: Generate Trading Plan on Dashboard**

Go to the **"🤖 ML Strategy Selector"** tab and click **"Generate Trading Plan"**. The system analyzes:

1. **Direction Prediction** (UP/DOWN with confidence %)
2. **Expected Move** (% magnitude in next 22 days)
3. **Tail Risk Probability** (extreme move likelihood)
4. **VIX Forecast** (from GARCH model)

---

### **ENTRY CONDITIONS BY STRATEGY TYPE**

#### **A. DIRECTIONAL TRADES (Long Call/Put or Spreads)**

**Entry Criteria:**
- ✅ **Confidence ≥ 60%** AND **Tail Risk ≤ 7%**
  - OR
- ✅ **Expected Move ≥ 4.0%** AND **Confidence ≥ 50%**

**Dashboard Indicators:**
- Strategy box shows: **"LONG_CALL"** or **"LONG_PUT"**
- Reasoning: *"High confidence (X%) [UP/DOWN] move with low tail risk"*
- VIX Forecast: Any level (but low VIX <15 preferred for buying options)

**What to Trade:**
- **If UP:** Buy Call Spread (ATM + strike distance shown)
- **If DOWN:** Buy Put Spread (ATM - strike distance shown)
- **Strike Distance:** Dashboard shows optimal distance (e.g., "Strike Distance: 150 from ATM")
- **Spread Width:** Dashboard shows (e.g., "Spread Width: 300")

**Example Entry:**
```
Dashboard Output:
- Direction: UP
- Confidence: 68%
- Expected Move: +4.5%
- Tail Risk: 5.2%
- Strategy: LONG_CALL
- Strike Distance: 200 from ATM
- Recommended Lots: 3

Action: Buy 3 lots of Bull Call Spread
- Buy Call: NIFTY Spot + 200 (e.g., 24,200 + 200 = 24,400 CE)
- Sell Call: NIFTY Spot + 500 (e.g., 24,200 + 500 = 24,700 CE)
- Expiry: 22 DTE (approximately 1 month)
```

---

#### **B. IRON CONDOR (Neutral/Range-bound)**

**Entry Criteria:**
- ✅ **Confidence < 40%** AND **Expected Move < 2.5%**

**Dashboard Indicators:**
- Strategy box shows: **"IRON_CONDOR"**
- Reasoning: *"Low confidence (X%) with limited move (Y%) - range-bound expected"*
- VIX Forecast: Preferably declining or stable

**What to Trade:**
- Sell premium on both sides with defined risk

**Strikes (from Dashboard):**
```
Dashboard Output:
- Short Call: ATM + 400
- Long Call:  ATM + 750
- Short Put:  ATM - 400
- Long Put:   ATM - 750
- Recommended Lots: 2

Action: Sell 2 lots of Iron Condor
- Sell Call: 24,200 + 400 = 24,600 CE
- Buy Call:  24,200 + 750 = 24,950 CE
- Sell Put:  24,200 - 400 = 23,800 PE
- Buy Put:   24,200 - 750 = 23,450 PE
- Expiry: 22 DTE
```

---

#### **C. SHORT STRANGLE (Moderate Confidence)**

**Entry Criteria:**
- ✅ **Confidence between 40-60%** OR **Expected Move 2.5-4.0%**
- ✅ **Tail Risk < 10%**

**Dashboard Indicators:**
- Strategy box shows: **"SHORT_STRANGLE"**
- Reasoning: *"Moderate confidence (X%) with Y% expected move"*

**What to Trade:**
```
Dashboard Output:
- Short Call: ATM + 500
- Short Put:  ATM - 500
- Recommended Lots: 2

Action: Sell 2 lots of Strangle
- Sell Call: 24,200 + 500 = 24,700 CE
- Sell Put:  24,200 - 500 = 23,700 PE
- Expiry: 22 DTE
```

---

#### **D. AVOID / NO TRADE**

**Entry Criteria:**
- ❌ **Tail Risk > 10%**

**Dashboard Indicators:**
- Strategy box shows: **"AVOID"**
- Reasoning: *"High tail risk (X%) - extreme move likely. Reduce exposure or hedge."*

**Action:** 
- **DO NOT ENTER** new positions
- **REDUCE** existing positions
- **HEDGE** with protective options
- Wait for tail risk to drop below 7%

---

## 💰 **POSITION SIZING (Automatic from Dashboard)**

The dashboard calculates optimal position size using **Kelly Criterion** adjusted for tail risk:

### **Dashboard Position Sizing Output:**
```
Position Sizing:
  Kelly Fraction: 18.5%
  Tail-Adjusted: 14.2%
  Position Size: ₹71,000
  Recommended Lots: 3
  Max Loss/Lot: ₹8,333
```

**How to Use:**
1. **Use "Recommended Lots"** as your position size
2. **Max Loss/Lot** = your stop loss per lot
3. **Total Capital at Risk** = Recommended Lots × Max Loss/Lot
4. Never exceed 2-3% of total capital (₹5L) = ₹10,000-₹15,000 max risk

---

## 🚪 **EXIT SIGNALS**

### **Exit Rule #1: Time-Based Exit**

**DEFAULT: Exit at 1-3 DTE (Days to Expiry)**

- **Entry at:** 22 DTE (approximately 1 month)
- **Exit at:** 1-3 DTE (last week before expiry)
- **Rationale:** Avoid expiry risk and capture theta decay

**Example:**
- Enter: Dec 26, 2024 (Jan 16 expiry = 21 DTE)
- Exit: Jan 13-15, 2025 (1-3 DTE)

---

### **Exit Rule #2: Profit Target**

**For Credit Strategies (Iron Condor, Short Strangle):**
- ✅ **Exit when profit = 50-70% of max credit**

**Example:**
```
Premium Collected: ₹10,000
Target Profit: ₹5,000-₹7,000 (50-70%)
Exit: When position shows ₹5K-₹7K profit
```

**For Debit Strategies (Long Calls/Puts, Spreads):**
- ✅ **Exit when profit = 100-150% of debit paid**

**Example:**
```
Debit Paid: ₹8,000
Target Profit: ₹8,000-₹12,000 (100-150%)
Exit: When position shows ₹8K-₹12K profit
```

---

### **Exit Rule #3: Stop Loss**

**Defined by Dashboard:**
- Maximum Loss/Lot is calculated
- **Exit when loss = 1.5-2x Max Loss/Lot**

**For Credit Spreads/Condors:**
- ✅ **Stop loss = 2× Max Credit Collected**

**Example:**
```
Max Credit Collected: ₹10,000
Stop Loss: -₹20,000
Exit immediately when loss reaches ₹20K
```

**For Debit Spreads:**
- ✅ **Stop loss = Full debit paid** (limited risk)

---

### **Exit Rule #4: Signal Reversal**

**Re-run Dashboard Analysis Weekly**

If the dashboard shows signal reversal:

1. **Directional → Neutral/Opposite:**
   - Close directional positions
   - May switch to Iron Condor if still within profit

2. **Neutral → Directional:**
   - Close Iron Condors early
   - Consider entering directional trade

3. **Tail Risk Spikes Above 10%:**
   - **IMMEDIATE EXIT** of all positions
   - Risk > Reward; protect capital

---

### **Exit Rule #5: VIX Spike/Crash**

**Monitor VIX Change:**

**For CREDIT Strategies (Short Strangle, IC):**
- ✅ **Exit if VIX spikes >30%** in a day
  - Reason: Positions likely underwater
  - Example: VIX jumps from 15 → 20+ (33% spike)

**For DEBIT Strategies (Long Calls/Puts):**
- ✅ **Exit if VIX drops >20%** in a day
  - Reason: Volatility crush kills option value
  - Example: VIX drops from 18 → 14 (22% drop)

---

## 📋 **COMPLETE TRADE WORKFLOW**

### **1. Pre-Trade Setup (Daily/Weekly)**

1. Open **Forecast Dashboard** ([forecast_dashboard.py](forecast_dashboard.py))
2. Click **"Train Model & Forecast"** (use 2 years training, 5-21 day horizon)
3. Go to **"ML Strategy Selector"** tab
4. Set Capital: ₹500,000
5. Set Confidence Threshold: 60%
6. Click **"Generate Trading Plan"**

---

### **2. Entry Checklist**

- [ ] Check Strategy Type (Directional/IC/Strangle/Avoid)
- [ ] Verify Entry Criteria met (see above)
- [ ] Note Recommended Lots
- [ ] Note Strike Distances from dashboard
- [ ] Calculate total capital at risk (≤2% of ₹5L)
- [ ] Set expiry: 22 DTE (approximately 1 month out)
- [ ] Execute trade on NSE (NIFTY options)

---

### **3. Trade Management**

**Daily:**
- Monitor P&L vs. profit target
- Monitor P&L vs. stop loss
- Check for VIX spikes (>30% move)

**Weekly:**
- Re-run dashboard analysis
- Check for signal reversal
- Check tail risk (exit if >10%)

---

### **4. Exit Checklist**

**Exit if ANY condition met:**
- [ ] Time: Reached 1-3 DTE
- [ ] Profit: Hit 50-70% (credit) or 100-150% (debit)
- [ ] Stop Loss: Hit 2x credit or full debit
- [ ] Signal Reversal: Dashboard shows opposite signal
- [ ] VIX Spike: >30% single-day move (for shorts)
- [ ] Tail Risk: Spikes above 10%

---

## 🎯 **EXAMPLE TRADES**

### **Example 1: Bullish Directional**

**Dashboard Output (Dec 19, 2024):**
```
Market Forecast:
  Direction: UP
  Confidence: 72%
  Expected Move: +5.2%
  Predicted NIFTY: 24,850
  Tail Risk: 4.1%

Strategy: LONG_CALL
Strike Distance: 200 from ATM
Spread Width: 400
Recommended Lots: 3
Max Loss/Lot: ₹9,500
```

**Trade Execution:**
```
Date: Dec 19, 2024
NIFTY Spot: 24,200

ENTRY:
Buy 3 lots Bull Call Spread (Jan 16 expiry, 28 DTE)
- Buy:  75 × 24,400 CE @ ₹180 = -₹13,500 per lot
- Sell: 75 × 24,800 CE @ ₹80  = +₹6,000 per lot
Net Debit: ₹7,500 per lot × 3 = -₹22,500 total
Max Profit: (400 - 150) × 75 × 3 = ₹56,250
Max Loss: ₹22,500 (limited)

EXIT PLAN:
- Time: Jan 13-15 (1-3 DTE)
- Profit Target: ₹22,500-₹33,750 (100-150%)
- Stop Loss: -₹22,500 (full debit, limited risk)
```

**Outcome (Jan 13):**
```
NIFTY closes at 24,720
- 24,400 CE = ₹320 (intrinsic value)
- 24,800 CE = ₹0 (worthless)
Spread value = ₹320 per share × 75 × 3 = ₹72,000
Exit value: ₹72,000
Profit: ₹72,000 - ₹22,500 = +₹49,500 ✅
ROI: 220%
```

---

### **Example 2: Iron Condor**

**Dashboard Output:**
```
Market Forecast:
  Direction: UP
  Confidence: 35%
  Expected Move: ±1.8%
  Tail Risk: 6.2%

Strategy: IRON_CONDOR
Short Call: ATM + 450
Long Call:  ATM + 800
Short Put:  ATM - 450
Long Put:   ATM - 800
Recommended Lots: 2
```

**Trade Execution:**
```
Date: Dec 19, 2024
NIFTY Spot: 24,200

ENTRY:
Sell 2 lots Iron Condor (Jan 16 expiry, 28 DTE)
Call Side:
- Sell: 50 × 24,650 CE @ ₹60 = +₹3,000 per lot
- Buy:  50 × 25,000 CE @ ₹20 = -₹1,000 per lot
Put Side:
- Sell: 50 × 23,750 PE @ ₹55 = +₹2,750 per lot
- Buy:  50 × 23,400 PE @ ₹15 = -₹750 per lot
Net Credit: ₹4,000 per lot × 2 = +₹8,000 total
Max Profit: ₹8,000
Max Loss: (350 - 40) × 50 × 2 = ₹31,000

EXIT PLAN:
- Time: Jan 13-15 (1-3 DTE)
- Profit Target: ₹4,000-₹5,600 (50-70% of credit)
- Stop Loss: -₹16,000 (2x credit)
```

**Outcome (Jan 10):**
```
Position shows +₹5,200 profit (65% of max)
Close all 4 legs
Profit: +₹5,200 ✅
ROI: 65%
```

---

## ⚠️ **RISK MANAGEMENT RULES**

### **Capital Allocation**
- ✅ **Max 2-3% risk per trade** = ₹10,000-₹15,000
- ✅ **Max 3-4 positions** simultaneously
- ✅ **Keep 20% cash** reserve (₹1,00,000)

### **Diversification**
- ✅ Mix directional and neutral strategies
- ✅ Vary expiry dates (don't put all in one expiry)
- ✅ Scale in (e.g., 1 lot → test → add 2 more lots)

### **Stop Discipline**
- ✅ **ALWAYS honor stop losses** (no exceptions)
- ✅ **Exit on tail risk spike** above 10%
- ✅ **Close before expiry week** (avoid gamma risk)

### **Review & Adapt**
- ✅ Track all trades in spreadsheet
- ✅ Calculate monthly win rate
- ✅ Adjust position sizing if losing streak
- ✅ Re-calibrate confidence threshold quarterly

---

## 📊 **DASHBOARD TABS TO MONITOR**

### **Tab 1: Current Forecast**
- Use for: Entry signals
- Key metrics: VIX forecast, expected change, confidence bands

### **Tab 2: Backtest Analysis**
- Use for: Validate model accuracy
- Key metrics: MAE, RMSE, direction accuracy

### **Tab 3: Model Diagnostics**
- Use for: Market regime detection
- Key metrics: VIX regime (Low/Normal/High/Panic)

### **Tab 4: ML Strategy Selector** ⭐ **MOST IMPORTANT**
- Use for: All trading signals
- Key sections:
  1. **Predictions**: Direction, move, tail risk
  2. **Strategy**: What to trade
  3. **Strikes**: Exact strike distances
  4. **Position Sizing**: Number of lots

---

## 🔄 **WEEKLY ROUTINE**

**Sunday Evening:**
1. Update all data: `python update_all_data.py`
2. Run dashboard: `python forecast_dashboard.py`
3. Generate trading plan for the week
4. Identify 2-3 best setups

**Monday-Friday:**
1. Monitor existing positions
2. Check for exit signals
3. Re-run forecast if major news/events
4. Scale in/out based on P&L

**Friday Evening:**
1. Review week's trades
2. Calculate win rate and ROI
3. Adjust strategy if needed

---

## 📞 **WHEN TO OVERRIDE SIGNALS**

**Trust the Dashboard EXCEPT:**

1. **Major Global Events:**
   - Elections, policy announcements
   - → Reduce position size by 50%

2. **Multi-Week Losing Streak:**
   - 3+ consecutive losses
   - → Pause trading, review model

3. **Extreme Market Conditions:**
   - Circuit breakers, crashes
   - → Close all positions, go to cash

4. **Low Liquidity:**
   - Bid-ask spread > 2% of premium
   - → Skip trade, wait for better liquidity

---

## 📈 **SUCCESS METRICS**

**Target Performance (Monthly):**
- ✅ Win Rate: >60%
- ✅ Average Win: >1.5× Average Loss
- ✅ Max Drawdown: <15% of capital
- ✅ Monthly ROI: 5-8% (₹25,000-₹40,000 on ₹5L)

**If Underperforming:**
- Increase confidence threshold (60% → 70%)
- Reduce position sizing
- Focus only on highest confidence trades

---

## 🎓 **LEARNING FROM THE DASHBOARD**

The dashboard provides educational insights:

1. **Backtest Tab**: See historical accuracy
2. **Feature Importance**: What drives predictions
3. **Regime Analysis**: How VIX behaves in different markets
4. **Error Distribution**: Understand model limitations

**Use these to:**
- Build intuition over time
- Identify when model works best
- Develop complementary strategies

---

## ✅ **FINAL CHECKLIST**

Before deploying ₹5 lacs:

- [ ] Understand all 4 strategy types
- [ ] Know all 5 exit rules
- [ ] Set up trade tracking spreadsheet
- [ ] Paper trade for 1 month first
- [ ] Start with 1-2 lots (₹50K-₹1L capital)
- [ ] Scale up only after 70%+ win rate
- [ ] Never violate risk management rules

---

## 📚 **RELATED FILES**

- [forecast_dashboard.py](forecast_dashboard.py) - Main dashboard
- [strategy_selector.py](strategy_selector.py) - ML strategy logic
- [ml_models.py](ml_models.py) - Prediction models
- [vix_forecaster.py](vix_forecaster.py) - GARCH forecasting
- [options_backtester.py](options_backtester.py) - Strategy backtesting

---

**Last Updated:** December 19, 2024  
**Capital:** ₹5,00,000  
**Risk per Trade:** 2-3% (₹10,000-₹15,000)  
**Expected Monthly Return:** 5-8% (₹25,000-₹40,000)

---

**⚠️ DISCLAIMER:** Options trading involves substantial risk. Past performance does not guarantee future results. Always use stop losses and never risk more than you can afford to lose. This guide is for educational purposes only.
