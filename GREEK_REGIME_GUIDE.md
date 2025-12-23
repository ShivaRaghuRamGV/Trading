# Greek Regime Classification - Step 2

## ⚠️ CRITICAL: Identify Market Regime Before Entry

You **MUST** classify the market regime before entering any trade. The regime determines which strategy will work best.

---

## 📊 Four Market Regimes

### 1. **DELTA-DRIVEN Regime** 📈
**Conditions:**
- ✅ Spot > VWAP (20-day average)
- ✅ Delta rising (spot trending)
- ✅ Gamma stable (no spikes)

**Characteristics:**
- Trending market with directional momentum
- Deltas are driving P&L more than other Greeks

**Best Strategies:**
- 🟢 Buy ITM/ATM options in trend direction
- Bull Call Spreads (uptrend)
- Bear Put Spreads (downtrend)
- Long Calls (strong uptrend)
- Long Puts (strong downtrend)

**Risk:**
- Reversals can be sharp
- Exits important - don't overstay

**Example:**
```
NIFTY: 25,900 
VWAP: 25,700 (spot > VWAP ✓)
Spot Change: +1.2% (rising ✓)
ATM Gamma: 0.0023 (normal ✓)
→ DELTA-DRIVEN REGIME
→ Action: Buy Call Spreads if bullish
```

---

### 2. **GAMMA-DRIVEN Regime** ⚡
**Conditions:**
- ✅ ATM Gamma spikes (>1.5x average)
- ✅ IV relatively flat (change < 2%)

**Characteristics:**
- High gamma environment
- Explosive moves possible
- Rapid P&L swings
- Near expiry or major event pending

**Best Strategies:**
- 🟢 Buy Straddles/Strangles (benefit from big moves)
- Gamma scalping (for professionals)
- **AVOID selling premium** (high risk)

**Risk:**
- Extreme volatility
- Can whipsaw both directions
- Premium buyers can still lose if move doesn't materialize

**Example:**
```
NIFTY: 24,200
ATM Gamma: 0.0045 (Average: 0.0028 → 1.6x spike! ✓)
IV Change: +0.8% (flat ✓)
→ GAMMA-DRIVEN REGIME  
→ Action: Buy ATM Straddle, expect big move
```

---

### 3. **VEGA-DRIVEN Regime** 🌊
**Conditions:**
- ✅ IV rising rapidly (>3%)
- ✅ IV change % > Spot change % (vol rising faster than price)

**Characteristics:**
- Volatility expanding
- Uncertainty rising
- Fear/greed increasing
- Event risk (budget, policy, geopolitics)

**Best Strategies:**
- 🟢 Buy options to benefit from IV expansion
- Calendar spreads (buy longer dated, sell near)
- Ratio spreads
- **AVOID selling premium naked**

**Risk:**
- Volatility crush if event passes without drama
- Time decay still hurts if you're long options

**Example:**
```
NIFTY: 24,100
NIFTY Change: +0.5%
VIX: 18.5
VIX Change: +4.2% (rising fast ✓)
IV% change (4.2%) > Spot% change (0.5%) ✓
→ VEGA-DRIVEN REGIME
→ Action: Buy options before VIX peaks, sell after event
```

---

### 4. **THETA-DRIVEN Regime** ⏰
**Conditions:**
- ✅ IV falling (<-2%)
- ✅ Spot range-bound (change < 0.5%)

**Characteristics:**
- Low volatility
- Sideways/choppy market
- Time decay dominant
- Post-event calm

**Best Strategies:**
- 🟢 Sell premium (Iron Condors, Short Strangles)
- Credit spreads
- Theta harvesting
- Range-bound strategies

**Risk:**
- Breakouts can cause losses
- Must exit before regime shifts

**Example:**
```
NIFTY: 24,300
NIFTY Change: +0.2% (range-bound ✓)
VIX: 13.2  
VIX Change: -3.1% (falling ✓)
→ THETA-DRIVEN REGIME
→ Action: Sell Iron Condor, collect premium
```

---

## 🔧 How to Calculate Regime (Manual Method)

### Step 1: Get Current Data
```
Current NIFTY Spot: _______
Previous NIFTY: _______
NIFTY Change %: _______ (= (Current - Prev) / Prev × 100)

Current VIX: _______
Previous VIX: _______
VIX Change %: _______ (= (Current - Prev) / Prev × 100)

20-Day VWAP: _______ (average of last 20 closes)
```

### Step 2: Calculate ATM Gamma
```
ATM Strike: _______ (round NIFTY to nearest 50)
ATM Call Gamma: _______ (from option chain)
ATM Put Gamma: _______
ATM Avg Gamma: _______ (average of call + put)

Overall Avg Gamma: _______ (average of all strikes)
Gamma Spike? _______ (YES if ATM > 1.5 × Overall)
```

### Step 3: Score Each Regime

**Delta-Driven Score:**
- [ ] Spot > VWAP? (+1 if yes)
- [ ] Spot rising >0.3%? (+1 if yes)
- [ ] Gamma stable (no spike)? (+1 if yes)
- **Total: ___ / 3**

**Gamma-Driven Score:**
- [ ] Gamma spike? (+2 if yes)
- [ ] IV flat (|change| < 2%)? (+1 if yes)
- **Total: ___ / 3**

**Vega-Driven Score:**
- [ ] IV rising >3%? (+2 if yes)
- [ ] |IV change| > |Spot change|? (+1 if yes)
- **Total: ___ / 3**

**Theta-Driven Score:**
- [ ] IV falling <-2%? (+2 if yes)
- [ ] Spot range-bound (|change| <0.5%)? (+1 if yes)
- **Total: ___ / 3**

### Step 4: Determine Regime
**Regime = Highest score above**

Confidence = (Highest Score / 3) × 100%

---

## 📋 Trading Workflow with Regime Check

### Pre-Trade Checklist

**1. Calculate Market Regime**
- [ ] Downloaded latest NIFTY & VIX data
- [ ] Calculated changes from previous day
- [ ] Computed VWAP
- [ ] Checked ATM Gamma for spikes
- [ ] Scored all 4 regimes
- [ ] Identified primary regime

**2. Match Strategy to Regime**
- [ ] Regime is: ______________
- [ ] Recommended strategy: ______________
- [ ] Confidence: _____%

**3. Only Enter If:**
- [ ] Regime confidence >60%
- [ ] Strategy matches regime
- [ ] GPI supports the play (from Greek dashboard)
- [ ] Tail risk <10% (from forecast dashboard)

---

## 🎯 Integration with Existing Dashboards

### Use All 3 Dashboards Together:

**1. Forecast Dashboard (Port 8051) - Direction & Tail Risk**
- Get directional bias (UP/DOWN)
- Check tail risk (<10% required)
- Note expected move magnitude

**2. Greek Dashboard (Port 8052) - GPI & Strikes**
- Calculate Greek Pressure Index
- Identify high/low GPI strikes
- Get entry/exit signals

**3. Regime Classification (Manual) - Strategy Selection**
- Determine current regime
- Match strategy to regime
- Verify alignment with GPI signals

### Example Combined Analysis:

```
DATE: Dec 19, 2025

STEP 1: Forecast Dashboard
- Direction: UP
- Confidence: 68%
- Expected Move: +4.5%
- Tail Risk: 5.2%
- ✅ Safe to trade (tail risk <10%)

STEP 2: Regime Classification (YOU CALCULATE)
- NIFTY: 25,900 (prev: 25,650)
- Change: +0.97%
- VIX: 10.8 (prev: 11.2)
- Change: -3.57%
- VWAP: 25,700
- Regime: DELTA-DRIVEN (Spot>VWAP, rising, gamma stable)
- ✅ Directional trades favored

STEP 3: Greek Dashboard
- GPI Analysis: ATM+200 Call has GPI=0.68 (HIGH)
- Signal: BUY
- ✅ Movement-sensitive strike identified

FINAL DECISION:
✅ All systems align!
- Forecast: Bullish +4.5%
- Regime: Delta-driven (trend following)
- GPI: High GPI on +200 Call (movement play)

ACTION: Buy Bull Call Spread
- Buy: 26,100 CE
- Sell: 26,500 CE
- Lots: 3
- Max Risk: ₹22,500
- Target: ₹50,000
```

---

## 🚨 Regime Shift Warnings

### Exit Immediately If Regime Changes:

**Scenario 1: Delta → Gamma**
- Gamma spike appears
- High volatility incoming
- **Exit directional positions**
- Consider switching to straddles

**Scenario 2: Theta → Vega**
- IV starts rising
- Range breaking down
- **Exit credit spreads**
- Volatility will hurt short premium

**Scenario 3: Any → Gamma**
- Gamma spike = danger zone
- **Reduce all positions**
- High risk period

---

## 📈 Regime Persistence

**Typical Regime Duration:**
- Delta-driven: 3-7 days (trend phase)
- Gamma-driven: 1-3 days (event/expiry proximity)
- Vega-driven: 2-5 days (pre/post events)
- Theta-driven: 5-10 days (calm periods)

**Check regime:**
- Daily (minimum)
- Intraday if major news
- Before every new trade

---

## 🎓 Quick Reference Card

| Regime | Spot vs VWAP | Spot Move | IV Move | Gamma | Strategy |
|--------|--------------|-----------|---------|-------|----------|
| Delta  | Above | Rising | Flat | Stable | Buy Directional |
| Gamma  | Any | Any | Flat | Spike | Buy Straddles |
| Vega   | Any | Small | Rising Fast | Any | Buy Options |
| Theta  | Any | Range | Falling | Low | Sell Premium |

---

## 💡 Pro Tips

1. **Never fight the regime** - If regime says Theta but you want to go directional, wait for regime shift

2. **Regime > GPI** - If regime and GPI conflict, trust the regime classification first

3. **Gamma spikes override** - If Gamma spikes, everything else is secondary. Reduce exposure.

4. **Track regime changes** - Keep a log. Regime shifts = opportunity or danger.

5. **Combine with forecast** - Regime tells you HOW to trade, forecast tells you WHICH direction

---

## 📊 Regime Log Template

```
Date: __________
NIFTY: ______ (Change: ____%)
VIX: ______ (Change: ____%)
VWAP: ______
ATM Gamma: ______ (Spike: Y/N)

Regime: ______________
Confidence: _____%
Strategy: ______________

Trades Entered:
- _______________________
- _______________________

Result: _________________
```

---

**Last Updated:** December 19, 2025  
**Integration:** Use with Forecast Dashboard (8051) & Greek Dashboard (8052)  
**Purpose:** Step 2 of Greek Regime Flip Model - MUST classify before every trade

