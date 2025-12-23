# Greek Pressure Index (GPI) Dashboard - User Guide

## 🚀 Dashboard is LIVE at http://localhost:8052

---

## 📊 **STEP 1: Greek Pressure Index**

This dashboard implements the core GPI calculation for identifying entry and exit points based on option Greeks.

---

## 🎯 **What is GPI?**

**Greek Pressure Index (GPI)** measures how sensitive an option is to price movements vs. time decay.

### Formula:
```
GPI = 0.4·Δn + 0.3·Γn + 0.2·Vn − 0.1·Θn
```

### Normalized Greeks:
- **Δn** = |Delta| (absolute delta)
- **Γn** = Gamma / Spot (gamma normalized by spot price)
- **Vn** = Vega / IV (vega normalized by implied volatility)
- **Θn** = |Theta| / Premium (theta normalized by option price)

### Interpretation:
- **High GPI (>0.6)** = MOVEMENT-SENSITIVE → Good for directional trades
- **Low GPI (<0.3)** = DECAY-SENSITIVE → Good for premium selling
- **Medium GPI (0.3-0.6)** = NEUTRAL → Mixed signals

---

## 🎮 **How to Use the Dashboard**

### Step 1: Set Parameters
1. **NIFTY Spot**: Current NIFTY price (auto-loaded: 25,898.55)
2. **VIX (IV %)**: Implied volatility (auto-loaded: 10.40%)
3. **Days to Expiry**: Select 1, 3, 7, 14, 21, or 30 days
4. Click **"Calculate GPI"**

### Step 2: View Summary
The dashboard shows:
- Current NIFTY Spot
- VIX / IV percentage
- Days to expiry
- **BUY Signals**: Number of movement-sensitive strikes (GPI > 0.6) within ±2%
- **SELL Signals**: Number of decay-sensitive strikes (GPI < 0.3) within ±2%

---

## 📈 **Dashboard Tabs**

### Tab 1: 📊 GPI Analysis

**GPI Bar Charts**
- Left: Calls (CE)
- Right: Puts (PE)
- **Green bars**: GPI > 0.6 (Movement-sensitive)
- **Orange bars**: GPI 0.3-0.6 (Neutral)
- **Red bars**: GPI < 0.6 (Decay-sensitive)
- Black dashed line: ATM strike
- Green dotted line: 0.6 threshold (movement)
- Orange dotted line: 0.3 threshold (decay)

**Greek Heatmaps**
- Shows all normalized Greeks (Δn, Γn, Vn, Θn) and final GPI
- Helps visualize which Greek is dominating each strike

---

### Tab 2: 🎯 Trading Signals

**Focus: ±2% Strikes Only**

Dashboard filters to show only strikes within 2% of current spot price (most liquid).

#### 🟢 BUY Signals (Movement-Sensitive)
- **GPI > 0.6**
- These options are sensitive to price moves
- **Use for**: Directional trades, buying options
- **Best when**: You expect movement in any direction

**Example:**
```
Strike: 26,000 CE
GPI: 0.68
Signal: BUY
→ This call is highly sensitive to upward moves
→ Good for bullish directional play
```

#### 🔴 SELL Signals (Decay-Sensitive)
- **GPI < 0.3**
- Theta decay dominates these options
- **Use for**: Premium selling, Iron Condors, Strangles
- **Best when**: You expect range-bound market

**Example:**
```
Strike: 26,500 CE
GPI: 0.24
Signal: SELL
→ This call decays fast, low movement sensitivity
→ Good for selling premium in neutral market
```

---

### Tab 3: 📋 Full Chain

Complete option chain with:
- All strikes generated
- Moneyness percentage
- Premium (option price)
- All Greeks (Delta, Gamma, Vega, Theta)
- Normalized Greeks (Δn, Γn, Vn, Θn)
- **GPI score**
- Classification (Movement/Neutral/Decay)
- Signal (BUY/HOLD/SELL)

**Features:**
- **Filterable columns**: Click column headers to filter
- **Sortable**: Click any column to sort
- **Color-coded**: Green = Movement, Red = Decay

---

## 💡 **Trading Strategy by GPI**

### High GPI (>0.6) - Movement Plays

**When to Use:**
- You have a directional view (bullish or bearish)
- Expecting significant price movement
- VIX/IV is low (cheap options)

**Strategies:**
- Buy Calls (if bullish)
- Buy Puts (if bearish)
- Bull/Bear Spreads
- Long Straddles (if direction uncertain but big move expected)

**Example:**
```
Market View: Bullish, expecting 3% up move
GPI Analysis: 26,100 CE has GPI = 0.72 (high!)
Action: Buy 26,100 CE or Bull Call Spread (26,100/26,500)
Rationale: High GPI means this option will move fast with spot
```

---

### Low GPI (<0.3) - Decay Plays

**When to Use:**
- Expecting range-bound market
- VIX/IV is high (expensive options)
- Time decay will work in your favor

**Strategies:**
- Sell Strangles
- Iron Condors
- Credit Spreads
- Sell Covered Calls

**Example:**
```
Market View: Neutral, expect ±1% range
GPI Analysis: 26,500 CE has GPI = 0.22 (low!)
Action: Sell 26,500 CE as part of Strangle or IC
Rationale: Low GPI means theta decay dominates, collect premium
```

---

### Medium GPI (0.3-0.6) - Neutral

**When to Use:**
- Mixed market signals
- Moderate volatility expected

**Strategies:**
- Butterfly spreads
- Ratio spreads
- Wait for clearer GPI signal

---

## 🔍 **Understanding the Greeks**

### Delta (Δ)
- Measures price sensitivity
- Call Delta: 0 to 1
- Put Delta: -1 to 0
- **Δn = |Delta|** (absolute value)

### Gamma (Γ)
- Rate of change of Delta
- Highest at ATM
- **Γn = Gamma / Spot** (normalized)

### Vega (V)
- Sensitivity to IV changes
- Higher when more time to expiry
- **Vn = Vega / IV** (normalized)

### Theta (Θ)
- Time decay per day
- Always negative for buyers
- **Θn = |Theta| / Premium** (normalized)

---

## 📊 **Practical Example**

### Scenario: Expecting a breakout

**Input:**
- NIFTY: 25,900
- VIX: 10.4%
- DTE: 7 days

**Dashboard Shows:**

**BUY Signals (High GPI):**
```
Strike    Type  Premium  Delta   Gamma   Vega   Theta   GPI    Signal
25,950    CE    125.3    0.52    0.0034  8.2    -12.5   0.68   BUY
25,900    CE    148.7    0.58    0.0036  8.5    -14.2   0.72   BUY
25,850    CE    175.2    0.63    0.0035  8.3    -15.8   0.70   BUY
```

**SELL Signals (Low GPI):**
```
Strike    Type  Premium  Delta   Gamma   Vega   Theta   GPI    Signal
26,400    CE     15.2    0.12    0.0008  2.1    -2.8    0.18   SELL
26,450    CE     10.8    0.09    0.0006  1.8    -2.1    0.14   SELL
26,500    CE      7.5    0.06    0.0004  1.4    -1.5    0.11   SELL
```

**Decision:**
- **If Bullish**: Buy 25,900 CE (GPI = 0.72, high movement sensitivity)
- **If Neutral**: Sell 26,400/26,500 CE as credit spread (low GPI, decay dominant)

---

## ⚙️ **Advanced Tips**

### Tip 1: Compare Call vs Put GPI
- If call GPI > put GPI at same strike → upside bias
- If put GPI > call GPI → downside bias

### Tip 2: GPI Changes with Time
- Closer to expiry → GPI of OTM drops (decay increases)
- Further from expiry → GPI more stable

### Tip 3: IV Impact on GPI
- High IV → Vega_n increases → GPI increases
- Low IV → Theta_n dominates → GPI decreases

### Tip 4: Focus on ±2% Strikes
- Most liquid
- Tightest spreads
- Best execution

---

## 🎯 **Integration with Other Dashboards**

### Complete Trading System:

**Step 1: Forecast Dashboard (Port 8051)**
- Get market direction (UP/DOWN)
- Check tail risk (<10%)
- Note expected move

**Step 2: GPI Dashboard (Port 8052)** ⭐ **YOU ARE HERE**
- Calculate GPI for all strikes
- Identify high GPI (movement) vs low GPI (decay)
- Get BUY/SELL signals

**Step 3: Manual Regime Check (GREEK_REGIME_GUIDE.md)**
- Classify market regime
- Verify strategy alignment

**Step 4: Execute**
- Enter trade based on combined signals
- Use position sizing from forecast dashboard

---

## 📋 **Quick Decision Matrix**

| Market View | VIX | GPI Signal | Strategy |
|-------------|-----|------------|----------|
| Bullish + Big Move | Low | BUY (High GPI CE) | Buy Call / Bull Spread |
| Bearish + Big Move | Low | BUY (High GPI PE) | Buy Put / Bear Spread |
| Neutral | High | SELL (Low GPI) | Iron Condor / Strangle |
| Uncertain | Medium | HOLD | Wait for clearer signal |

---

## 🚨 **Important Notes**

1. **±2% Focus**: Dashboard filters signals to ±2% strikes for best liquidity
2. **Refresh Daily**: GPI changes daily as Greeks evolve
3. **Combine Signals**: Don't use GPI alone - check forecast + regime
4. **Respect Thresholds**: 
   - GPI > 0.6 = Clear movement play
   - GPI < 0.3 = Clear decay play
   - 0.3-0.6 = Wait for better setup

---

## 📞 **Dashboard Features**

✅ Real-time GPI calculation
✅ Black-Scholes Greeks (all 5: Delta, Gamma, Vega, Theta, Rho)
✅ Normalized Greek metrics
✅ Color-coded signals
✅ Focus on ±2% liquid strikes
✅ Sortable/filterable tables
✅ Visual heatmaps
✅ Top 10 BUY/SELL opportunities

---

**Dashboard URL**: http://localhost:8052
**Current Data**: NIFTY=25,898.55, VIX=10.40%
**Last Updated**: December 19, 2025

---

**Next Step**: Add Step 2 - Market Regime Classification
