# Greek Regime Flip Model - Interpretation Guide

## Understanding Market Regimes and Dominant Greeks

### What Are the Two Different Concepts?

1. **Dominant Greek** - Which Greek has the highest Greek Dominance Score (GDS)
2. **Market Regime** - The trading environment classification based on Greek scores + market conditions

These can be DIFFERENT! You might see "Vega-Driven Regime" with "Gamma as Dominant Greek" - here's why:

---

## Greek Dominance Score (GDS) Calculation

Each option Greek gets a weighted score:
- **DELTA**: 45% weight × normalized Delta
- **GAMMA**: 40% weight × normalized Gamma  
- **VEGA**: 30% weight × normalized Vega
- **THETA**: -35% weight × normalized Theta (negative because time decay)

The **Dominant Greek** is simply whichever has the highest GDS value at that moment.

---

## Market Regime Classification (More Complex)

The model looks at **Greek scores + market conditions** to classify the regime:

### 1. GAMMA-DRIVEN REGIME ⚡
**Trigger Conditions:**
- Gamma GDS > 0.15 threshold
- IV change < 2.0% (low volatility change)

**Market Characteristics:**
- **High gamma exposure** - large position changes with small spot moves
- **Range-bound market** - spot isn't moving much
- **Volatility stable** - VIX/IV relatively calm
- **High convexity risk** - small spot moves create big hedge requirements

**Trading Implications:**
- Market makers have significant gamma exposure
- Expect choppy, range-bound price action
- Spot moves get amplified through gamma hedging
- Good for **scalping** and **range trading**
- Risky for directional bets (gamma can flip positions quickly)

**Example:** Options are ATM (at-the-money) with stable IV. Small NIFTY moves force market makers to hedge frequently, creating choppy price action within a range.

---

### 2. VEGA-DRIVEN REGIME 🌊
**Trigger Conditions:**
- Vega GDS > 0.15 threshold  
- IV change > 1.2% (rising volatility)

**Market Characteristics:**
- **High volatility sensitivity** - option prices moving more from IV than spot
- **Increasing uncertainty** - VIX rising, fear increasing
- **Volatility trading dominates** - traders buying/selling vol, not direction
- **Premium expansion** - all options getting more expensive

**Trading Implications:**
- **Volatility is the main driver**, not spot price
- Rising VIX environment (fear/uncertainty)
- Good for **long straddles/strangles** (benefit from vol expansion)
- **Sell premium carefully** - vol can spike further
- Focus on **IV rank/percentile** rather than spot levels

**Example:** News event pending, earnings volatility, geopolitical uncertainty - IV jumps from 15 to 18. Option prices surge even if NIFTY doesn't move much.

---

### 3. THETA-DRIVEN REGIME ⏰
**Trigger Conditions:**
- Theta GDS > 0.15 threshold
- Spot change < 0.5% (very stable market)

**Market Characteristics:**
- **Time decay dominates** - options losing value primarily due to time passage
- **Extremely low volatility** - market going nowhere
- **Quiet, boring market** - no catalysts, low volume
- **Premium compression** - options bleeding value daily

**Trading Implications:**
- **Best for premium sellers** - collect theta systematically
- **Sell credit spreads, iron condors** - profit from time decay
- **Avoid long options** - theta eats your premium daily
- Good for **systematic income strategies**
- Low risk environment (until it breaks!)

**Example:** During holiday periods or low-volume days. NIFTY moves ±0.2% daily, IV drops, options just decay. Short strangles profit.

---

### 4. DELTA-DRIVEN REGIME 📈📉
**Trigger Conditions:**
- None of the above conditions met
- Usually high spot change (> 0.5%)

**Market Characteristics:**
- **Directional movement dominates** - spot moving significantly
- **Trending market** - up or down, but moving
- **Delta hedging active** - market makers hedging directional risk
- **Breakout/breakdown scenarios**

**Trading Implications:**
- **Trend following works** - ride the momentum
- **Directional options** - buy calls in uptrend, puts in downtrend
- **Delta-hedge your positions** - directional risk is primary
- Good for **spread trading** matching your directional view

**Example:** NIFTY breaks support/resistance, moves 2% in a day. Options move primarily due to spot change (Delta), not time or volatility.

---

## Why "Vega-Driven" with "Gamma Dominant"?

This happens when:
1. **Gamma has the highest GDS score** (highest weighted Greek exposure)
2. **BUT IV is spiking** (>1.2% change) which triggers Vega-Driven regime

**Interpretation:**
- Your **position structure** is gamma-heavy (likely ATM options)
- But the **market environment** is volatility-driven (IV expanding)
- The regime classification prioritizes **what's driving prices** (Vega) over **what your position is sensitive to** (Gamma)

**Real Example:**
- You have ATM straddles (high gamma)
- Market is uncertain, VIX jumps 3 points
- Your P&L moves mostly from **IV expansion** (Vega), not spot moves
- **Regime: Vega-Driven** (market condition)
- **Dominant Greek: Gamma** (position sensitivity)

---

## Confidence Score

The model also gives a **Confidence %** for each regime:
- **70-95%**: High confidence - regime conditions clearly met
- **50-70%**: Medium confidence - borderline conditions
- **<50%**: Low confidence - transitioning between regimes

**Calculation varies by regime:**
- Gamma: Base 70 + gamma_percentile/3
- Vega: Base 60 + abs(IV_change)*5  
- Theta: Base 70 + (1-abs(spot_change))*20
- Delta: Base 50 + abs(spot_change)*10

---

## Trading Strategy Selection by Regime

| Regime | Best Strategies | Avoid |
|--------|----------------|-------|
| **Gamma-Driven** | Range trading, scalping, gamma scalping, short-term hedging | Directional bets, holding overnight |
| **Vega-Driven** | Long straddles/strangles, vol trading, VIX calls | Short premium, naked options |
| **Theta-Driven** | Credit spreads, iron condors, short strangles, theta harvesting | Long options, long gamma |
| **Delta-Driven** | Directional spreads, trend following, delta-hedged positions | Range strategies, theta plays |

---

## Key Takeaways

1. **Dominant Greek** = What your position is most sensitive to (positional metric)
2. **Market Regime** = What's actually driving option prices (market metric)
3. They can differ because position sensitivity ≠ market driver
4. **Trade the regime**, not just the dominant Greek
5. Regime changes signal strategy shifts - be nimble!

---

## Example Scenarios

### Scenario 1: Gamma-Driven + Gamma Dominant ✅
- **Perfect alignment** - gamma is both your risk and market's driver
- Range-bound, choppy market
- ATM options, stable IV
- **Strategy:** Gamma scalping, delta hedging frequently

### Scenario 2: Vega-Driven + Gamma Dominant ⚠️
- **Misalignment** - your gamma exposure in a vol-expanding environment  
- Rising VIX, increasing uncertainty
- Your ATM options gaining mostly from Vega, not Gamma
- **Strategy:** Consider if you want vol exposure or should hedge Vega

### Scenario 3: Delta-Driven + Theta Dominant 🤔
- Strong trend, but your position bleeds theta
- Spot moving 2%/day, but you're short theta (long options)
- **Problem:** Time decay fighting your directional profit
- **Strategy:** Roll to longer dates or switch to spreads

### Scenario 4: Theta-Driven + Delta Dominant ❌  
- Quiet market (good for theta), but you have directional exposure
- If theta-driven continues, your delta positions go nowhere
- **Strategy:** Consider reducing delta, adding theta collection

---

**Remember:** The Greek Regime Model helps you **trade the market as it is**, not as you think it should be!
