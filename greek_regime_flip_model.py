"""
THE GREEK REGIME FLIP MODEL (GRFM)
Greeks-Driven Entry–Exit System for NIFTY Options

Based on: Greek_Regime_Flip_Model_Entry_Exit_System.pdf
"""

import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta

app = dash.Dash(__name__)
app.title = "Greek Regime Flip Model"


class GreekRegimeFlipModel:
    """
    Greek Regime Flip Model for systematic options trading
    Generates entry/exit signals based on Greek dominance
    """
    
    def __init__(self, spot, r=0.07):
        self.spot = spot
        self.r = r
        self.iv_history = []
        self.greek_history = []
        
    def bs_greeks(self, K, T, sigma, otype='call'):
        """Calculate Black-Scholes Greeks"""
        S, r = self.spot, self.r
        if T <= 0 or sigma <= 0:
            return {'price': 0, 'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0}
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if otype == 'call':
            price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            delta = norm.cdf(d1)
            theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2))/365
        else:
            price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
            delta = -norm.cdf(-d1)
            theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2))/365
        
        gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
        vega = S*norm.pdf(d1)*np.sqrt(T)/100
        
        return {'price': price, 'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta}
    
    def normalize_greeks(self, greeks, iv, option_price):
        """
        Step 3: Normalize Greeks for comparison
        Delta_n = |Delta|
        Gamma_n = Gamma × Spot
        Vega_n = Vega / max(IV, 0.01)
        Theta_n = |Theta| / Option Price
        """
        delta_n = abs(greeks['delta'])
        gamma_n = greeks['gamma'] * self.spot
        vega_n = greeks['vega'] / max(iv/100, 0.01)
        theta_n = abs(greeks['theta']) / max(option_price, 0.01) if option_price > 0.01 else 0
        
        return {
            'delta_n': delta_n,
            'gamma_n': gamma_n,
            'vega_n': vega_n,
            'theta_n': theta_n
        }
    
    def calculate_gds(self, normalized):
        """
        Step 4: Greek Dominance Score (GDS)
        GDS_DELTA = 0.45 × Delta_n
        GDS_GAMMA = 0.40 × Gamma_n
        GDS_VEGA = 0.30 × Vega_n
        GDS_THETA = -0.35 × Theta_n
        """
        gds = {
            'DELTA': 0.45 * normalized['delta_n'],
            'GAMMA': 0.40 * normalized['gamma_n'],
            'VEGA': 0.30 * normalized['vega_n'],
            'THETA': -0.35 * normalized['theta_n']
        }
        
        # Find dominant Greek
        dominant = max(gds, key=gds.get)
        dominant_value = gds[dominant]
        
        return {
            'gds_scores': gds,
            'dominant_greek': dominant,
            'dominant_value': dominant_value
        }
    
    def classify_regime(self, gds_scores, iv_change, spot_change, gamma_percentile):
        """
        Step 5: Market Regime Classification
        - GAMMA-DRIVEN: High gamma, small IV change
        - VEGA-DRIVEN: High vega, rising IV
        - THETA-DRIVEN: High theta, range-bound spot
        - DELTA-DRIVEN: Default
        """
        threshold = 0.15  # GDS threshold
        
        if gds_scores['GAMMA'] > threshold and abs(iv_change) < 2.0:
            regime = 'GAMMA-DRIVEN'
            conf = min(95, 70 + gamma_percentile/3)
        elif gds_scores['VEGA'] > threshold and iv_change > 1.2:
            regime = 'VEGA-DRIVEN'
            conf = min(95, 60 + abs(iv_change)*5)
        elif gds_scores['THETA'] > threshold and abs(spot_change) < 0.5:
            regime = 'THETA-DRIVEN'
            conf = min(95, 70 + (1-abs(spot_change))*20)
        else:
            regime = 'DELTA-DRIVEN'
            conf = min(95, 50 + abs(spot_change)*10)
        
        return {
            'regime': regime,
            'confidence': conf,
            'iv_change': iv_change,
            'spot_change': spot_change,
            'gamma_percentile': gamma_percentile
        }
    
    def check_long_gamma_vega_entry(self, regime, delta, gamma_pct, theta_hourly, iv_change):
        """
        Step 6.1: Long Gamma/Vega Breakout Entry
        Conditions:
        - Regime is GAMMA-DRIVEN or VEGA-DRIVEN
        - Delta between 0.45 and 0.55
        - ATM Gamma percentile > 70
        - Theta decay < 1.2% per hour
        - IV increasing by at least 1.2%
        """
        if regime not in ['GAMMA-DRIVEN', 'VEGA-DRIVEN']:
            return {'signal': False, 'reason': f'Regime={regime}'}
        
        if not (0.45 <= abs(delta) <= 0.55):
            return {'signal': False, 'reason': f'Delta={delta:.3f} not 0.45-0.55'}
        
        if gamma_pct <= 70:
            return {'signal': False, 'reason': f'Gamma%ile={gamma_pct:.0f} ≤ 70'}
        
        if theta_hourly >= 1.2:
            return {'signal': False, 'reason': f'Theta={theta_hourly:.2f}% ≥ 1.2%'}
        
        if iv_change < 1.2:
            return {'signal': False, 'reason': f'IV Δ={iv_change:.2f}% < 1.2%'}
        
        return {
            'signal': True,
            'type': 'LONG_GAMMA_VEGA',
            'reason': f'✓ {regime}, Δ={delta:.3f}, Γ%={gamma_pct:.0f}, Θ={theta_hourly:.2f}%, IV↑{iv_change:.2f}%'
        }
    
    def check_vega_expansion_entry(self, front_iv, back_iv, spot_change, vega_score):
        """
        Step 6.2: Vega Expansion (Calendar/Diagonal) Entry
        Conditions:
        - Front-month IV lower than back-month IV by at least 2%
        - Spot price stable
        - Vega dominance score high
        """
        iv_spread = back_iv - front_iv
        
        if iv_spread < 2.0:
            return {'signal': False, 'reason': f'IV Spread={iv_spread:.2f}% < 2%'}
        
        if abs(spot_change) > 0.5:
            return {'signal': False, 'reason': f'Spot not stable: {spot_change:.2f}%'}
        
        if vega_score < 0.15:
            return {'signal': False, 'reason': f'Vega score={vega_score:.3f} low'}
        
        return {
            'signal': True,
            'type': 'VEGA_EXPANSION',
            'reason': f'✓ IV Spread={iv_spread:.2f}%, Spot stable, Vega={vega_score:.3f}'
        }
    
    def check_theta_harvest_entry(self, regime, iv_percentile, delta, theta_hourly):
        """
        Step 6.3: Theta Harvest (Short Premium) Entry
        Conditions:
        - THETA-DRIVEN regime
        - IV percentile above 65
        - Absolute Delta below 0.25
        - Theta decay exceeds 1.8% per hour
        """
        if regime != 'THETA-DRIVEN':
            return {'signal': False, 'reason': f'Regime={regime}'}
        
        if iv_percentile <= 65:
            return {'signal': False, 'reason': f'IV%ile={iv_percentile:.0f} ≤ 65'}
        
        if abs(delta) >= 0.25:
            return {'signal': False, 'reason': f'|Δ|={abs(delta):.3f} ≥ 0.25'}
        
        if theta_hourly < 1.8:
            return {'signal': False, 'reason': f'Θ={theta_hourly:.2f}% < 1.8%'}
        
        return {
            'signal': True,
            'type': 'THETA_HARVEST',
            'reason': f'✓ THETA regime, IV%={iv_percentile:.0f}, |Δ|={abs(delta):.3f}, Θ={theta_hourly:.2f}%'
        }
    
    def check_long_exit(self, gamma_drop_pct, iv_stalled, theta_hourly):
        """
        Step 7: Exit Long Options
        Exit if:
        - Gamma drops more than 25% from peak
        - IV stops rising for two consecutive bars
        - Theta decay exceeds 2% per hour
        """
        exits = []
        
        if gamma_drop_pct > 25:
            exits.append(f'Γ drop {gamma_drop_pct:.1f}%')
        
        if iv_stalled:
            exits.append('IV stalled 2 bars')
        
        if theta_hourly > 2.0:
            exits.append(f'Θ {theta_hourly:.2f}%/hr')
        
        if exits:
            return {'signal': True, 'reason': ' | '.join(exits)}
        
        return {'signal': False, 'reason': 'Hold'}
    
    def check_short_exit(self, gamma_percentile, spot_move_pct, implied_move_pct, iv_expand):
        """
        Step 7: Exit Short Options
        Exit if:
        - ATM Gamma percentile exceeds 80
        - Spot moves more than 60% of implied move
        - IV expands by more than 1.5%
        """
        exits = []
        
        if gamma_percentile > 80:
            exits.append(f'Γ%={gamma_percentile:.0f}')
        
        if abs(spot_move_pct) > 0.6 * implied_move_pct:
            exits.append(f'Spot {spot_move_pct:.2f}% > {0.6*implied_move_pct:.2f}%')
        
        if iv_expand > 1.5:
            exits.append(f'IV↑{iv_expand:.2f}%')
        
        if exits:
            return {'signal': True, 'reason': ' | '.join(exits)}
        
        return {'signal': False, 'reason': 'Hold'}
    
    def calculate_position_size(self, capital, theta, vega, risk_pct=0.6):
        """
        Step 8: Position Sizing (Greeks-Based)
        Position size = Risk / (Theta × 2 + Vega × 1.5)
        Risk per trade = 0.5–0.75% of capital
        """
        risk_amount = capital * (risk_pct / 100)
        greek_risk = abs(theta) * 2 + vega * 1.5
        
        if greek_risk > 0:
            lots = int(risk_amount / greek_risk)
        else:
            lots = 1
        
        return max(1, min(lots, 10))  # Cap between 1-10 lots
    
    def generate_chain(self, iv, dte, n=21):
        """Generate option chain with Greeks"""
        T = dte/365
        sigma = iv/100
        atm = round(self.spot/50)*50
        rng = (n//2)*50
        strikes = np.arange(atm-rng, atm+rng+50, 50)
        
        opts = []
        for K in strikes:
            moneyness = ((K-self.spot)/self.spot)*100
            
            for otype, label in [('call', 'CE'), ('put', 'PE')]:
                greeks = self.bs_greeks(K, T, sigma, otype)
                normalized = self.normalize_greeks(greeks, iv, greeks['price'])
                gds = self.calculate_gds(normalized)
                
                # Theta hourly rate
                theta_hourly = (abs(greeks['theta']) / greeks['price'] * 100) if greeks['price'] > 0.01 else 0
                
                opts.append({
                    'Strike': K,
                    'Type': label,
                    'Moneyness_%': moneyness,
                    'Premium': greeks['price'],
                    'IV': iv,
                    'Delta': greeks['delta'],
                    'Gamma': greeks['gamma'],
                    'Vega': greeks['vega'],
                    'Theta': greeks['theta'],
                    'Delta_n': normalized['delta_n'],
                    'Gamma_n': normalized['gamma_n'],
                    'Vega_n': normalized['vega_n'],
                    'Theta_n': normalized['theta_n'],
                    'GDS_DELTA': gds['gds_scores']['DELTA'],
                    'GDS_GAMMA': gds['gds_scores']['GAMMA'],
                    'GDS_VEGA': gds['gds_scores']['VEGA'],
                    'GDS_THETA': gds['gds_scores']['THETA'],
                    'Dominant': gds['dominant_greek'],
                    'Dominant_Value': gds['dominant_value'],
                    'Theta_Hourly_%': theta_hourly
                })
        
        return pd.DataFrame(opts)


try:
    nifty = pd.read_csv('nifty_history.csv')
    vix = pd.read_csv('india_vix_history.csv')
    SPOT = nifty['Close'].iloc[-1]
    VIX = vix['Close'].iloc[-1]
    SPOT_PREV = nifty['Close'].iloc[-2] if len(nifty) > 1 else SPOT * 0.99
    VIX_PREV = vix['Close'].iloc[-2] if len(vix) > 1 else VIX * 1.02
    print(f"✓ NIFTY={SPOT:.0f}, VIX={VIX:.1f}%")
except:
    SPOT, VIX = 25900, 10.4
    SPOT_PREV, VIX_PREV = 25800, 10.8
    print(f"⚠️ Using defaults: NIFTY={SPOT}, VIX={VIX}")


app.layout = html.Div([
    html.Div([
        html.H1("🎯 GREEK REGIME FLIP MODEL", style={'color':'white','margin':0}),
        html.P("Greeks-Driven Entry–Exit System for NIFTY Options", 
               style={'color':'#ecf0f1','fontSize':16,'marginTop':5})
    ], style={'background':'linear-gradient(135deg, #667eea 0%, #764ba2 100%)','padding':25,'marginBottom':25}),
    
    html.Div([
        html.Div([
            html.Label('NIFTY Spot:', style={'fontWeight':'bold','display':'block','marginBottom':5}),
            dcc.Input(id='spot', type='number', value=SPOT, min=10000, max=100000,
                     style={'width':'100%','padding':8})
        ], style={'flex':1,'marginRight':15}),
        
        html.Div([
            html.Label('Current IV (%):', style={'fontWeight':'bold','display':'block','marginBottom':5}),
            dcc.Input(id='iv', type='number', value=VIX, step=0.1, min=1, max=200,
                     style={'width':'100%','padding':8})
        ], style={'flex':1,'marginRight':15}),
        
        html.Div([
            html.Label('Days to Expiry:', style={'fontWeight':'bold','display':'block','marginBottom':5}),
            dcc.Dropdown(id='dte', 
                        options=[{'label':f'{d} days','value':d} for d in [1,3,7,14,21,30]],
                        value=7, style={'width':'100%'})
        ], style={'flex':1,'marginRight':15}),
        
        html.Div([
            html.Label('Capital (₹):', style={'fontWeight':'bold','display':'block','marginBottom':5}),
            dcc.Input(id='capital', type='number', value=500000, step=10000, 
                     min=10000, max=100000000,
                     style={'width':'100%','padding':8})
        ], style={'flex':1,'marginRight':15}),
        
        html.Div([
            html.Label('\u00a0', style={'display':'block','marginBottom':5}),
            html.Button('🚀 ANALYZE', id='analyze', n_clicks=0,
                       style={'width':'100%','padding':10,'background':'#10b981','color':'white',
                             'border':'none','borderRadius':5,'fontWeight':'bold','fontSize':16,
                             'cursor':'pointer'})
        ], style={'flex':1})
    ], style={'display':'flex','padding':'20px 30px','background':'#f8fafc','borderRadius':10,'marginBottom':25}),
    
    html.Div(id='regime-box', style={'marginBottom':25}),
    
    dcc.Tabs([
        dcc.Tab(label='📊 Entry Signals', children=[
            html.Div(id='entry-signals', style={'padding':25})
        ]),
        
        dcc.Tab(label='🚪 Exit Signals', children=[
            html.Div(id='exit-signals', style={'padding':25})
        ]),
        
        dcc.Tab(label='📈 GDS Analysis', children=[
            html.Div([
                dcc.Graph(id='gds-chart'),
                html.Div([
                    html.Div([dcc.Graph(id='greek-heatmap-ce')], style={'width':'48%','display':'inline-block'}),
                    html.Div([dcc.Graph(id='greek-heatmap-pe')], style={'width':'48%','display':'inline-block','marginLeft':'4%'})
                ])
            ], style={'padding':20})
        ]),
        
        dcc.Tab(label='📋 Full Chain', children=[
            html.Div(id='full-chain', style={'padding':25})
        ])
    ])
])


@app.callback(
    [Output('regime-box','children'),
     Output('entry-signals','children'),
     Output('exit-signals','children'),
     Output('gds-chart','figure'),
     Output('greek-heatmap-ce','figure'),
     Output('greek-heatmap-pe','figure'),
     Output('full-chain','children')],
    [Input('analyze','n_clicks')],
    [State('spot','value'), State('iv','value'), State('dte','value'), State('capital','value')]
)
def analyze(n, spot, iv, dte, capital):
    if n == 0:
        empty = go.Figure()
        empty.update_layout(title="Click ANALYZE to begin")
        return None, None, None, empty, empty, empty, None
    
    # Validate inputs
    validator = SecurityValidator()
    spot = validator.validate_numeric(spot, 10000, 100000, "spot")
    iv = validator.validate_numeric(iv, 1, 200, "IV")
    dte = validator.validate_numeric(dte, 1, 365, "DTE")
    capital = validator.validate_numeric(capital, 10000, 100000000, "capital")
    
    # Initialize model
    model = GreekRegimeFlipModel(spot)
    df = model.generate_chain(iv, dte)
    
    # Calculate regime
    spot_change = ((spot - SPOT_PREV) / SPOT_PREV) * 100
    iv_change = ((iv - VIX_PREV) / VIX_PREV) * 100
    
    # Calculate gamma percentile (simplified)
    atm_strikes = df[(df['Moneyness_%'].abs() <= 1.0)]
    gamma_pct = 75  # Simplified - would need historical data for real percentile
    
    # Get ATM option for GDS calculation
    atm = df.iloc[(df['Moneyness_%'].abs()).argmin()]
    normalized = model.normalize_greeks(
        {'delta': atm['Delta'], 'gamma': atm['Gamma'], 'vega': atm['Vega'], 'theta': atm['Theta']},
        iv, atm['Premium']
    )
    gds = model.calculate_gds(normalized)
    regime_data = model.classify_regime(gds['gds_scores'], iv_change, spot_change, gamma_pct)
    
    # Regime Box
    regime_colors = {
        'DELTA-DRIVEN': '#3b82f6',
        'GAMMA-DRIVEN': '#ef4444',
        'VEGA-DRIVEN': '#a855f7',
        'THETA-DRIVEN': '#10b981'
    }
    
    regime_icons = {
        'DELTA-DRIVEN': '📈',
        'GAMMA-DRIVEN': '⚡',
        'VEGA-DRIVEN': '🌊',
        'THETA-DRIVEN': '⏰'
    }
    
    regime = regime_data['regime']
    regime_box = html.Div([
        html.Div([
            html.Div([
                html.H2(f"{regime_icons[regime]} {regime}", 
                       style={'color':'white','marginBottom':10}),
                html.P(f"Confidence: {regime_data['confidence']:.1f}%", 
                      style={'color':'#ecf0f1','fontSize':20,'marginBottom':15}),
                html.Div([
                    html.P(f"Spot Change: {spot_change:+.2f}%", style={'color':'white','marginBottom':5}),
                    html.P(f"IV Change: {iv_change:+.2f}%", style={'color':'white','marginBottom':5}),
                    html.P(f"Dominant Greek: {gds['dominant_greek']}", 
                          style={'color':'#fbbf24','fontSize':18,'fontWeight':'bold'})
                ])
            ], style={'flex':1,'padding':20}),
            
            html.Div([
                html.H4('Greek Dominance Scores (GDS)', style={'color':'white','marginBottom':15}),
                html.Div([
                    html.Div([
                        html.P(f"DELTA: {gds['gds_scores']['DELTA']:.3f}", 
                              style={'color':'white' if gds['dominant_greek']=='DELTA' else '#cbd5e1','fontSize':14}),
                        html.P(f"GAMMA: {gds['gds_scores']['GAMMA']:.3f}", 
                              style={'color':'white' if gds['dominant_greek']=='GAMMA' else '#cbd5e1','fontSize':14}),
                    ], style={'flex':1}),
                    html.Div([
                        html.P(f"VEGA: {gds['gds_scores']['VEGA']:.3f}", 
                              style={'color':'white' if gds['dominant_greek']=='VEGA' else '#cbd5e1','fontSize':14}),
                        html.P(f"THETA: {gds['gds_scores']['THETA']:.3f}", 
                              style={'color':'white' if gds['dominant_greek']=='THETA' else '#cbd5e1','fontSize':14}),
                    ], style={'flex':1})
                ], style={'display':'flex'})
            ], style={'flex':1,'padding':20,'borderLeft':'2px solid rgba(255,255,255,0.3)'})
        ], style={'display':'flex'})
    ], style={'background':regime_colors[regime],'borderRadius':10,'padding':20,'marginBottom':25,
             'boxShadow':'0 10px 25px rgba(0,0,0,0.2)'})
    
    # Entry Signals
    entry_cards = []
    
    # Check each entry type for ATM options
    atm_opts = df[df['Moneyness_%'].abs() <= 2.0]
    
    for idx, row in atm_opts.iterrows():
        theta_hourly = row['Theta_Hourly_%']
        iv_percentile = 70  # Simplified
        
        # Long Gamma/Vega
        long_entry = model.check_long_gamma_vega_entry(
            regime, row['Delta'], gamma_pct, theta_hourly, iv_change
        )
        
        # Theta Harvest
        theta_entry = model.check_theta_harvest_entry(
            regime, iv_percentile, row['Delta'], theta_hourly
        )
        
        # Position sizing
        lots = model.calculate_position_size(capital, row['Theta'], row['Vega'])
        
        if long_entry['signal'] or theta_entry['signal']:
            signal_type = long_entry['type'] if long_entry['signal'] else theta_entry['type']
            signal_reason = long_entry['reason'] if long_entry['signal'] else theta_entry['reason']
            signal_color = '#10b981' if long_entry['signal'] else '#f59e0b'
            
            entry_cards.append(html.Div([
                html.Div([
                    html.H4(f"{row['Strike']} {row['Type']}", 
                           style={'color':'white','marginBottom':5}),
                    html.P(f"Entry: {signal_type}", 
                          style={'color':'#fbbf24','fontSize':16,'fontWeight':'bold','marginBottom':10}),
                    html.P(signal_reason, style={'color':'#ecf0f1','fontSize':13,'marginBottom':15}),
                    html.Div([
                        html.Div([
                            html.P('Position', style={'color':'#94a3b8','fontSize':11}),
                            html.P(f"{lots} lots", style={'color':'white','fontSize':16,'fontWeight':'bold'})
                        ], style={'flex':1}),
                        html.Div([
                            html.P('Premium', style={'color':'#94a3b8','fontSize':11}),
                            html.P(f"₹{row['Premium']:.2f}", style={'color':'white','fontSize':16,'fontWeight':'bold'})
                        ], style={'flex':1}),
                        html.Div([
                            html.P('Dominant', style={'color':'#94a3b8','fontSize':11}),
                            html.P(row['Dominant'], style={'color':'#fbbf24','fontSize':16,'fontWeight':'bold'})
                        ], style={'flex':1})
                    ], style={'display':'flex','gap':20})
                ], style={'padding':20,'background':signal_color,'borderRadius':8,
                         'marginBottom':15,'boxShadow':'0 4px 6px rgba(0,0,0,0.1)'})
            ]))
    
    if not entry_cards:
        entry_cards = [html.Div([
            html.H3('⏸️ No Entry Signals', style={'color':'#64748b','textAlign':'center','padding':40})
        ])]
    
    entry_section = html.Div(entry_cards)
    
    # Exit Signals (simplified - would need position tracking)
    exit_section = html.Div([
        html.Div([
            html.H3('Exit Criteria', style={'marginBottom':20}),
            html.Div([
                html.Div([
                    html.H4('🟢 LONG OPTIONS', style={'color':'#10b981','marginBottom':15}),
                    html.Ul([
                        html.Li('Gamma drops > 25% from peak', style={'marginBottom':8}),
                        html.Li('IV stops rising for 2 consecutive bars', style={'marginBottom':8}),
                        html.Li('Theta decay exceeds 2% per hour', style={'marginBottom':8})
                    ], style={'fontSize':14,'color':'#475569'})
                ], style={'flex':1,'padding':20,'background':'#f0fdf4','borderRadius':8,'marginRight':15}),
                
                html.Div([
                    html.H4('🔴 SHORT OPTIONS', style={'color':'#ef4444','marginBottom':15}),
                    html.Ul([
                        html.Li('ATM Gamma percentile > 80', style={'marginBottom':8}),
                        html.Li('Spot moves > 60% of implied move', style={'marginBottom':8}),
                        html.Li('IV expands > 1.5%', style={'marginBottom':8})
                    ], style={'fontSize':14,'color':'#475569'})
                ], style={'flex':1,'padding':20,'background':'#fef2f2','borderRadius':8})
            ], style={'display':'flex'})
        ], style={'padding':20,'background':'white','borderRadius':10})
    ])
    
    # GDS Chart
    ce = df[df['Type']=='CE'].sort_values('Strike')
    pe = df[df['Type']=='PE'].sort_values('Strike')
    
    fig_gds = make_subplots(rows=2, cols=2,
                           subplot_titles=('CE - GDS by Greek', 'PE - GDS by Greek',
                                         'CE - Dominant Greek', 'PE - Dominant Greek'),
                           vertical_spacing=0.15)
    
    # GDS stacked bars
    for greek, color in [('GDS_DELTA','#3b82f6'), ('GDS_GAMMA','#ef4444'), 
                         ('GDS_VEGA','#a855f7'), ('GDS_THETA','#10b981')]:
        fig_gds.add_trace(go.Bar(x=ce['Strike'], y=ce[greek], name=greek.replace('GDS_',''),
                                marker_color=color, legendgroup='gds'), row=1, col=1)
        fig_gds.add_trace(go.Bar(x=pe['Strike'], y=pe[greek], name=greek.replace('GDS_',''),
                                marker_color=color, showlegend=False, legendgroup='gds'), row=1, col=2)
    
    # Dominant Greek
    dom_colors_ce = [regime_colors.get(d+'-DRIVEN', '#64748b') for d in ce['Dominant']]
    dom_colors_pe = [regime_colors.get(d+'-DRIVEN', '#64748b') for d in pe['Dominant']]
    
    fig_gds.add_trace(go.Bar(x=ce['Strike'], y=ce['Dominant_Value'], marker_color=dom_colors_ce,
                            text=ce['Dominant'], textposition='outside'), row=2, col=1)
    fig_gds.add_trace(go.Bar(x=pe['Strike'], y=pe['Dominant_Value'], marker_color=dom_colors_pe,
                            text=pe['Dominant'], textposition='outside'), row=2, col=2)
    
    fig_gds.add_vline(x=spot, line_dash="dash", line_color="black", annotation_text="ATM")
    fig_gds.update_layout(height=700, title='Greek Dominance Score Analysis', barmode='stack')
    fig_gds.update_xaxes(title_text="Strike")
    fig_gds.update_yaxes(title_text="GDS")
    
    # Heatmaps
    def create_heatmap(data, title):
        cols = ['Delta_n', 'Gamma_n', 'Vega_n', 'Theta_n']
        z = data[cols].T.values
        return go.Figure(data=go.Heatmap(
            z=z, x=data['Strike'].values, y=cols,
            colorscale='RdYlGn', text=np.round(z, 3),
            texttemplate='%{text}', textfont={"size":9}
        )).update_layout(title=title, height=300, xaxis_title='Strike')
    
    hm_ce = create_heatmap(ce, 'CE - Normalized Greeks')
    hm_pe = create_heatmap(pe, 'PE - Normalized Greeks')
    
    # Full Chain Table
    display_cols = ['Strike','Type','Moneyness_%','Premium','Delta','Gamma','Vega','Theta',
                   'Theta_Hourly_%','Dominant','Dominant_Value']
    
    df_display = df[display_cols].copy()
    for c in ['Moneyness_%','Premium','Delta','Gamma','Vega','Theta','Theta_Hourly_%','Dominant_Value']:
        df_display[c] = df_display[c].round(3)
    
    full_chain_table = dash_table.DataTable(
        data=df_display.to_dict('records'),
        columns=[{'name': c, 'id': c} for c in display_cols],
        style_cell={'textAlign':'center','padding':'10px','fontSize':12},
        style_header={'background':'#1e293b','color':'white','fontWeight':'bold'},
        style_data_conditional=[
            {'if':{'filter_query':'{Dominant}="DELTA"'},'backgroundColor':'#dbeafe'},
            {'if':{'filter_query':'{Dominant}="GAMMA"'},'backgroundColor':'#fee2e2'},
            {'if':{'filter_query':'{Dominant}="VEGA"'},'backgroundColor':'#f3e8ff'},
            {'if':{'filter_query':'{Dominant}="THETA"'},'backgroundColor':'#d1fae5'}
        ],
        filter_action='native',
        sort_action='native',
        page_size=25
    )
    
    return regime_box, entry_section, exit_section, fig_gds, hm_ce, hm_pe, full_chain_table


if __name__ == '__main__':
    print("="*70)
    print("GREEK REGIME FLIP MODEL")
    print("Greeks-Driven Entry–Exit System for NIFTY Options")
    print("="*70)
    print("Dashboard: http://127.0.0.1:8054")
    print("="*70)
    app.run(host='127.0.0.1', port=8054, debug=True)
