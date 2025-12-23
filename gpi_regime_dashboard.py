"""
Greek Regime Flip Dashboard
Step 1: GPI Calculation
Step 2: Market Regime Classification
"""

import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy.stats import norm

app = dash.Dash(__name__)
app.title = "Greek Regime Flip"


# ============== SECURITY VALIDATOR ==============
class SecurityValidator:
    """Validate and sanitize all inputs"""
    
    @staticmethod
    def validate_numeric(value, min_val, max_val, name="value"):
        """Validate numeric input is within acceptable bounds"""
        try:
            num = float(value)
            if not (min_val <= num <= max_val):
                return min(max(num, min_val), max_val)  # Clamp to range
            if not np.isfinite(num):
                raise ValueError(f"{name} must be a finite number")
            return num
        except (TypeError, ValueError):
            return (min_val + max_val) / 2  # Return midpoint as fallback


class GreekSystem:
    def __init__(self, spot, r=0.07):
        self.spot = spot
        self.r = r
    
    def bs_greeks(self, K, T, sigma, otype='call'):
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
    
    def chain(self, iv, dte, n=21):
        T = dte/365
        sigma = iv/100
        atm = round(self.spot/50)*50
        rng = (n//2)*50
        strikes = np.arange(atm-rng, atm+rng+50, 50)
        
        opts = []
        for K in strikes:
            m = ((K-self.spot)/self.spot)*100
            for t, g in [('CE', self.bs_greeks(K,T,sigma,'call')), ('PE', self.bs_greeks(K,T,sigma,'put'))]:
                opts.append({
                    'Strike': K, 'Type': t, 'Moneyness_%': m, 'Premium': g['price'],
                    'Delta': g['delta'], 'Gamma': g['gamma'], 'Vega': g['vega'], 
                    'Theta': g['theta'], 'IV': iv
                })
        return pd.DataFrame(opts)
    
    def gpi(self, df):
        df = df.copy()
        df['Delta_n'] = np.abs(df['Delta'])
        df['Gamma_n'] = df['Gamma']/self.spot
        df['Vega_n'] = df['Vega']/(df['IV']/100)
        df['Theta_n'] = np.where(df['Premium']>0.01, np.abs(df['Theta'])/df['Premium'], 0)
        df['GPI'] = 0.4*df['Delta_n'] + 0.3*df['Gamma_n'] + 0.2*df['Vega_n'] - 0.1*df['Theta_n']
        df['Class'] = pd.cut(df['GPI'], bins=[-np.inf,0.3,0.6,np.inf], 
                            labels=['DECAY','NEUTRAL','MOVEMENT'])
        df['Signal'] = 'HOLD'
        df.loc[df['GPI']>0.6, 'Signal'] = 'BUY'
        df.loc[df['GPI']<0.3, 'Signal'] = 'SELL'
        df['Dist_%'] = np.abs(df['Moneyness_%'])
        return df
    
    def regime(self, df, iv):
        try:
            nifty = pd.read_csv('nifty_history.csv')
            vix = pd.read_csv('india_vix_history.csv')
            sp = nifty['Close'].iloc[-2] if len(nifty)>1 else self.spot*0.99
            vp = vix['Close'].iloc[-2] if len(vix)>1 else iv*1.02
            vw = nifty['Close'].tail(20).mean() if len(nifty)>=20 else self.spot*0.98
        except:
            sp, vp, vw = self.spot*0.99, iv*1.02, self.spot*0.98
        
        sc = ((self.spot-sp)/sp)*100
        vc = ((iv-vp)/vp)*100
        
        atm = round(self.spot/50)*50
        ao = df[df['Strike']==atm]
        ag = ao['Gamma'].mean() if len(ao)>0 else df['Gamma'].mean()
        avg = df['Gamma'].mean()
        spike = ag > avg*1.5
        
        s = {'Delta-driven':0, 'Gamma-driven':0, 'Vega-driven':0, 'Theta-driven':0}
        c = {k:[] for k in s}
        
        # Delta
        if self.spot>vw:
            s['Delta-driven']+=1
            c['Delta-driven'].append(f"✓ Spot>{vw:.0f}")
        else:
            c['Delta-driven'].append(f"✗ Spot<{vw:.0f}")
        if sc>0.3:
            s['Delta-driven']+=1
            c['Delta-driven'].append(f"✓ Rising+{sc:.2f}%")
        else:
            c['Delta-driven'].append(f"✗ {sc:+.2f}%")
        if not spike:
            s['Delta-driven']+=1
            c['Delta-driven'].append("✓ Γ stable")
        else:
            c['Delta-driven'].append("✗ Γ spike")
        
        # Gamma
        if spike:
            s['Gamma-driven']+=2
            c['Gamma-driven'].append(f"✓ Γ spike {ag:.4f}")
        else:
            c['Gamma-driven'].append(f"✗ No spike")
        if abs(vc)<2:
            s['Gamma-driven']+=1
            c['Gamma-driven'].append(f"✓ IV flat {vc:+.2f}%")
        else:
            c['Gamma-driven'].append(f"✗ IV {vc:+.2f}%")
        
        # Vega
        if vc>3:
            s['Vega-driven']+=2
            c['Vega-driven'].append(f"✓ IV+{vc:.2f}%")
        else:
            c['Vega-driven'].append(f"✗ IV{vc:+.2f}%")
        if abs(vc)>abs(sc):
            s['Vega-driven']+=1
            c['Vega-driven'].append("✓ IV>Spot")
        else:
            c['Vega-driven'].append("✗ Spot>IV")
        
        # Theta
        if vc<-2:
            s['Theta-driven']+=2
            c['Theta-driven'].append(f"✓ IV{vc:.2f}%")
        else:
            c['Theta-driven'].append(f"✗ IV{vc:+.2f}%")
        if abs(sc)<0.5:
            s['Theta-driven']+=1
            c['Theta-driven'].append(f"✓ Range{sc:+.2f}%")
        else:
            c['Theta-driven'].append(f"✗ Move{sc:+.2f}%")
        
        r = max(s, key=s.get)
        conf = (s[r]/3)*100
        
        st = {
            'Delta-driven': {'icon':'📈','color':'#3498db','desc':'Trending','strat':'BUY directional',
                            'trades':['Long Calls/Puts','Spreads'],'risk':'Reversals'},
            'Gamma-driven': {'icon':'⚡','color':'#e74c3c','desc':'Explosive moves',
                            'strat':'BUY straddles, AVOID selling','trades':['Straddles','Strangles'],
                            'risk':'Extreme vol'},
            'Vega-driven': {'icon':'🌊','color':'#9b59b6','desc':'Vol expanding',
                           'strat':'BUY options for IV','trades':['Long options','Calendars'],
                           'risk':'Vol crush'},
            'Theta-driven': {'icon':'⏰','color':'#27ae60','desc':'Range-bound',
                            'strat':'SELL premium','trades':['Iron Condor','Short Strangle'],
                            'risk':'Breakouts'}
        }
        
        return {
            'regime': r, 'conf': conf, 'scores': s, 'conds': c, 'strat': st[r],
            'sc': sc, 'vc': vc, 'vw': vw, 'ag': ag, 'spike': spike
        }


try:
    nifty = pd.read_csv('nifty_history.csv')
    vix = pd.read_csv('india_vix_history.csv')
    SPOT = nifty['Close'].iloc[-1]
    VIX = vix['Close'].iloc[-1]
    print(f"✓ NIFTY={SPOT:.0f}, VIX={VIX:.1f}")
except:
    SPOT, VIX = 24200, 15
    print(f"⚠️ Defaults: NIFTY={SPOT}, VIX={VIX}")


app.layout = html.Div([
    html.Div([
        html.H1("⚡ Greek Regime Flip Dashboard", style={'color':'white','margin':0}),
        html.P("Step 1: GPI | Step 2: Market Regime", style={'color':'#bdc3c7','fontSize':16})
    ], style={'background':'#2c3e50','padding':20,'marginBottom':20}),
    
    html.Div([
        html.Label('NIFTY:', style={'fontWeight':'bold','marginRight':10}),
        dcc.Input(id='spot', type='number', value=SPOT, min=10000, max=100000,
                 style={'width':120,'padding':5,'marginRight':30}),
        html.Label('VIX:', style={'fontWeight':'bold','marginRight':10}),
        dcc.Input(id='vix', type='number', value=VIX, min=1, max=200, step=0.1,
                 style={'width':100,'padding':5,'marginRight':30}),
        html.Label('DTE:', style={'fontWeight':'bold','marginRight':10}),
        dcc.Dropdown(id='dte', options=[{'label':f'{d}d','value':d} for d in [1,3,7,14,21,30]],
                    value=7, style={'width':100,'marginRight':30,'display':'inline-block'}),
        html.Button('Analyze', id='btn', n_clicks=0, style={'padding':'10px 30px','background':'#e74c3c',
                   'color':'white','border':'none','borderRadius':5,'fontWeight':'bold','cursor':'pointer'})
    ], style={'padding':20,'background':'#ecf0f1','marginBottom':20}),
    
    html.Div(id='sum', style={'padding':'0 20px','marginBottom':20}),
    html.Div(id='reg', style={'padding':'0 20px','marginBottom':30}),
    
    dcc.Tabs([
        dcc.Tab(label='📊 GPI', children=[
            html.Div([dcc.Graph(id='gpi'), 
                     html.Div([html.Div([html.H4('CE',style={'textAlign':'center','color':'#27ae60'}),
                                        dcc.Graph(id='hc')],style={'width':'49%','display':'inline-block'}),
                              html.Div([html.H4('PE',style={'textAlign':'center','color':'#e74c3c'}),
                                       dcc.Graph(id='hp')],style={'width':'49%','display':'inline-block','marginLeft':'2%'})])
                    ], style={'padding':20})
        ]),
        dcc.Tab(label='🎯 Signals', children=[
            html.Div([
                html.H3('±2% Strikes Only',style={'color':'#2c3e50','marginBottom':20}),
                html.Div([html.H4('🟢 BUY (GPI>0.6)',style={'color':'#27ae60'}),
                         html.P('Movement-sensitive',style={'fontSize':13,'color':'#7f8c8d','marginBottom':15}),
                         html.Div(id='buy')],style={'marginBottom':40}),
                html.Div([html.H4('🔴 SELL (GPI<0.3)',style={'color':'#e74c3c'}),
                         html.P('Decay-sensitive',style={'fontSize':13,'color':'#7f8c8d','marginBottom':15}),
                         html.Div(id='sell')])
            ],style={'padding':20})
        ]),
        dcc.Tab(label='📋 Chain', children=[
            html.Div([html.H3('Full Option Chain',style={'color':'#2c3e50','marginBottom':20}),
                     html.Div(id='full')],style={'padding':20})
        ])
    ])
])


@app.callback(
    [Output('sum','children'), Output('reg','children'), Output('gpi','figure'),
     Output('hc','figure'), Output('hp','figure'), Output('buy','children'),
     Output('sell','children'), Output('full','children')],
    [Input('btn','n_clicks')],
    [State('spot','value'), State('vix','value'), State('dte','value')]
)
def update(n, spot, vix, dte):
    if n==0:
        e = go.Figure()
        e.update_layout(title="Click Analyze", height=400)
        return None, None, e, e, e, None, None, None
    
    # Validate inputs
    validator = SecurityValidator()
    spot = validator.validate_numeric(spot, 10000, 100000, "NIFTY")
    vix = validator.validate_numeric(vix, 1, 200, "VIX")
    dte = validator.validate_numeric(dte, 1, 365, "DTE")
    
    gs = GreekSystem(spot)
    df = gs.chain(vix, dte)
    df = gs.gpi(df)
    rd = gs.regime(df, vix)
    
    df2 = df[df['Dist_%']<=2.0]
    bc = len(df2[df2['Signal']=='BUY'])
    sc = len(df2[df2['Signal']=='SELL'])
    
    summ = html.Div([html.Div([
        html.Div([html.P("NIFTY",style={'color':'#7f8c8d','fontSize':14}),
                 html.P(f"₹{spot:,.0f}",style={'fontSize':28,'fontWeight':'bold','color':'#2c3e50'})],
                style={'flex':1,'textAlign':'center','padding':15,'background':'#ecf0f1','borderRadius':8}),
        html.Div([html.P("VIX",style={'color':'#7f8c8d','fontSize':14}),
                 html.P(f"{vix:.1f}%",style={'fontSize':28,'fontWeight':'bold','color':'#e67e22'})],
                style={'flex':1,'textAlign':'center','padding':15,'background':'#ecf0f1','borderRadius':8}),
        html.Div([html.P("DTE",style={'color':'#7f8c8d','fontSize':14}),
                 html.P(f"{dte}",style={'fontSize':28,'fontWeight':'bold','color':'#9b59b6'})],
                style={'flex':1,'textAlign':'center','padding':15,'background':'#ecf0f1','borderRadius':8}),
        html.Div([html.P("BUY",style={'color':'#7f8c8d','fontSize':14}),
                 html.P(f"{bc}",style={'fontSize':28,'fontWeight':'bold','color':'#27ae60'})],
                style={'flex':1,'textAlign':'center','padding':15,'background':'#d5f4e6','borderRadius':8}),
        html.Div([html.P("SELL",style={'color':'#7f8c8d','fontSize':14}),
                 html.P(f"{sc}",style={'fontSize':28,'fontWeight':'bold','color':'#e74c3c'})],
                style={'flex':1,'textAlign':'center','padding':15,'background':'#fadbd8','borderRadius':8})
    ],style={'display':'flex','gap':15})])
    
    r, st = rd['regime'], rd['strat']
    regbox = html.Div([html.Div([
        html.H3(f"{st['icon']} STEP 2: {r.upper()}",style={'color':'white','marginBottom':10,'textAlign':'center'}),
        html.Div([
            html.Div([
                html.H2(r,style={'color':'white','marginBottom':5}),
                html.P(f"Confidence: {rd['conf']:.0f}%",style={'color':'#ecf0f1','fontSize':18,'marginBottom':10}),
                html.P(st['desc'],style={'color':'#ecf0f1','fontSize':14,'fontStyle':'italic','marginBottom':15}),
                html.Div([
                    html.Strong('Metrics:',style={'color':'#f39c12','display':'block','marginBottom':8}),
                    html.P(f"Spot: {rd['sc']:+.2f}%",style={'color':'white','fontSize':13,'marginBottom':3}),
                    html.P(f"IV: {rd['vc']:+.2f}%",style={'color':'white','fontSize':13,'marginBottom':3}),
                    html.P(f"VWAP: ₹{rd['vw']:,.0f}",style={'color':'white','fontSize':13,'marginBottom':3}),
                    html.P(f"ATM Γ: {rd['ag']:.4f} {'🔥' if rd['spike'] else ''}",
                          style={'color':'#e74c3c' if rd['spike'] else 'white','fontSize':13,'fontWeight':'bold' if rd['spike'] else 'normal'})
                ])
            ],style={'flex':1,'padding':'0 20px'}),
            html.Div([
                html.Strong('🎯 Strategy:',style={'color':'#f39c12','fontSize':16,'display':'block','marginBottom':8}),
                html.P(st['strat'],style={'color':'white','fontSize':15,'fontWeight':'bold','marginBottom':15}),
                html.Strong('Trades:',style={'color':'#f39c12','fontSize':14,'display':'block','marginBottom':5}),
                html.Ul([html.Li(t,style={'color':'white','fontSize':13,'marginBottom':3}) for t in st['trades']],
                       style={'marginLeft':20,'marginBottom':10}),
                html.Strong('⚠️ Risk:',style={'color':'#e74c3c','fontSize':14,'display':'block','marginBottom':5}),
                html.P(st['risk'],style={'color':'#ecf0f1','fontSize':13})
            ],style={'flex':1,'padding':'0 20px','borderLeft':'2px solid white'})
        ],style={'display':'flex','marginBottom':15}),
        html.Div([
            html.Strong('Conditions:',style={'color':'#f39c12','fontSize':14,'display':'block','marginBottom':10}),
            html.Div([html.Div([html.Strong(f"{k}: {rd['scores'][k]}/3",
                                           style={'color':'white' if k==r else '#95a5a6','fontSize':12,'display':'block','marginBottom':3}),
                               html.Ul([html.Li(x,style={'fontSize':11,'color':'#ecf0f1','marginBottom':2}) for x in rd['conds'][k]],
                                      style={'marginLeft':15})],style={'flex':1,'marginRight':10})
                     for k in ['Delta-driven','Gamma-driven','Vega-driven','Theta-driven']],
                    style={'display':'flex','gap':10})
        ])
    ],style={'padding':25,'background':st['color'],'borderRadius':10,'boxShadow':'0 4px 6px rgba(0,0,0,0.3)'})])
    
    ce = df[df['Type']=='CE'].sort_values('Strike')
    pe = df[df['Type']=='PE'].sort_values('Strike')
    
    fg = make_subplots(rows=1, cols=2, subplot_titles=('CE','PE'))
    cc = ['#27ae60' if x>0.6 else '#f39c12' if x>0.3 else '#e74c3c' for x in ce['GPI']]
    cp = ['#27ae60' if x>0.6 else '#f39c12' if x>0.3 else '#e74c3c' for x in pe['GPI']]
    fg.add_trace(go.Bar(x=ce['Strike'],y=ce['GPI'],marker_color=cc,name='CE',
                       text=ce['GPI'].round(3),textposition='outside'),row=1,col=1)
    fg.add_trace(go.Bar(x=pe['Strike'],y=pe['GPI'],marker_color=cp,name='PE',
                       text=pe['GPI'].round(3),textposition='outside'),row=1,col=2)
    fg.add_vline(x=spot,line_dash="dash",line_color="black",annotation_text="ATM",row=1,col=1)
    fg.add_vline(x=spot,line_dash="dash",line_color="black",annotation_text="ATM",row=1,col=2)
    fg.add_hline(y=0.6,line_dash="dot",line_color="green",annotation_text=">0.6",row=1,col=1)
    fg.add_hline(y=0.3,line_dash="dot",line_color="orange",annotation_text="<0.3",row=1,col=1)
    fg.add_hline(y=0.6,line_dash="dot",line_color="green",row=1,col=2)
    fg.add_hline(y=0.3,line_dash="dot",line_color="orange",row=1,col=2)
    fg.update_layout(title='GPI by Strike',height=500,showlegend=False)
    fg.update_xaxes(title_text="Strike",row=1,col=1)
    fg.update_xaxes(title_text="Strike",row=1,col=2)
    fg.update_yaxes(title_text="GPI",row=1,col=1)
    fg.update_yaxes(title_text="GPI",row=1,col=2)
    
    def hm(data, ttl):
        g = ['Delta_n','Gamma_n','Vega_n','Theta_n','GPI']
        h = data[g].T
        return go.Figure(data=go.Heatmap(z=h.values, x=data['Strike'].values, y=g,
                                        colorscale='RdYlGn', text=np.round(h.values,3),
                                        texttemplate='%{text}', textfont={"size":9})).update_layout(title=ttl,height=300,xaxis_title='Strike')
    
    hc = hm(ce, 'CE Heatmap')
    hp = hm(pe, 'PE Heatmap')
    
    def tbl(data, sig):
        if len(data)==0:
            return html.P(f"No {sig}",style={'textAlign':'center','color':'#95a5a6','padding':20})
        cols = ['Strike','Type','Premium','Delta','Gamma','Vega','Theta','GPI','Class','Signal']
        d = data[cols].copy()
        for c in ['Premium','Delta','Gamma','Vega','Theta','GPI']:
            d[c] = d[c].round(3)
        return dash_table.DataTable(
            data=d.to_dict('records'),
            columns=[{'name':c,'id':c} for c in cols],
            style_cell={'textAlign':'center','padding':'10px','fontSize':12},
            style_header={'background':'#34495e','color':'white','fontWeight':'bold'},
            style_data_conditional=[
                {'if':{'filter_query':'{Signal}="BUY"'},'background':'#d5f4e6','color':'#27ae60'},
                {'if':{'filter_query':'{Signal}="SELL"'},'background':'#fadbd8','color':'#e74c3c'}
            ],
            page_size=10
        )
    
    bd = df2[df2['Signal']=='BUY'].nlargest(10,'GPI')
    sd = df2[df2['Signal']=='SELL'].nsmallest(10,'GPI')
    bt = tbl(bd, 'BUY')
    st = tbl(sd, 'SELL')
    
    cols = ['Strike','Type','Moneyness_%','Premium','IV','Delta','Gamma','Vega','Theta',
           'Delta_n','Gamma_n','Vega_n','Theta_n','GPI','Class','Signal']
    dd = df[cols].copy()
    for c in ['Moneyness_%','Premium','IV','Delta','Gamma','Vega','Theta','Delta_n','Gamma_n','Vega_n','Theta_n','GPI']:
        dd[c] = dd[c].round(3)
    ft = dash_table.DataTable(
        data=dd.to_dict('records'),
        columns=[{'name':c,'id':c} for c in cols],
        style_cell={'textAlign':'center','padding':'8px','fontSize':11},
        style_header={'background':'#2c3e50','color':'white','fontWeight':'bold'},
        style_data_conditional=[
            {'if':{'filter_query':'{Class}="MOVEMENT"'},'background':'#d5f4e6'},
            {'if':{'filter_query':'{Class}="DECAY"'},'background':'#fadbd8'},
            {'if':{'column_id':'GPI'},'fontWeight':'bold'}
        ],
        filter_action='native',
        sort_action='native',
        page_size=20
    )
    
    return summ, regbox, fg, hc, hp, bt, st, ft


if __name__ == '__main__':
    print("="*60)
    print("Greek Regime Flip Dashboard")
    print("="*60)
    print("http://127.0.0.1:8053")
    print("="*60)
    app.run(host='127.0.0.1', port=8053, debug=True)
