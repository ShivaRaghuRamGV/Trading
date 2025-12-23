"""
Greek Regime Flip Dashboard
============================

Entry/Exit system based on Option Greeks analysis
Computes Greek Pressure Index (GPI) for each strike

Author: Trading System
Date: December 2025
"""

import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Greek Regime Flip Dashboard"


class GreekAnalyzer:
    """
    Analyze option Greeks and compute Greek Pressure Index (GPI)
    """
    
    def __init__(self):
        self.spot_price = None
        self.risk_free_rate = 0.07  # 7% for India
        self.options_data = None
        self.vwap = None
        self.iv_prev = None
        self.spot_prev = None
        
    def black_scholes_greeks(self, S, K, T, r, sigma, option_type='call'):
        """
        Calculate Black-Scholes Greeks
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Implied volatility
            option_type: 'call' or 'put'
        
        Returns:
            dict with Greeks
        """
        if T <= 0 or sigma <= 0:
            return {
                'price': 0,
                'delta': 0,
                'gamma': 0,
                'vega': 0,
                'theta': 0,
                'rho': 0
            }
        
        # d1 and d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Price
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    - r * K * np.exp(-r * T) * norm.cdf(d2))
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = -norm.cdf(-d1)
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r * T) * norm.cdf(-d2))
        
        # Gamma (same for call and put)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Vega (same for call and put)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% change in IV
        
        # Theta (annualized, convert to daily)
        theta = theta / 365
        
        # Rho
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }
    
    def generate_option_chain(self, spot, iv, days_to_expiry, num_strikes=21):
        """
        Generate synthetic option chain with Greeks
        
        Args:
            spot: Current NIFTY spot price
            iv: Implied volatility (as decimal, e.g., 0.15 for 15%)
            days_to_expiry: Days until expiry
            num_strikes: Number of strikes to generate
        
        Returns:
            DataFrame with option chain and Greeks
        """
        self.spot_price = spot
        T = days_to_expiry / 365  # Convert to years
        
        # Generate strikes around ATM (±10% in increments of 50)
        atm_strike = round(spot / 50) * 50
        strike_range = (num_strikes // 2) * 50
        strikes = np.arange(atm_strike - strike_range, 
                          atm_strike + strike_range + 50, 
                          50)
        
        options = []
        
        for strike in strikes:
            # Calculate Greeks for Call
            call_greeks = self.black_scholes_greeks(
                spot, strike, T, self.risk_free_rate, iv, 'call'
            )
            
            # Calculate Greeks for Put
            put_greeks = self.black_scholes_greeks(
                spot, strike, T, self.risk_free_rate, iv, 'put'
            )
            
            # Moneyness
            moneyness = (strike - spot) / spot * 100
            
            options.append({
                'Strike': strike,
                'Moneyness_%': moneyness,
                'Type': 'CE',
                'Premium': call_greeks['price'],
                'Delta': call_greeks['delta'],
                'Gamma': call_greeks['gamma'],
                'Vega': call_greeks['vega'],
                'Theta': call_greeks['theta'],
                'IV': iv * 100
            })
            
            options.append({
                'Strike': strike,
                'Moneyness_%': moneyness,
                'Type': 'PE',
                'Premium': put_greeks['price'],
                'Delta': put_greeks['delta'],
                'Gamma': put_greeks['gamma'],
                'Vega': put_greeks['vega'],
                'Theta': put_greeks['theta'],
                'IV': iv * 100
            })
        
        df = pd.DataFrame(options)
        self.options_data = df
        return df
    
    def calculate_greek_pressure_index(self, df=None):
        """
        Calculate Greek Pressure Index (GPI) for each option
        
        GPI = 0.4·Δn + 0.3·Γn + 0.2·Vn − 0.1·Θn
        
        Where:
        - Δn = |Delta|
        - Γn = Gamma / Spot
        - Vn = Vega / IV
        - Θn = |Theta| / Premium
        
        Args:
            df: Options DataFrame
        
        Returns:
            DataFrame with GPI scores
        """
        if df is None:
            df = self.options_data.copy()
        else:
            df = df.copy()
        
        # Normalize Greeks
        df['Delta_n'] = np.abs(df['Delta'])
        df['Gamma_n'] = df['Gamma'] / (self.spot_price / 100)  # Normalized by spot/100
        df['Vega_n'] = df['Vega'] / (df['IV'] / 100)  # Normalized by IV
        
        # Handle division by zero for Theta normalization
        df['Theta_n'] = np.where(
            df['Premium'] > 0.01,
            np.abs(df['Theta']) / df['Premium'],
            0
        )
        
        # Calculate GPI
        df['GPI'] = (
            0.4 * df['Delta_n'] + 
            0.3 * df['Gamma_n'] + 
            0.2 * df['Vega_n'] - 
            0.1 * df['Theta_n']
        )
        
        # Greek Regime Classification
        df['Regime'] = pd.cut(
            df['GPI'],
            bins=[-np.inf, 0.3, 0.6, np.inf],
            labels=['DECAY', 'NEUTRAL', 'MOVEMENT']
        )
        
        # Entry/Exit Signals based on GPI
        df['Signal'] = 'HOLD'
        
        # Movement-sensitive (high GPI) → BUY for directional plays
        df.loc[df['GPI'] > 0.6, 'Signal'] = 'BUY'
        
        # Decay-sensitive (low GPI) → SELL premium
        df.loc[df['GPI'] < 0.3, 'Signal'] = 'SELL'
        
        # Filter to ±2% strikes
        df['Distance_%'] = np.abs(df['Moneyness_%'])
        df_filtered = df[df['Distance_%'] <= 2.0].copy()
        
        self.options_data = df
        return df, df_filtered
    
    def get_top_opportunities(self, df, n=10):
        """
        Get top trading opportunities based on GPI
        
        Args:
            df: Options DataFrame with GPI
            n: Number of top opportunities
        
        Returns:
            Top BUY and SELL opportunities
        """
        # Top BUY signals (highest GPI for movement plays)
        top_buys = df[df['Signal'] == 'BUY'].nlargest(n, 'GPI')
        
        # Top SELL signals (lowest GPI for premium selling)
        top_sells = df[df['Signal'] == 'SELL'].nsmallest(n, 'GPI')
        
        return top_buys, top_sells


class GreekDashboard:
    """
    Dashboard for Greek Regime Flip analysis
    """
    
    def __init__(self):
        self.analyzer = GreekAnalyzer()
        self.app = app
        self.nifty_spot = 24200  # Default
        self.vix = 15  # Default
        
        # Load latest NIFTY data
        self.load_latest_data()
        
        print("="*60)
        print("Greek Regime Flip Dashboard")
        print("="*60)
        print(f"NIFTY Spot: {self.nifty_spot:.2f}")
        print(f"VIX: {self.vix:.2f}")
        print("="*60)
        
        self.build_layout()
        self.setup_callbacks()
    
    def load_latest_data(self):
        """Load latest NIFTY and VIX data"""
        try:
            nifty_df = pd.read_csv('nifty_history.csv')
            vix_df = pd.read_csv('india_vix_history.csv')
            
            self.nifty_spot = nifty_df['Close'].iloc[-1]
            self.vix = vix_df['Close'].iloc[-1]
            
            # Store previous values for regime detection
            if len(nifty_df) > 1:
                self.nifty_prev = nifty_df['Close'].iloc[-2]
            else:
                self.nifty_prev = self.nifty_spot * 0.99
            
            if len(vix_df) > 1:
                self.vix_prev = vix_df['Close'].iloc[-2]
            else:
                self.vix_prev = self.vix * 1.02
            
            # Calculate VWAP (20-day average as proxy)
            if len(nifty_df) >= 20:
                self.vwap = nifty_df['Close'].tail(20).mean()
            else:
                self.vwap = self.nifty_spot * 0.98
            
            print(f"✓ Loaded latest data: NIFTY={self.nifty_spot:.2f}, VIX={self.vix:.2f}")
            print(f"  Previous: NIFTY={self.nifty_prev:.2f}, VIX={self.vix_prev:.2f}")
            print(f"  VWAP (20-day): {self.vwap:.2f}")
        except Exception as e:
            print(f"⚠️  Using default values: {e}")
            self.nifty_prev = self.nifty_spot * 0.99
            self.vix_prev = self.vix * 1.02
            self.vwap = self.nifty_spot * 0.98
    
    def build_layout(self):
        """Build dashboard layout"""
        
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("⚡ Greek Regime Flip Dashboard", 
                       style={'color': 'white', 'marginBottom': 0}),
                html.P("Entry/Exit System Based on Option Greeks Analysis",
                      style={'color': '#bdc3c7', 'fontSize': 16})
            ], style={'backgroundColor': '#34495e', 'padding': '20px 40px', 'marginBottom': 30}),
            
            # Controls
            html.Div([
                html.Div([
                    html.Label('NIFTY Spot:', 
                              style={'fontWeight': 'bold', 'marginRight': 10}),
                    dcc.Input(id='spot-price', type='number', value=self.nifty_spot,
                             style={'width': 120, 'padding': 5, 'marginRight': 30})
                ], style={'display': 'inline-block'}),
                
                html.Div([
                    html.Label('VIX (IV):', 
                              style={'fontWeight': 'bold', 'marginRight': 10}),
                    dcc.Input(id='vix-input', type='number', value=self.vix,
                             style={'width': 100, 'padding': 5, 'marginRight': 30})
                ], style={'display': 'inline-block'}),
                
                html.Div([
                    html.Label('Days to Expiry:', 
                              style={'fontWeight': 'bold', 'marginRight': 10}),
                    dcc.Dropdown(
                        id='dte-select',
                        options=[
                            {'label': '1 Day', 'value': 1},
                rket Regime (Step 2)
            html.Div(id='market-regime', style={'padding': '0 40px', 'marginBottom': 30}),
            
            # Ma            {'label': '3 Days', 'value': 3},
                            {'label': '7 Days (Weekly)', 'value': 7},
                            {'label': '14 Days', 'value': 14},
                            {'label': '21 Days (Monthly)', 'value': 21},
                            {'label': '30 Days', 'value': 30}
                        ],
                        value=7,
                        style={'width': 200, 'marginRight': 30}
                    )
                ], style={'display': 'inline-block'}),
                
                html.Div([
                    html.Button('Analyze Greeks', id='analyze-btn', n_clicks=0,
                               style={'padding': '10px 30px', 'fontSize': 16, 'fontWeight': 'bold',
                                     'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none',
                                     'borderRadius': 5, 'cursor': 'pointer'})
                ], style={'display': 'inline-block'})
            ], style={'padding': '20px 40px', 'backgroundColor': '#ecf0f1', 'marginBottom': 20}),
            
            # Summary Metrics
            html.Div(id='greek-summary', style={'padding': '0 40px', 'marginBottom': 20}),
            
            # Main Content
            dcc.Tabs(id='greek-tabs', value='gpi-tab', children=[
                # Tab 1: GPI Analysis
                dcc.Tab(label='📊 GPI Analysis', value='gpi-tab', children=[
                    html.Div([
                        html.H3('Greek Pressure Index by Strike', 
                               style={'color': '#2c3e50', 'marginBottom': 20}),
                        
                        dcc.Graph(id='gpi-chart'),
                        
                        html.Div([
                            html.Div([
                                html.H4('Calls (CE)', style={'textAlign': 'center', 'color': '#27ae60'}),
                                dcc.Graph(id='gpi-calls-heatmap')
                            ], style={'width': '49%', 'display': 'inline-block'}),
                            
                            html.Div([
                                html.H4('Puts (PE)', style={'textAlign': 'center', 'color': '#e74c3c'}),
                                dcc.Graph(id='gpi-puts-heatmap')
                            ], style={'width': '49%', 'display': 'inline-block', 'marginLeft': '2%'})
                        ])
                    ], style={'padding': 20})
                ]),
                
                # Tab 2: Entry/Exit Signals
                dcc.Tab(label='🎯 Entry/Exit Signals', value='signals-tab', children=[
                    html.Div([
                        html.H3('Trading Opportunities', 
                               style={'color': '#2c3e50', 'marginBottom': 20}),
                        
                        html.Div([
                            html.Div([
                                html.H4('🟢 TOP BUY SIGNALS (Movement Plays)', 
                                       style={'color': '#27ae60', 'marginBottom': 10}),
                                html.P('High GPI (>0.6) - Movement-sensitive options for directional trades',
                                      style={'fontSize': 13, 'color': '#7f8c8d', 'marginBottom': 15}),
                                html.Div(id='buy-signals-table')
                            ], style={'marginBottom': 40}),
                            
                            html.Div([
                                html.H4('🔴 TOP SELL SIGNALS (Premium Decay)', 
                                       style={'color': '#e74c3c', 'marginBottom': 10}),
                                html.P('Low GPI (<0.3) - Decay-sensitive options for premium selling',
                                      style={'fontSize': 13, 'color': '#7f8c8d', 'marginBottom': 15}),
                                html.Div(id='sell-signals-table')
                            ])
                        ])
                    ], style={'padding': 20})
                ]),
                
                # Tab 3: Greeks Breakdown
                dcc.Tab(label='📈 Greeks Breakdown', value='greeks-tab', children=[
                    html.Div([
                        html.H3('Individual Greeks Analysis', 
                               style={'color': '#2c3e50', 'marginBottom': 20}),
                     market-regime', 'children'),
             Output('gpi-chart', 'figure'),
             Output('gpi-calls-heatmap', 'figure'),
             Output('gpi-puts-heatmap', 'figure'),
             Output('buy-signals-table', 'children'),
             Output('sell-signals-table', 'children'),
             Output('delta-chart', 'figure'),
             Output('gamma-chart', 'figure'),
             Output('vega-chart', 'figure'),
             Output('theta-chart', 'figure'),
             Output('option-chain-table', 'children')],
            [Input('analyze-btn', 'n_clicks')],
            [State('spot-price', 'value'),
             State('vix-input', 'value'),
             State('dte-select', 'value')]
        )
        def analyze_greeks(n_clicks, spot, vix, dte):
            if n_clicks == 0:
                empty_fig = go.Figure()
                empty_fig.update_layout(
                    title="Click 'Analyze Greeks' to generate analysis",
                    height=400
                )
                return (None, 
        @self.app.callback(
            [Output('greek-summary', 'children'),
             Output('gpi-chart', 'figure'),
             Output('gpi-calls-heatmap', 'figure'),
             Output('gpi-puts-heatmap', 'figure'),
             Output('buy-signals-table', 'children'),
             Output('sell-signals-table', 'children'),
             Output('delta-chart', 'figure'),
             OIdentify Market Regime (Step 2)
            regime_data = self.analyzer.identify_market_regime(
                df_full, spot, vix, 
                vwap=self.vwap, 
                spot_prev=self.nifty_prev, 
                iv_prev=self.vix_prev
            )
            
            # Summary metrics
            summary = self.create_summary(df_full, df_filtered, spot, vix, dte)
            
            # Market regime display
            regime_display = self.create_regime_display(regime_data
             Output('theta-chart', 'figure'),
             Output('option-chain-table', 'children')],
            [Input('analyze-btn', 'n_clicks')],
            [State('spot-price', 'value'),
             State('vix-input', 'value'),
             State('dte-select', 'value')]
        )
        def analyze_greeks(n_clicks, spot, vix, dte):
            if n_clicks == 0:
                empty_fig = gregime_display, o.Figure()
                empty_fig.update_layout(
                    title="Click 'Analyze Greeks' to generate analysis",
                    height=400
                )
                return (None, empty_fig, empty_fig, empty_fig, None, None, 
                       empty_fig, empty_fig, empty_fig, empty_fig, None)
            
            # Generate option chain
            iv = vix / 100  # Convert VIX to IV decimal
            df = self.analyzer.generate_option_chain(spot, iv, dte, num_strikes=21)
            
            # Calculate GPI
            df_full, df_filtered = self.analyzer.calculate_greek_pressure_index(df)
            
            # Summary metrics
            summary = self.create_summary(df_full, df_filtered, spot, vix, dte)
            
            # Charts
            gpi_chart = self.create_gpi_chart(df_full, spot)
            calls_heatmap = self.create_greek_heatmap(df_full[df_full['Type'] == 'CE'], 'CE')
            puts_heatmap = self.create_greek_heatmap(df_full[df_full['Type'] == 'PE'], 'PE')
            
            # Signals tables
            top_buys, top_sells = self.analyzer.get_top_opportunities(df_filtered, n=10)
            buy_table = self.create_signals_table(top_buys, 'BUY')
            sell_table = self.create_signals_table(top_sells, 'SELL')
            
            # Greeks charts
            delta_chart = self.create_greek_chart(df_full, 'Delta', spot)
            gamma_chart = self.create_greek_chart(df_full, 'Gamma', spot)
            vega_chart = self.create_greek_chart(df_full, 'Vega', spot)
            theta_chart = self.create_greek_chart(df_full, 'Theta', spot)
            
            # Option chain table
            chain_table = self.create_option_chain_table(df_full)
            
            return (summary, gpi_chart, calls_heatmap, puts_heatmap, 
                   buy_table, sell_table, delta_chart, gamma_chart, 
                   vega_chart, theta_chart, chain_table)
    
    def create_summary(self, df_full, df_filtered, spot, vix, dte):
        """Create summary metrics"""
        
        buy_signals = len(df_filtered[df_filtered['Signal'] == 'BUY'])
        sell_signals = len(df_filtered[df_filtered['Signal'] == 'SELL'])
        
        avg_gpi = df_full['GPI'].mean()
        max_gpi = df_full['GPI'].max()
        min_gpi = df_full['GPI'].min()
        
        return html.Div([
            html.Div([
                html.Div([
                    html.P("Current Spot", style={'color': '#7f8c8d', 'fontSize': 14}),
                    html.P(f"₹{spot:,.2f}", 
                          style={'fontSize': 28, 'fontWeight': 'bold', 'color': '#2c3e50'})
                ], style={'flex': 1, 'textAlign': 'center', 'padding': 15, 
                         'backgroundColor': '#ecf0f1', 'borderRadius': 8, 'margin': '0 10px'}),
                
               regime_display(self, regime_data):
        """Create market regime display (Step 2)"""
        
        regime = regime_data['regime']
        confidence = regime_data['confidence']
        strategy = regime_data['strategy_recommendation']
        
        # Color coding by regime
        regime_colors = {
            'Delta-driven': '#3498db',
            'Gamma-driven': '#e74c3c',
            'Vega-driven': '#9b59b6',
            'Theta-driven': '#27ae60'
        }
        
        regime_icons = {
            'Delta-driven': '📈',
            'Gamma-driven': '⚡',
            'Vega-driven': '🌊',
            'Theta-driven': '⏰'
        }
        
        bg_color = regime_colors.get(regime, '#34495e')
        icon = regime_icons.get(regime, '📊')
        
        return html.Div([
            html.Div([
                html.H3([
                    f"{icon} STEP 2: MARKET REGIME CLASSIFICATION",
                ], style={'color': 'white', 'marginBottom': 10, 'textAlign': 'center'}),
                
                html.Div([
                    # Left side - Regime info
                    html.Div([
                        html.H2(regime, style={'color': 'white', 'marginBottom': 5}),
                        html.P(f"Confidence: {confidence:.0f}%", 
                              style={'color': '#ecf0f1', 'fontSize': 18, 'marginBottom': 15}),
                        html.P(strategy['description'], 
                              style={'color': '#ecf0f1', 'fontSize': 14, 'fontStyle': 'italic'})
                    ], style={'flex': 1, 'padding': '0 20px'}),
                    
                    # Right side - Strategy recommendation
                    html.Div([
                        html.Div([
                            html.Strong('🎯 Recommended Strategy:', 
                                      style={'color': '#f39c12', 'fontSize': 16, 'display': 'block', 'marginBottom': 8}),
                            html.P(strategy['strategy'], 
                                  style={'color': 'white', 'fontSize': 15, 'fontWeight': 'bold', 'marginBottom': 10}),
                        ]),
                        html.Div([
                            html.Strong('⚠️ Risk:', style={'color': '#e74c3c', 'fontSize': 14, 'display': 'block', 'marginBottom': 5}),
                            html.P(strategy['risk'], style={'color': '#ecf0f1', 'fontSize': 13})
                        ])
                    ], style={'flex': 1, 'padding': '0 20px', 'borderLeft': '2px solid white'})
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': 20}),
                
                # Market conditions
                html.Div([
                    html.Div([
                        html.Strong('Market Conditions:', style={'color': '#f39c12', 'marginBottom': 10, 'display': 'block'}),
                        html.Div([
                            html.Div([
                                html.Span(f"Spot vs VWAP: ", style={'color': '#bdc3c7'}),
                                html.Span(f"{regime_data['spot_vs_vwap']} (₹{regime_data['vwap']:,.0f})", 
                                         style={'color': 'white', 'fontWeight': 'bold'})
                            ], style={'marginBottom': 5}),
                            html.Div([
                                html.Span(f"Spot Change: ", style={'color': '#bdc3c7'}),
                                html.Span(f"{regime_data['spot_change_pct']:+.2f}%", 
                                         style={'color': '#27ae60' if regime_data['spot_change_pct'] > 0 else '#e74c3c', 
                                               'fontWeight': 'bold'})
                            ], style={'marginBottom': 5}),
                            html.Div([
                                html.Span(f"IV Change: ", style={'color': '#bdc3c7'}),
                                html.Span(f"{regime_data['iv_change_pct']:+.2f}%", 
                                         style={'color': '#27ae60' if regime_data['iv_change_pct'] > 0 else '#e74c3c', 
                                               'fontWeight': 'bold'})
                            ], style={'marginBottom': 5}),
                            html.Div([
                                html.Span(f"ATM Gamma: ", style={'color': '#bdc3c7'}),
                                html.Span(f"{regime_data['atm_gamma']:.4f} {'(SPIKE!)' if regime_data['gamma_spike'] else '(Normal)'}", 
                                         style={'color': '#e74c3c' if regime_data['gamma_spike'] else 'white', 
                                               'fontWeight': 'bold'})
                            ])
                        ])
                    ], style={'flex': 1}),
                    
                    html.Div([
                        html.Strong('Best Plays for This Regime:', style={'color': '#f39c12', 'marginBottom': 10, 'display': 'block'}),
                        html.Ul([
                            html.Li(play, style={'color': 'white', 'marginBottom': 5}) 
                            for play in strategy['best_plays']
                        ], style={'marginLeft': 20})
                    ], style={'flex': 1})
                ], style={'display': 'flex', 'gap': 40})
                
            ], style={'padding': 25, 'backgroundColor': bg_color, 'borderRadius': 10,
                     'boxShadow': '0 4px 6px rgba(0,0,0,0.3)'})
        ])
    
    def create_summary(self, df_full, df_filtered, spot, vix, dte):
        """Create summary metrics"""
        
        buy_signals = len(df_filtered[df_filtered['Signal'] == 'BUY'])
        sell_signals = len(df_filtered[df_filtered['Signal'] == 'SELL'])
        
        avg_gpi = df_full['GPI'].mean()
        max_gpi = df_full['GPI'].max()
        min_gpi = df_full['GPI'].min()
        
        return html.Div([
            html.Div([
                html.Div([
                    html.P("Current Spot", style={'color': '#7f8c8d', 'fontSize': 14}),
                    html.P(f"₹{spot:,.2f}", 
                          style={'fontSize': 28, 'fontWeight': 'bold', 'color': '#2c3e50'})
                ], style={'flex': 1, 'textAlign': 'center', 'padding': 15, 
                         'backgroundColor': '#ecf0f1', 'borderRadius': 8, 'margin': '0 10px'}),
                
                html.Div([
                    html.P("VIX (IV)", style={'color': '#7f8c8d', 'fontSize': 14}),
                    html.P(f"{vix:.2f}%", 
                          style={'fontSize': 28, 'fontWeight': 'bold', 'color': '#e67e22'})
                ], style={'flex': 1, 'textAlign': 'center', 'padding': 15, 
                         'backgroundColor': '#ecf0f1', 'borderRadius': 8, 'margin': '0 10px'}),
                
                html.Div([
                    html.P("Days to Expiry", style={'color': '#7f8c8d', 'fontSize': 14}),
                    html.P(f"{dte}", 
                          style={'fontSize': 28, 'fontWeight': 'bold', 'color': '#9b59b6'})
                ], style={'flex': 1, 'textAlign': 'center', 'padding': 15, 
                         'backgroundColor': '#ecf0f1', 'borderRadius': 8, 'margin': '0 10px'}),
                
                html.Div([
                    html.P("BUY Signals", style={'color': '#7f8c8d', 'fontSize': 14}),
                    html.P(f"{buy_signals}", 
                          style={'fontSize': 28, 'fontWeight': 'bold', 'color': '#27ae60'})
                ], style={'flex': 1, 'textAlign': 'center', 'padding': 15, 
                         'backgroundColor': '#d5f4e6', 'borderRadius': 8, 'margin': '0 10px'}),
                
                html.Div([
                    html.P("SELL Signals", style={'color': '#7f8c8d', 'fontSize': 14}),
                    html.P(f"{sell_signals}", 
                          style={'fontSize': 28, 'fontWeight': 'bold', 'color': '#e74c3c'})
                ], style={'flex': 1, 'textAlign': 'center', 'padding': 15, 
                         'backgroundColor': '#fadbd8', 'borderRadius': 8, 'margin': '0 10px'}),
                
                html.Div([
                    html.P("Avg GPI", style={'color': '#7f8c8d', 'fontSize': 14}),
                    html.P(f"{avg_gpi:.3f}", 
                          style={'fontSize': 28, 'fontWeight': 'bold', 'color': '#3498db'})
                ], style={'flex': 1, 'textAlign': 'center', 'padding': 15, 
                         'backgroundColor': '#ecf0f1', 'borderRadius': 8, 'margin': '0 10px'})
            ], style={'display': 'flex', 'gap': 10})
        ])
    
    def create_gpi_chart(self, df, spot):
        """Create GPI comparison chart"""
        
        calls = df[df['Type'] == 'CE'].sort_values('Strike')
        puts = df[df['Type'] == 'PE'].sort_values('Strike')
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Calls (CE)', 'Puts (PE)'),
            specs=[[{'secondary_y': False}, {'secondary_y': False}]]
        )
        
        # Calls
        fig.add_trace(
            go.Bar(x=calls['Strike'], y=calls['GPI'], 
                  name='GPI (CE)',
                  marker_color=['#27ae60' if gpi > 0.6 else '#f39c12' if gpi > 0.3 else '#e74c3c' 
                               for gpi in calls['GPI']],
                  text=calls['GPI'].round(3),
                  textposition='outside'),
            row=1, col=1
        )
        
        # Puts
        fig.add_trace(
            go.Bar(x=puts['Strike'], y=puts['GPI'], 
                  name='GPI (PE)',
                  marker_color=['#27ae60' if gpi > 0.6 else '#f39c12' if gpi > 0.3 else '#e74c3c' 
                               for gpi in puts['GPI']],
                  text=puts['GPI'].round(3),
                  textposition='outside'),
            row=1, col=2
        )
        
        # Add ATM line
        fig.add_vline(x=spot, line_dash="dash", line_color="black", 
                     annotation_text="ATM", row=1, col=1)
        fig.add_vline(x=spot, line_dash="dash", line_color="black", 
                     annotation_text="ATM", row=1, col=2)
        
        # Add GPI threshold lines
        fig.add_hline(y=0.6, line_dash="dot", line_color="green", 
                     annotation_text="Movement (>0.6)", row=1, col=1)
        fig.add_hline(y=0.3, line_dash="dot", line_color="orange", 
                     annotation_text="Decay (<0.3)", row=1, col=1)
        
        fig.add_hline(y=0.6, line_dash="dot", line_color="green", 
                     annotation_text="Movement (>0.6)", row=1, col=2)
        fig.add_hline(y=0.3, line_dash="dot", line_color="orange", 
                     annotation_text="Decay (<0.3)", row=1, col=2)
        
        fig.update_layout(
            title='Greek Pressure Index (GPI) by Strike',
            height=500,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Strike Price", row=1, col=1)
        fig.update_xaxes(title_text="Strike Price", row=1, col=2)
        fig.update_yaxes(title_text="GPI", row=1, col=1)
        fig.update_yaxes(title_text="GPI", row=1, col=2)
        
        return fig
    
    def create_greek_heatmap(self, df, option_type):
        """Create heatmap of Greeks"""
        
        df_sorted = df.sort_values('Strike')
        
        # Prepare data for heatmap
        greeks = ['Delta_n', 'Gamma_n', 'Vega_n', 'Theta_n', 'GPI']
        heatmap_data = df_sorted[greeks].T
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=df_sorted['Strike'].values,
            y=greeks,
            colorscale='RdYlGn',
            text=np.round(heatmap_data.values, 3),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Value")
        ))
        
        fig.update_layout(
            title=f'{option_type} Greeks Heatmap',
            xaxis_title='Strike Price',
            yaxis_title='Greek',
            height=300
        )
        
        return fig
    
    def create_signals_table(self, df, signal_type):
        """Create signals table"""
        
        if len(df) == 0:
            return html.P(f"No {signal_type} signals found", 
                         style={'textAlign': 'center', 'color': '#95a5a6', 'padding': 20})
        
        # Select columns to display
        display_cols = ['Strike', 'Type', 'Premium', 'Delta', 'Gamma', 'Vega', 'Theta', 
                       'GPI', 'Regime', 'Signal']
        
        df_display = df[display_cols].copy()
        df_display['Premium'] = df_display['Premium'].round(2)
        df_display['Delta'] = df_display['Delta'].round(3)
        df_display['Gamma'] = df_display['Gamma'].round(4)
        df_display['Vega'] = df_display['Vega'].round(2)
        df_display['Theta'] = df_display['Theta'].round(2)
        df_display['GPI'] = df_display['GPI'].round(3)
        
        return dash_table.DataTable(
            data=df_display.to_dict('records'),
            columns=[{'name': col, 'id': col} for col in display_cols],
            style_cell={
                'textAlign': 'center',
                'padding': '10px',
                'fontSize': 12
            },
            style_header={
                'backgroundColor': '#34495e',
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'filter_query': '{Signal} = "BUY"'},
                    'backgroundColor': '#d5f4e6',
                    'color': '#27ae60'
                },
                {
                    'if': {'filter_query': '{Signal} = "SELL"'},
                    'backgroundColor': '#fadbd8',
                    'color': '#e74c3c'
                }
            ],
            page_size=10
        )
    
    def create_greek_chart(self, df, greek_name, spot):
        """Create individual Greek chart"""
        
        calls = df[df['Type'] == 'CE'].sort_values('Strike')
        puts = df[df['Type'] == 'PE'].sort_values('Strike')
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=calls['Strike'], y=calls[greek_name],
            mode='lines+markers',
            name=f'{greek_name} (CE)',
            line=dict(color='#27ae60', width=2),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=puts['Strike'], y=puts[greek_name],
            mode='lines+markers',
            name=f'{greek_name} (PE)',
            line=dict(color='#e74c3c', width=2),
            marker=dict(size=6)
        ))
        
        # Add ATM line
        fig.add_vline(x=spot, line_dash="dash", line_color="black", 
                     annotation_text="ATM")
        
        fig.update_layout(
            title=f'{greek_name} by Strike Price',
            xaxis_title='Strike Price',
            yaxis_title=greek_name,
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def create_option_chain_table(self, df):
        """Create full option chain table"""
        
        # Select and order columns
        display_cols = ['Strike', 'Type', 'Moneyness_%', 'Premium', 'IV',
                       'Delta', 'Gamma', 'Vega', 'Theta',
                       'Delta_n', 'Gamma_n', 'Vega_n', 'Theta_n',
                       'GPI', 'Regime', 'Signal']
        
        df_display = df[display_cols].copy()
        
        # Round numerical columns
        df_display['Moneyness_%'] = df_display['Moneyness_%'].round(2)
        df_display['Premium'] = df_display['Premium'].round(2)
        df_display['IV'] = df_display['IV'].round(2)
        df_display['Delta'] = df_display['Delta'].round(3)
        df_display['Gamma'] = df_display['Gamma'].round(4)
        df_display['Vega'] = df_display['Vega'].round(2)
        df_display['Theta'] = df_display['Theta'].round(2)
        df_display['Delta_n'] = df_display['Delta_n'].round(3)
        df_display['Gamma_n'] = df_display['Gamma_n'].round(3)
        df_display['Vega_n'] = df_display['Vega_n'].round(3)
        df_display['Theta_n'] = df_display['Theta_n'].round(3)
        df_display['GPI'] = df_display['GPI'].round(3)
        
        return dash_table.DataTable(
            data=df_display.to_dict('records'),
            columns=[{'name': col, 'id': col} for col in display_cols],
            style_cell={
                'textAlign': 'center',
                'padding': '8px',
                'fontSize': 11
            },
            style_header={
                'backgroundColor': '#2c3e50',
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'filter_query': '{Regime} = "MOVEMENT"'},
                    'backgroundColor': '#d5f4e6'
                },
                {
                    'if': {'filter_query': '{Regime} = "DECAY"'},
                    'backgroundColor': '#fadbd8'
                },
                {
                    'if': {'column_id': 'GPI'},
                    'fontWeight': 'bold'
                }
            ],
            filter_action='native',
            sort_action='native',
            page_size=20
        )
    
    def run(self, host='127.0.0.1', port=8052, debug=True):
        """Run the dashboard"""
        print(f"\n{'='*60}")
        print("🚀 Greek Regime Flip Dashboard ready!")
        print(f"Opening at http://{host}:{port}")
        print("="*60)
        self.app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    dashboard = GreekDashboard()
    dashboard.run()
