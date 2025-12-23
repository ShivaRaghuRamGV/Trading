"""
Comprehensive Trading Dashboard
Interactive dashboard for NIFTY 50 and INDIA VIX analysis
"""

import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Input, Output, State, dash_table
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from analysis import TradingAnalyzer
from event_analysis import EventAnalyzer
from hypothesis_parser import HypothesisParser


# ============== SECURITY VALIDATOR ==============
class SecurityValidator:
    """Validate and sanitize all inputs"""
    
    @staticmethod
    def validate_numeric(value, min_val, max_val, name="value"):
        """Validate numeric input is within acceptable bounds"""
        try:
            num = float(value)
            if not (min_val <= num <= max_val):
                return min(max(num, min_val), max_val)
            if not np.isfinite(num):
                raise ValueError(f"{name} must be a finite number")
            return num
        except (TypeError, ValueError):
            return (min_val + max_val) / 2
    
    @staticmethod
    def validate_file_path(path, allowed_files):
        """Validate file path is in allowed list"""
        import os
        base_name = os.path.basename(path)
        if base_name not in allowed_files:
            raise ValueError(f"File {base_name} not allowed")
        return path


class TradingDashboard:
    def __init__(self, nifty_df, vix_df):
        """Initialize dashboard with data"""
        self.nifty_df = nifty_df
        self.vix_df = vix_df
        self.analyzer = TradingAnalyzer(nifty_df, vix_df)
        self.event_analyzer = EventAnalyzer(self.analyzer.merged_df)
        
        # Pre-calculate all analyses
        self._prepare_data()
        
        # Initialize Dash app
        self.app = Dash(__name__, suppress_callback_exceptions=True)
        self.setup_layout()
        self.setup_callbacks()
    
    def _prepare_data(self):
        """Pre-calculate all analysis results"""
        print("Preparing dashboard data...")
        
        # Remove timezone info from dates if present
        if hasattr(self.analyzer.merged_df['Date'].dtype, 'tz') and self.analyzer.merged_df['Date'].dtype.tz is not None:
            self.analyzer.merged_df['Date'] = self.analyzer.merged_df['Date'].dt.tz_localize(None)
        
        # Basic returns
        self.df_returns, self.weekly, self.monthly = self.analyzer.calculate_returns()
        
        # Drawdowns
        self.df_dd, self.max_dd, self.max_dd_date = self.analyzer.calculate_drawdowns()
        
        # Trend analysis
        self.df_trend = self.analyzer.label_trend()
        
        # VIX analysis
        self.df_vix = self.analyzer.analyze_vix()
        
        # Correlation
        self.df_corr, self.overall_corr = self.analyzer.correlation_analysis()
        
        # Lead-lag
        self.lead_lag_df = self.analyzer.lead_lag_analysis()
        
        # Granger causality
        self.granger_results = self.analyzer.granger_causality()
        
        # IV-RV spread
        self.df_iv_rv = self.analyzer.calculate_iv_rv_spread()
        
        # Tail risk
        self.nifty_stats, self.vix_stats, self.extreme_moves = self.analyzer.tail_risk_analysis()
        
        # Trading insights
        self.insights, self.latest_data = self.analyzer.generate_trading_insights()
        
        # Event analysis
        self.df_events, self.event_results = self.event_analyzer.analyze_event_impact()
        self.df_expiry, self.expiry_stats, self.expiry_day_stats, self.vix_decay = \
            self.event_analyzer.analyze_expiry_behavior()
        
        # Strategy backtests - Don't run at startup, run on-demand in callback
        print("\nPreparing strategy backtests...")
        
        # Just initialize placeholders - backtests will run on user request
        self.df_strangle = None
        self.strangle_metrics = None
        self.df_straddle = None
        self.straddle_metrics = None
        self.df_condor = None
        self.condor_metrics = None
        
        # Get options data date range for strategy tab
        self.options_min_date = None
        self.options_max_date = None
        if hasattr(self.event_analyzer, 'df') and hasattr(self.event_analyzer.df, 'index'):
            try:
                from options_backtester import OptionsBacktester
                temp_backtester = OptionsBacktester(self.event_analyzer.df)
                if temp_backtester.ce_data is not None and temp_backtester.pe_data is not None:
                    self.options_min_date = max(temp_backtester.ce_date_min, temp_backtester.pe_date_min)
                    self.options_max_date = min(temp_backtester.ce_date_max, temp_backtester.pe_date_max)
            except:
                pass
        
        # If options data not available, use full NIFTY range
        if self.options_min_date is None:
            self.options_min_date = self.nifty_df['Date'].min()
            self.options_max_date = self.nifty_df['Date'].max()
        
        print("✓ Data preparation complete")
    
    def _convert_trades_to_daily(self, trades_df, daily_df):
        """Convert trade-level data to daily format for charting"""
        # Create daily dataframe with cumulative P&L
        result = daily_df[['Date']].copy()
        result['Strategy_Return'] = 0.0
        result['Cumulative_Strategy_Return'] = 0.0
        result['Signal'] = 0
        
        # Map trade P&L to exit dates
        for _, trade in trades_df.iterrows():
            exit_date = pd.Timestamp(trade['Exit_Date'])
            mask = result['Date'] == exit_date
            if mask.any():
                # Calculate return as percentage of spot price
                pnl_pct = trade['Return_%']
                result.loc[mask, 'Strategy_Return'] = pnl_pct
                result.loc[mask, 'Signal'] = 1
        
        # Calculate cumulative returns
        result['Cumulative_Strategy_Return'] = result['Strategy_Return'].cumsum()
        
        return result
    
    def setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = html.Div([
            html.Div([
                html.H1("📊 NIFTY 50 & INDIA VIX Trading Dashboard", 
                       style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 10}),
                html.P(f"Data Range: {self.df_returns['Date'].min().date()} to {self.df_returns['Date'].max().date()}",
                      style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': 14}),
            ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': 20}),
            
            # Trading Insights Section
            html.Div([
                html.H2("🎯 Current Trading Insights", style={'color': '#2c3e50'}),
                html.Div(id='current-insights', style={
                    'backgroundColor': '#fff', 'padding': 20, 'borderRadius': 10,
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'marginBottom': 20
                }),
            ]),
            
            # Key Metrics Row
            html.Div([
                html.Div([
                    html.H3("NIFTY 50", style={'textAlign': 'center', 'color': '#27ae60'}),
                    html.P(f"{self.latest_data['Close_nifty']:.2f}", 
                          style={'fontSize': 32, 'fontWeight': 'bold', 'textAlign': 'center'}),
                    html.P(f"Trend: {self.latest_data['Trend']}", 
                          style={'textAlign': 'center', 'color': '#7f8c8d'}),
                ], style={'flex': 1, 'backgroundColor': '#fff', 'padding': 20, 'margin': 10, 
                         'borderRadius': 10, 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
                
                html.Div([
                    html.H3("INDIA VIX", style={'textAlign': 'center', 'color': '#e74c3c'}),
                    html.P(f"{self.latest_data['Close_vix']:.2f}", 
                          style={'fontSize': 32, 'fontWeight': 'bold', 'textAlign': 'center'}),
                    html.P(f"Regime: {self.latest_data['VIX_Regime']}", 
                          style={'textAlign': 'center', 'color': '#7f8c8d'}),
                ], style={'flex': 1, 'backgroundColor': '#fff', 'padding': 20, 'margin': 10,
                         'borderRadius': 10, 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
                
                html.Div([
                    html.H3("Max Drawdown", style={'textAlign': 'center', 'color': '#e67e22'}),
                    html.P(f"{self.max_dd:.2f}%", 
                          style={'fontSize': 32, 'fontWeight': 'bold', 'textAlign': 'center'}),
                    html.P(f"Date: {self.max_dd_date.date()}", 
                          style={'textAlign': 'center', 'color': '#7f8c8d'}),
                ], style={'flex': 1, 'backgroundColor': '#fff', 'padding': 20, 'margin': 10,
                         'borderRadius': 10, 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
                
                html.Div([
                    html.H3("Correlation", style={'textAlign': 'center', 'color': '#9b59b6'}),
                    html.P(f"{self.overall_corr:.3f}", 
                          style={'fontSize': 32, 'fontWeight': 'bold', 'textAlign': 'center'}),
                    html.P("NIFTY vs VIX", 
                          style={'textAlign': 'center', 'color': '#7f8c8d'}),
                ], style={'flex': 1, 'backgroundColor': '#fff', 'padding': 20, 'margin': 10,
                         'borderRadius': 10, 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': 20}),
            
            # Date range selector and frequency selector
            html.Div([
                html.Div([
                    html.Label('Select Date Range:', style={'fontWeight': 'bold', 'marginRight': 10}),
                    dcc.DatePickerRange(
                        id='date-range',
                        start_date=self.df_returns['Date'].max() - timedelta(days=365),
                        end_date=self.df_returns['Date'].max(),
                        min_date_allowed=self.df_returns['Date'].min(),
                        max_date_allowed=self.df_returns['Date'].max(),
                        display_format='YYYY-MM-DD'
                    ),
                ], style={'display': 'inline-block', 'marginRight': 30}),
                html.Div([
                    html.Label('Frequency:', style={'fontWeight': 'bold', 'marginRight': 10}),
                    dcc.Dropdown(
                        id='frequency-selector',
                        options=[
                            {'label': 'Daily (1D)', 'value': '1D'},
                            {'label': 'Weekly (1W)', 'value': '1W'},
                            {'label': 'Monthly (1M)', 'value': '1M'}
                        ],
                        value='1D',
                        clearable=False,
                        style={'width': 200}
                    ),
                ], style={'display': 'inline-block', 'verticalAlign': 'top'}),
            ], style={'marginBottom': 20, 'padding': 10, 'backgroundColor': '#fff', 
                     'borderRadius': 10, 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            # Tabs for different analyses
            dcc.Tabs([
                # Tab 1: Price & Returns
                dcc.Tab(label='📈 Price & Returns', children=[
                    dcc.Graph(id='price-chart'),
                    dcc.Graph(id='returns-distribution'),
                    dcc.Graph(id='rolling-returns'),
                ]),
                
                # Tab 2: VIX Analysis
                dcc.Tab(label='📊 VIX Analysis', children=[
                    dcc.Graph(id='vix-chart'),
                    dcc.Graph(id='vix-regime-pie'),
                    dcc.Graph(id='vix-histogram'),
                ]),
                
                # Tab 3: Correlation & Lead-Lag
                dcc.Tab(label='🔗 Correlation Analysis', children=[
                    dcc.Graph(id='correlation-chart'),
                    dcc.Graph(id='lead-lag-chart'),
                    dcc.Graph(id='scatter-nifty-vix'),
                ]),
                
                # Tab 4: IV-RV Spread
                dcc.Tab(label='📉 IV-RV Analysis', children=[
                    dcc.Graph(id='iv-rv-spread-chart'),
                    dcc.Graph(id='iv-rv-zscore-chart'),
                    dcc.Graph(id='realized-vol-comparison'),
                ]),
                
                # Tab 5: Event Analysis
                dcc.Tab(label='📅 Event Analysis', children=[
                    dcc.Graph(id='expiry-analysis'),
                    dcc.Graph(id='vix-decay-chart'),
                    html.Div(id='event-stats'),
                ]),
                
                # Tab 6: Strategy Backtests
                dcc.Tab(label='💰 Strategy Backtests', children=[
                    html.Div([
                        html.Div([
                            html.Label('Select Strategy:', style={'fontWeight': 'bold', 'marginRight': 10}),
                            dcc.Dropdown(
                                id='strategy-selector',
                                options=[
                                    {'label': 'Short Strangle', 'value': 'short_strangle'},
                                    {'label': 'Long Straddle', 'value': 'long_straddle'},
                                    {'label': 'Iron Condor', 'value': 'iron_condor'}
                                ],
                                value='short_strangle',
                                style={'width': 300}
                            ),
                        ], style={'display': 'inline-block', 'marginRight': 30}),
                        html.Div([
                            html.Label('Backtest Date Range (Max 6 months):', style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': 5}),
                            dcc.DatePickerRange(
                                id='strategy-date-range',
                                min_date_allowed=self.options_min_date,
                                max_date_allowed=self.options_max_date,
                                start_date=self.options_max_date - pd.Timedelta(days=180),
                                end_date=self.options_max_date,
                                display_format='DD-MMM-YYYY',
                                style={'display': 'inline-block', 'marginRight': 10}
                            ),
                            html.Button('Load', id='strategy-load-btn', n_clicks=0,
                                       style={'padding': '10px 30px', 'fontSize': 16, 'fontWeight': 'bold',
                                              'backgroundColor': '#3498db', 'color': 'white', 'border': 'none',
                                              'borderRadius': 5, 'cursor': 'pointer', 'verticalAlign': 'middle'}),
                        ], style={'display': 'inline-block', 'verticalAlign': 'top'}),
                    ], style={'padding': 20, 'backgroundColor': '#f8f9fa', 'borderRadius': 5}),
                    
                    # Strategy Parameters - will be dynamically updated based on strategy selection
                    html.Div(id='strategy-params', style={'padding': 20, 'backgroundColor': '#fff', 'borderRadius': 5, 'margin': '0 20px 20px 20px'}),
                    
                    # Hidden placeholder inputs for Dash callback validation
                    html.Div([
                        dcc.Input(id='vix-threshold', type='hidden', value=18),
                        dcc.Input(id='strike-distance', type='hidden', value=200),
                        dcc.Input(id='dte-entry', type='hidden', value=7),
                        dcc.Input(id='dte-exit', type='hidden', value=1),
                        dcc.Input(id='long-strike-distance', type='hidden', value=400),
                    ], style={'display': 'none'}),
                    
                    html.Div(id='strategy-date-warning', style={'padding': '10px 20px', 'color': '#e74c3c'}),
                    html.Div(id='strategy-metrics'),
                    dcc.Graph(id='strategy-equity-curve'),
                    dcc.Graph(id='strategy-signals'),
                ]),
                
                # Tab 7: Risk Analysis
                dcc.Tab(label='⚠️ Risk Analysis', children=[
                    dcc.Graph(id='drawdown-chart'),
                    dcc.Graph(id='tail-risk-chart'),
                    html.Div(id='risk-metrics'),
                ]),
                
                # Tab 8: Hypothesis Testing
                dcc.Tab(label='🔬 Hypothesis Testing', children=[
                    html.Div([
                        html.H3('Test Your Trading Hypothesis', style={'color': '#2c3e50', 'marginBottom': 20}),
                        html.P('Enter your hypothesis in plain English. The system will understand natural language!', 
                              style={'color': '#7f8c8d', 'marginBottom': 10, 'fontSize': 16}),
                        html.P('Example hypotheses you can test:', 
                              style={'color': '#7f8c8d', 'marginBottom': 10, 'fontWeight': 'bold'}),
                        html.Ul([
                            html.Li('nifty return is greater than 1%', style={'color': '#7f8c8d', 'marginBottom': 5}),
                            html.Li('vix is above 20', style={'color': '#7f8c8d', 'marginBottom': 5}),
                            html.Li('nifty return is more than mean + 2 standard deviations', style={'color': '#7f8c8d', 'marginBottom': 5}),
                            html.Li('last month return is positive and this month return is also positive', style={'color': '#7f8c8d', 'marginBottom': 5}),
                            html.Li('vix change is positive', style={'color': '#7f8c8d', 'marginBottom': 5}),
                            html.Li('nifty return is between 0.5 and 2', style={'color': '#7f8c8d', 'marginBottom': 5}),
                        ], style={'marginBottom': 20}),
                        
                        html.Div([
                            html.Div([
                                html.Label('Hypothesis:', style={'fontWeight': 'bold', 'marginBottom': 5, 'display': 'block'}),
                                dcc.Input(
                                    id='hypothesis-input',
                                    type='text',
                                    placeholder='e.g., nifty return is more than mean + 2 standard deviations',
                                    style={'width': '100%', 'padding': 10, 'fontSize': 14, 'borderRadius': 5}
                                ),
                            ], style={'flex': 3, 'marginRight': 20}),
                            
                            html.Div([
                                html.Label('Timeframe:', style={'fontWeight': 'bold', 'marginBottom': 5, 'display': 'block'}),
                                dcc.Dropdown(
                                    id='hypothesis-timeframe',
                                    options=[
                                        {'label': 'Daily', 'value': 'daily'},
                                        {'label': 'Weekly', 'value': 'weekly'},
                                        {'label': 'Monthly', 'value': 'monthly'}
                                    ],
                                    value='daily',
                                    clearable=False,
                                    style={'width': 200}
                                ),
                            ], style={'flex': 1, 'marginRight': 20}),
                            
                            html.Div([
                                html.Label('\u00A0', style={'display': 'block', 'marginBottom': 5}),
                                html.Button('Test Hypothesis', id='test-hypothesis-btn', n_clicks=0,
                                          style={'padding': '10px 20px', 'fontSize': 14, 'fontWeight': 'bold',
                                                'backgroundColor': '#3498db', 'color': 'white', 'border': 'none',
                                                'borderRadius': 5, 'cursor': 'pointer'}),
                            ], style={'flex': 1}),
                        ], style={'display': 'flex', 'alignItems': 'flex-end', 'marginBottom': 30}),
                        
                        html.Div([
                            html.P('You can use natural language like "is greater than", "is above", "is more than", "is positive", etc.',
                                  style={'fontSize': 12, 'color': '#95a5a6', 'fontStyle': 'italic', 'marginBottom': 5}),
                            html.P('Supported: nifty, nifty return, vix, vix change, mean, standard deviation, median, last month/week/day, this month/week/day',
                                  style={'fontSize': 12, 'color': '#95a5a6', 'fontStyle': 'italic'}),
                        ]),
                    ], style={'padding': 20, 'backgroundColor': '#fff', 'borderRadius': 10, 'marginBottom': 20}),
                    
                    html.Div(id='hypothesis-results', style={'padding': 20}),
                    dcc.Graph(id='hypothesis-chart'),
                    dcc.Graph(id='hypothesis-distribution'),
                ]),
            ]),
        ], style={'fontFamily': 'Arial, sans-serif', 'padding': '20px', 'backgroundColor': '#f5f6fa'})
    
    def setup_callbacks(self):
        """Setup all dashboard callbacks"""
        
        @self.app.callback(
            Output('current-insights', 'children'),
            Input('date-range', 'start_date')
        )
        def update_insights(start_date):
            insights_html = []
            for insight in self.insights:
                insights_html.append(html.P(insight, style={
                    'fontSize': 16, 'margin': 10, 'padding': 10,
                    'backgroundColor': '#e8f8f5', 'borderLeft': '4px solid #16a085',
                    'borderRadius': 5
                }))
            return insights_html
        
        @self.app.callback(
            [Output('price-chart', 'figure'),
             Output('vix-chart', 'figure'),
             Output('correlation-chart', 'figure')],
            [Input('date-range', 'start_date'),
             Input('date-range', 'end_date'),
             Input('frequency-selector', 'value')]
        )
        def update_main_charts(start_date, end_date, frequency):
            # Filter data
            mask = (self.df_corr['Date'] >= start_date) & (self.df_corr['Date'] <= end_date)
            df_filtered = self.df_corr[mask].copy()
            
            # Resample based on frequency
            if frequency != '1D':
                df_filtered = df_filtered.set_index('Date')
                freq_map = {'1W': 'W-FRI', '1M': 'ME'}
                df_filtered = df_filtered.resample(freq_map[frequency]).agg({
                    'Close_nifty': 'last',
                    'Close_vix': 'last',
                    'Rolling_Corr_30': 'last'
                }).dropna().reset_index()
            else:
                df_filtered = df_filtered.copy()
            
            # Ensure we have the Drawdown column
            if 'Drawdown' not in df_filtered.columns:
                df_filtered['Cumulative_Max'] = df_filtered['Close_nifty'].cummax()
                df_filtered['Drawdown'] = ((df_filtered['Close_nifty'] - df_filtered['Cumulative_Max']) / df_filtered['Cumulative_Max']) * 100
            
            # Price chart with drawdown
            fig_price = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.7, 0.3],
                subplot_titles=('NIFTY 50 Price', 'Drawdown %')
            )
            
            fig_price.add_trace(
                go.Scatter(x=df_filtered['Date'], y=df_filtered['Close_nifty'],
                          name='NIFTY 50', line=dict(color='#2ecc71', width=2)),
                row=1, col=1
            )
            
            fig_price.add_trace(
                go.Scatter(x=df_filtered['Date'], y=df_filtered['Drawdown'],
                          name='Drawdown', fill='tozeroy', 
                          line=dict(color='#e74c3c', width=1)),
                row=2, col=1
            )
            
            fig_price.update_layout(height=600, showlegend=True, 
                                   title_text="NIFTY 50 Price & Drawdown")
            
            # VIX chart with regimes
            fig_vix = go.Figure()
            
            fig_vix.add_trace(go.Scatter(
                x=df_filtered['Date'], y=df_filtered['Close_vix'],
                name='India VIX', line=dict(color='#e74c3c', width=2)
            ))
            
            # Add regime zones
            fig_vix.add_hrect(y0=0, y1=12, fillcolor="green", opacity=0.1, 
                             annotation_text="Low VIX", annotation_position="top left")
            fig_vix.add_hrect(y0=12, y1=18, fillcolor="yellow", opacity=0.1,
                             annotation_text="Normal", annotation_position="top left")
            fig_vix.add_hrect(y0=18, y1=25, fillcolor="orange", opacity=0.1,
                             annotation_text="High", annotation_position="top left")
            fig_vix.add_hrect(y0=25, y1=100, fillcolor="red", opacity=0.1,
                             annotation_text="Panic", annotation_position="top left")
            
            fig_vix.update_layout(
                title="India VIX with Regime Zones",
                height=400,
                yaxis_title="VIX Level"
            )
            
            # Correlation chart
            fig_corr = go.Figure()
            
            # Check if correlation columns exist
            if 'Rolling_Corr_30' in df_filtered.columns:
                fig_corr.add_trace(go.Scatter(
                    x=df_filtered['Date'], y=df_filtered['Rolling_Corr_30'],
                    name='30-Day Rolling Correlation',
                    line=dict(color='#9b59b6', width=2)
                ))
            
            fig_corr.add_hline(y=0, line_dash="dash", line_color="gray")
            
            fig_corr.update_layout(
                title="Rolling 30-Day Correlation: NIFTY Returns vs VIX Change",
                height=400,
                yaxis_title="Correlation",
                yaxis_range=[-1, 1]
            )
            
            return fig_price, fig_vix, fig_corr
        
        @self.app.callback(
            Output('returns-distribution', 'figure'),
            [Input('date-range', 'start_date'),
             Input('date-range', 'end_date'),
             Input('frequency-selector', 'value')]
        )
        def update_returns_dist(start_date, end_date, frequency):
            mask = (self.df_returns['Date'] >= start_date) & (self.df_returns['Date'] <= end_date)
            df_filtered = self.df_returns[mask].copy()
            
            # Set frequency label
            freq_label = {'1D': 'Daily', '1W': 'Weekly', '1M': 'Monthly'}[frequency]
            
            # Resample based on frequency
            if frequency != '1D':
                df_filtered = df_filtered.set_index('Date')
                freq_map = {'1W': 'W-FRI', '1M': 'ME'}
                df_filtered = df_filtered.resample(freq_map[frequency]).agg({
                    'Close_nifty': 'last'
                }).dropna().reset_index()
                # Recalculate returns for the selected frequency
                df_filtered['Return_Simple'] = df_filtered['Close_nifty'].pct_change() * 100
                df_filtered['Return_Log'] = np.log(df_filtered['Close_nifty'] / df_filtered['Close_nifty'].shift(1)) * 100
            else:
                df_filtered = df_filtered.copy()
                # Use existing daily returns or calculate
                if 'Daily_Return_Simple' in df_filtered.columns:
                    df_filtered['Return_Simple'] = df_filtered['Daily_Return_Simple']
                else:
                    df_filtered['Return_Simple'] = df_filtered['Close_nifty'].pct_change() * 100
                if 'Daily_Return_Log' in df_filtered.columns:
                    df_filtered['Return_Log'] = df_filtered['Daily_Return_Log']
                else:
                    df_filtered['Return_Log'] = np.log(df_filtered['Close_nifty'] / df_filtered['Close_nifty'].shift(1)) * 100
            
            # Calculate statistics
            returns_clean = df_filtered['Return_Simple'].dropna()
            mean_return = returns_clean.mean()
            median_return = returns_clean.median()
            std_return = returns_clean.std()
            
            # Calculate mode (most frequent bin)
            if len(returns_clean) > 0:
                hist_values, bin_edges = np.histogram(returns_clean, bins=50)
                mode_idx = hist_values.argmax()
                mode_return = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2
            else:
                mode_return = 0
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=(f'{freq_label} Returns Distribution', 'Log vs Simple Returns')
            )
            
            # Histogram
            fig.add_trace(
                go.Histogram(x=returns_clean, 
                            nbinsx=50, 
                            name=f'{freq_label} Returns (Mean: {mean_return:.3f}%, Median: {median_return:.3f}%, Mode: {mode_return:.3f}%, Std: {std_return:.3f}%)',
                            marker_color='#3498db',
                            legendgroup='histogram'),
                row=1, col=1
            )
            
            # Add shaded regions for ±1 std dev
            fig.add_vrect(x0=mean_return - std_return, x1=mean_return + std_return,
                         fillcolor="rgba(39, 174, 96, 0.1)", line_width=0,
                         annotation_text=f"±1σ",
                         annotation_position="top left", row=1, col=1)
            
            # Log vs Simple
            fig.add_trace(
                go.Scatter(x=df_filtered['Return_Simple'].dropna(),
                          y=df_filtered['Return_Log'].dropna(),
                          mode='markers', name='Log vs Simple',
                          marker=dict(size=3, color='#e74c3c', opacity=0.5)),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=True,
                            title_text=f"{freq_label} Returns Analysis | Mean: {mean_return:.3f}% | Median: {median_return:.3f}% | Std Dev: {std_return:.3f}%")
            
            return fig
            
            # Histogram
            fig.add_trace(
                go.Histogram(x=returns_clean, 
                            nbinsx=50, 
                            name=f'Daily Returns (Mean: {mean_return:.3f}%, Median: {median_return:.3f}%, Mode: {mode_return:.3f}%, Std: {std_return:.3f}%)',
                            marker_color='#3498db',
                            legendgroup='histogram'),
                row=1, col=1
            )
            
            # Add shaded regions for ±1 std dev
            fig.add_vrect(x0=mean_return - std_return, x1=mean_return + std_return,
                         fillcolor="rgba(39, 174, 96, 0.1)", line_width=0,
                         annotation_text=f"±1σ",
                         annotation_position="top left", row=1, col=1)
            
            # Log vs Simple
            fig.add_trace(
                go.Scatter(x=df_filtered['Daily_Return_Simple'].dropna(),
                          y=df_filtered['Daily_Return_Log'].dropna(),
                          mode='markers', name='Log vs Simple',
                          marker=dict(size=3, color='#e74c3c', opacity=0.5)),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=True,
                            title_text=f"Returns Analysis | Mean: {mean_return:.3f}% | Median: {median_return:.3f}% | Std Dev: {std_return:.3f}%")
            
            return fig
        
        @self.app.callback(
            Output('rolling-returns', 'figure'),
            [Input('date-range', 'start_date'),
             Input('date-range', 'end_date'),
             Input('frequency-selector', 'value')]
        )
        def update_rolling_returns(start_date, end_date, frequency):
            mask = (self.df_returns['Date'] >= start_date) & (self.df_returns['Date'] <= end_date)
            df_filtered = self.df_returns[mask].copy()
            
            # Set frequency label and MA windows
            freq_label = {'1D': 'Daily', '1W': 'Weekly', '1M': 'Monthly'}[frequency]
            ma_windows = {'1D': (5, 21, 50), '1W': (4, 13, 26), '1M': (3, 6, 12)}[frequency]
            
            # Resample based on frequency
            if frequency != '1D':
                df_filtered = df_filtered.set_index('Date')
                freq_map = {'1W': 'W-FRI', '1M': 'ME'}
                df_filtered = df_filtered.resample(freq_map[frequency]).agg({
                    'Close_nifty': 'last'
                }).dropna().reset_index()
                # Recalculate returns and moving averages
                df_filtered['Return'] = df_filtered['Close_nifty'].pct_change() * 100
                df_filtered['Return_MA_1'] = df_filtered['Return'].rolling(ma_windows[0]).mean()
                df_filtered['Return_MA_2'] = df_filtered['Return'].rolling(ma_windows[1]).mean()
                df_filtered['Return_MA_3'] = df_filtered['Return'].rolling(ma_windows[2]).mean()
            else:
                df_filtered = df_filtered.copy()
                if 'Daily_Return_Simple' in df_filtered.columns:
                    df_filtered['Return'] = df_filtered['Daily_Return_Simple']
                else:
                    df_filtered['Return'] = df_filtered['Close_nifty'].pct_change() * 100
                # Use existing MAs or calculate
                if 'Return_MA_5' in df_filtered.columns:
                    df_filtered['Return_MA_1'] = df_filtered['Return_MA_5']
                else:
                    df_filtered['Return_MA_1'] = df_filtered['Return'].rolling(ma_windows[0]).mean()
                if 'Return_MA_21' in df_filtered.columns:
                    df_filtered['Return_MA_2'] = df_filtered['Return_MA_21']
                else:
                    df_filtered['Return_MA_2'] = df_filtered['Return'].rolling(ma_windows[1]).mean()
                if 'Return_MA_50' in df_filtered.columns:
                    df_filtered['Return_MA_3'] = df_filtered['Return_MA_50']
                else:
                    df_filtered['Return_MA_3'] = df_filtered['Return'].rolling(ma_windows[2]).mean()
            
            fig = go.Figure()
            
            if 'Return_MA_1' in df_filtered.columns:
                fig.add_trace(go.Scatter(
                    x=df_filtered['Date'], y=df_filtered['Return_MA_1'].dropna(),
                    name=f'{ma_windows[0]}-Period MA', line=dict(color='#3498db', width=1)
                ))
            
            if 'Return_MA_2' in df_filtered.columns:
                fig.add_trace(go.Scatter(
                    x=df_filtered['Date'], y=df_filtered['Return_MA_2'].dropna(),
                    name=f'{ma_windows[1]}-Period MA', line=dict(color='#e74c3c', width=2)
                ))
            
            if 'Return_MA_3' in df_filtered.columns:
                fig.add_trace(go.Scatter(
                    x=df_filtered['Date'], y=df_filtered['Return_MA_3'].dropna(),
                    name=f'{ma_windows[2]}-Period MA', line=dict(color='#27ae60', width=2)
                ))
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            
            fig.update_layout(
                title=f"Rolling Mean of {freq_label} Returns",
                height=400,
                yaxis_title="Return (%)",
                xaxis_title="Date",
                showlegend=True
            )
            
            return fig
        
        @self.app.callback(
            Output('iv-rv-spread-chart', 'figure'),
            [Input('date-range', 'start_date'),
             Input('date-range', 'end_date'),
             Input('frequency-selector', 'value')]
        )
        def update_iv_rv(start_date, end_date, frequency):
            mask = (self.df_iv_rv['Date'] >= start_date) & (self.df_iv_rv['Date'] <= end_date)
            df_filtered = self.df_iv_rv[mask].copy()
            
            # Resample based on frequency
            if frequency != '1D':
                df_filtered = df_filtered.set_index('Date')
                freq_map = {'1W': 'W-FRI', '1M': 'ME'}
                df_filtered = df_filtered.resample(freq_map[frequency]).agg({
                    'Close_nifty': 'last',
                    'Close_vix': 'last',
                    'RV_21D': 'last',
                    'IV_RV_Spread': 'last'
                }).dropna().reset_index()
            else:
                df_filtered = df_filtered.copy()
            
            # Ensure required columns exist
            if 'RV_21D' not in df_filtered.columns:
                df_filtered['Daily_Return_Log'] = np.log(df_filtered['Close_nifty'] / df_filtered['Close_nifty'].shift(1))
                df_filtered['RV_21D'] = df_filtered['Daily_Return_Log'].rolling(21).std() * np.sqrt(252) * 100
            if 'IV_RV_Spread' not in df_filtered.columns:
                df_filtered['IV_RV_Spread'] = df_filtered['Close_vix'] - df_filtered['RV_21D']
            
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=('India VIX vs Realized Volatility', 'IV-RV Spread')
            )
            
            # IV vs RV
            fig.add_trace(
                go.Scatter(x=df_filtered['Date'], y=df_filtered['Close_vix'],
                          name='Implied Vol (VIX)', line=dict(color='#e74c3c', width=2)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df_filtered['Date'], y=df_filtered['RV_21D'].dropna(),
                          name='Realized Vol (21D)', line=dict(color='#3498db', width=2)),
                row=1, col=1
            )
            
            # Spread
            fig.add_trace(
                go.Scatter(x=df_filtered['Date'], y=df_filtered['IV_RV_Spread'].dropna(),
                          name='IV-RV Spread', fill='tozeroy',
                          line=dict(color='#9b59b6', width=2)),
                row=2, col=1
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
            
            fig.update_layout(height=600, title_text="Implied vs Realized Volatility Analysis")
            
            return fig
        
        # Callback to update parameter inputs based on strategy selection
        @self.app.callback(
            Output('strategy-params', 'children'),
            Input('strategy-selector', 'value')
        )
        def update_strategy_params(strategy_type):
            """Update parameter inputs based on selected strategy"""
            
            if strategy_type == 'iron_condor':
                # Iron Condor specific parameters
                return [
                    html.H4('Iron Condor Parameters', style={'color': '#2c3e50', 'marginBottom': 15}),
                    html.Div([
                        html.Div([
                            html.Label('VIX Threshold:', style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': 5}),
                            dcc.Input(id='vix-threshold', type='number', value=18, min=10, max=40, step=1,
                                     style={'width': 100, 'padding': 5})
                        ], style={'display': 'inline-block', 'marginRight': 30}),
                        html.Div([
                            html.Label('Short Strike Distance:', style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': 5}),
                            dcc.Input(id='strike-distance', type='number', value=200, min=50, max=500, step=50,
                                     style={'width': 100, 'padding': 5})
                        ], style={'display': 'inline-block', 'marginRight': 30}),
                        html.Div([
                            html.Label('Long Strike Distance:', style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': 5}),
                            dcc.Input(id='long-strike-distance', type='number', value=400, min=100, max=800, step=50,
                                     style={'width': 100, 'padding': 5})
                        ], style={'display': 'inline-block', 'marginRight': 30}),
                    ], style={'marginTop': 10}),
                    html.Div([
                        html.Div([
                            html.Label('DTE Entry:', style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': 5}),
                            dcc.Input(id='dte-entry', type='number', value=7, min=1, max=30, step=1,
                                     style={'width': 100, 'padding': 5})
                        ], style={'display': 'inline-block', 'marginRight': 30}),
                        html.Div([
                            html.Label('DTE Exit:', style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': 5}),
                            dcc.Input(id='dte-exit', type='number', value=1, min=0, max=10, step=1,
                                     style={'width': 100, 'padding': 5})
                        ], style={'display': 'inline-block'}),
                    ], style={'marginTop': 10}),
                    html.P('Iron Condor: Sell ATM±Short Distance, Buy ATM±Long Distance (both CE and PE)',
                          style={'color': '#7f8c8d', 'fontSize': 14, 'marginTop': 10, 'fontStyle': 'italic'})
                ]
            
            elif strategy_type == 'long_straddle':
                # Long Straddle parameters
                return [
                    html.H4('Long Straddle Parameters', style={'color': '#2c3e50', 'marginBottom': 15}),
                    html.Div([
                        html.Div([
                            html.Label('VIX Threshold (enter when VIX <):', style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': 5}),
                            dcc.Input(id='vix-threshold', type='number', value=12, min=10, max=40, step=1,
                                     style={'width': 100, 'padding': 5})
                        ], style={'display': 'inline-block', 'marginRight': 30}),
                        html.Div([
                            html.Label('DTE Entry:', style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': 5}),
                            dcc.Input(id='dte-entry', type='number', value=7, min=1, max=30, step=1,
                                     style={'width': 100, 'padding': 5})
                        ], style={'display': 'inline-block', 'marginRight': 30}),
                        html.Div([
                            html.Label('DTE Exit:', style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': 5}),
                            dcc.Input(id='dte-exit', type='number', value=1, min=0, max=10, step=1,
                                     style={'width': 100, 'padding': 5})
                        ], style={'display': 'inline-block'}),
                    ], style={'marginTop': 10}),
                    html.P('Long Straddle: Buy ATM Call and ATM Put (enter when VIX is low)',
                          style={'color': '#7f8c8d', 'fontSize': 14, 'marginTop': 10, 'fontStyle': 'italic'})
                ]
            
            else:  # short_strangle
                # Short Strangle parameters
                return [
                    html.H4('Short Strangle Parameters', style={'color': '#2c3e50', 'marginBottom': 15}),
                    html.Div([
                        html.Div([
                            html.Label('VIX Threshold (enter when VIX >):', style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': 5}),
                            dcc.Input(id='vix-threshold', type='number', value=18, min=10, max=40, step=1,
                                     style={'width': 100, 'padding': 5})
                        ], style={'display': 'inline-block', 'marginRight': 30}),
                        html.Div([
                            html.Label('Strike Distance (ATM ±):', style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': 5}),
                            dcc.Input(id='strike-distance', type='number', value=200, min=50, max=500, step=50,
                                     style={'width': 100, 'padding': 5})
                        ], style={'display': 'inline-block', 'marginRight': 30}),
                        html.Div([
                            html.Label('DTE Entry:', style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': 5}),
                            dcc.Input(id='dte-entry', type='number', value=7, min=1, max=30, step=1,
                                     style={'width': 100, 'padding': 5})
                        ], style={'display': 'inline-block', 'marginRight': 30}),
                        html.Div([
                            html.Label('DTE Exit:', style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': 5}),
                            dcc.Input(id='dte-exit', type='number', value=1, min=0, max=10, step=1,
                                     style={'width': 100, 'padding': 5})
                        ], style={'display': 'inline-block'}),
                    ], style={'marginTop': 10}),
                    html.P('Short Strangle: Sell OTM Call and OTM Put at ATM ± distance',
                          style={'color': '#7f8c8d', 'fontSize': 14, 'marginTop': 10, 'fontStyle': 'italic'})
                ]
        
        @self.app.callback(
            [Output('strategy-metrics', 'children'),
             Output('strategy-equity-curve', 'figure'),
             Output('strategy-signals', 'figure'),
             Output('strategy-date-warning', 'children')],
            [Input('strategy-load-btn', 'n_clicks')],
            [State('strategy-selector', 'value'),
             State('strategy-date-range', 'start_date'),
             State('strategy-date-range', 'end_date'),
             State('frequency-selector', 'value'),
             State('vix-threshold', 'value'),
             State('strike-distance', 'value'),
             State('dte-entry', 'value'),
             State('dte-exit', 'value'),
             State('long-strike-distance', 'value')]
        )
        def update_strategy(n_clicks, strategy_type, start_date, end_date, frequency, 
                          vix_threshold, strike_distance, dte_entry, dte_exit, long_strike_distance):
            # Don't load until button clicked
            if n_clicks == 0:
                placeholder_msg = html.Div([
                    html.H3("Ready to Run Backtest", style={'color': '#3498db'}),
                    html.P("Select a strategy and date range, then click 'Load' to run the backtest.", 
                          style={'fontSize': 16})
                ], style={'backgroundColor': '#fff', 'padding': 40, 'borderRadius': 10, 'textAlign': 'center'})
                
                empty_fig = go.Figure()
                empty_fig.update_layout(title="Click 'Load' to generate chart")
                return placeholder_msg, empty_fig, empty_fig, ""
            
            # Validate date range (max 6 months)
            warning_msg = ""
            if start_date and end_date:
                start = pd.Timestamp(start_date)
                end = pd.Timestamp(end_date)
                days_diff = (end - start).days
                
                if days_diff > 180:
                    # Adjust to 6 months from end date
                    start = end - pd.Timedelta(days=180)
                    start_date = start
                    warning_msg = "⚠️ Date range limited to 6 months. Showing last 6 months from selected end date."
                
                # Set defaults if not provided
                if vix_threshold is None:
                    vix_threshold = 18 if strategy_type == 'short_strangle' else 12
                if strike_distance is None:
                    strike_distance = 200
                if dte_entry is None:
                    dte_entry = 7
                if dte_exit is None:
                    dte_exit = 1
                
                # Run backtest for selected date range
                print(f"\nRunning {strategy_type} backtest for {start_date} to {end_date}")
                print(f"Parameters: VIX={vix_threshold}, Strike Distance={strike_distance}, DTE Entry={dte_entry}, DTE Exit={dte_exit}")
                
                # Create filtered data for the date range using the merged dataframe
                filtered_df = self.analyzer.merged_df[
                    (self.analyzer.merged_df['Date'] >= start_date) & 
                    (self.analyzer.merged_df['Date'] <= end_date)
                ].copy()
                
                if filtered_df.empty:
                    error_msg = html.Div([
                        html.H3("No Data Available", style={'color': '#e74c3c'}),
                        html.P(f"No data found for the selected date range: {start_date} to {end_date}")
                    ], style={'backgroundColor': '#fff', 'padding': 20, 'borderRadius': 10})
                    
                    empty_fig = go.Figure()
                    empty_fig.update_layout(title="No Data Available")
                    return error_msg, empty_fig, empty_fig, warning_msg
                
                # Run backtest with filtered data
                from event_analysis import EventAnalyzer
                temp_analyzer = EventAnalyzer(filtered_df)
                
                if strategy_type == 'short_strangle':
                    result = temp_analyzer.backtest_with_real_options(
                        strategy_type='short_strangle',
                        vix_threshold=vix_threshold,
                        strike_distance=strike_distance,
                        dte_entry=dte_entry,
                        dte_exit=dte_exit,
                        start_date=start_date,
                        end_date=end_date
                    )
                elif strategy_type == 'long_straddle':
                    result = temp_analyzer.backtest_with_real_options(
                        strategy_type='long_straddle',
                        vix_threshold=vix_threshold,
                        dte_entry=dte_entry,
                        dte_exit=dte_exit,
                        start_date=start_date,
                        end_date=end_date
                    )
                else:
                    result = temp_analyzer.backtest_simple_strategy('iron_condor')
                
                if result and len(result) == 2:
                    if strategy_type in ['short_strangle', 'long_straddle']:
                        trades_df, metrics = result
                        df_strat = self._convert_trades_to_daily(trades_df, filtered_df)
                    else:
                        # Iron Condor - simplified strategy returns daily data already
                        df_strat, metrics = result
                        # Ensure it has Date column for merging
                        if 'Date' not in df_strat.columns and df_strat.index.name == 'Date':
                            df_strat = df_strat.reset_index()
                else:
                    # Fallback to simplified
                    df_strat, metrics = temp_analyzer.backtest_simple_strategy(strategy_type)
                    if 'Date' not in df_strat.columns and df_strat.index.name == 'Date':
                        df_strat = df_strat.reset_index()
            else:
                # No date range specified
                error_msg = html.Div([
                    html.H3("Please Select Date Range", style={'color': '#e74c3c'}),
                    html.P("Select a date range (max 6 months) to run the backtest.")
                ], style={'backgroundColor': '#fff', 'padding': 20, 'borderRadius': 10})
                
                empty_fig = go.Figure()
                empty_fig.update_layout(title="No Data Available")
                return error_msg, empty_fig, empty_fig, ""
            
            # Handle missing metrics or data
            if df_strat is None or df_strat.empty or metrics is None:
                error_msg = html.Div([
                    html.H3("Strategy Data Not Available", style={'color': '#e74c3c'}),
                    html.P("Unable to load strategy backtest data. Please check the data files.")
                ], style={'backgroundColor': '#fff', 'padding': 20, 'borderRadius': 10})
                
                empty_fig = go.Figure()
                empty_fig.update_layout(title="No Data Available")
                return error_msg, empty_fig, empty_fig, ""
            
            # Filter by date
            mask = (df_strat['Date'] >= start_date) & (df_strat['Date'] <= end_date)
            df_filtered = df_strat[mask]
            
            # Also filter the market data for the signals chart
            market_filtered = filtered_df[
                (filtered_df['Date'] >= start_date) & 
                (filtered_df['Date'] <= end_date)
            ]
            
            # Metrics display
            metrics_html = html.Div([
                html.H3(f"{metrics.get('Strategy', strategy_type).replace('_', ' ').title()} Performance", 
                       style={'color': '#2c3e50'}),
                html.Div([
                    html.P(f"Parameters: VIX>{vix_threshold}, DTE {dte_entry}→{dte_exit}" + 
                          (f", ATM±{strike_distance}" if strategy_type == 'short_strangle' else ""),
                          style={'color': '#7f8c8d', 'fontSize': 14, 'marginBottom': 15, 'fontStyle': 'italic'})
                ]),
                html.Div([
                    html.Div([
                        html.P("Total Trades", style={'color': '#7f8c8d', 'fontSize': 12}),
                        html.P(f"{metrics.get('Total_Trades', 0)}", 
                              style={'fontSize': 24, 'fontWeight': 'bold', 'color': '#2c3e50'})
                    ], style={'flex': 1, 'textAlign': 'center'}),
                    html.Div([
                        html.P("Win Rate", style={'color': '#7f8c8d', 'fontSize': 12}),
                        html.P(f"{metrics.get('Win_Rate_%', 0):.1f}%", 
                              style={'fontSize': 24, 'fontWeight': 'bold', 'color': '#27ae60'})
                    ], style={'flex': 1, 'textAlign': 'center'}),
                    html.Div([
                        html.P("Total P&L" if 'Total_PnL' in metrics else "Avg Return/Trade", 
                              style={'color': '#7f8c8d', 'fontSize': 12}),
                        html.P(f"₹{metrics.get('Total_PnL', 0):,.0f}" if 'Total_PnL' in metrics 
                              else f"{metrics.get('Avg_Return_Per_Trade', 0):.2f}%", 
                              style={'fontSize': 24, 'fontWeight': 'bold', 
                                    'color': '#27ae60' if metrics.get('Total_PnL', metrics.get('Avg_Return_Per_Trade', 0)) > 0 else '#e74c3c'})
                    ], style={'flex': 1, 'textAlign': 'center'}),
                    html.Div([
                        html.P("Sharpe Ratio", style={'color': '#7f8c8d', 'fontSize': 12}),
                        html.P(f"{metrics.get('Sharpe_Ratio', 0):.2f}", 
                              style={'fontSize': 24, 'fontWeight': 'bold', 'color': '#3498db'})
                    ], style={'flex': 1, 'textAlign': 'center'}),
                    html.Div([
                        html.P("Max Drawdown" if 'Max_Drawdown' in metrics else "Total Return", 
                              style={'color': '#7f8c8d', 'fontSize': 12}),
                        html.P(f"₹{metrics.get('Max_Drawdown', 0):,.0f}" if 'Max_Drawdown' in metrics 
                              else f"{metrics.get('Total_Return', 0):.2f}%", 
                              style={'fontSize': 24, 'fontWeight': 'bold', 'color': '#e74c3c'})
                    ], style={'flex': 1, 'textAlign': 'center'}),
                ], style={'display': 'flex', 'gap': 20, 'marginTop': 20})
            ], style={'backgroundColor': '#fff', 'padding': 20, 'borderRadius': 10,
                     'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'marginBottom': 20})
            
            # Equity curve
            fig_equity = go.Figure()
            fig_equity.add_trace(go.Scatter(
                x=df_filtered['Date'], y=df_filtered['Cumulative_Strategy_Return'],
                name='Strategy Returns', fill='tozeroy',
                line=dict(color='#27ae60', width=2)
            ))
            fig_equity.update_layout(
                title=f"{strategy_type.replace('_', ' ').title()} - Cumulative Returns",
                height=400,
                yaxis_title="Cumulative Return (%)"
            )
            
            # Signals chart
            fig_signals = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.6, 0.4],
                subplot_titles=('NIFTY 50 with Entry Signals', 'VIX Level')
            )
            
            # NIFTY with signals
            fig_signals.add_trace(
                go.Scatter(x=market_filtered['Date'], y=market_filtered['Close_nifty'],
                          name='NIFTY 50', line=dict(color='#3498db', width=1)),
                row=1, col=1
            )
            
            # Add entry points if Signal column exists
            if 'Signal' in df_filtered.columns:
                entries = df_filtered[df_filtered['Signal'] == 1]
                if not entries.empty:
                    # Merge with market data to get Close_nifty prices
                    entries_with_price = entries.merge(market_filtered[['Date', 'Close_nifty']], on='Date', how='left')
                    fig_signals.add_trace(
                        go.Scatter(x=entries_with_price['Date'], y=entries_with_price['Close_nifty'],
                                  mode='markers', name='Entry Signal',
                                  marker=dict(color='#27ae60', size=8, symbol='triangle-up')),
                        row=1, col=1
                    )
            
            # VIX
            fig_signals.add_trace(
                go.Scatter(x=market_filtered['Date'], y=market_filtered['Close_vix'],
                          name='VIX', line=dict(color='#e74c3c', width=1)),
                row=2, col=1
            )
            
            fig_signals.update_layout(height=600, title_text="Strategy Entry Signals")
            
            return metrics_html, fig_equity, fig_signals, warning_msg
        
        # Additional callbacks for other charts...
        @self.app.callback(
            Output('lead-lag-chart', 'figure'),
            Input('date-range', 'start_date')
        )
        def update_lead_lag(start_date):
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=self.lead_lag_df['Lag'],
                y=self.lead_lag_df['Correlation'],
                marker_color=np.where(self.lead_lag_df['Correlation'] > 0, '#27ae60', '#e74c3c')
            ))
            fig.update_layout(
                title="Lead-Lag Cross-Correlation: NIFTY Returns vs VIX Change",
                xaxis_title="Lag (days, negative = VIX leads)",
                yaxis_title="Correlation",
                height=400
            )
            return fig
        
        @self.app.callback(
            [Output('vix-regime-pie', 'figure'),
             Output('vix-histogram', 'figure')],
            [Input('date-range', 'start_date'),
             Input('date-range', 'end_date'),
             Input('frequency-selector', 'value')]
        )
        def update_vix_analysis(start_date, end_date, frequency):
            mask = (self.df_vix['Date'] >= start_date) & (self.df_vix['Date'] <= end_date)
            df_filtered = self.df_vix[mask].copy()
            
            # Resample based on frequency
            if frequency != '1D':
                df_filtered = df_filtered.set_index('Date')
                freq_map = {'1W': 'W-FRI', '1M': 'ME'}
                df_filtered = df_filtered.resample(freq_map[frequency]).agg({
                    'Close_vix': 'last',
                    'VIX_Regime': 'last'
                }).dropna().reset_index()
            else:
                df_filtered = df_filtered.copy()
            
            # Pie chart for VIX regimes
            if 'VIX_Regime' in df_filtered.columns:
                regime_counts = df_filtered['VIX_Regime'].value_counts()
                fig_pie = go.Figure(data=[go.Pie(
                    labels=regime_counts.index,
                    values=regime_counts.values,
                    marker=dict(colors=['#27ae60', '#f39c12', '#e67e22', '#e74c3c'])
                )])
                fig_pie.update_layout(
                    title="VIX Regime Distribution",
                    height=400
                )
            else:
                fig_pie = go.Figure()
                fig_pie.add_annotation(text="VIX Regime data not available", 
                                      showarrow=False, font=dict(size=16))
            
            # Histogram of VIX values
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=df_filtered['Close_vix'],
                nbinsx=50,
                marker_color='#e74c3c',
                name='VIX Distribution'
            ))
            
            # Add regime lines
            fig_hist.add_vline(x=12, line_dash="dash", line_color="green", annotation_text="Low/Normal")
            fig_hist.add_vline(x=18, line_dash="dash", line_color="orange", annotation_text="Normal/High")
            fig_hist.add_vline(x=25, line_dash="dash", line_color="red", annotation_text="High/Panic")
            
            fig_hist.update_layout(
                title="India VIX Distribution with Regime Thresholds",
                xaxis_title="VIX Level",
                yaxis_title="Frequency",
                height=400
            )
            
            return fig_pie, fig_hist
        
        @self.app.callback(
            Output('scatter-nifty-vix', 'figure'),
            [Input('date-range', 'start_date'),
             Input('date-range', 'end_date'),
             Input('frequency-selector', 'value')]
        )
        def update_scatter(start_date, end_date, frequency):
            mask = (self.df_corr['Date'] >= start_date) & (self.df_corr['Date'] <= end_date)
            df_filtered = self.df_corr[mask].copy()
            
            # Set frequency label
            freq_label = {'1D': 'Daily', '1W': 'Weekly', '1M': 'Monthly'}[frequency]
            
            # Resample based on frequency
            if frequency != '1D':
                df_filtered = df_filtered.set_index('Date')
                freq_map = {'1W': 'W-FRI', '1M': 'ME'}
                df_filtered = df_filtered.resample(freq_map[frequency]).agg({
                    'Close_nifty': 'last',
                    'Close_vix': 'last'
                }).dropna().reset_index()
                # Recalculate returns
                df_filtered['Return'] = df_filtered['Close_nifty'].pct_change() * 100
                df_filtered['VIX_Change'] = df_filtered['Close_vix'].diff()
            else:
                df_filtered = df_filtered.copy()
                # Calculate returns if not present
                if 'Daily_Return' not in df_filtered.columns:
                    df_filtered['Return'] = df_filtered['Close_nifty'].pct_change() * 100
                else:
                    df_filtered['Return'] = df_filtered['Daily_Return']
                if 'VIX_Change' not in df_filtered.columns:
                    df_filtered['VIX_Change'] = df_filtered['Close_vix'].diff()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_filtered['Return'].dropna(),
                y=df_filtered['VIX_Change'].dropna(),
                mode='markers',
                marker=dict(size=4, color='#3498db', opacity=0.5),
                name=f'{freq_label} Data'
            ))
            
            # Add trend line
            from scipy import stats
            clean_data = df_filtered[['Return', 'VIX_Change']].dropna()
            if len(clean_data) > 10:
                slope, intercept, r_value, _, _ = stats.linregress(clean_data['Return'], clean_data['VIX_Change'])
                x_trend = np.linspace(clean_data['Return'].min(), clean_data['Return'].max(), 100)
                y_trend = slope * x_trend + intercept
                fig.add_trace(go.Scatter(
                    x=x_trend, y=y_trend,
                    mode='lines',
                    line=dict(color='#e74c3c', width=2),
                    name=f'Trend (R²={r_value**2:.3f})'
                ))
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            
            fig.update_layout(
                title=f"NIFTY {freq_label} Returns vs VIX Change Scatter",
                xaxis_title=f"NIFTY {freq_label} Return (%)",
                yaxis_title="VIX Change",
                height=400
            )
            
            return fig
        
        @self.app.callback(
            [Output('iv-rv-zscore-chart', 'figure'),
             Output('realized-vol-comparison', 'figure')],
            [Input('date-range', 'start_date'),
             Input('date-range', 'end_date'),
             Input('frequency-selector', 'value')]
        )
        def update_iv_rv_additional(start_date, end_date, frequency):
            mask = (self.df_iv_rv['Date'] >= start_date) & (self.df_iv_rv['Date'] <= end_date)
            df_filtered = self.df_iv_rv[mask].copy()
            
            # Resample based on frequency
            if frequency != '1D':
                df_filtered = df_filtered.set_index('Date')
                freq_map = {'1W': 'W-FRI', '1M': 'ME'}
                # Only resample columns that exist
                agg_dict = {'Close_nifty': 'last', 'Close_vix': 'last'}
                for col in ['RV_21D', 'IV_RV_Spread', 'Spread_Zscore']:
                    if col in df_filtered.columns:
                        agg_dict[col] = 'last'
                df_filtered = df_filtered.resample(freq_map[frequency]).agg(agg_dict).dropna().reset_index()
            else:
                df_filtered = df_filtered.copy()
            
            # Ensure columns exist
            if 'RV_21D' not in df_filtered.columns:
                df_filtered['Daily_Return_Log'] = np.log(df_filtered['Close_nifty'] / df_filtered['Close_nifty'].shift(1))
                df_filtered['RV_21D'] = df_filtered['Daily_Return_Log'].rolling(21).std() * np.sqrt(252) * 100
            if 'IV_RV_Spread' not in df_filtered.columns:
                df_filtered['IV_RV_Spread'] = df_filtered['Close_vix'] - df_filtered['RV_21D']
            if 'Spread_Zscore' not in df_filtered.columns:
                spread_mean = df_filtered['IV_RV_Spread'].mean()
                spread_std = df_filtered['IV_RV_Spread'].std()
                df_filtered['Spread_Zscore'] = (df_filtered['IV_RV_Spread'] - spread_mean) / spread_std
            
            # Z-score chart
            fig_zscore = go.Figure()
            fig_zscore.add_trace(go.Scatter(
                x=df_filtered['Date'], y=df_filtered['Spread_Zscore'].dropna(),
                name='Spread Z-Score', fill='tozeroy',
                line=dict(color='#9b59b6', width=2)
            ))
            
            fig_zscore.add_hline(y=2, line_dash="dash", line_color="red", annotation_text="+2σ")
            fig_zscore.add_hline(y=-2, line_dash="dash", line_color="red", annotation_text="-2σ")
            fig_zscore.add_hline(y=1, line_dash="dot", line_color="orange", annotation_text="+1σ")
            fig_zscore.add_hline(y=-1, line_dash="dot", line_color="orange", annotation_text="-1σ")
            fig_zscore.add_hline(y=0, line_dash="solid", line_color="gray")
            
            fig_zscore.update_layout(
                title="IV-RV Spread Z-Score (Mean Reversion Signal)",
                yaxis_title="Z-Score",
                height=400
            )
            
            # Realized volatility comparison (different windows)
            fig_rv_comp = go.Figure()
            
            # Calculate multiple RV windows if not present
            if 'RV_10D' not in df_filtered.columns:
                df_filtered['Daily_Return_Log'] = np.log(df_filtered['Close_nifty'] / df_filtered['Close_nifty'].shift(1))
                df_filtered['RV_10D'] = df_filtered['Daily_Return_Log'].rolling(10).std() * np.sqrt(252) * 100
            if 'RV_21D' not in df_filtered.columns:
                df_filtered['RV_21D'] = df_filtered['Daily_Return_Log'].rolling(21).std() * np.sqrt(252) * 100
            if 'RV_63D' not in df_filtered.columns:
                df_filtered['RV_63D'] = df_filtered['Daily_Return_Log'].rolling(63).std() * np.sqrt(252) * 100
            
            fig_rv_comp.add_trace(go.Scatter(
                x=df_filtered['Date'], y=df_filtered['Close_vix'],
                name='Implied Vol (VIX)', line=dict(color='#e74c3c', width=2)
            ))
            
            if 'RV_10D' in df_filtered.columns:
                fig_rv_comp.add_trace(go.Scatter(
                    x=df_filtered['Date'], y=df_filtered['RV_10D'].dropna(),
                    name='RV 10-Day', line=dict(color='#3498db', width=1, dash='dot')
                ))
            
            if 'RV_21D' in df_filtered.columns:
                fig_rv_comp.add_trace(go.Scatter(
                    x=df_filtered['Date'], y=df_filtered['RV_21D'].dropna(),
                    name='RV 21-Day', line=dict(color='#27ae60', width=1.5)
                ))
            
            if 'RV_63D' in df_filtered.columns:
                fig_rv_comp.add_trace(go.Scatter(
                    x=df_filtered['Date'], y=df_filtered['RV_63D'].dropna(),
                    name='RV 63-Day', line=dict(color='#f39c12', width=1, dash='dash')
                ))
            
            fig_rv_comp.update_layout(
                title="Realized Volatility Comparison (Multiple Windows)",
                yaxis_title="Volatility (%)",
                height=400
            )
            
            return fig_zscore, fig_rv_comp
        
        @self.app.callback(
            [Output('expiry-analysis', 'figure'),
             Output('vix-decay-chart', 'figure')],
            [Input('date-range', 'start_date'),
             Input('date-range', 'end_date'),
             Input('frequency-selector', 'value')]
        )
        def update_event_analysis(start_date, end_date, frequency):
            mask = (self.df_events['Date'] >= start_date) & (self.df_events['Date'] <= end_date)
            df_filtered = self.df_events[mask].copy()
            
            # Set frequency label
            freq_label = {'1D': 'Daily', '1W': 'Weekly', '1M': 'Monthly'}[frequency]
            
            # Resample based on frequency
            if frequency != '1D':
                df_filtered = df_filtered.set_index('Date')
                freq_map = {'1W': 'W-FRI', '1M': 'ME'}
                # Only resample columns that exist
                agg_dict = {'Close_nifty': 'last', 'Close_vix': 'last'}
                for col in ['Is_Expiry_Week', 'VIX_Spike']:
                    if col in df_filtered.columns:
                        agg_dict[col] = 'max'
                df_filtered = df_filtered.resample(freq_map[frequency]).agg(agg_dict).dropna().reset_index()
                # Recalculate returns
                df_filtered['Return'] = df_filtered['Close_nifty'].pct_change() * 100
            else:
                df_filtered = df_filtered.copy()
                if 'Daily_Return' in df_filtered.columns:
                    df_filtered['Return'] = df_filtered['Daily_Return']
                else:
                    df_filtered['Return'] = df_filtered['Close_nifty'].pct_change() * 100
            
            # Expiry week analysis
            fig_expiry = go.Figure()
            
            if 'Is_Expiry_Week' in df_filtered.columns and 'Return' in df_filtered.columns:
                expiry_weeks = df_filtered[df_filtered['Is_Expiry_Week'] == 1]
                non_expiry = df_filtered[df_filtered['Is_Expiry_Week'] == 0]
                
                fig_expiry.add_trace(go.Box(
                    y=expiry_weeks['Return'].dropna(),
                    name='Expiry Week',
                    marker_color='#e74c3c'
                ))
                
                fig_expiry.add_trace(go.Box(
                    y=non_expiry['Return'].dropna(),
                    name='Non-Expiry Week',
                    marker_color='#3498db'
                ))
                
                fig_expiry.update_layout(
                    title=f"{freq_label} Returns: Expiry Week vs Non-Expiry Week",
                    yaxis_title=f"{freq_label} Return (%)",
                    height=400
                )
            else:
                fig_expiry.add_annotation(text="Expiry week data not available", 
                                         showarrow=False, font=dict(size=16))
            
            # VIX decay after spike
            fig_decay = go.Figure()
            
            if 'VIX_Spike' in df_filtered.columns:
                # Find spike events
                spikes = df_filtered[df_filtered['VIX_Spike'] == 1]
                
                if len(spikes) > 0:
                    # Average decay pattern
                    decay_data = []
                    for idx_pos, idx in enumerate(spikes.index):
                        # Get next 20 days after this spike
                        current_pos = df_filtered.index.get_loc(idx)
                        if current_pos + 20 < len(df_filtered):
                            spike_vix = df_filtered.loc[idx, 'Close_vix']
                            # Get the next 20 rows
                            future_data = df_filtered.iloc[current_pos+1:current_pos+21]
                            if len(future_data) > 0:
                                future_vix = future_data['Close_vix'].values
                                decay_pct = ((future_vix - spike_vix) / spike_vix * 100)
                                decay_data.append(decay_pct)
                    
                    if decay_data and len(decay_data) > 0:
                        # Average across all spikes
                        min_len = min(len(d) for d in decay_data)
                        if min_len > 0:
                            decay_array = np.array([d[:min_len] for d in decay_data])
                            avg_decay = decay_array.mean(axis=0)
                            std_decay = decay_array.std(axis=0)
                            
                            days = np.arange(1, len(avg_decay) + 1)
                            
                            fig_decay.add_trace(go.Scatter(
                                x=days, y=avg_decay,
                                name='Average Decay',
                                line=dict(color='#e74c3c', width=2)
                            ))
                            
                            fig_decay.add_trace(go.Scatter(
                                x=days, y=avg_decay + std_decay,
                                name='+1 Std Dev',
                                line=dict(color='#e74c3c', width=0),
                                showlegend=False
                            ))
                            
                            fig_decay.add_trace(go.Scatter(
                                x=days, y=avg_decay - std_decay,
                                name='±1 Std Dev',
                                fill='tonexty',
                                line=dict(color='#e74c3c', width=0),
                                fillcolor='rgba(231, 76, 60, 0.2)'
                            ))
                            
                            fig_decay.add_hline(y=0, line_dash="dash", line_color="gray")
                        else:
                            fig_decay.add_annotation(text=f"Found {len(spikes)} VIX spikes but insufficient data for decay analysis", 
                                                   showarrow=False, font=dict(size=14))
                    else:
                        fig_decay.add_annotation(text=f"Found {len(spikes)} VIX spikes but insufficient future data", 
                                               showarrow=False, font=dict(size=14))
                else:
                    fig_decay.add_annotation(text="No VIX spike events found in selected date range", 
                                           showarrow=False, font=dict(size=14))
            else:
                fig_decay.add_annotation(text="VIX spike data not available", 
                                       showarrow=False, font=dict(size=14))
            
            fig_decay.update_layout(
                title="Average VIX Decay Pattern After Spikes",
                xaxis_title="Days After Spike",
                yaxis_title="% Change from Spike Level",
                height=400
            )
            
            return fig_expiry, fig_decay
        
        @self.app.callback(
            [Output('drawdown-chart', 'figure'),
             Output('tail-risk-chart', 'figure')],
            [Input('date-range', 'start_date'),
             Input('date-range', 'end_date'),
             Input('frequency-selector', 'value')]
        )
        def update_risk_analysis(start_date, end_date, frequency):
            mask = (self.df_returns['Date'] >= start_date) & (self.df_returns['Date'] <= end_date)
            df_filtered = self.df_returns[mask].copy()
            
            # Set frequency label
            freq_label = {'1D': 'Daily', '1W': 'Weekly', '1M': 'Monthly'}[frequency]
            
            # Resample based on frequency
            if frequency != '1D':
                df_filtered = df_filtered.set_index('Date')
                freq_map = {'1W': 'W-FRI', '1M': 'ME'}
                df_filtered = df_filtered.resample(freq_map[frequency]).agg({
                    'Close_nifty': 'last'
                }).dropna().reset_index()
                # Recalculate returns and drawdowns
                df_filtered['Return_Simple'] = df_filtered['Close_nifty'].pct_change() * 100
                df_filtered['Cumulative_Max'] = df_filtered['Close_nifty'].cummax()
                df_filtered['Drawdown'] = ((df_filtered['Close_nifty'] - df_filtered['Cumulative_Max']) / df_filtered['Cumulative_Max']) * 100
            else:
                df_filtered = df_filtered.copy()
                # Use existing columns or calculate
                if 'Daily_Return_Simple' in df_filtered.columns:
                    df_filtered['Return_Simple'] = df_filtered['Daily_Return_Simple']
                else:
                    df_filtered['Return_Simple'] = df_filtered['Close_nifty'].pct_change() * 100
                if 'Drawdown' not in df_filtered.columns:
                    df_filtered['Cumulative_Max'] = df_filtered['Close_nifty'].cummax()
                    df_filtered['Drawdown'] = ((df_filtered['Close_nifty'] - df_filtered['Cumulative_Max']) / df_filtered['Cumulative_Max']) * 100
            
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=df_filtered['Date'], y=df_filtered['Drawdown'],
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='#e74c3c', width=2),
                fillcolor='rgba(231, 76, 60, 0.3)'
            ))
            
            fig_dd.add_hline(y=0, line_dash="solid", line_color="gray")
            fig_dd.add_hline(y=-10, line_dash="dash", line_color="orange", annotation_text="-10%")
            fig_dd.add_hline(y=-20, line_dash="dash", line_color="red", annotation_text="-20%")
            
            fig_dd.update_layout(
                title="NIFTY 50 Drawdown (Underwater Plot)",
                yaxis_title="Drawdown (%)",
                height=400
            )
            
            # Tail risk chart (distribution of extreme moves)
            returns = df_filtered['Return_Simple'].dropna()
            
            # Identify tail events
            if len(returns) > 0:
                threshold_down = returns.quantile(0.05)  # 5th percentile
                threshold_up = returns.quantile(0.95)    # 95th percentile
            else:
                threshold_down = 0
                threshold_up = 0
            
            fig_tail = go.Figure()
            
            # Main distribution
            fig_tail.add_trace(go.Histogram(
                x=returns,
                nbinsx=50,
                name='All Returns',
                marker_color='#3498db',
                opacity=0.7
            ))
            
            # Highlight tails
            if len(returns) > 0:
                fig_tail.add_vline(x=threshold_down, line_dash="dash", line_color="red", 
                                  annotation_text=f"5th %ile: {threshold_down:.2f}%")
                fig_tail.add_vline(x=threshold_up, line_dash="dash", line_color="green", 
                                  annotation_text=f"95th %ile: {threshold_up:.2f}%")
            
            fig_tail.update_layout(
                title=f"Tail Risk Analysis - {freq_label} Returns (5th/95th Percentiles)",
                xaxis_title=f"{freq_label} Return (%)",
                yaxis_title="Frequency",
                height=400
            )
            
            return fig_dd, fig_tail
        
        @self.app.callback(
            [Output('hypothesis-results', 'children'),
             Output('hypothesis-chart', 'figure'),
             Output('hypothesis-distribution', 'figure')],
            [Input('test-hypothesis-btn', 'n_clicks')],
            [Input('hypothesis-input', 'value'),
             Input('hypothesis-timeframe', 'value')]
        )
        def test_hypothesis(n_clicks, hypothesis, timeframe):
            if n_clicks == 0 or not hypothesis:
                empty_msg = html.Div([
                    html.P("Enter a hypothesis in plain English and click 'Test Hypothesis'", 
                          style={'textAlign': 'center', 'color': '#7f8c8d', 'padding': 40})
                ])
                return empty_msg, go.Figure(), go.Figure()
            
            try:
                # Prepare data based on timeframe
                df = self.df_corr.copy()
                
                # Create parser instance
                parser = HypothesisParser(df)
                
                # Parse the natural language hypothesis
                result, variables = parser.parse(hypothesis, timeframe)
                
                # Get the processed dataframe from parser
                df = parser.df
                
                # Resample if needed
                if timeframe == 'weekly':
                    df = df.set_index('Date')
                    df['Hypothesis_True'] = result.astype(int)  # Convert boolean to int for aggregation
                    df = df.resample('W-FRI').agg({
                        'Close_nifty': 'last',
                        'Close_vix': 'last',
                        'Hypothesis_True': 'max'
                    }).reset_index()
                    df = df.dropna(subset=['Close_nifty', 'Close_vix'])  # Only drop if price data is missing
                    df['Hypothesis_True'] = df['Hypothesis_True'].astype(bool)  # Convert back to boolean
                    df['nifty_return'] = df['Close_nifty'].pct_change() * 100
                elif timeframe == 'monthly':
                    df = df.set_index('Date')
                    df['Hypothesis_True'] = result.astype(int)  # Convert boolean to int for aggregation
                    df = df.resample('ME').agg({
                        'Close_nifty': 'last',
                        'Close_vix': 'last',
                        'Hypothesis_True': 'max'
                    }).reset_index()
                    df = df.dropna(subset=['Close_nifty', 'Close_vix'])  # Only drop if price data is missing
                    df['Hypothesis_True'] = df['Hypothesis_True'].astype(bool)  # Convert back to boolean
                    df['nifty_return'] = df['Close_nifty'].pct_change() * 100
                    print(f"DEBUG (monthly): df shape = {df.shape}, columns = {df.columns.tolist()}")
                else:
                    df['Hypothesis_True'] = result
                    df['nifty_return'] = df['nifty_return'] if 'nifty_return' in df.columns else df['Close_nifty'].pct_change() * 100
                    print(f"DEBUG (daily): df shape = {df.shape}, columns = {df.columns.tolist()}")
                
                print(f"DEBUG: About to calculate statistics. 'nifty_return' in df: {'nifty_return' in df.columns}")
                # Calculate statistics
                total_periods = len(df)
                true_count = df['Hypothesis_True'].sum()
                false_count = total_periods - true_count
                success_rate = (true_count / total_periods * 100) if total_periods > 0 else 0
                
                # Calculate returns when hypothesis is true vs false
                true_returns = df[df['Hypothesis_True']]['nifty_return'].dropna()
                false_returns = df[~df['Hypothesis_True']]['nifty_return'].dropna()
                
                true_avg_return = true_returns.mean() if len(true_returns) > 0 else 0
                false_avg_return = false_returns.mean() if len(false_returns) > 0 else 0
                
                true_win_rate = (true_returns > 0).sum() / len(true_returns) * 100 if len(true_returns) > 0 else 0
                false_win_rate = (false_returns > 0).sum() / len(false_returns) * 100 if len(false_returns) > 0 else 0
                
                # Results summary
                results_div = html.Div([
                    html.H3('📊 Hypothesis Test Results', style={'color': '#2c3e50', 'marginBottom': 20}),
                    
                    html.Div([
                        html.Div([
                            html.Div([
                                html.P('Total Periods Tested', style={'color': '#7f8c8d', 'fontSize': 12, 'marginBottom': 5}),
                                html.P(f"{total_periods:,}", style={'fontSize': 28, 'fontWeight': 'bold', 'color': '#2c3e50', 'margin': 0}),
                            ], style={'textAlign': 'center', 'padding': 20, 'backgroundColor': '#ecf0f1', 'borderRadius': 10, 'flex': 1, 'margin': 10}),
                            
                            html.Div([
                                html.P('Hypothesis TRUE', style={'color': '#7f8c8d', 'fontSize': 12, 'marginBottom': 5}),
                                html.P(f"{true_count:,}", style={'fontSize': 28, 'fontWeight': 'bold', 'color': '#27ae60', 'margin': 0}),
                                html.P(f"{success_rate:.2f}%", style={'fontSize': 14, 'color': '#27ae60', 'margin': 0}),
                            ], style={'textAlign': 'center', 'padding': 20, 'backgroundColor': '#d5f4e6', 'borderRadius': 10, 'flex': 1, 'margin': 10}),
                            
                            html.Div([
                                html.P('Hypothesis FALSE', style={'color': '#7f8c8d', 'fontSize': 12, 'marginBottom': 5}),
                                html.P(f"{false_count:,}", style={'fontSize': 28, 'fontWeight': 'bold', 'color': '#e74c3c', 'margin': 0}),
                                html.P(f"{100-success_rate:.2f}%", style={'fontSize': 14, 'color': '#e74c3c', 'margin': 0}),
                            ], style={'textAlign': 'center', 'padding': 20, 'backgroundColor': '#fadbd8', 'borderRadius': 10, 'flex': 1, 'margin': 10}),
                        ], style={'display': 'flex'}),
                        
                        html.Div([
                            html.H4('Performance Comparison', style={'color': '#2c3e50', 'marginTop': 30, 'marginBottom': 15}),
                            html.Div([
                                html.Div([
                                    html.P('When Hypothesis is TRUE:', style={'fontWeight': 'bold', 'color': '#27ae60', 'marginBottom': 10}),
                                    html.P(f"Avg Return: {true_avg_return:.3f}%", style={'margin': 5}),
                                    html.P(f"Win Rate: {true_win_rate:.2f}%", style={'margin': 5}),
                                    html.P(f"Sample Size: {len(true_returns)}", style={'margin': 5, 'fontSize': 12, 'color': '#7f8c8d'}),
                                ], style={'flex': 1, 'padding': 20, 'backgroundColor': '#d5f4e6', 'borderRadius': 10, 'marginRight': 10}),
                                
                                html.Div([
                                    html.P('When Hypothesis is FALSE:', style={'fontWeight': 'bold', 'color': '#e74c3c', 'marginBottom': 10}),
                                    html.P(f"Avg Return: {false_avg_return:.3f}%", style={'margin': 5}),
                                    html.P(f"Win Rate: {false_win_rate:.2f}%", style={'margin': 5}),
                                    html.P(f"Sample Size: {len(false_returns)}", style={'margin': 5, 'fontSize': 12, 'color': '#7f8c8d'}),
                                ], style={'flex': 1, 'padding': 20, 'backgroundColor': '#fadbd8', 'borderRadius': 10}),
                            ], style={'display': 'flex'}),
                        ]),
                    ]),
                ], style={'backgroundColor': '#fff', 'padding': 20, 'borderRadius': 10, 'marginBottom': 20})
                
                # Chart 1: Time series showing when hypothesis was true
                fig_chart = go.Figure()
                
                # NIFTY price with colored background
                fig_chart.add_trace(go.Scatter(
                    x=df['Date'], y=df['Close_nifty'],
                    name='NIFTY 50',
                    line=dict(color='#3498db', width=2)
                ))
                
                # Add colored regions for true/false
                for i in range(len(df)):
                    if df.iloc[i]['Hypothesis_True']:
                        fig_chart.add_vrect(
                            x0=df.iloc[i]['Date'],
                            x1=df.iloc[i+1]['Date'] if i+1 < len(df) else df.iloc[i]['Date'],
                            fillcolor='rgba(39, 174, 96, 0.1)',
                            line_width=0,
                            layer='below'
                        )
                
                fig_chart.update_layout(
                    title=f"NIFTY 50 Price with Hypothesis Regions (Green = True)",
                    xaxis_title="Date",
                    yaxis_title="NIFTY Price",
                    height=400,
                    showlegend=True
                )
                
                # Chart 2: Distribution comparison
                fig_dist = go.Figure()
                
                if len(true_returns) > 0:
                    fig_dist.add_trace(go.Histogram(
                        x=true_returns,
                        name=f'Returns when TRUE (n={len(true_returns)})',
                        marker_color='#27ae60',
                        opacity=0.7,
                        nbinsx=50
                    ))
                
                if len(false_returns) > 0:
                    fig_dist.add_trace(go.Histogram(
                        x=false_returns,
                        name=f'Returns when FALSE (n={len(false_returns)})',
                        marker_color='#e74c3c',
                        opacity=0.7,
                        nbinsx=50
                    ))
                
                fig_dist.update_layout(
                    title="Return Distribution: Hypothesis TRUE vs FALSE",
                    xaxis_title=f"{timeframe.capitalize()} Return (%)",
                    yaxis_title="Frequency",
                    height=400,
                    barmode='overlay'
                )
                
                return results_div, fig_chart, fig_dist
                
            except Exception as e:
                error_div = html.Div([
                    html.H4('❌ Error Testing Hypothesis', style={'color': '#e74c3c'}),
                    html.P(f"Error: {str(e)}", style={'color': '#7f8c8d', 'padding': 20, 'backgroundColor': '#fadbd8', 'borderRadius': 10}),
                    html.P('Examples of valid hypotheses:',
                          style={'fontSize': 14, 'color': '#7f8c8d', 'marginTop': 10, 'fontWeight': 'bold'}),
                    html.Ul([
                        html.Li('nifty return is greater than 1%'),
                        html.Li('vix is above 20'),
                        html.Li('nifty return is more than mean + 2 standard deviations'),
                        html.Li('last month return is positive and this month return is also positive'),
                    ], style={'color': '#7f8c8d'}),
                ], style={'padding': 20})
                
                return error_div, go.Figure(), go.Figure()
    
    def run(self, debug=True, port=8050):
        """Run the dashboard"""
        self.app.run(debug=debug, port=port)


def main():
    """Main function to run dashboard"""
    print("=" * 60)
    print("Loading Trading Dashboard...")
    print("=" * 60)
    
    # Load data
    try:
        nifty_df = pd.read_csv('nifty_history.csv', parse_dates=['Date'])
        vix_df = pd.read_csv('india_vix_history.csv', parse_dates=['Date'])
        
        print(f"✓ Loaded {len(nifty_df)} NIFTY records")
        print(f"✓ Loaded {len(vix_df)} VIX records")
        
        # Create and run dashboard
        dashboard = TradingDashboard(nifty_df, vix_df)
        
        print("\n" + "=" * 60)
        print("🚀 Dashboard ready!")
        print("Opening at http://localhost:8050")
        print("=" * 60)
        
        dashboard.run(debug=True, port=8050)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n⚠️  Please run 'python nse_data_fetcher.py' first to download data.")


if __name__ == "__main__":
    main()
