"""
VIX Forecasting Dashboard
Lightweight dashboard focused on GARCH-X forecasting
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from vix_forecaster import VIXForecaster
from strategy_selector import MLStrategySelector

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "VIX Forecasting Dashboard"


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
    
    @staticmethod
    def validate_file_path(path, allowed_files):
        """Validate file path is in allowed list"""
        import os
        base_name = os.path.basename(path)
        if base_name not in allowed_files:
            raise ValueError(f"File {base_name} not allowed")
        return path


class ForecastDashboard:
    def __init__(self):
        print("="*60)
        print("Loading VIX Forecasting Dashboard...")
        print("="*60)
        
        # Load data
        self.df = self.load_data()
        
        # Initialize forecaster
        self.forecaster = VIXForecaster(self.df)
        self.forecaster.prepare_features()
        
        # Initialize ML strategy selector
        try:
            self.ml_selector = MLStrategySelector()
            print("✓ ML models loaded")
        except Exception as e:
            print(f"⚠️  ML models not available: {e}")
            self.ml_selector = None
        
        print("✓ Dashboard initialization complete\n")
        
        # Build layout
        self.app = app
        self.build_layout()
        self.setup_callbacks()
    
    def load_data(self):
        """Load NIFTY and VIX data with security validation"""
        validator = SecurityValidator()
        allowed_files = ['nifty_history.csv', 'india_vix_history.csv']
        
        try:
            # Validate and load NIFTY data
            nifty_path = validator.validate_file_path('nifty_history.csv', allowed_files)
            nifty_df = pd.read_csv(nifty_path)
            nifty_df['Date'] = pd.to_datetime(nifty_df['Date'], errors='coerce').dt.tz_localize(None)
            nifty_df = nifty_df.rename(columns={'Close': 'Close_nifty'})
            
            # Validate and load VIX data
            vix_path = validator.validate_file_path('india_vix_history.csv', allowed_files)
            vix_df = pd.read_csv(vix_path)
            vix_df['Date'] = pd.to_datetime(vix_df['Date'], errors='coerce').dt.tz_localize(None)
            vix_df = vix_df.rename(columns={'Close': 'Close_vix'})
            
            # Merge on Date
            df = pd.merge(nifty_df[['Date', 'Close_nifty']], 
                         vix_df[['Date', 'Close_vix']], 
                         on='Date', how='inner')
            df = df.sort_values('Date').reset_index(drop=True)
            
            print(f"✓ Loaded {len(df)} records")
            print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def build_layout(self):
        """Build dashboard layout"""
        
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("📈 VIX Forecasting Dashboard", 
                       style={'color': 'white', 'marginBottom': 0}),
                html.P("GARCH(1,1)-X Model with Exogenous Variables",
                      style={'color': '#bdc3c7', 'fontSize': 16})
            ], style={'backgroundColor': '#2c3e50', 'padding': '20px 40px', 'marginBottom': 30}),
            
            # Controls
            html.Div([
                html.Div([
                    html.Label('Forecast Horizon (days):', 
                              style={'fontWeight': 'bold', 'marginBottom': 5, 'display': 'block'}),
                    dcc.Dropdown(
                        id='forecast-horizon',
                        options=[
                            {'label': '1 Day', 'value': 1},
                            {'label': '5 Days', 'value': 5},
                            {'label': '21 Days (1 Month)', 'value': 21},
                            {'label': '63 Days (3 Months)', 'value': 63}
                        ],
                        value=5,
                        style={'width': 200}
                    )
                ], style={'display': 'inline-block', 'marginRight': 30}),
                
                html.Div([
                    html.Label('Training Window (years):', 
                              style={'fontWeight': 'bold', 'marginBottom': 5, 'display': 'block'}),
                    dcc.Dropdown(
                        id='training-window',
                        options=[
                            {'label': '1 Year', 'value': 252},
                            {'label': '2 Years', 'value': 504},
                            {'label': '3 Years', 'value': 756},
                            {'label': 'All Data', 'value': 0}
                        ],
                        value=504,
                        style={'width': 200}
                    )
                ], style={'display': 'inline-block', 'marginRight': 30}),
                
                html.Div([
                    html.Button('Train Model & Forecast', id='train-btn', n_clicks=0,
                               style={'padding': '10px 30px', 'fontSize': 16, 'fontWeight': 'bold',
                                     'backgroundColor': '#27ae60', 'color': 'white', 'border': 'none',
                                     'borderRadius': 5, 'cursor': 'pointer', 'marginTop': 22})
                ], style={'display': 'inline-block'}),
            ], style={'padding': '20px 40px', 'backgroundColor': '#ecf0f1', 'marginBottom': 20}),
            
            # Model Status
            html.Div(id='model-status', style={'padding': '0 40px', 'marginBottom': 20}),
            
            # Tabs
            dcc.Tabs(id='forecast-tabs', value='forecast-tab', children=[
                # Tab 1: Current Forecast
                dcc.Tab(label='📊 Current Forecast', value='forecast-tab', children=[
                    html.Div([
                        # Lookback Period Recommendation
                        html.Div(id='lookback-recommendation', style={'marginBottom': 20,
                                                                      'padding': 15,
                                                                      'backgroundColor': '#e8f5e9',
                                                                      'borderRadius': 8,
                                                                      'border': '2px solid #4caf50'}),
                        
                        html.Div(id='forecast-metrics', style={'marginBottom': 20}),
                        dcc.Graph(id='forecast-chart'),
                    ], style={'padding': 20})
                ]),
                
                # Tab 2: Backtest Results
                dcc.Tab(label='🔬 Backtest Analysis', value='backtest-tab', children=[
                    html.Div([
                        html.Div([
                            html.Label('Backtest Step (days):', 
                                      style={'fontWeight': 'bold', 'marginRight': 10}),
                            dcc.Input(id='backtest-step', type='number', value=21, 
                                     min=1, max=63, step=1, style={'width': 80, 'padding': 5}),
                            html.Button('Run Backtest', id='backtest-btn', n_clicks=0,
                                       style={'padding': '8px 20px', 'marginLeft': 20,
                                             'backgroundColor': '#3498db', 'color': 'white',
                                             'border': 'none', 'borderRadius': 5, 'cursor': 'pointer'})
                        ], style={'marginBottom': 20}),
                        
                        html.Div(id='backtest-metrics', style={'marginBottom': 20}),
                        dcc.Graph(id='backtest-chart'),
                        dcc.Graph(id='error-distribution'),
                    ], style={'padding': 20})
                ]),
                
                # Tab 3: Model Diagnostics
                dcc.Tab(label='⚙️ Model Diagnostics', value='diagnostics-tab', children=[
                    html.Div([
                        html.H3('Volatility Regime Analysis', style={'color': '#2c3e50'}),
                        html.Div(id='regime-metrics', style={'marginBottom': 20}),
                        dcc.Graph(id='regime-chart'),
                        
                        html.H3('Exogenous Variables', style={'color': '#2c3e50', 'marginTop': 30}),
                        html.Div(id='exog-info', style={'marginBottom': 30}),
                        
                        html.H3('Model Statistics', style={'color': '#2c3e50'}),
                        html.Div(id='model-stats'),
                        
                        dcc.Graph(id='residuals-chart'),
                    ], style={'padding': 20})
                ]),
                
                # Tab 4: ML Strategy Selector
                dcc.Tab(label='🤖 ML Strategy Selector', value='ml-tab', children=[
                    html.Div([
                        html.H3('Machine Learning-Based Strategy Selection', 
                               style={'color': '#2c3e50', 'marginBottom': 20}),
                        
                        html.Div([
                            html.Div([
                                html.Label('Capital (₹):', 
                                          style={'fontWeight': 'bold', 'marginRight': 10}),
                                dcc.Input(id='ml-capital', type='number', value=500000, 
                                         min=10000, max=100000000, step=50000, 
                                         style={'width': 150, 'padding': 5}),
                            ], style={'display': 'inline-block', 'marginRight': 30}),
                            
                            html.Div([
                                html.Label('Confidence Threshold (%):', 
                                          style={'fontWeight': 'bold', 'marginRight': 10}),
                                dcc.Slider(id='ml-confidence', min=40, max=80, value=60, step=5,
                                          marks={i: f'{i}%' for i in range(40, 81, 10)},
                                          tooltip={"placement": "bottom", "always_visible": True}),
                            ], style={'display': 'inline-block', 'width': 300, 'verticalAlign': 'top'}),
                            
                            html.Button('Generate Trading Plan', id='ml-generate-btn', n_clicks=0,
                                       style={'padding': '10px 30px', 'marginLeft': 30,
                                             'backgroundColor': '#e74c3c', 'color': 'white',
                                             'border': 'none', 'borderRadius': 5, 'cursor': 'pointer',
                                             'fontWeight': 'bold'})
                        ], style={'marginBottom': 30, 'padding': 20, 'backgroundColor': '#ecf0f1',
                                 'borderRadius': 10}),
                        
                        # ML Predictions
                        html.Div(id='ml-predictions', style={'marginBottom': 30}),
                        
                        # Strategy Recommendation
                        html.Div(id='ml-strategy', style={'marginBottom': 30}),
                        
                        # Strike Selection
                        html.Div(id='ml-strikes', style={'marginBottom': 30}),
                        
                        # Position Sizing
                        html.Div(id='ml-position', style={'marginBottom': 30}),
                        
                        # Feature Importance
                        dcc.Graph(id='ml-feature-importance'),
                        
                        # Backtest Performance
                        html.Div([
                            html.H3('Backtest: ML vs Static Rules', 
                                   style={'color': '#2c3e50', 'marginTop': 40}),
                            
                            html.Div([
                                html.Div([
                                    html.Label('Backtest Period:', 
                                              style={'fontWeight': 'bold', 'marginRight': 10}),
                                    dcc.DatePickerRange(
                                        id='ml-backtest-dates',
                                        start_date='2020-01-01',
                                        end_date='2024-12-31',
                                        display_format='YYYY-MM-DD',
                                        style={'marginRight': 20}
                                    ),
                                ], style={'display': 'inline-block', 'marginRight': 20}),
                                
                                html.Button('Run ML Backtest', id='ml-backtest-btn', n_clicks=0,
                                           style={'padding': '10px 30px',
                                                 'backgroundColor': '#9b59b6', 'color': 'white',
                                                 'border': 'none', 'borderRadius': 5, 'cursor': 'pointer',
                                                 'fontWeight': 'bold'})
                            ], style={'marginBottom': 20}),
                            
                            html.Div(id='ml-backtest-results'),
                            dcc.Graph(id='ml-backtest-chart'),
                        ])
                    ], style={'padding': 20})
                ])
            ]),
            
            # Hidden div to store backtest results
            dcc.Store(id='backtest-data'),
            dcc.Store(id='model-trained', data=False)
        ])
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        # Setup ML callbacks
        self.setup_ml_callbacks()
        
        @self.app.callback(
            [Output('model-status', 'children'),
             Output('lookback-recommendation', 'children'),
             Output('forecast-metrics', 'children'),
             Output('forecast-chart', 'figure'),
             Output('regime-metrics', 'children'),
             Output('regime-chart', 'figure'),
             Output('model-stats', 'children'),
             Output('exog-info', 'children'),
             Output('residuals-chart', 'figure'),
             Output('model-trained', 'data')],
            [Input('train-btn', 'n_clicks')],
            [State('forecast-horizon', 'value'),
             State('training-window', 'value')]
        )
        def train_and_forecast(n_clicks, horizon, window):
            if n_clicks == 0:
                empty_fig = go.Figure()
                empty_fig.update_layout(
                    title="Click 'Train Model & Forecast' to generate predictions",
                    height=500
                )
                return (None, None, None, empty_fig, None, empty_fig, None, None, empty_fig, False)
                return "", "", empty_fig, "", empty_fig, "", "", empty_fig, False
            
            # Train model
            try:
                # Get lookback recommendation
                lookback_rec = self.forecaster.recommend_lookback_period()
                
                # Create recommendation display
                rec_html = html.Div([
                    html.H4([
                        "💡 Recommended Lookback Period: ",
                        html.Span(f"{lookback_rec['recommended_period']} Years", 
                                 style={'color': '#2ecc71', 'fontWeight': 'bold'})
                    ], style={'marginBottom': 10}),
                    html.P(f"Confidence: {lookback_rec['confidence']:.0f}%", 
                          style={'fontSize': 14, 'color': '#7f8c8d', 'marginBottom': 10}),
                    html.P([
                        html.Strong("Current Market: "),
                        f"VIX={lookback_rec['current_vix']:.2f} ({lookback_rec['vix_regime']} regime), ",
                        f"63-day MA={lookback_rec['vix_mean_63d']:.2f}"
                    ], style={'fontSize': 13, 'marginBottom': 10}),
                    html.Div([
                        html.Strong("Reasoning:", style={'display': 'block', 'marginBottom': 5}),
                        html.Ul([html.Li(reason, style={'fontSize': 12}) for reason in lookback_rec['reasons']])
                    ])
                ])
                
                if window > 0:
                    train_end = self.df['Date'].max() - timedelta(days=horizon)
                    train_start = train_end - timedelta(days=window)
                    train_data = self.df[
                        (self.df['Date'] >= train_start) & 
                        (self.df['Date'] <= train_end)
                    ]
                    temp_forecaster = VIXForecaster(train_data)
                    temp_forecaster.prepare_features()
                    temp_forecaster.select_exogenous_variables()
                    temp_forecaster.train_model()
                else:
                    self.forecaster.select_exogenous_variables()
                    self.forecaster.train_model()
                    temp_forecaster = self.forecaster
                
                # Generate forecast
                forecast_results = temp_forecaster.forecast(horizon=horizon)
                
                # Model status
                status = html.Div([
                    html.H4("✓ Model Trained Successfully", style={'color': '#27ae60', 'marginBottom': 10}),
                    html.P(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                ], style={'backgroundColor': '#d5f4e6', 'padding': 15, 'borderRadius': 5})
                
                # Forecast metrics
                last_vix = forecast_results['last_vix']
                actual_latest_vix = self.df['Close_vix'].iloc[-1]  # Latest VIX from data
                actual_latest_date = self.df['Date'].iloc[-1]  # Latest date
                forecast_vix = forecast_results['vix_forecast'][-1]
                change = forecast_vix - last_vix
                change_pct = (change / last_vix) * 100
                kappa = forecast_results.get('kappa', 0.1)
                long_run_mean = forecast_results.get('long_run_mean', last_vix)
                
                metrics = html.Div([
                    html.Div([
                        html.Div([
                            html.P("Latest VIX (Actual)", style={'color': '#7f8c8d', 'fontSize': 14}),
                            html.P(f"{actual_latest_vix:.2f}", 
                                  style={'fontSize': 32, 'fontWeight': 'bold', 'color': '#2c3e50'}),
                            html.P(f"As of {actual_latest_date.strftime('%d %b %Y')}", 
                                  style={'fontSize': 11, 'color': '#95a5a6'})
                        ], style={'flex': 1, 'textAlign': 'center'}),
                        
                        html.Div([
                            html.P("Training Cutoff VIX", style={'color': '#7f8c8d', 'fontSize': 14}),
                            html.P(f"{last_vix:.2f}", 
                                  style={'fontSize': 28, 'fontWeight': 'bold', 'color': '#7f8c8d'}),
                            html.P(f"({horizon} days before latest)", 
                                  style={'fontSize': 11, 'color': '#95a5a6'})
                        ], style={'flex': 1, 'textAlign': 'center'}),
                        
                        html.Div([
                            html.P(f"{horizon}-Day Forecast", style={'color': '#7f8c8d', 'fontSize': 14}),
                            html.P(f"{forecast_vix:.2f}", 
                                  style={'fontSize': 32, 'fontWeight': 'bold', 
                                        'color': '#e74c3c' if forecast_vix > last_vix else '#27ae60'})
                        ], style={'flex': 1, 'textAlign': 'center'}),
                        
                        html.Div([
                            html.P("Expected Change", style={'color': '#7f8c8d', 'fontSize': 14}),
                            html.P(f"{change:+.2f} ({change_pct:+.1f}%)", 
                                  style={'fontSize': 32, 'fontWeight': 'bold', 
                                        'color': '#e74c3c' if change > 0 else '#27ae60'})
                        ], style={'flex': 1, 'textAlign': 'center'}),
                    ], style={'display': 'flex', 'gap': 20}),
                    html.Div([
                        html.Div([
                            html.P("Mean Reversion Speed (κ)", style={'color': '#7f8c8d', 'fontSize': 12}),
                            html.P(f"{kappa:.4f}", 
                                  style={'fontSize': 20, 'fontWeight': 'bold', 'color': '#3498db'})
                        ], style={'flex': 1, 'textAlign': 'center'}),
                        html.Div([
                            html.P("Long-run Mean VIX", style={'color': '#7f8c8d', 'fontSize': 12}),
                            html.P(f"{long_run_mean:.2f}", 
                                  style={'fontSize': 20, 'fontWeight': 'bold', 'color': '#3498db'})
                        ], style={'flex': 1, 'textAlign': 'center'}),
                        html.Div([
                            html.P("Half-life (days)", style={'color': '#7f8c8d', 'fontSize': 12}),
                            html.P(f"{(0.693 / kappa):.1f}" if kappa > 0 else "∞", 
                                  style={'fontSize': 20, 'fontWeight': 'bold', 'color': '#3498db'})
                        ], style={'flex': 1, 'textAlign': 'center'}),
                    ], style={'display': 'flex', 'gap': 20, 'marginTop': 15, 'paddingTop': 15, 'borderTop': '1px solid #ecf0f1'})
                ], style={'backgroundColor': '#fff', 'padding': 20, 'borderRadius': 5, 'marginBottom': 20})
                
                # Forecast chart
                forecast_fig = self.create_forecast_chart(forecast_results)
                
                # Model diagnostics
                diagnostics = temp_forecaster.get_model_diagnostics()
                stats_html = self.create_diagnostics_html(diagnostics)
                
                # Exogenous variables info
                exog_html = html.Div([
                    html.P(f"✓ {var}", style={'fontSize': 14, 'marginBottom': 5}) 
                    for var in temp_forecaster.exog_vars
                ])
                
                # Residuals chart
                residuals_fig = self.create_residuals_chart(temp_forecaster)
                
                # Regime analysis
                regime_metrics = self.create_regime_metrics()
                regime_fig = self.create_regime_chart()
                
                return status, rec_html, metrics, forecast_fig, regime_metrics, regime_fig, stats_html, exog_html, residuals_fig, True
                
            except Exception as e:
                error_status = html.Div([
                    html.H4("❌ Error Training Model", style={'color': '#e74c3c'}),
                    html.P(str(e))
                ], style={'backgroundColor': '#fadbd8', 'padding': 15, 'borderRadius': 5})
                
                empty_fig = go.Figure()
                return error_status, None, "", empty_fig, "", empty_fig, "", "", empty_fig, False
        
        @self.app.callback(
            [Output('backtest-metrics', 'children'),
             Output('backtest-chart', 'figure'),
             Output('error-distribution', 'figure'),
             Output('backtest-data', 'data')],
            [Input('backtest-btn', 'n_clicks')],
            [State('training-window', 'value'),
             State('forecast-horizon', 'value'),
             State('backtest-step', 'value'),
             State('model-trained', 'data')]
        )
        def run_backtest(n_clicks, window, horizon, step, model_trained):
            if n_clicks == 0:
                empty_fig = go.Figure()
                empty_fig.update_layout(title="Click 'Run Backtest' to start analysis")
                return "", empty_fig, empty_fig, None
            
            try:
                # Run rolling forecast
                window_size = window if window > 0 else 252 * 2
                backtest_results = self.forecaster.rolling_forecast(
                    window_size=window_size,
                    horizon=horizon,
                    step=step
                )
                
                if backtest_results.empty:
                    return "No backtest results", go.Figure(), go.Figure(), None
                
                # Calculate metrics
                mae = backtest_results['Forecast_Error'].abs().mean()
                rmse = np.sqrt((backtest_results['Forecast_Error']**2).mean())
                mape = backtest_results['Forecast_Error_Pct'].abs().mean()
                
                # Direction accuracy
                actual_direction = (backtest_results['Actual_VIX'].diff() > 0).astype(int)
                forecast_direction = (backtest_results['Forecast_VIX'].diff() > 0).astype(int)
                direction_accuracy = (actual_direction == forecast_direction).sum() / len(actual_direction) * 100
                
                metrics = html.Div([
                    html.H4("Backtest Results", style={'color': '#2c3e50', 'marginBottom': 15}),
                    html.Div([
                        html.Div([
                            html.P("MAE", style={'color': '#7f8c8d', 'fontSize': 12}),
                            html.P(f"{mae:.2f}", style={'fontSize': 24, 'fontWeight': 'bold', 'color': '#3498db'})
                        ], style={'flex': 1, 'textAlign': 'center'}),
                        html.Div([
                            html.P("RMSE", style={'color': '#7f8c8d', 'fontSize': 12}),
                            html.P(f"{rmse:.2f}", style={'fontSize': 24, 'fontWeight': 'bold', 'color': '#3498db'})
                        ], style={'flex': 1, 'textAlign': 'center'}),
                        html.Div([
                            html.P("MAPE", style={'color': '#7f8c8d', 'fontSize': 12}),
                            html.P(f"{mape:.1f}%", style={'fontSize': 24, 'fontWeight': 'bold', 'color': '#3498db'})
                        ], style={'flex': 1, 'textAlign': 'center'}),
                        html.Div([
                            html.P("Direction Accuracy", style={'color': '#7f8c8d', 'fontSize': 12}),
                            html.P(f"{direction_accuracy:.1f}%", 
                                  style={'fontSize': 24, 'fontWeight': 'bold', 'color': '#27ae60'})
                        ], style={'flex': 1, 'textAlign': 'center'}),
                    ], style={'display': 'flex', 'gap': 20})
                ], style={'backgroundColor': '#fff', 'padding': 20, 'borderRadius': 5})
                
                # Charts
                backtest_fig = self.create_backtest_chart(backtest_results)
                error_fig = self.create_error_distribution(backtest_results)
                
                # Store results
                data = backtest_results.to_dict('records')
                
                return metrics, backtest_fig, error_fig, data
                
            except Exception as e:
                error_msg = html.Div([
                    html.P(f"❌ Error running backtest: {str(e)}", style={'color': '#e74c3c'})
                ])
                return error_msg, go.Figure(), go.Figure(), None
    
    def create_forecast_chart(self, forecast_results):
        """Create forecast visualization"""
        # Get historical VIX
        hist_vix = self.forecaster.df_features[['Date', 'Close_vix']].tail(252)
        
        # Forecast values
        forecast_dates = forecast_results['forecast_dates']
        forecast_vix = forecast_results['vix_forecast']
        
        fig = go.Figure()
        
        # Historical VIX
        fig.add_trace(go.Scatter(
            x=hist_vix['Date'],
            y=hist_vix['Close_vix'],
            mode='lines',
            name='Historical VIX',
            line=dict(color='#3498db', width=2)
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_vix,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#e74c3c', width=3, dash='dash'),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=f"VIX Forecast ({forecast_results['forecast_horizon']} days)",
            xaxis_title="Date",
            yaxis_title="VIX",
            height=500,
            hovermode='x unified',
            legend=dict(x=0, y=1)
        )
        
        return fig
    
    def create_backtest_chart(self, backtest_results):
        """Create backtest results chart"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Forecast vs Actual', 'Forecast Errors'),
            vertical_spacing=0.12,
            row_heights=[0.6, 0.4]
        )
        
        # Actual vs Forecast
        fig.add_trace(
            go.Scatter(x=backtest_results['Date'], y=backtest_results['Actual_VIX'],
                      name='Actual VIX', line=dict(color='#3498db', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=backtest_results['Date'], y=backtest_results['Forecast_VIX'],
                      name='Forecast', line=dict(color='#e74c3c', width=2, dash='dash')),
            row=1, col=1
        )
        
        # Errors
        fig.add_trace(
            go.Scatter(x=backtest_results['Date'], y=backtest_results['Forecast_Error'],
                      name='Forecast Error', fill='tozeroy',
                      line=dict(color='#9b59b6', width=1)),
            row=2, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        fig.update_layout(height=700, hovermode='x unified')
        
        return fig
    
    def create_error_distribution(self, backtest_results):
        """Create error distribution chart"""
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=backtest_results['Forecast_Error'],
            name='Error Distribution',
            nbinsx=30,
            marker_color='#3498db'
        ))
        
        fig.update_layout(
            title="Forecast Error Distribution",
            xaxis_title="Forecast Error",
            yaxis_title="Frequency",
            height=400
        )
        
        return fig
    
    def create_diagnostics_html(self, diagnostics):
        """Create model diagnostics HTML"""
        if diagnostics is None:
            return html.P("No diagnostics available")
        
        return html.Div([
            html.Div([
                html.Div([
                    html.P("AIC", style={'color': '#7f8c8d', 'fontSize': 12}),
                    html.P(f"{diagnostics['AIC']:.2f}", 
                          style={'fontSize': 20, 'fontWeight': 'bold', 'color': '#2c3e50'})
                ], style={'flex': 1}),
                html.Div([
                    html.P("BIC", style={'color': '#7f8c8d', 'fontSize': 12}),
                    html.P(f"{diagnostics['BIC']:.2f}", 
                          style={'fontSize': 20, 'fontWeight': 'bold', 'color': '#2c3e50'})
                ], style={'flex': 1}),
                html.Div([
                    html.P("Log Likelihood", style={'color': '#7f8c8d', 'fontSize': 12}),
                    html.P(f"{diagnostics['Log_Likelihood']:.2f}", 
                          style={'fontSize': 20, 'fontWeight': 'bold', 'color': '#2c3e50'})
                ], style={'flex': 1}),
            ], style={'display': 'flex', 'gap': 20, 'marginBottom': 20})
        ])
    
    def create_residuals_chart(self, forecaster):
        """Create residuals analysis chart"""
        if forecaster.model_fit is None:
            return go.Figure()
        
        residuals = forecaster.model_fit.resid
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Residuals Over Time', 'Residuals Distribution')
        )
        
        # Residuals time series
        fig.add_trace(
            go.Scatter(y=residuals, mode='lines', name='Residuals',
                      line=dict(color='#3498db', width=1)),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        
        # Distribution
        fig.add_trace(
            go.Histogram(x=residuals, name='Distribution', nbinsx=30,
                        marker_color='#9b59b6'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        
        return fig
    
    def create_regime_metrics(self):
        """Create regime analysis metrics"""
        # Calculate regime statistics
        df = self.df.copy()
        df['VIX_MA_63'] = df['Close_vix'].rolling(63).mean()
        df['VIX_regime'] = (df['Close_vix'] > df['VIX_MA_63']).astype(int)
        df = df.dropna()
        
        # Current regime
        current_regime = df['VIX_regime'].iloc[-1]
        current_vix = df['Close_vix'].iloc[-1]
        current_ma = df['VIX_MA_63'].iloc[-1]
        
        # Regime statistics
        high_vol_pct = (df['VIX_regime'] == 1).sum() / len(df) * 100
        low_vol_pct = (df['VIX_regime'] == 0).sum() / len(df) * 100
        
        # Calculate average regime duration
        df['regime_change'] = df['VIX_regime'].diff().fillna(0)
        regime_starts = df[df['regime_change'] != 0].index.tolist()
        regime_durations = [regime_starts[i+1] - regime_starts[i] for i in range(len(regime_starts)-1)]
        avg_duration = np.mean(regime_durations) if regime_durations else 0
        
        # Distance to regime threshold
        distance_to_threshold = abs(current_vix - current_ma)
        distance_pct = (distance_to_threshold / current_ma) * 100
        
        regime_html = html.Div([
            html.Div([
                html.Div([
                    html.P("Current Regime", style={'color': '#7f8c8d', 'fontSize': 14}),
                    html.P("HIGH VOLATILITY" if current_regime == 1 else "LOW VOLATILITY", 
                          style={'fontSize': 28, 'fontWeight': 'bold', 
                                'color': '#e74c3c' if current_regime == 1 else '#27ae60'})
                ], style={'flex': 1, 'textAlign': 'center'}),
                
                html.Div([
                    html.P("VIX vs 63-Day MA", style={'color': '#7f8c8d', 'fontSize': 14}),
                    html.P(f"{current_vix:.2f} vs {current_ma:.2f}", 
                          style={'fontSize': 20, 'fontWeight': 'bold', 'color': '#2c3e50'}),
                    html.P(f"{'Above' if current_regime == 1 else 'Below'} threshold by {distance_pct:.1f}%",
                          style={'fontSize': 12, 'color': '#7f8c8d'})
                ], style={'flex': 1, 'textAlign': 'center'}),
                
                html.Div([
                    html.P("Historical Distribution", style={'color': '#7f8c8d', 'fontSize': 14}),
                    html.P(f"{high_vol_pct:.1f}% High Vol", 
                          style={'fontSize': 16, 'color': '#e74c3c'}),
                    html.P(f"{low_vol_pct:.1f}% Low Vol", 
                          style={'fontSize': 16, 'color': '#27ae60'})
                ], style={'flex': 1, 'textAlign': 'center'}),
                
                html.Div([
                    html.P("Avg Regime Duration", style={'color': '#7f8c8d', 'fontSize': 14}),
                    html.P(f"{avg_duration:.0f} days", 
                          style={'fontSize': 20, 'fontWeight': 'bold', 'color': '#3498db'})
                ], style={'flex': 1, 'textAlign': 'center'}),
            ], style={'display': 'flex', 'gap': 20}),
        ], style={'backgroundColor': '#fff', 'padding': 20, 'borderRadius': 5, 'marginBottom': 20})
        
        return regime_html
    
    def create_regime_chart(self):
        """Create volatility regime visualization"""
        df = self.df.copy()
        df['VIX_MA_63'] = df['Close_vix'].rolling(63).mean()
        df['VIX_regime'] = (df['Close_vix'] > df['VIX_MA_63']).astype(int)
        df = df.dropna()
        
        # Take last 2 years for better visibility
        df_recent = df.tail(504)  # ~2 years of trading days
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=('VIX with Regime Threshold (63-Day MA)', 'Volatility Regime'),
            row_heights=[0.7, 0.3]
        )
        
        # VIX and threshold
        fig.add_trace(
            go.Scatter(x=df_recent['Date'], y=df_recent['Close_vix'],
                      name='India VIX', line=dict(color='#3498db', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df_recent['Date'], y=df_recent['VIX_MA_63'],
                      name='63-Day MA (Threshold)', line=dict(color='#e74c3c', width=2, dash='dash')),
            row=1, col=1
        )
        
        # Regime indicator
        colors = ['#27ae60' if r == 0 else '#e74c3c' for r in df_recent['VIX_regime']]
        fig.add_trace(
            go.Bar(x=df_recent['Date'], y=df_recent['VIX_regime'],
                  name='Regime', marker_color=colors, showlegend=False),
            row=2, col=1
        )
        
        fig.update_yaxes(title_text="VIX Level", row=1, col=1)
        fig.update_yaxes(title_text="Regime", ticktext=['Low Vol', 'High Vol'], 
                        tickvals=[0, 1], row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        fig.update_layout(height=600, title_text="Volatility Regime Analysis (Last 2 Years)")
        
        return fig
    
    def setup_ml_callbacks(self):
        """Setup ML strategy selector callbacks"""
        
        @self.app.callback(
            [Output('ml-predictions', 'children'),
             Output('ml-strategy', 'children'),
             Output('ml-strikes', 'children'),
             Output('ml-position', 'children'),
             Output('ml-feature-importance', 'figure')],
            [Input('ml-generate-btn', 'n_clicks')],
            [State('ml-capital', 'value'),
             State('ml-confidence', 'value')]
        )
        def generate_ml_plan(n_clicks, capital, confidence_threshold):
            if n_clicks == 0 or self.ml_selector is None:
                empty_fig = go.Figure()
                empty_fig.update_layout(title="Generate trading plan to see feature importance")
                return "", "", "", "", empty_fig
            
            # Validate inputs
            validator = SecurityValidator()
            capital = validator.validate_numeric(capital, 10000, 100000000, "capital")
            confidence_threshold = validator.validate_numeric(confidence_threshold, 0, 100, "confidence")
            
            try:
                # Prepare features
                self.ml_selector.prepare_current_features()
                
                # Get predictions
                predictions = self.ml_selector.get_predictions()
                
                # Predictions display
                pred_html = html.Div([
                    html.H4('📊 Market Forecast (22-day horizon)', 
                           style={'color': '#2c3e50', 'marginBottom': 15}),
                    html.Div([
                        html.Div([
                            html.P("Direction", style={'color': '#7f8c8d', 'fontSize': 12, 'marginBottom': 5}),
                            html.P(predictions['direction'], 
                                  style={'fontSize': 24, 'fontWeight': 'bold', 
                                        'color': '#27ae60' if predictions['direction'] == 'UP' else '#e74c3c'})
                        ], style={'flex': 1, 'textAlign': 'center'}),
                        
                        html.Div([
                            html.P("Confidence", style={'color': '#7f8c8d', 'fontSize': 12, 'marginBottom': 5}),
                            html.P(f"{predictions['confidence']:.1f}%", 
                                  style={'fontSize': 24, 'fontWeight': 'bold', 'color': '#2c3e50'})
                        ], style={'flex': 1, 'textAlign': 'center'}),
                        
                        html.Div([
                            html.P("Expected Move", style={'color': '#7f8c8d', 'fontSize': 12, 'marginBottom': 5}),
                            html.P(f"{'+' if predictions['direction'] == 'UP' else '-'}{predictions['expected_move']:.2f}%", 
                                  style={'fontSize': 24, 'fontWeight': 'bold', 
                                        'color': '#27ae60' if predictions['direction'] == 'UP' else '#e74c3c'})
                        ], style={'flex': 1, 'textAlign': 'center'}),
                        
                        html.Div([
                            html.P("Tail Risk", style={'color': '#7f8c8d', 'fontSize': 12, 'marginBottom': 5}),
                            html.P(f"{predictions['tail_risk_prob']:.2f}%", 
                                  style={'fontSize': 24, 'fontWeight': 'bold',
                                        'color': '#e74c3c' if predictions['tail_risk_prob'] > 5 else '#27ae60'})
                        ], style={'flex': 1, 'textAlign': 'center'}),
                    ], style={'display': 'flex', 'gap': 20}),
                    
                    html.Div([
                        html.P([html.Strong("Predicted NIFTY: "), 
                               f"{predictions['predicted_nifty']:.2f}"], 
                              style={'fontSize': 14, 'color': '#34495e', 'marginTop': 15}),
                        html.P([html.Strong("Predicted Range: "), 
                               f"{predictions['predicted_range_low']:.2f} - {predictions['predicted_range_high']:.2f}"], 
                              style={'fontSize': 14, 'color': '#34495e'}),
                    ])
                ], style={'padding': 20, 'backgroundColor': '#ecf0f1', 'borderRadius': 10})
                
                # Strategy selection
                strategy_rec = self.ml_selector.select_strategy(predictions, confidence_threshold)
                
                strategy_html = html.Div([
                    html.H4('🎯 Strategy Recommendation', 
                           style={'color': '#2c3e50', 'marginBottom': 15}),
                    html.Div([
                        html.H2(strategy_rec['strategy'], 
                               style={'color': '#e74c3c', 'marginBottom': 10}),
                        html.P(strategy_rec['reasoning'], 
                              style={'fontSize': 16, 'color': '#34495e'})
                    ], style={'padding': 20, 'backgroundColor': '#fff', 
                             'border': '2px solid #e74c3c', 'borderRadius': 10})
                ], style={'padding': 20, 'backgroundColor': '#ecf0f1', 'borderRadius': 10})
                
                # Strike selection
                strikes = self.ml_selector.calculate_strike_distances(predictions)
                
                current_nifty = self.ml_selector.current_nifty
                atm_strike = round(current_nifty / 50) * 50
                
                strike_html = html.Div([
                    html.H4('📍 Strike Selection', 
                           style={'color': '#2c3e50', 'marginBottom': 15}),
                    html.Div([
                        html.P([html.Strong("Current NIFTY: "), f"{current_nifty:.2f}"], 
                              style={'fontSize': 14, 'marginBottom': 5}),
                        html.P([html.Strong("ATM Strike: "), f"{atm_strike}"], 
                              style={'fontSize': 14, 'marginBottom': 15}),
                        
                        html.Div([
                            html.Div([
                                html.P("Iron Condor", style={'fontWeight': 'bold', 'fontSize': 16, 'marginBottom': 10}),
                                html.P(f"Short Call: {atm_strike + strikes['ic_short_call']}", style={'fontSize': 14}),
                                html.P(f"Long Call: {atm_strike + strikes['ic_long_call']}", style={'fontSize': 14}),
                                html.P(f"Short Put: {atm_strike - strikes['ic_short_put']}", style={'fontSize': 14}),
                                html.P(f"Long Put: {atm_strike - strikes['ic_long_put']}", style={'fontSize': 14}),
                            ], style={'flex': 1, 'padding': 15, 'backgroundColor': '#fff', 'borderRadius': 5}),
                            
                            html.Div([
                                html.P("Short Strangle", style={'fontWeight': 'bold', 'fontSize': 16, 'marginBottom': 10}),
                                html.P(f"Short Call: {atm_strike + strikes['strangle_call']}", style={'fontSize': 14}),
                                html.P(f"Short Put: {atm_strike - strikes['strangle_put']}", style={'fontSize': 14}),
                            ], style={'flex': 1, 'padding': 15, 'backgroundColor': '#fff', 'borderRadius': 5}),
                            
                            html.Div([
                                html.P("Directional", style={'fontWeight': 'bold', 'fontSize': 16, 'marginBottom': 10}),
                                html.P(f"Strike: {atm_strike + strikes['directional_distance']}", style={'fontSize': 14}),
                                html.P(f"Spread Width: {strikes['spread_width']}", style={'fontSize': 14}),
                            ], style={'flex': 1, 'padding': 15, 'backgroundColor': '#fff', 'borderRadius': 5}),
                        ], style={'display': 'flex', 'gap': 15})
                    ], style={'padding': 20, 'backgroundColor': '#ecf0f1', 'borderRadius': 10})
                ])
                
                # Position sizing
                position = self.ml_selector.calculate_position_size(predictions, capital)
                
                position_html = html.Div([
                    html.H4('💰 Position Sizing (Kelly Criterion)', 
                           style={'color': '#2c3e50', 'marginBottom': 15}),
                    html.Div([
                        html.Div([
                            html.P("Kelly Fraction", style={'color': '#7f8c8d', 'fontSize': 12, 'marginBottom': 5}),
                            html.P(f"{position['kelly_fraction']:.2f}%", 
                                  style={'fontSize': 20, 'fontWeight': 'bold', 'color': '#2c3e50'})
                        ], style={'flex': 1, 'textAlign': 'center'}),
                        
                        html.Div([
                            html.P("Tail-Adjusted", style={'color': '#7f8c8d', 'fontSize': 12, 'marginBottom': 5}),
                            html.P(f"{position['tail_adjusted_fraction']:.2f}%", 
                                  style={'fontSize': 20, 'fontWeight': 'bold', 'color': '#2c3e50'})
                        ], style={'flex': 1, 'textAlign': 'center'}),
                        
                        html.Div([
                            html.P("Position Size", style={'color': '#7f8c8d', 'fontSize': 12, 'marginBottom': 5}),
                            html.P(f"₹{position['position_size']:,.0f}", 
                                  style={'fontSize': 20, 'fontWeight': 'bold', 'color': '#27ae60'})
                        ], style={'flex': 1, 'textAlign': 'center'}),
                        
                        html.Div([
                            html.P("Recommended Lots", style={'color': '#7f8c8d', 'fontSize': 12, 'marginBottom': 5}),
                            html.P(f"{position['recommended_lots']}", 
                                  style={'fontSize': 20, 'fontWeight': 'bold', 'color': '#e74c3c'})
                        ], style={'flex': 1, 'textAlign': 'center'}),
                    ], style={'display': 'flex', 'gap': 20})
                ], style={'padding': 20, 'backgroundColor': '#ecf0f1', 'borderRadius': 10})
                
                # Feature importance chart
                importance_data = [
                    ('direction', self.ml_selector.ml.direction_importance),
                    ('magnitude', self.ml_selector.ml.magnitude_importance),
                    ('tail_risk', self.ml_selector.ml.tail_importance)
                ]
                
                fig = make_subplots(
                    rows=1, cols=3,
                    subplot_titles=('Direction Model', 'Magnitude Model', 'Tail Risk Model')
                )
                
                for idx, (name, importance_df) in enumerate(importance_data, 1):
                    top_10 = importance_df.head(10)
                    fig.add_trace(
                        go.Bar(x=top_10['importance'], y=top_10['feature'],
                              orientation='h', name=name,
                              marker_color=['#3498db', '#27ae60', '#e74c3c'][idx-1]),
                        row=1, col=idx
                    )
                
                fig.update_layout(height=500, showlegend=False, title_text="Top 10 Features by Model")
                fig.update_yaxes(autorange="reversed")
                
                return pred_html, strategy_html, strike_html, position_html, fig
                
            except Exception as e:
                import traceback
                error_msg = html.Div([
                    html.P(f"Error generating plan: {str(e)}", style={'color': '#e74c3c'}),
                    html.Pre(traceback.format_exc(), style={'fontSize': 10, 'backgroundColor': '#ecf0f1', 'padding': 10})
                ])
                empty_fig = go.Figure()
                return error_msg, "", "", "", empty_fig
        
        @self.app.callback(
            [Output('ml-backtest-results', 'children'),
             Output('ml-backtest-chart', 'figure')],
            [Input('ml-backtest-btn', 'n_clicks')],
            [State('ml-backtest-dates', 'start_date'),
             State('ml-backtest-dates', 'end_date')]
        )
        def run_ml_backtest(n_clicks, start_date, end_date):
            if n_clicks == 0 or self.ml_selector is None:
                empty_fig = go.Figure()
                empty_fig.update_layout(title="Click 'Run ML Backtest' to see performance")
                return "", empty_fig
            
            try:
                # Run backtest with user-selected dates
                backtest_df = self.ml_selector.backtest_strategy_selection(
                    start_date=start_date if start_date else '2020-01-01',
                    end_date=end_date if end_date else '2024-12-31'
                )
                
                # Ensure we have the right column names
                if 'Date' in backtest_df.columns:
                    backtest_df['date'] = backtest_df['Date']
                
                # Add calculated columns needed for visualization
                if 'actual_magnitude' in backtest_df.columns and 'pred_magnitude' in backtest_df.columns:
                    backtest_df['actual_move'] = backtest_df['actual_magnitude']
                    backtest_df['predicted_move'] = backtest_df['pred_magnitude']
                
                # Calculate metrics
                ml_accuracy = backtest_df['ml_correct'].mean() * 100
                ml_mag_error = backtest_df['ml_mag_error'].mean()
                
                ml_strategy_counts = backtest_df['ml_strategy'].value_counts()
                static_strategy_counts = backtest_df['static_strategy'].value_counts()
                
                # Create backtest visualization
                backtest_fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.08,
                    subplot_titles=(
                        'Direction Prediction Accuracy Over Time',
                        'Magnitude Prediction: Actual vs Predicted',
                        'Strategy Selection Distribution'
                    ),
                    row_heights=[0.35, 0.35, 0.3]
                )
                
                # Row 1: Direction accuracy (rolling)
                backtest_df['ml_correct_rolling'] = backtest_df['ml_correct'].rolling(20, min_periods=1).mean() * 100
                backtest_fig.add_trace(
                    go.Scatter(x=backtest_df['date'], y=backtest_df['ml_correct_rolling'],
                              name='ML Direction Accuracy (20-day MA)', 
                              line=dict(color='#27ae60', width=2)),
                    row=1, col=1
                )
                backtest_fig.add_hline(y=50, line_dash="dash", line_color="gray", 
                                      annotation_text="Random (50%)", row=1, col=1)
                backtest_fig.add_hline(y=ml_accuracy, line_dash="dot", line_color="#e74c3c",
                                      annotation_text=f"Average: {ml_accuracy:.1f}%", row=1, col=1)
                
                # Row 2: Actual vs Predicted magnitude
                backtest_fig.add_trace(
                    go.Scatter(x=backtest_df['date'], y=backtest_df['actual_move'],
                              name='Actual Move', mode='markers',
                              marker=dict(color='#3498db', size=6)),
                    row=2, col=1
                )
                backtest_fig.add_trace(
                    go.Scatter(x=backtest_df['date'], y=backtest_df['predicted_move'],
                              name='Predicted Move', mode='markers',
                              marker=dict(color='#e74c3c', size=6, symbol='x')),
                    row=2, col=1
                )
                
                # Row 3: Strategy distribution over time
                strategy_colors = {
                    'DIRECTIONAL': '#27ae60',
                    'LONG_CALL': '#27ae60',
                    'LONG_PUT': '#c0392b',
                    'IRON_CONDOR': '#3498db',
                    'SHORT_STRANGLE': '#f39c12',
                    'AVOID': '#e74c3c'
                }
                
                for strategy in backtest_df['ml_strategy'].unique():
                    strategy_mask = backtest_df['ml_strategy'] == strategy
                    backtest_fig.add_trace(
                        go.Scatter(x=backtest_df[strategy_mask]['date'],
                                  y=[strategy] * strategy_mask.sum(),
                                  mode='markers',
                                  name=strategy,
                                  marker=dict(color=strategy_colors.get(strategy, '#95a5a6'), 
                                            size=8, symbol='square')),
                        row=3, col=1
                    )
                
                backtest_fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)
                backtest_fig.update_yaxes(title_text="Move (%)", row=2, col=1)
                backtest_fig.update_yaxes(title_text="Strategy", row=3, col=1)
                backtest_fig.update_xaxes(title_text="Date", row=3, col=1)
                
                # Get actual date range from data
                date_range_str = f"{backtest_df['date'].min().strftime('%Y-%m-%d')} to {backtest_df['date'].max().strftime('%Y-%m-%d')}"
                
                backtest_fig.update_layout(
                    height=900, 
                    title_text=f"ML Market Forecast Backtest Performance ({date_range_str})",
                    showlegend=True
                )
                
                return html.Div([
                    html.H4(f'Backtest Results ({date_range_str})', style={'color': '#2c3e50', 'marginBottom': 20}),
                    
                    html.Div([
                        html.Div([
                            html.H5('ML System', style={'color': '#27ae60', 'marginBottom': 10}),
                            html.P(f"Direction Accuracy: {ml_accuracy:.2f}%", style={'fontSize': 14}),
                            html.P(f"Magnitude MAE: {ml_mag_error:.2f}%", style={'fontSize': 14}),
                            html.P(f"Total Signals: {len(backtest_df)}", style={'fontSize': 14}),
                            html.Hr(),
                            html.P("Strategy Distribution:", style={'fontWeight': 'bold', 'marginTop': 10}),
                            *[html.P(f"{strategy}: {count} ({count/len(backtest_df)*100:.1f}%)", 
                                    style={'fontSize': 13, 'marginLeft': 10})
                              for strategy, count in ml_strategy_counts.items()]
                        ], style={'flex': 1, 'padding': 20, 'backgroundColor': '#ecf0f1', 'borderRadius': 10}),
                        
                        html.Div([
                            html.H5('Static VIX Rules', style={'color': '#95a5a6', 'marginBottom': 10}),
                            html.P(f"Direction Accuracy: ~50% (no prediction)", style={'fontSize': 14}),
                            html.P(f"Magnitude: Fixed distances", style={'fontSize': 14}),
                            html.P(f"Total Signals: {len(backtest_df)}", style={'fontSize': 14}),
                            html.Hr(),
                            html.P("Strategy Distribution:", style={'fontWeight': 'bold', 'marginTop': 10}),
                            *[html.P(f"{strategy}: {count} ({count/len(backtest_df)*100:.1f}%)", 
                                    style={'fontSize': 13, 'marginLeft': 10})
                              for strategy, count in static_strategy_counts.items()]
                        ], style={'flex': 1, 'padding': 20, 'backgroundColor': '#ecf0f1', 'borderRadius': 10}),
                    ], style={'display': 'flex', 'gap': 20}),
                    
                    html.Div([
                        html.H5('Key Advantages', style={'color': '#2c3e50', 'marginTop': 20, 'marginBottom': 10}),
                        html.Ul([
                            html.Li(f"✓ {ml_accuracy:.1f}% direction accuracy vs 50% random"),
                            html.Li(f"✓ ±{ml_mag_error:.2f}% magnitude prediction for dynamic strikes"),
                            html.Li(f"✓ Avoided {(backtest_df['ml_strategy'] == 'AVOID').sum()} high-risk periods"),
                            html.Li(f"✓ {(backtest_df['ml_strategy'] == 'DIRECTIONAL').sum()} directional trades (vs 0 in static)"),
                        ], style={'fontSize': 14, 'color': '#27ae60'})
                    ], style={'padding': 20, 'backgroundColor': '#fff', 'borderRadius': 10, 'marginTop': 20})
                ]), backtest_fig
                
            except Exception as e:
                import traceback
                empty_fig = go.Figure()
                return html.Div([
                    html.P(f"Error running backtest: {str(e)}", style={'color': '#e74c3c'}),
                    html.Pre(traceback.format_exc(), style={'fontSize': 10, 'backgroundColor': '#ecf0f1', 'padding': 10})
                ]), empty_fig
    
    def run(self, debug=True, port=8051):
        """Run the dashboard"""
        print("="*60)
        print(f"🚀 Forecast Dashboard ready!")
        print(f"Opening at http://localhost:{port}")
        print("="*60)
        self.app.run(debug=debug, port=port)


if __name__ == "__main__":
    dashboard = ForecastDashboard()
    dashboard.run()
