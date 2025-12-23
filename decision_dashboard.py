"""
DECISION DASHBOARD - Integrated Options Trading Strategy Selector
Combines signals from:
- Dashboard.py (port 8050): Volatility regime, NIFTY return distribution
- VIX Forecasting Dashboard (port 8051): VIX forecast, NIFTY forecast
- Greek Regime Flip Model (port 8055): Greek regime, dominant Greeks

Outputs: Strategy recommendations and strike selections
"""

import dash
from dash import dcc, html, dash_table, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import warnings
warnings.filterwarnings('ignore')

# Import VIX forecaster and ML selector for real forecasts
from vix_forecaster import VIXForecaster
from strategy_selector import MLStrategySelector

# Strategy definitions with conditions
STRATEGY_MATRIX = {
    'LONG_CALL': {
        'conditions': {
            'vix_forecast': 'rising',
            'nifty_forecast': 'bullish',
            'greek_regime': ['Delta-driven', 'Gamma-driven'],
            'volatility_regime': ['Low', 'Medium'],
            'return_skew': 'positive'
        },
        'avoid_when': {
            'vix_forecast': 'falling',
            'greek_regime': ['Vega-driven'],
            'volatility_regime': ['High']
        },
        'strike_selection': 'ATM to slightly ITM (spot - 100 to spot + 100)',
        'risk': 'Limited to premium',
        'max_profit': 'Unlimited'
    },
    'LONG_PUT': {
        'conditions': {
            'vix_forecast': 'rising',
            'nifty_forecast': 'bearish',
            'greek_regime': ['Delta-driven', 'Gamma-driven'],
            'volatility_regime': ['Low', 'Medium'],
            'return_skew': 'negative'
        },
        'avoid_when': {
            'vix_forecast': 'falling',
            'greek_regime': ['Vega-driven'],
            'volatility_regime': ['High']
        },
        'strike_selection': 'ATM to slightly ITM (spot - 100 to spot + 100)',
        'risk': 'Limited to premium',
        'max_profit': 'Substantial'
    },
    'BULL_CALL_SPREAD': {
        'conditions': {
            'nifty_forecast': 'moderately_bullish',
            'greek_regime': ['Delta-driven', 'Theta-driven'],
            'volatility_regime': ['Medium', 'High'],
            'vix_forecast': 'stable'
        },
        'avoid_when': {
            'nifty_forecast': 'bearish',
            'greek_regime': ['Gamma-driven'],
            'volatility_regime': ['Low']
        },
        'strike_selection': 'Buy ATM, Sell OTM (spot + 200)',
        'risk': 'Limited',
        'max_profit': 'Limited'
    },
    'BEAR_PUT_SPREAD': {
        'conditions': {
            'nifty_forecast': 'moderately_bearish',
            'greek_regime': ['Delta-driven', 'Theta-driven'],
            'volatility_regime': ['Medium', 'High'],
            'vix_forecast': 'stable'
        },
        'avoid_when': {
            'nifty_forecast': 'bullish',
            'greek_regime': ['Gamma-driven']
        },
        'strike_selection': 'Buy ATM, Sell OTM (spot - 200)',
        'risk': 'Limited',
        'max_profit': 'Limited'
    },
    'LONG_STRADDLE': {
        'conditions': {
            'vix_forecast': 'rising',
            'greek_regime': ['Gamma-driven', 'Vega-driven'],
            'volatility_regime': ['Low'],
            'nifty_forecast': 'uncertain'
        },
        'avoid_when': {
            'vix_forecast': 'falling',
            'greek_regime': ['Theta-driven'],
            'volatility_regime': ['High']
        },
        'strike_selection': 'ATM (spot ± 50)',
        'risk': 'Limited to total premium',
        'max_profit': 'Unlimited'
    },
    'SHORT_STRADDLE': {
        'conditions': {
            'vix_forecast': 'falling',
            'greek_regime': ['Theta-driven'],
            'volatility_regime': ['High'],
            'nifty_forecast': 'range_bound'
        },
        'avoid_when': {
            'vix_forecast': 'rising',
            'greek_regime': ['Gamma-driven', 'Vega-driven'],
            'volatility_regime': ['Low']
        },
        'strike_selection': 'ATM (spot ± 50)',
        'risk': 'Unlimited',
        'max_profit': 'Limited to premium collected'
    },
    'IRON_CONDOR': {
        'conditions': {
            'nifty_forecast': 'range_bound',
            'greek_regime': ['Theta-driven'],
            'volatility_regime': ['Medium', 'High'],
            'vix_forecast': 'stable_to_falling'
        },
        'avoid_when': {
            'greek_regime': ['Gamma-driven', 'Vega-driven'],
            'nifty_forecast': 'trending',
            'volatility_regime': ['Low']
        },
        'strike_selection': 'Sell OTM Put (spot - 300), Buy OTM Put (spot - 500), Sell OTM Call (spot + 300), Buy OTM Call (spot + 500)',
        'risk': 'Limited',
        'max_profit': 'Limited to credit'
    },
    'LONG_STRANGLE': {
        'conditions': {
            'vix_forecast': 'rising',
            'greek_regime': ['Vega-driven', 'Gamma-driven'],
            'volatility_regime': ['Low'],
            'nifty_forecast': 'uncertain'
        },
        'avoid_when': {
            'vix_forecast': 'falling',
            'greek_regime': ['Theta-driven'],
            'volatility_regime': ['High']
        },
        'strike_selection': 'OTM Put (spot - 200), OTM Call (spot + 200)',
        'risk': 'Limited to premium',
        'max_profit': 'Unlimited'
    },
    'CALENDAR_SPREAD': {
        'conditions': {
            'greek_regime': ['Vega-driven', 'Theta-driven'],
            'volatility_regime': ['Medium'],
            'vix_forecast': 'stable',
            'nifty_forecast': 'range_bound'
        },
        'avoid_when': {
            'greek_regime': ['Gamma-driven'],
            'nifty_forecast': 'strongly_trending'
        },
        'strike_selection': 'Sell near-term ATM, Buy far-term ATM',
        'risk': 'Limited',
        'max_profit': 'Limited'
    },
    'BUTTERFLY_SPREAD': {
        'conditions': {
            'nifty_forecast': 'range_bound',
            'greek_regime': ['Theta-driven', 'Delta-driven'],
            'volatility_regime': ['Medium'],
            'vix_forecast': 'stable'
        },
        'avoid_when': {
            'greek_regime': ['Gamma-driven'],
            'nifty_forecast': 'trending'
        },
        'strike_selection': 'Buy ITM (spot - 200), Sell 2x ATM, Buy OTM (spot + 200)',
        'risk': 'Limited',
        'max_profit': 'Limited'
    }
}

class DecisionDashboard:
    def __init__(self):
        self.app = dash.Dash(__name__)
        
        # Load initial data
        self.auto_data = self.load_auto_data()
        
        self.setup_layout()
        self.setup_callbacks()
        
        # Cache for dashboard data
        self.cache = {
            'dashboard_8050': None,
            'forecast_8051': None,
            'greek_8055': None,
            'timestamp': None
        }
    
    def fetch_dashboard_data(self, port, endpoint='/'):
        """Fetch data from running dashboards"""
        try:
            response = requests.get(f'http://127.0.0.1:{port}{endpoint}', timeout=5)
            if response.status_code == 200:
                return {'status': 'running', 'data': response.text}
            return {'status': 'error', 'message': f'HTTP {response.status_code}'}
        except requests.exceptions.ConnectionError:
            return {'status': 'offline', 'message': 'Dashboard not running'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def load_auto_data(self):
        """Load all data automatically from files and generate forecasts"""
        data = {}
        
        try:
            # Load VIX history
            vix_df = pd.read_csv('india_vix_history.csv')
            vix_df['Date'] = pd.to_datetime(vix_df['Date'], errors='coerce')
            vix_df = vix_df.dropna(subset=['Date'])
            vix_df = vix_df.sort_values('Date')
            vix_df = vix_df.rename(columns={'Close': 'Close_vix'})
            
            latest_vix = vix_df.iloc[-1]['Close_vix']
            
            # Determine VIX regime
            if latest_vix < 12:
                vix_regime = 'Low'
            elif latest_vix < 18:
                vix_regime = 'Medium'
            else:
                vix_regime = 'High'
            
            data['vix_current'] = latest_vix
            data['vix_regime'] = vix_regime
            
            # Load NIFTY history
            nifty_df = pd.read_csv('nifty_history.csv')
            nifty_df['Date'] = pd.to_datetime(nifty_df['Date'], errors='coerce')
            nifty_df = nifty_df.dropna(subset=['Date'])
            nifty_df = nifty_df.sort_values('Date')
            nifty_df = nifty_df.rename(columns={'Close': 'Close_nifty'})
            
            latest_nifty = nifty_df.iloc[-1]['Close_nifty']
            data['nifty_current'] = latest_nifty
            
            # Calculate monthly returns for skew
            nifty_df_temp = nifty_df.copy()
            nifty_df_temp['Returns'] = nifty_df_temp['Close_nifty'].pct_change()
            monthly_returns = nifty_df_temp.set_index('Date').resample('ME')['Returns'].apply(
                lambda x: (1 + x).prod() - 1
            )
            
            data['monthly_return_mean'] = monthly_returns.mean() * 100
            data['monthly_return_std'] = monthly_returns.std() * 100
            data['monthly_return_skew'] = monthly_returns.skew()
            data['monthly_return_kurt'] = monthly_returns.kurtosis()
            
            # ===== USE REAL VIX FORECASTER (from port 8051) =====
            # Merge NIFTY and VIX data
            df_merged = pd.merge(nifty_df[['Date', 'Close_nifty']], 
                                vix_df[['Date', 'Close_vix']], 
                                on='Date', how='inner')
            
            # Initialize VIX forecaster
            forecaster = VIXForecaster(df_merged)
            forecaster.prepare_features()
            forecaster.select_exogenous_variables()
            forecaster.train_model()
            
            # Generate 5-day forecast (matching the dashboard)
            forecast_results = forecaster.forecast(horizon=5)
            
            # Extract forecast values
            forecast_vix_5day = forecast_results['vix_forecast'][-1]  # 5-day ahead
            current_vix_for_forecast = forecast_results['last_vix']
            
            # Calculate VIX change percentage
            vix_forecast_pct = ((forecast_vix_5day - current_vix_for_forecast) / current_vix_for_forecast) * 100
            data['vix_forecast_pct'] = vix_forecast_pct
            data['vix_forecast_value'] = forecast_vix_5day
            
            # ===== USE REAL ML STRATEGY SELECTOR FOR NIFTY FORECAST =====
            try:
                ml_selector = MLStrategySelector()
                ml_selector.prepare_current_features()
                predictions = ml_selector.get_predictions()
                
                # Extract NIFTY forecast from ML predictions
                nifty_forecast_pct = predictions['expected_move']
                if predictions['direction'] == 'DOWN':
                    nifty_forecast_pct = -nifty_forecast_pct
                
                data['nifty_forecast_pct'] = nifty_forecast_pct
                data['nifty_ml_confidence'] = predictions['confidence']
                data['nifty_direction'] = predictions['direction']
            except Exception as e:
                print(f"  ⚠ ML models not available, using simple forecast: {e}")
                # Fallback to simple momentum
                nifty_recent = nifty_df.tail(22)['Close_nifty'].values
                recent_trend = (nifty_recent[-1] - nifty_recent[0]) / nifty_recent[0]
                nifty_forecast_pct = recent_trend * 100 * 1.5
                data['nifty_forecast_pct'] = nifty_forecast_pct
            
            # ===== GET GREEK REGIME FROM ACTUAL OPTION CHAIN DATA (port 8055) =====
            try:
                import glob
                import os
                from datetime import datetime, timedelta
                
                # Find most recent option chain CSV file
                csv_pattern = 'nifty_option_excel/option-chain-ED-NIFTY-*.csv'
                csv_files = glob.glob(csv_pattern)
                
                if csv_files:
                    # Sort by modification time to get latest
                    latest_csv = max(csv_files, key=os.path.getmtime)
                    
                    # Parse filename to extract dates
                    # Format: option-chain-ED-NIFTY-{expiry}_{trading_date}.csv
                    basename = os.path.basename(latest_csv)
                    # Example: option-chain-ED-NIFTY-26-Dec-2024_19 Dec 2024.csv
                    
                    # Load CSV
                    df_options = pd.read_csv(latest_csv, skiprows=1)
                    df_options['STRIKE'] = pd.to_numeric(df_options['STRIKE'].astype(str).str.replace(',', ''), errors='coerce')
                    
                    # Extract CALL data
                    calls = df_options[['OI', 'CHNG IN OI', 'VOLUME', 'IV', 'LTP', 'STRIKE']].copy()
                    calls.columns = ['OI', 'CHNG_IN_OI', 'VOLUME', 'IV', 'Premium', 'Strike']
                    calls['Type'] = 'CE'
                    
                    # Extract PUT data
                    puts = df_options[['STRIKE', 'LTP.1', 'IV.1', 'VOLUME.1', 'CHNG IN OI.1', 'OI.1']].copy()
                    puts.columns = ['Strike', 'Premium', 'IV', 'VOLUME', 'CHNG_IN_OI', 'OI']
                    puts['Type'] = 'PE'
                    
                    # Combine
                    combined = pd.concat([calls, puts], ignore_index=True)
                    
                    # Clean numeric fields
                    for col in ['IV', 'Premium', 'Strike', 'VOLUME', 'OI']:
                        combined[col] = pd.to_numeric(combined[col].astype(str).str.replace(',', ''), errors='coerce')
                    
                    combined = combined.dropna(subset=['Strike', 'IV'])
                    
                    # Calculate average IV
                    avg_iv = combined['IV'].mean()
                    
                    # Determine Greek regime based on IV levels (simple classification)
                    # This matches the logic from greek_regime_flip_live.py
                    if avg_iv > 20:
                        greek_regime = 'Vega-driven'
                    elif avg_iv < 12:
                        greek_regime = 'Delta-driven'
                    elif latest_vix < 12:
                        greek_regime = 'Theta-driven'
                    else:
                        greek_regime = 'Gamma-driven'
                    
                    data['greek_regime'] = greek_regime
                    data['avg_iv'] = avg_iv
                    
                    print(f"  ✓ Greek regime from option chain: {greek_regime} (Avg IV: {avg_iv:.1f}%)")
                    
                else:
                    # Fallback to parquet files
                    parquet_files = glob.glob('nifty_option_excel/NIFTY_options_*.parquet')
                    if parquet_files:
                        df_opts = pd.read_parquet(parquet_files[-1])
                        avg_iv = df_opts['IV'].mean() if 'IV' in df_opts.columns else 20
                        
                        if avg_iv > 25:
                            greek_regime = 'Vega-driven'
                        elif avg_iv < 15:
                            greek_regime = 'Delta-driven'
                        elif latest_vix > 15:
                            greek_regime = 'Gamma-driven'
                        else:
                            greek_regime = 'Theta-driven'
                        
                        data['greek_regime'] = greek_regime
                    else:
                        # Final fallback based on VIX
                        if latest_vix < 12:
                            data['greek_regime'] = 'Delta-driven'
                        elif latest_vix < 15:
                            data['greek_regime'] = 'Theta-driven'
                        elif latest_vix < 20:
                            data['greek_regime'] = 'Vega-driven'
                        else:
                            data['greek_regime'] = 'Gamma-driven'
            except Exception as e:
                print(f"  ⚠ Could not load option chain: {e}")
                # Fallback Greek regime based on VIX
                if latest_vix < 12:
                    data['greek_regime'] = 'Delta-driven'
                elif latest_vix < 15:
                    data['greek_regime'] = 'Theta-driven'
                elif latest_vix < 20:
                    data['greek_regime'] = 'Vega-driven'
                else:
                    data['greek_regime'] = 'Gamma-driven'
            
            print(f"\n✓ Auto-loaded data (integrated with VIX & ML forecasters):")
            print(f"  NIFTY Spot: ₹{data['nifty_current']:,.2f}")
            print(f"  VIX Current: {data['vix_current']:.2f} ({data['vix_regime']})")
            print(f"  VIX Forecast (5-day): {data['vix_forecast_pct']:+.2f}% (→ {data['vix_forecast_value']:.2f})")
            print(f"  NIFTY Forecast (ML): {data['nifty_forecast_pct']:+.2f}% ({data.get('nifty_direction', 'N/A')})")
            print(f"  Greek Regime: {data['greek_regime']}")
            
            return data
            
        except Exception as e:
            print(f"Error loading auto data: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'nifty_current': 24900,
                'vix_current': 14,
                'vix_regime': 'Medium',
                'vix_forecast_pct': 0,
                'nifty_forecast_pct': 0,
                'greek_regime': 'Delta-driven',
                'monthly_return_skew': 0
            }
    
    def load_local_data(self):
        """Alias for load_auto_data for backward compatibility"""
        return self.auto_data if hasattr(self, 'auto_data') else self.load_auto_data()
    
    def classify_nifty_forecast(self, forecast_pct):
        """Classify NIFTY forecast direction"""
        if forecast_pct > 3:
            return 'bullish'
        elif forecast_pct > 1:
            return 'moderately_bullish'
        elif forecast_pct < -3:
            return 'bearish'
        elif forecast_pct < -1:
            return 'moderately_bearish'
        else:
            return 'range_bound'
    
    def classify_vix_forecast(self, vix_change_pct):
        """Classify VIX forecast trend"""
        if vix_change_pct > 10:
            return 'rising'
        elif vix_change_pct < -10:
            return 'falling'
        else:
            return 'stable'
    
    def calculate_strategy_scores(self, signals):
        """Score each strategy based on current signals"""
        scores = {}
        
        for strategy_name, strategy_def in STRATEGY_MATRIX.items():
            score = 0
            max_score = 0
            avoid = False
            reasons = []
            
            # Check avoid conditions first
            for condition, value in strategy_def['avoid_when'].items():
                max_score += 1
                if condition in signals:
                    if isinstance(value, list):
                        if signals[condition] in value:
                            avoid = True
                            reasons.append(f"❌ {condition}: {signals[condition]} (avoid)")
                        else:
                            score += 1
                            reasons.append(f"✓ {condition}: {signals[condition]}")
                    else:
                        if signals[condition] == value:
                            avoid = True
                            reasons.append(f"❌ {condition}: {signals[condition]} (avoid)")
                        else:
                            score += 1
                            reasons.append(f"✓ {condition}: {signals[condition]}")
                else:
                    reasons.append(f"⚠ {condition}: not available")
            
            # Check favorable conditions
            for condition, value in strategy_def['conditions'].items():
                max_score += 1
                if condition in signals:
                    if isinstance(value, list):
                        if signals[condition] in value:
                            score += 1
                            reasons.append(f"✓ {condition}: {signals[condition]}")
                        else:
                            reasons.append(f"⚠ {condition}: {signals[condition]} (expected: {', '.join(value)})")
                    else:
                        if signals[condition] == value:
                            score += 1
                            reasons.append(f"✓ {condition}: {signals[condition]}")
                        else:
                            reasons.append(f"⚠ {condition}: {signals[condition]} (expected: {value})")
                else:
                    reasons.append(f"⚠ {condition}: not available")
            
            # Calculate percentage score
            score_pct = (score / max_score * 100) if max_score > 0 else 0
            
            # Calculate strikes for this strategy
            strikes = self.calculate_strikes(strategy_name, signals.get('spot', 24900))
            
            scores[strategy_name] = {
                'score': score_pct,
                'avoid': avoid,
                'reasons': reasons,
                'strikes': strikes,
                'risk': strategy_def.get('risk', 'N/A'),
                'max_profit': strategy_def.get('max_profit', 'N/A')
            }
        
        return scores
    
    def load_latest_option_chain(self):
        """Load the most recent option chain CSV file"""
        try:
            import glob
            import os
            from datetime import datetime
            
            # Find all option chain CSV files
            csv_pattern = 'nifty_option_excel/option-chain-ED-NIFTY-*.csv'
            csv_files = glob.glob(csv_pattern)
            
            if not csv_files:
                return None
            
            # Sort by modification time to get latest
            latest_csv = max(csv_files, key=os.path.getmtime)
            
            # Load CSV
            df_options = pd.read_csv(latest_csv, skiprows=1)
            df_options['STRIKE'] = pd.to_numeric(df_options['STRIKE'].astype(str).str.replace(',', ''), errors='coerce')
            
            # Extract CALL data
            calls = df_options[['OI', 'CHNG IN OI', 'VOLUME', 'IV', 'LTP', 'STRIKE']].copy()
            calls.columns = ['OI', 'CHNG_IN_OI', 'VOLUME', 'IV', 'Premium', 'Strike']
            calls['Type'] = 'CE'
            
            # Extract PUT data
            puts = df_options[['STRIKE', 'LTP.1', 'IV.1', 'VOLUME.1', 'CHNG IN OI.1', 'OI.1']].copy()
            puts.columns = ['Strike', 'Premium', 'IV', 'VOLUME', 'CHNG_IN_OI', 'OI']
            puts['Type'] = 'PE'
            
            # Combine
            combined = pd.concat([calls, puts], ignore_index=True)
            
            # Clean numeric fields
            for col in ['IV', 'Premium', 'Strike']:
                combined[col] = pd.to_numeric(combined[col].astype(str).str.replace(',', ''), errors='coerce')
            
            combined = combined.dropna(subset=['Strike', 'Premium'])
            
            return combined
            
        except Exception as e:
            print(f"Error loading option chain: {e}")
            return None
    
    def get_option_premium(self, strike, option_type, df_options):
        """Get premium for a specific strike and option type"""
        if df_options is None:
            return None
        
        match = df_options[(df_options['Strike'] == strike) & (df_options['Type'] == option_type)]
        if not match.empty:
            return match.iloc[0]['Premium']
        return None
    
    def calculate_strategy_cost(self, strategy_name, strikes, df_options, lot_size=65):
        """Calculate total cost of a strategy based on actual premiums"""
        if df_options is None:
            return None
        
        cost = 0
        legs = []
        
        if 'LONG_CALL' in strategy_name:
            prem = self.get_option_premium(strikes['call'], 'CE', df_options)
            if prem:
                cost = prem * lot_size
                legs.append(f"Buy {strikes['call']} CE @ ₹{prem:.2f}")
                
        elif 'LONG_PUT' in strategy_name:
            prem = self.get_option_premium(strikes['put'], 'PE', df_options)
            if prem:
                cost = prem * lot_size
                legs.append(f"Buy {strikes['put']} PE @ ₹{prem:.2f}")
                
        elif 'BULL_CALL_SPREAD' in strategy_name:
            buy_prem = self.get_option_premium(strikes['buy_call'], 'CE', df_options)
            sell_prem = self.get_option_premium(strikes['sell_call'], 'CE', df_options)
            if buy_prem and sell_prem:
                cost = (buy_prem - sell_prem) * lot_size
                legs.append(f"Buy {strikes['buy_call']} CE @ ₹{buy_prem:.2f}")
                legs.append(f"Sell {strikes['sell_call']} CE @ ₹{sell_prem:.2f}")
                
        elif 'BEAR_PUT_SPREAD' in strategy_name:
            buy_prem = self.get_option_premium(strikes['buy_put'], 'PE', df_options)
            sell_prem = self.get_option_premium(strikes['sell_put'], 'PE', df_options)
            if buy_prem and sell_prem:
                cost = (buy_prem - sell_prem) * lot_size
                legs.append(f"Buy {strikes['buy_put']} PE @ ₹{buy_prem:.2f}")
                legs.append(f"Sell {strikes['sell_put']} PE @ ₹{sell_prem:.2f}")
                
        elif 'STRADDLE' in strategy_name:
            call_prem = self.get_option_premium(strikes['call'], 'CE', df_options)
            put_prem = self.get_option_premium(strikes['put'], 'PE', df_options)
            if call_prem and put_prem:
                if 'SHORT' in strategy_name:
                    cost = -(call_prem + put_prem) * lot_size  # Credit received
                else:
                    cost = (call_prem + put_prem) * lot_size  # Debit paid
                legs.append(f"{'Sell' if 'SHORT' in strategy_name else 'Buy'} {strikes['call']} CE @ ₹{call_prem:.2f}")
                legs.append(f"{'Sell' if 'SHORT' in strategy_name else 'Buy'} {strikes['put']} PE @ ₹{put_prem:.2f}")
                
        elif 'STRANGLE' in strategy_name:
            call_prem = self.get_option_premium(strikes['call'], 'CE', df_options)
            put_prem = self.get_option_premium(strikes['put'], 'PE', df_options)
            if call_prem and put_prem:
                cost = (call_prem + put_prem) * lot_size
                legs.append(f"Buy {strikes['call']} CE @ ₹{call_prem:.2f}")
                legs.append(f"Buy {strikes['put']} PE @ ₹{put_prem:.2f}")
                
        elif 'IRON_CONDOR' in strategy_name:
            sell_put_prem = self.get_option_premium(strikes['sell_put'], 'PE', df_options)
            buy_put_prem = self.get_option_premium(strikes['buy_put'], 'PE', df_options)
            sell_call_prem = self.get_option_premium(strikes['sell_call'], 'CE', df_options)
            buy_call_prem = self.get_option_premium(strikes['buy_call'], 'CE', df_options)
            
            if all([sell_put_prem, buy_put_prem, sell_call_prem, buy_call_prem]):
                cost = -(sell_put_prem + sell_call_prem - buy_put_prem - buy_call_prem) * lot_size
                legs.append(f"Sell {strikes['sell_put']} PE @ ₹{sell_put_prem:.2f}")
                legs.append(f"Buy {strikes['buy_put']} PE @ ₹{buy_put_prem:.2f}")
                legs.append(f"Sell {strikes['sell_call']} CE @ ₹{sell_call_prem:.2f}")
                legs.append(f"Buy {strikes['buy_call']} CE @ ₹{buy_call_prem:.2f}")
                
        elif 'BUTTERFLY' in strategy_name:
            buy_lower_prem = self.get_option_premium(strikes['buy_lower'], 'CE', df_options)
            sell_atm_prem = self.get_option_premium(strikes['sell_atm_1'], 'CE', df_options)
            buy_upper_prem = self.get_option_premium(strikes['buy_upper'], 'CE', df_options)
            
            if all([buy_lower_prem, sell_atm_prem, buy_upper_prem]):
                cost = (buy_lower_prem - 2*sell_atm_prem + buy_upper_prem) * lot_size
                legs.append(f"Buy {strikes['buy_lower']} CE @ ₹{buy_lower_prem:.2f}")
                legs.append(f"Sell 2x {strikes['sell_atm_1']} CE @ ₹{sell_atm_prem:.2f}")
                legs.append(f"Buy {strikes['buy_upper']} CE @ ₹{buy_upper_prem:.2f}")
        
        return {'total_cost': cost, 'premium_legs': legs}
    
    def calculate_kelly_position(self, win_prob, win_amount, loss_amount, capital=500000, tail_risk=30):
        """Calculate Kelly criterion position sizing"""
        # Kelly formula: f = (p*b - q) / b
        # where p = win probability, q = loss probability, b = win/loss ratio
        
        if loss_amount == 0:
            return {
                'kelly_fraction': 0,
                'adjusted_kelly': 0,
                'position_size': 0,
                'tail_risk': tail_risk
            }
        
        b = abs(win_amount / loss_amount)
        p = win_prob
        q = 1 - p
        
        kelly_fraction = (p * b - q) / b
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        # Tail risk adjustment (reduce by 50% for safety)
        adjusted_kelly = kelly_fraction * 0.5
        
        position_size = capital * adjusted_kelly
        
        return {
            'kelly_fraction': kelly_fraction,
            'adjusted_kelly': adjusted_kelly,
            'position_size': position_size,
            'tail_risk': tail_risk
        }
    
    def calculate_strikes(self, strategy_name, spot):
        """Calculate option strikes based on strategy and spot price"""
        strikes = {}
        atm = round(spot / 50) * 50  # Round to nearest 50
        
        if 'LONG_CALL' in strategy_name:
            strikes['call'] = atm
        elif 'LONG_PUT' in strategy_name:
            strikes['put'] = atm
        elif 'BULL_CALL_SPREAD' in strategy_name:
            strikes['buy_call'] = atm
            strikes['sell_call'] = atm + 200
        elif 'BEAR_PUT_SPREAD' in strategy_name:
            strikes['buy_put'] = atm
            strikes['sell_put'] = atm - 200
        elif 'LONG_STRADDLE' in strategy_name:
            strikes['call'] = atm
            strikes['put'] = atm
        elif 'SHORT_STRADDLE' in strategy_name:
            strikes['call'] = atm
            strikes['put'] = atm
        elif 'STRANGLE' in strategy_name:
            strikes['call'] = atm + 100
            strikes['put'] = atm - 100
        elif 'CALENDAR_SPREAD' in strategy_name:
            strikes['call'] = atm + 100
            strikes['put'] = atm - 200
        elif 'IRON_CONDOR' in strategy_name:
            strikes['sell_put'] = atm - 300
            strikes['buy_put'] = atm - 500
            strikes['sell_call'] = atm + 300
            strikes['buy_call'] = atm + 500
        elif 'BUTTERFLY' in strategy_name:
            strikes['buy_lower'] = atm - 200
            strikes['sell_atm_1'] = atm
            strikes['sell_atm_2'] = atm
            strikes['buy_upper'] = atm + 200
        
        return strikes
    
    def setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("🎯 DECISION DASHBOARD", 
                       style={'textAlign':'center','color':'white','marginBottom':5}),
                html.P("Integrated Options Strategy Selector", 
                      style={'textAlign':'center','color':'#94a3b8','fontSize':18})
            ], style={'padding':'20px','background':'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                     'borderRadius':'10px','marginBottom':25}),
            
            # Auto-populated Data Display
            html.Div([
                html.Div([
                    html.Label("NIFTY Spot (₹):", style={'fontWeight':'bold','marginBottom':5,'fontSize':12}),
                    html.Div(f"₹{self.auto_data.get('nifty_current', 24900):,.2f}",
                            id='display-spot',
                            style={'padding':'10px','background':'white','borderRadius':5,
                                  'border':'2px solid #667eea','fontSize':16,'fontWeight':'bold',
                                  'color':'#667eea','textAlign':'center'})
                ], style={'flex':1,'marginRight':15}),
                
                html.Div([
                    html.Label("VIX Forecast (%):", style={'fontWeight':'bold','marginBottom':5,'fontSize':12}),
                    html.Div(f"{self.auto_data.get('vix_forecast_pct', 0):+.1f}%",
                            id='display-vix-forecast',
                            style={'padding':'10px','background':'white','borderRadius':5,
                                  'border':'2px solid #10b981','fontSize':16,'fontWeight':'bold',
                                  'color':'#10b981','textAlign':'center'})
                ], style={'flex':1,'marginRight':15}),
                
                html.Div([
                    html.Label("NIFTY Forecast (%):", style={'fontWeight':'bold','marginBottom':5,'fontSize':12}),
                    html.Div(f"{self.auto_data.get('nifty_forecast_pct', 0):+.1f}%",
                            id='display-nifty-forecast',
                            style={'padding':'10px','background':'white','borderRadius':5,
                                  'border':'2px solid #f59e0b','fontSize':16,'fontWeight':'bold',
                                  'color':'#f59e0b','textAlign':'center'})
                ], style={'flex':1,'marginRight':15}),
                
                html.Div([
                    html.Label("Greek Regime:", style={'fontWeight':'bold','marginBottom':5,'fontSize':12}),
                    html.Div(self.auto_data.get('greek_regime', 'Delta-driven'),
                            id='display-greek-regime',
                            style={'padding':'10px','background':'white','borderRadius':5,
                                  'border':'2px solid #a855f7','fontSize':14,'fontWeight':'bold',
                                  'color':'#a855f7','textAlign':'center'})
                ], style={'flex':1,'marginRight':15}),
                
                html.Div([
                    html.Label("Volatility Regime:", style={'fontWeight':'bold','marginBottom':5,'fontSize':12}),
                    html.Div(f"{self.auto_data.get('vix_regime', 'Medium')} (VIX {self.auto_data.get('vix_current', 14):.1f})",
                            id='display-vol-regime',
                            style={'padding':'10px','background':'white','borderRadius':5,
                                  'border':'2px solid #8b5cf6','fontSize':14,'fontWeight':'bold',
                                  'color':'#8b5cf6','textAlign':'center'})
                ], style={'flex':1,'marginRight':15}),
                
                html.Div([
                    html.Label('🔄', style={'display':'block','marginBottom':5,'fontSize':12,'color':'transparent'}),
                    html.Button('🔍 ANALYZE', id='analyze-btn', n_clicks=1,
                               style={'width':'100%','padding':10,'background':'#667eea',
                                     'color':'white','border':'none','borderRadius':5,
                                     'fontWeight':'bold','fontSize':16,'cursor':'pointer'})
                ], style={'flex':1})
            ], style={'display':'flex','padding':20,'background':'#f8fafc',
                     'borderRadius':10,'marginBottom':25}),
            
            # Hidden stores for data
            dcc.Store(id='auto-spot', data=self.auto_data.get('nifty_current', 24900)),
            dcc.Store(id='auto-vix-forecast', data=self.auto_data.get('vix_forecast_pct', 0)),
            dcc.Store(id='auto-nifty-forecast', data=self.auto_data.get('nifty_forecast_pct', 0)),
            dcc.Store(id='auto-greek-regime', data=self.auto_data.get('greek_regime', 'Delta-driven')),
            dcc.Store(id='auto-vol-regime', data=self.auto_data.get('vix_regime', 'Medium')),
            
            # Signal Summary
            html.Div(id='signal-summary', style={'marginBottom':25}),
            
            # Strategy Recommendations
            html.Div([
                html.H3("📊 STRATEGY RECOMMENDATIONS", style={'color':'#1e293b','marginBottom':20}),
                html.Div(id='strategy-recommendations')
            ], style={'marginBottom':25}),
            
            # Detailed Analysis
            dcc.Tabs([
                dcc.Tab(label='✅ Recommended Strategies', children=[
                    html.Div(id='recommended-strategies', style={'padding':20})
                ]),
                dcc.Tab(label='⚠️ Caution Strategies', children=[
                    html.Div(id='caution-strategies', style={'padding':20})
                ]),
                dcc.Tab(label='❌ Avoid Strategies', children=[
                    html.Div(id='avoid-strategies', style={'padding':20})
                ]),
                dcc.Tab(label='📈 Signal Analysis', children=[
                    html.Div(id='signal-analysis', style={'padding':20})
                ])
            ])
            
        ], style={'padding':'30px','maxWidth':'1600px','margin':'0 auto',
                 'fontFamily':'system-ui, -apple-system, sans-serif'})
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output('signal-summary', 'children'),
             Output('strategy-recommendations', 'children'),
             Output('recommended-strategies', 'children'),
             Output('caution-strategies', 'children'),
             Output('avoid-strategies', 'children'),
             Output('signal-analysis', 'children')],
            [Input('analyze-btn', 'n_clicks')],
            [State('auto-spot', 'data'),
             State('auto-vix-forecast', 'data'),
             State('auto-nifty-forecast', 'data'),
             State('auto-greek-regime', 'data'),
             State('auto-vol-regime', 'data')]
        )
        def analyze_strategies(n_clicks, spot, vix_forecast_pct, nifty_forecast_pct, 
                              greek_regime, vol_regime):
            # Auto-run on load (n_clicks starts at 1)
            
            # Load local data
            local_data = self.load_local_data()
            
            # Build signals dictionary
            signals = {
                'spot': spot if spot else local_data.get('nifty_current', 24900),
                'vix_forecast': self.classify_vix_forecast(vix_forecast_pct or 0),
                'nifty_forecast': self.classify_nifty_forecast(nifty_forecast_pct or 0),
                'greek_regime': greek_regime or 'Delta-driven',
                'volatility_regime': vol_regime or 'Medium',
                'return_skew': 'positive' if local_data.get('monthly_return_skew', 0) > 0 else 'negative'
            }
            
            # Calculate strategy scores
            strategy_scores = self.calculate_strategy_scores(signals)
            
            # Categorize strategies
            recommended = []
            caution = []
            avoid = []
            
            for strat_name, strat_data in sorted(strategy_scores.items(), 
                                                 key=lambda x: x[1]['score'], reverse=True):
                if strat_data['avoid']:
                    avoid.append((strat_name, strat_data))
                elif strat_data['score'] >= 70:
                    recommended.append((strat_name, strat_data))
                elif strat_data['score'] >= 40:
                    caution.append((strat_name, strat_data))
                else:
                    avoid.append((strat_name, strat_data))
            
            # Signal Summary
            signal_summary = html.Div([
                html.H3("🎯 Current Market Signals", style={'marginBottom':15}),
                html.Div([
                    html.Div([
                        html.Strong("NIFTY Spot: "),
                        html.Span(f"₹{signals['spot']:,.2f}", style={'fontSize':18,'color':'#667eea'})
                    ], style={'marginBottom':10}),
                    html.Div([
                        html.Strong("VIX Forecast: "),
                        html.Span(f"{vix_forecast_pct:+.1f}%", 
                                 style={'fontSize':18,'color':'#10b981' if vix_forecast_pct < 0 else '#ef4444'}),
                        html.Span(f" ({signals['vix_forecast']})", style={'marginLeft':10,'color':'#64748b'})
                    ], style={'marginBottom':10}),
                    html.Div([
                        html.Strong("NIFTY Forecast: "),
                        html.Span(f"{nifty_forecast_pct:+.1f}%", 
                                 style={'fontSize':18,'color':'#10b981' if nifty_forecast_pct > 0 else '#ef4444'}),
                        html.Span(f" ({signals['nifty_forecast']})", style={'marginLeft':10,'color':'#64748b'})
                    ], style={'marginBottom':10}),
                    html.Div([
                        html.Strong("Greek Regime: "),
                        html.Span(signals['greek_regime'], style={'fontSize':18,'color':'#f59e0b'})
                    ], style={'marginBottom':10}),
                    html.Div([
                        html.Strong("Volatility Regime: "),
                        html.Span(signals['volatility_regime'], style={'fontSize':18,'color':'#8b5cf6'})
                    ], style={'marginBottom':10}),
                    html.Div([
                        html.Strong("Return Skew: "),
                        html.Span(signals['return_skew'], style={'fontSize':18,'color':'#06b6d4'})
                    ])
                ])
            ], style={'padding':20,'background':'#f0fdf4','borderRadius':10,'border':'2px solid #10b981'})
            
            # Strategy Summary Cards
            strategy_cards = html.Div([
                html.Div([
                    html.H2(str(len(recommended)), style={'fontSize':36,'margin':0,'color':'#10b981'}),
                    html.P("Recommended", style={'margin':0,'color':'#64748b'})
                ], style={'flex':1,'padding':20,'background':'#f0fdf4','borderRadius':8,
                         'textAlign':'center','border':'2px solid #10b981'}),
                html.Div([
                    html.H2(str(len(caution)), style={'fontSize':36,'margin':0,'color':'#f59e0b'}),
                    html.P("Caution", style={'margin':0,'color':'#64748b'})
                ], style={'flex':1,'padding':20,'background':'#fffbeb','borderRadius':8,
                         'textAlign':'center','border':'2px solid #f59e0b','marginLeft':15}),
                html.Div([
                    html.H2(str(len(avoid)), style={'fontSize':36,'margin':0,'color':'#ef4444'}),
                    html.P("Avoid", style={'margin':0,'color':'#64748b'})
                ], style={'flex':1,'padding':20,'background':'#fef2f2','borderRadius':8,
                         'textAlign':'center','border':'2px solid #ef4444','marginLeft':15})
            ], style={'display':'flex','marginBottom':20})
            
            # Load option chain for premium calculations
            df_options = self.load_latest_option_chain()
            lot_size = 65
            capital = 500000
            
            # Recommended Strategies Detail
            rec_content = []
            for strat_name, strat_data in recommended:
                strikes = self.calculate_strikes(strat_name, signals['spot'])
                strike_text = ', '.join([f"{k.replace('_', ' ').title()}: {v}" 
                                        for k, v in strikes.items()])
                
                # Calculate costs and Kelly position
                cost_data = self.calculate_strategy_cost(strat_name, strikes, df_options, lot_size)
                
                # Use strategy score as win probability
                win_prob = strat_data['score'] / 100.0
                # Estimate potential profit as 2x cost (conservative for options)
                potential_profit = abs(cost_data['total_cost']) * 2 if cost_data['total_cost'] != 0 else 10000
                kelly_data = self.calculate_kelly_position(
                    win_prob=win_prob,
                    win_amount=potential_profit,
                    loss_amount=abs(cost_data['total_cost']),
                    capital=capital,
                    tail_risk=local_data.get('tail_risk', 30)
                )
                
                # Build premium breakdown
                premium_breakdown = []
                for leg in cost_data['premium_legs']:
                    premium_breakdown.append(html.Li(leg, style={'marginBottom':5}))
                
                rec_content.append(html.Div([
                    html.Div([
                        html.H4(strat_name.replace('_', ' ').title(), 
                               style={'color':'#10b981','marginBottom':10}),
                        html.Div([
                            html.Strong(f"Match Score: {strat_data['score']:.0f}%"),
                            html.Div(style={'width':f"{strat_data['score']}%",'height':6,
                                          'background':'#10b981','borderRadius':3,'marginTop':5})
                        ]),
                        html.Div([
                            html.Strong("Strikes: ", style={'color':'#1e293b'}),
                            html.Span(strike_text, style={'color':'#667eea','fontSize':16})
                        ], style={'marginTop':15,'padding':10,'background':'#eff6ff',
                                 'borderRadius':5}),
                        
                        # Premium & Cost Section
                        html.Div([
                            html.H5("💰 Premium & Cost", style={'color':'#1e293b','marginBottom':10}),
                            html.Ul(premium_breakdown, style={'marginLeft':20,'marginBottom':10}),
                            html.Div([
                                html.Strong("Total Cost: "),
                                html.Span(f"₹{abs(cost_data['total_cost']):,.0f}", 
                                         style={'fontSize':18,'color':'#ef4444' if cost_data['total_cost'] > 0 else '#10b981',
                                               'fontWeight':'bold'})
                            ], style={'padding':10,'background':'#fef2f2' if cost_data['total_cost'] > 0 else '#f0fdf4',
                                     'borderRadius':5,'marginTop':10})
                        ], style={'marginTop':15,'padding':15,'background':'#f8fafc','borderRadius':8}),
                        
                        # Kelly Criterion Position Sizing
                        html.Div([
                            html.H5("📊 Position Sizing (Kelly Criterion)", 
                                   style={'color':'#1e293b','marginBottom':10}),
                            html.Div([
                                html.Div([
                                    html.Strong("Kelly Fraction: "),
                                    html.Span(f"{kelly_data['kelly_fraction']:.1%}", 
                                             style={'color':'#667eea','fontSize':16})
                                ], style={'marginBottom':8}),
                                html.Div([
                                    html.Strong("Adjusted Kelly: "),
                                    html.Span(f"{kelly_data['adjusted_kelly']:.1%}", 
                                             style={'color':'#8b5cf6','fontSize':16}),
                                    html.Span(" (50% reduction for safety)", 
                                             style={'fontSize':12,'color':'#64748b','marginLeft':5})
                                ], style={'marginBottom':8}),
                                html.Div([
                                    html.Strong("Recommended Position: "),
                                    html.Span(f"₹{kelly_data['position_size']:,.0f}", 
                                             style={'fontSize':18,'color':'#10b981','fontWeight':'bold'}),
                                    html.Span(f" ({kelly_data['position_size']/capital*100:.1f}% of capital)", 
                                             style={'fontSize':12,'color':'#64748b','marginLeft':5})
                                ], style={'marginBottom':8}),
                                html.Div([
                                    html.Strong("Tail Risk: "),
                                    html.Span(f"{kelly_data['tail_risk']:.1f}%", 
                                             style={'fontSize':16,'color':'#ef4444'}),
                                    html.Div(style={'width':'100%','background':'#e2e8f0','borderRadius':3,
                                                   'height':8,'marginTop':5}, children=[
                                        html.Div(style={'width':f"{min(kelly_data['tail_risk'], 100)}%",
                                                       'background':'#ef4444','borderRadius':3,'height':8})
                                    ])
                                ])
                            ])
                        ], style={'marginTop':15,'padding':15,'background':'#eff6ff','borderRadius':8}),
                        
                        html.Div([
                            html.Strong("Risk: "), html.Span(strat_data['risk']),
                            html.Br(),
                            html.Strong("Max Profit: "), html.Span(strat_data['max_profit'])
                        ], style={'marginTop':10,'fontSize':14,'color':'#64748b'}),
                        html.Details([
                            html.Summary("View Analysis", style={'cursor':'pointer','color':'#667eea'}),
                            html.Ul([html.Li(reason) for reason in strat_data['reasons']],
                                   style={'marginTop':10})
                        ], style={'marginTop':10})
                    ])
                ], style={'padding':20,'background':'white','borderRadius':8,
                         'border':'2px solid #10b981','marginBottom':15}))
            
            # Caution Strategies Detail
            caution_content = []
            for strat_name, strat_data in caution:
                strikes = self.calculate_strikes(strat_name, signals['spot'])
                strike_text = ', '.join([f"{k.replace('_', ' ').title()}: {v}" 
                                        for k, v in strikes.items()])
                
                # Calculate costs and Kelly position
                cost_data = self.calculate_strategy_cost(strat_name, strikes, df_options, lot_size)
                win_prob = strat_data['score'] / 100.0
                potential_profit = abs(cost_data['total_cost']) * 1.5 if cost_data['total_cost'] != 0 else 8000
                kelly_data = self.calculate_kelly_position(
                    win_prob=win_prob,
                    win_amount=potential_profit,
                    loss_amount=abs(cost_data['total_cost']),
                    capital=capital,
                    tail_risk=local_data.get('tail_risk', 30)
                )
                
                premium_breakdown = []
                for leg in cost_data['premium_legs']:
                    premium_breakdown.append(html.Li(leg, style={'marginBottom':5}))
                
                caution_content.append(html.Div([
                    html.Div([
                        html.H4(strat_name.replace('_', ' ').title(), 
                               style={'color':'#f59e0b','marginBottom':10}),
                        html.Div([
                            html.Strong(f"Match Score: {strat_data['score']:.0f}%"),
                            html.Div(style={'width':f"{strat_data['score']}%",'height':6,
                                          'background':'#f59e0b','borderRadius':3,'marginTop':5})
                        ]),
                        html.Div([
                            html.Strong("Strikes: "),
                            html.Span(strike_text, style={'color':'#667eea'})
                        ], style={'marginTop':15,'padding':10,'background':'#fffbeb',
                                 'borderRadius':5}),
                        
                        # Premium & Cost Section
                        html.Div([
                            html.H5("💰 Premium & Cost", style={'color':'#1e293b','marginBottom':10}),
                            html.Ul(premium_breakdown, style={'marginLeft':20,'marginBottom':10}),
                            html.Div([
                                html.Strong("Total Cost: "),
                                html.Span(f"₹{abs(cost_data['total_cost']):,.0f}", 
                                         style={'fontSize':18,'color':'#ef4444' if cost_data['total_cost'] > 0 else '#10b981',
                                               'fontWeight':'bold'})
                            ], style={'padding':10,'background':'#fef2f2' if cost_data['total_cost'] > 0 else '#f0fdf4',
                                     'borderRadius':5,'marginTop':10})
                        ], style={'marginTop':15,'padding':15,'background':'#f8fafc','borderRadius':8}),
                        
                        # Kelly Position Sizing (with warning)
                        html.Div([
                            html.H5("📊 Position Sizing (Kelly Criterion)", 
                                   style={'color':'#1e293b','marginBottom':10}),
                            html.Div([
                                html.Div("⚠️ CAUTION: Lower confidence strategy", 
                                        style={'color':'#f59e0b','fontWeight':'bold','marginBottom':10,
                                              'padding':8,'background':'#fffbeb','borderRadius':5}),
                                html.Div([
                                    html.Strong("Kelly Fraction: "),
                                    html.Span(f"{kelly_data['kelly_fraction']:.1%}", 
                                             style={'color':'#667eea','fontSize':16})
                                ], style={'marginBottom':8}),
                                html.Div([
                                    html.Strong("Adjusted Kelly: "),
                                    html.Span(f"{kelly_data['adjusted_kelly']:.1%}", 
                                             style={'color':'#8b5cf6','fontSize':16})
                                ], style={'marginBottom':8}),
                                html.Div([
                                    html.Strong("Recommended Position: "),
                                    html.Span(f"₹{kelly_data['position_size']:,.0f}", 
                                             style={'fontSize':18,'color':'#f59e0b','fontWeight':'bold'})
                                ], style={'marginBottom':8}),
                                html.Div([
                                    html.Strong("Tail Risk: "),
                                    html.Span(f"{kelly_data['tail_risk']:.1f}%", 
                                             style={'fontSize':16,'color':'#ef4444'}),
                                    html.Div(style={'width':'100%','background':'#e2e8f0','borderRadius':3,
                                                   'height':8,'marginTop':5}, children=[
                                        html.Div(style={'width':f"{min(kelly_data['tail_risk'], 100)}%",
                                                       'background':'#ef4444','borderRadius':3,'height':8})
                                    ])
                                ])
                            ])
                        ], style={'marginTop':15,'padding':15,'background':'#fffbeb','borderRadius':8}),
                        
                        html.Details([
                            html.Summary("View Analysis", style={'cursor':'pointer','color':'#667eea'}),
                            html.Ul([html.Li(reason) for reason in strat_data['reasons']],
                                   style={'marginTop':10})
                        ], style={'marginTop':10})
                    ])
                ], style={'padding':20,'background':'white','borderRadius':8,
                         'border':'2px solid #f59e0b','marginBottom':15}))
            
            # Avoid Strategies Detail
            avoid_content = []
            for strat_name, strat_data in avoid:
                strikes = self.calculate_strikes(strat_name, signals['spot'])
                strike_text = ', '.join([f"{k.replace('_', ' ').title()}: {v}" 
                                        for k, v in strikes.items()])
                
                # Calculate cost but not Kelly (since we're avoiding)
                cost_data = self.calculate_strategy_cost(strat_name, strikes, df_options, lot_size)
                
                premium_breakdown = []
                for leg in cost_data['premium_legs']:
                    premium_breakdown.append(html.Li(leg, style={'marginBottom':5}))
                
                avoid_content.append(html.Div([
                    html.Div([
                        html.H4(strat_name.replace('_', ' ').title(), 
                               style={'color':'#ef4444','marginBottom':10}),
                        html.Div([
                            html.Strong(f"Match Score: {strat_data['score']:.0f}%"),
                            html.Div(style={'width':f"{strat_data['score']}%",'height':6,
                                          'background':'#ef4444','borderRadius':3,'marginTop':5})
                        ]),
                        html.Div([
                            html.Strong("Strikes: "),
                            html.Span(strike_text, style={'color':'#667eea'})
                        ], style={'marginTop':15,'padding':10,'background':'#fef2f2',
                                 'borderRadius':5}),
                        html.P("⛔ Not recommended under current conditions", 
                              style={'color':'#ef4444','marginTop':10,'fontWeight':'bold'}),
                        
                        # Show cost for reference only
                        html.Details([
                            html.Summary("View Cost (Reference Only)", 
                                       style={'cursor':'pointer','color':'#667eea'}),
                            html.Div([
                                html.Ul(premium_breakdown, style={'marginLeft':20,'marginBottom':10}),
                                html.Div([
                                    html.Strong("Total Cost: "),
                                    html.Span(f"₹{abs(cost_data['total_cost']):,.0f}", 
                                             style={'fontSize':16,'color':'#64748b'})
                                ], style={'padding':8,'background':'#f8fafc','borderRadius':5})
                            ], style={'marginTop':10})
                        ], style={'marginTop':15}),
                        
                        html.Details([
                            html.Summary("View Why Avoid", style={'cursor':'pointer','color':'#667eea'}),
                            html.Ul([html.Li(reason) for reason in strat_data['reasons']],
                                   style={'marginTop':10})
                        ], style={'marginTop':10})
                    ])
                ], style={'padding':20,'background':'white','borderRadius':8,
                         'border':'2px solid #ef4444','marginBottom':15}))
            
            # Signal Analysis
            signal_analysis = html.Div([
                html.H4("Market Condition Matrix", style={'marginBottom':20}),
                html.Table([
                    html.Tr([
                        html.Th("Signal", style={'padding':10,'background':'#1e293b','color':'white'}),
                        html.Th("Current Value", style={'padding':10,'background':'#1e293b','color':'white'}),
                        html.Th("Interpretation", style={'padding':10,'background':'#1e293b','color':'white'})
                    ]),
                    html.Tr([
                        html.Td("VIX Forecast", style={'padding':10}),
                        html.Td(f"{vix_forecast_pct:+.1f}%", style={'padding':10,'fontWeight':'bold'}),
                        html.Td(signals['vix_forecast'], style={'padding':10,'color':'#667eea'})
                    ], style={'background':'#f8fafc'}),
                    html.Tr([
                        html.Td("NIFTY Forecast", style={'padding':10}),
                        html.Td(f"{nifty_forecast_pct:+.1f}%", style={'padding':10,'fontWeight':'bold'}),
                        html.Td(signals['nifty_forecast'], style={'padding':10,'color':'#667eea'})
                    ]),
                    html.Tr([
                        html.Td("Greek Regime", style={'padding':10}),
                        html.Td(signals['greek_regime'], style={'padding':10,'fontWeight':'bold'}),
                        html.Td("Greeks dominating option pricing", style={'padding':10,'color':'#667eea'})
                    ], style={'background':'#f8fafc'}),
                    html.Tr([
                        html.Td("Volatility Regime", style={'padding':10}),
                        html.Td(signals['volatility_regime'], style={'padding':10,'fontWeight':'bold'}),
                        html.Td("Current VIX level classification", style={'padding':10,'color':'#667eea'})
                    ]),
                    html.Tr([
                        html.Td("Return Skew", style={'padding':10}),
                        html.Td(signals['return_skew'], style={'padding':10,'fontWeight':'bold'}),
                        html.Td("Monthly return distribution bias", style={'padding':10,'color':'#667eea'})
                    ], style={'background':'#f8fafc'})
                ], style={'width':'100%','borderCollapse':'collapse','border':'1px solid #e2e8f0'})
            ])
            
            return (signal_summary, strategy_cards, rec_content, 
                   caution_content, avoid_content, signal_analysis)
    
    def run(self, debug=True, port=8060):
        """Run the dashboard"""
        print("\n" + "="*70)
        print("DECISION DASHBOARD")
        print("="*70)
        print(f"Dashboard: http://127.0.0.1:{port}")
        print("="*70 + "\n")
        self.app.run(debug=debug, port=port)

if __name__ == "__main__":
    dashboard = DecisionDashboard()
    dashboard.run(debug=True, port=8060)
