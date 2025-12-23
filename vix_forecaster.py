"""
VIX Forecasting using GARCH(1,1)-X Model
Tier 1: Using existing data (lagged VIX, NIFTY returns, realized volatility)
"""

import pandas as pd
import numpy as np
from arch import arch_model
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class VIXForecaster:
    """
    GARCH(1,1)-X model for VIX forecasting
    
    Model: σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁ + γ·Xₜ
    
    Exogenous variables (Tier 1):
    - Lagged VIX (VIX_lag1, VIX_lag5)
    - NIFTY returns (absolute)
    - Realized volatility (21-day rolling)
    - Return momentum (5-day, 21-day)
    """
    
    def __init__(self, df):
        """
        Initialize forecaster with merged NIFTY + VIX data
        
        Args:
            df: DataFrame with columns ['Date', 'Close_nifty', 'Close_vix']
        """
        self.df = df.copy()
        self.model = None
        self.model_fit = None
        self.exog_vars = []
        
        # Ensure required columns exist
        if 'Close_nifty' not in self.df.columns or 'Close_vix' not in self.df.columns:
            raise ValueError("DataFrame must have 'Close_nifty' and 'Close_vix' columns")
        
    def prepare_features(self):
        """Create exogenous variables from existing data"""
        df = self.df.copy()
        
        # Load and merge Tier 2 macro data
        try:
            macro_data = pd.read_csv('macro_data.csv')
            macro_data['Date'] = pd.to_datetime(macro_data['Date']).dt.tz_localize(None)
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            df = pd.merge(df, macro_data, on='Date', how='left')
            print(f"✓ Merged {len(macro_data)} macro records (Tier 2)")
        except Exception as e:
            print(f"⚠️  Macro data not available: {e}")
        
        # 1. VIX features
        df['VIX_lag1'] = df['Close_vix'].shift(1)
        df['VIX_lag5'] = df['Close_vix'].shift(5)
        df['VIX_change'] = df['Close_vix'].pct_change()
        df['VIX_momentum'] = df['Close_vix'] - df['Close_vix'].shift(5)
        
        # 2. NIFTY returns
        df['NIFTY_return'] = df['Close_nifty'].pct_change() * 100
        df['NIFTY_return_abs'] = df['NIFTY_return'].abs()
        df['NIFTY_return_lag1'] = df['NIFTY_return'].shift(1)
        
        # 3. Realized volatility (multiple windows)
        df['Realized_Vol_5D'] = df['NIFTY_return'].rolling(window=5).std() * np.sqrt(252)
        df['Realized_Vol_21D'] = df['NIFTY_return'].rolling(window=21).std() * np.sqrt(252)
        df['Vol_trend'] = df['Realized_Vol_5D'] - df['Realized_Vol_21D']
        
        # 4. Return momentum and acceleration
        df['Return_5D'] = df['NIFTY_return'].rolling(window=5).mean()
        df['Return_21D'] = df['NIFTY_return'].rolling(window=21).mean()
        df['Return_accel'] = df['Return_5D'] - df['Return_21D']
        
        # 5. High-Low range and volatility of volatility
        df['Range_5D'] = df['NIFTY_return'].rolling(window=5).apply(lambda x: x.max() - x.min())
        df['VIX_vol'] = df['Close_vix'].rolling(window=21).std()
        
        # 6. VIX mean reversion indicators
        df['VIX_MA_21'] = df['Close_vix'].rolling(window=21).mean()
        df['VIX_MA_63'] = df['Close_vix'].rolling(window=63).mean()
        df['VIX_deviation'] = (df['Close_vix'] - df['VIX_MA_21']) / df['VIX_MA_21']
        df['VIX_regime'] = (df['Close_vix'] > df['VIX_MA_63']).astype(int)  # 1 = high vol regime
        
        # 7. Squared returns (volatility clustering)
        df['NIFTY_return_sq'] = df['NIFTY_return'] ** 2
        
        # Drop NaN rows
        df = df.dropna()
        
        self.df_features = df
        return df
    
    def select_exogenous_variables(self, variables=None):
        """
        Select which exogenous variables to include in GARCH-X
        
        Args:
            variables: List of variable names. If None, uses default set.
        """
        if variables is None:
            # Enhanced: Tier 1 + Tier 2 predictive variables
            self.exog_vars = [
                # Tier 1: Derived from NIFTY/VIX
                'VIX_lag1',
                'VIX_momentum',
                'NIFTY_return_abs',
                'NIFTY_return_lag1',
                'Realized_Vol_5D',
                'Vol_trend',
                'VIX_deviation',
                'VIX_regime',
                'NIFTY_return_sq'
            ]
            
            # Tier 2: Add macro variables if available
            if hasattr(self, 'df_features'):
                macro_vars = ['US_VIX', 'US_VIX_change', 'USDINR_vol', 'Crude_vol', 'SP500_vol']
                for var in macro_vars:
                    if var in self.df_features.columns and self.df_features[var].notna().sum() > 100:
                        self.exog_vars.append(var)
                        print(f"  + Added {var} to model")
        else:
            self.exog_vars = variables
        
        print(f"Selected exogenous variables: {', '.join(self.exog_vars)}")
    
    def estimate_mean_reversion(self):
        """
        Estimate mean reversion speed (kappa) from historical VIX data
        
        Uses AR(1) regression: VIX(t) - VIX(t-1) = alpha + beta * VIX(t-1) + error
        Then: kappa = -beta, long_run_mean = -alpha/beta
        
        Returns:
            dict with kappa, mean, half_life, r_squared
        """
        if not hasattr(self, 'df_features'):
            self.prepare_features()
        
        df = self.df_features.copy()
        
        # VIX level and its lag
        vix = df['Close_vix'].values[1:]  # Current VIX
        vix_lag = df['Close_vix'].values[:-1]  # Previous day VIX
        
        # VIX change
        vix_change = vix - vix_lag
        
        # AR(1) regression: ΔVIX = alpha + beta * VIX(t-1)
        # This is equivalent to: VIX(t) = alpha + (1+beta) * VIX(t-1)
        # Mean reverting form: ΔVIX = -kappa * (VIX(t-1) - mu)
        # Expanding: ΔVIX = -kappa * VIX(t-1) + kappa * mu
        # So: alpha = kappa * mu, beta = -kappa
        
        from scipy import stats
        
        # Regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(vix_lag, vix_change)
        
        # Extract parameters
        kappa = -slope  # Mean reversion speed
        long_run_mean = -intercept / slope if slope != 0 else df['Close_vix'].mean()
        
        # Half-life: time for VIX to close half the gap to mean
        # half_life = ln(2) / kappa
        import math
        half_life = math.log(2) / kappa if kappa > 0 else float('inf')
        
        r_squared = r_value ** 2
        
        results = {
            'kappa': kappa,
            'long_run_mean': long_run_mean,
            'half_life_days': half_life,
            'r_squared': r_squared,
            'p_value': p_value
        }
        
        print("\n" + "="*60)
        print("Mean Reversion Analysis")
        print("="*60)
        print(f"Kappa (mean reversion speed): {kappa:.4f}")
        print(f"Long-run mean VIX: {long_run_mean:.2f}")
        print(f"Half-life: {half_life:.1f} days")
        print(f"R-squared: {r_squared:.4f}")
        print(f"P-value: {p_value:.4e}")
        print("="*60)
        
        # Store for later use
        self.kappa = kappa
        self.vix_long_run_mean = long_run_mean
        
        return results
    
    def recommend_lookback_period(self):
        """
        Automatically recommend optimal lookback period (2-year vs 5-year)
        based on current market conditions and model performance
        
        Returns:
            dict with recommendation and reasons
        """
        if not hasattr(self, 'df_features'):
            self.prepare_features()
        
        df = self.df_features.dropna()
        current_vix = df['Close_vix'].iloc[-1]
        vix_mean_63d = df['Close_vix'].rolling(63).mean().iloc[-1]
        vix_mean_252d = df['Close_vix'].rolling(252).mean().iloc[-1]
        
        # Criteria scores (0-10 scale)
        scores = {
            '2_year': 0,
            '5_year': 0
        }
        reasons = []
        
        # 1. Current VIX Level (30% weight)
        if current_vix < 12:
            scores['5_year'] += 3
            reasons.append(f"VIX={current_vix:.2f} is low → 5-year captures tail risk better")
        elif current_vix > 18:
            scores['2_year'] += 3
            reasons.append(f"VIX={current_vix:.2f} is elevated → 2-year captures recent regime")
        else:
            scores['2_year'] += 1.5
            scores['5_year'] += 1.5
            reasons.append(f"VIX={current_vix:.2f} is neutral → both periods viable")
        
        # 2. VIX Regime Stability (25% weight)
        vix_recent = df['Close_vix'].iloc[-252:] if len(df) >= 252 else df['Close_vix']
        vix_cv = vix_recent.std() / vix_recent.mean()  # Coefficient of variation
        
        if vix_cv < 0.3:
            scores['2_year'] += 2.5
            reasons.append(f"VIX stable (CV={vix_cv:.2f}) → 2-year sufficient")
        else:
            scores['5_year'] += 2.5
            reasons.append(f"VIX volatile (CV={vix_cv:.2f}) → 5-year includes regime transitions")
        
        # 3. Distance from Long-term Mean (20% weight)
        vix_ltm = df['Close_vix'].mean()
        distance_pct = abs(current_vix - vix_ltm) / vix_ltm * 100
        
        if distance_pct > 20:
            scores['5_year'] += 2
            reasons.append(f"VIX {distance_pct:.1f}% from LT mean → 5-year for mean reversion")
        else:
            scores['2_year'] += 2
            reasons.append(f"VIX near LT mean → 2-year for current dynamics")
        
        # 4. Recent Trend (15% weight)
        vix_trend = (current_vix - vix_mean_63d) / vix_mean_63d * 100
        
        if abs(vix_trend) > 10:
            scores['2_year'] += 1.5
            reasons.append(f"VIX trending {vix_trend:+.1f}% → 2-year captures momentum")
        else:
            scores['5_year'] += 1.5
            reasons.append("VIX range-bound → 5-year for structural view")
        
        # 5. Data Recency Preference (10% weight)
        # Check if there were major events in last 2 years
        recent_max = vix_recent.max()
        historical_max = df['Close_vix'].max()
        
        if recent_max / historical_max > 0.7:
            scores['2_year'] += 1
            reasons.append("Recent volatility spike → 2-year captures current risks")
        else:
            scores['5_year'] += 1
            reasons.append("Recent period calm → 5-year prevents underestimation")
        
        # Determine recommendation
        if scores['5_year'] > scores['2_year']:
            recommendation = '5_year'
            confidence = min(95, 50 + (scores['5_year'] - scores['2_year']) * 10)
        else:
            recommendation = '2_year'
            confidence = min(95, 50 + (scores['2_year'] - scores['5_year']) * 10)
        
        # Convert to years for display
        lookback_years = 5 if recommendation == '5_year' else 2
        
        return {
            'recommended_period': lookback_years,
            'confidence': confidence,
            'scores': scores,
            'reasons': reasons,
            'current_vix': current_vix,
            'vix_regime': 'LOW' if current_vix < 15 else 'HIGH' if current_vix > 20 else 'MEDIUM',
            'vix_mean_63d': vix_mean_63d,
            'vix_mean_252d': vix_mean_252d
        }
    
    def train_model(self, train_end_date=None, p=1, q=1):
        """
        Train GARCH(p,q)-X model
        
        Args:
            train_end_date: End date for training data (None = use all data)
            p: GARCH order
            q: ARCH order
        
        Returns:
            Model fit results
        """
        if not hasattr(self, 'df_features'):
            self.prepare_features()
        
        if not self.exog_vars:
            self.select_exogenous_variables()
        
        # Prepare data
        df_train = self.df_features.copy()
        if train_end_date:
            df_train = df_train[df_train['Date'] <= train_end_date]
        
        # Target: VIX changes (or VIX levels)
        y = df_train['Close_vix'].values
        
        # Exogenous variables
        X = df_train[self.exog_vars].values
        
        print(f"\nTraining GARCH({p},{q})-X model...")
        print(f"Training samples: {len(y)}")
        print(f"Date range: {df_train['Date'].min().date()} to {df_train['Date'].max().date()}")
        
        # Build GARCH model with exogenous variables
        # Note: arch package expects returns, we'll model VIX changes
        vix_returns = df_train['VIX_change'].dropna().values * 100
        X_aligned = X[1:]  # Align with VIX returns (1 less row due to differencing)
        
        try:
            # GARCH(1,1) with exogenous variables
            model = arch_model(
                vix_returns,
                x=X_aligned,
                vol='GARCH',
                p=p,
                q=q,
                dist='normal'
            )
            
            self.model_fit = model.fit(disp='off')
            self.model = model
            
            print("\n" + "="*60)
            print("GARCH-X Model Summary")
            print("="*60)
            print(self.model_fit.summary())
            
            return self.model_fit
            
        except Exception as e:
            print(f"❌ Error training GARCH-X model: {e}")
            print("Falling back to standard GARCH(1,1) without exogenous variables...")
            
            # Fallback: Standard GARCH without exogenous
            model = arch_model(vix_returns, vol='GARCH', p=p, q=q, dist='normal')
            self.model_fit = model.fit(disp='off')
            self.model = model
            
            return self.model_fit
    
    def forecast(self, horizon=1):
        """
        Generate VIX forecast using GARCH model + mean reversion
        
        Args:
            horizon: Forecast horizon in days (1, 5, 21, etc.)
        
        Returns:
            Dictionary with forecast results
        """
        if self.model_fit is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Estimate mean reversion parameters if not already done
        if not hasattr(self, 'kappa'):
            print("Estimating mean reversion parameters...")
            self.estimate_mean_reversion()
        
        # Get last actual VIX
        last_vix = self.df_features['Close_vix'].iloc[-1]
        last_date = self.df_features['Date'].iloc[-1]
        
        # Use estimated parameters
        vix_mean = self.vix_long_run_mean
        kappa = self.kappa
        
        # Get forecast from GARCH model
        forecast_obj = self.model_fit.forecast(horizon=horizon)
        
        # Get recent features for regime detection
        recent_features = self.df_features.tail(1)
        vix_regime = recent_features['VIX_regime'].iloc[0]
        vix_momentum = recent_features['VIX_momentum'].iloc[0]
        vol_trend = recent_features['Vol_trend'].iloc[0]
        
        # Adjust kappa based on regime (faster reversion in high vol)
        kappa_adjusted = kappa * (1.5 if vix_regime == 1 else 1.0)
        
        # Generate forecasts with regime-dependent mean reversion
        vix_forecast = np.zeros(horizon)
        current_vix = last_vix
        
        # Get GARCH variance forecast for drift adjustment
        variance_forecast = forecast_obj.variance.iloc[-1].values
        volatility_forecast = np.sqrt(variance_forecast)
        
        for i in range(horizon):
            # Base mean reversion
            mr_change = kappa_adjusted * (vix_mean - current_vix)
            
            # Momentum component (short-term continuation)
            momentum_factor = 0.3 if horizon <= 5 else 0.1
            momentum_change = momentum_factor * vix_momentum / (i + 1)  # Decay over horizon
            
            # Volatility trend adjustment
            vol_adjustment = 0.2 * vol_trend if vol_trend > 0 else 0.1 * vol_trend
            
            # Combined forecast
            expected_change = mr_change + momentum_change + vol_adjustment
            current_vix = current_vix + expected_change
            
            # Ensure VIX stays positive and reasonable
            current_vix = max(8, min(current_vix, 60))
            
            vix_forecast[i] = current_vix
        
        results = {
            'last_date': last_date,
            'last_vix': last_vix,
            'forecast_horizon': horizon,
            'vix_forecast': vix_forecast,
            'vix_volatility': volatility_forecast,  # Uncertainty in forecast
            'kappa': kappa,
            'long_run_mean': vix_mean,
            'forecast_dates': [last_date + timedelta(days=i+1) for i in range(horizon)]
        }
        
        return results
    
    def rolling_forecast(self, window_size=252*2, horizon=1, step=21):
        """
        Generate rolling window forecasts for backtesting
        
        Args:
            window_size: Training window size in days (default: 2 years)
            horizon: Forecast horizon (default: 1 day)
            step: Days between forecasts (default: 21 for monthly)
        
        Returns:
            DataFrame with forecasts and actual values
        """
        if not hasattr(self, 'df_features'):
            self.prepare_features()
        
        df = self.df_features.copy()
        forecasts = []
        
        # Start from window_size + 50 to ensure enough data
        start_idx = window_size
        end_idx = len(df) - horizon
        
        print(f"\nRunning rolling forecasts...")
        print(f"Window size: {window_size} days")
        print(f"Forecast horizon: {horizon} day(s)")
        print(f"Step: {step} days")
        
        for i in range(start_idx, end_idx, step):
            # Training data: window ending at i
            train_data = df.iloc[i-window_size:i].copy()
            
            # Forecast target date
            forecast_date = df.iloc[i + horizon - 1]['Date']
            actual_vix = df.iloc[i + horizon - 1]['Close_vix']
            
            try:
                # Build temporary model for this window
                y_train = train_data['VIX_change'].dropna().values * 100
                X_train = train_data[self.exog_vars].iloc[1:].values  # Align with returns
                
                model = arch_model(y_train, x=X_train, vol='GARCH', p=1, q=1, dist='normal')
                fit = model.fit(disp='off')
                
                # Estimate mean reversion for this window
                vix_window = train_data['Close_vix'].values[1:]
                vix_lag_window = train_data['Close_vix'].values[:-1]
                vix_change_window = vix_window - vix_lag_window
                
                from scipy import stats
                slope, intercept, _, _, _ = stats.linregress(vix_lag_window, vix_change_window)
                kappa_window = -slope
                vix_mean_window = -intercept / slope if slope != 0 else train_data['Close_vix'].mean()
                
                # Ensure kappa is positive and reasonable
                if kappa_window <= 0 or kappa_window > 1:
                    kappa_window = 0.1  # Fallback to default
                
                # Get last VIX and features in training window
                last_vix = train_data['Close_vix'].iloc[-1]
                last_features = train_data.iloc[-1]
                vix_regime = last_features['VIX_regime']
                vix_momentum = last_features['VIX_momentum']
                vol_trend = last_features['Vol_trend']
                
                # Adjust kappa based on regime
                kappa_adjusted = kappa_window * (1.5 if vix_regime == 1 else 1.0)
                
                # Mean reversion forecast with regime adjustments
                current_vix = last_vix
                for h in range(horizon):
                    mr_change = kappa_adjusted * (vix_mean_window - current_vix)
                    momentum_factor = 0.3 if horizon <= 5 else 0.1
                    momentum_change = momentum_factor * vix_momentum / (h + 1)
                    vol_adjustment = 0.2 * vol_trend if vol_trend > 0 else 0.1 * vol_trend
                    
                    expected_change = mr_change + momentum_change + vol_adjustment
                    current_vix = current_vix + expected_change
                    current_vix = max(8, min(current_vix, 60))  # Bounds
                
                vix_fcast = current_vix
                
                forecasts.append({
                    'Date': forecast_date,
                    'Actual_VIX': actual_vix,
                    'Forecast_VIX': vix_fcast,
                    'Forecast_Error': actual_vix - vix_fcast,
                    'Forecast_Error_Pct': ((actual_vix - vix_fcast) / actual_vix) * 100
                })
                
                if len(forecasts) % 10 == 0:
                    print(f"  Completed {len(forecasts)} forecasts...")
                    
            except Exception as e:
                print(f"  ⚠️ Skipping forecast for {forecast_date}: {e}")
                continue
        
        df_forecasts = pd.DataFrame(forecasts)
        
        if not df_forecasts.empty:
            print(f"\n✓ Generated {len(df_forecasts)} forecasts")
            print(f"Date range: {df_forecasts['Date'].min().date()} to {df_forecasts['Date'].max().date()}")
            
            # Calculate accuracy metrics
            mae = df_forecasts['Forecast_Error'].abs().mean()
            rmse = np.sqrt((df_forecasts['Forecast_Error']**2).mean())
            mape = df_forecasts['Forecast_Error_Pct'].abs().mean()
            
            print(f"\nForecast Accuracy:")
            print(f"  MAE:  {mae:.2f}")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  MAPE: {mape:.2f}%")
        
        return df_forecasts
    
    def get_model_diagnostics(self):
        """Get model diagnostics and statistics"""
        if self.model_fit is None:
            return None
        
        diagnostics = {
            'AIC': self.model_fit.aic,
            'BIC': self.model_fit.bic,
            'Log_Likelihood': self.model_fit.loglikelihood,
            'Parameters': self.model_fit.params.to_dict(),
            'P_values': self.model_fit.pvalues.to_dict()
        }
        
        return diagnostics


# Example usage
if __name__ == "__main__":
    # Load data
    nifty_df = pd.read_csv('nifty_history.csv')
    nifty_df['Date'] = pd.to_datetime(nifty_df['Date'])
    nifty_df = nifty_df.rename(columns={'Close': 'Close_nifty'})
    
    vix_df = pd.read_csv('india_vix_history.csv')
    vix_df['Date'] = pd.to_datetime(vix_df['Date'])
    vix_df = vix_df.rename(columns={'Close': 'Close_vix'})
    
    df = pd.merge(nifty_df[['Date', 'Close_nifty']], 
                 vix_df[['Date', 'Close_vix']], 
                 on='Date', how='inner')
    
    # Initialize forecaster
    forecaster = VIXForecaster(df)
    
    # Prepare features
    forecaster.prepare_features()
    
    # Train model
    forecaster.train_model()
    
    # Generate forecast
    forecast_results = forecaster.forecast(horizon=5)
    print(f"\n5-day VIX Forecast: {forecast_results['vix_forecast']}")
    
    # Rolling forecasts for backtesting
    backtest_results = forecaster.rolling_forecast(window_size=252*2, horizon=1, step=21)
    print(backtest_results.head())
