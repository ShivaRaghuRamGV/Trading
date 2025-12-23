"""
Machine Learning Models for Options Strategy Selection
=======================================================

Three-Model System:
1. XGBoost Classifier: Predict 22-day direction (up/down) → Strategy selection
2. XGBoost Regressor: Predict 22-day % move → Strike selection & spread width
3. Tail-Risk Model: Predict extreme moves (>2 SD) → Position sizing & hedging

Author: Trading System
Date: December 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')


class OptionsMLModels:
    """
    ML-based options strategy selection and risk management
    """
    
    def __init__(self, data_file='nifty_history.csv', vix_file='india_vix_history.csv', 
                 macro_file='macro_data.csv'):
        """
        Initialize with historical data
        
        Args:
            data_file: NIFTY historical data
            vix_file: VIX historical data
            macro_file: Macro variables (Tier 2)
        """
        self.data_file = data_file
        self.vix_file = vix_file
        self.macro_file = macro_file
        
        # Models
        self.direction_model = None  # XGBoost Classifier
        self.magnitude_model = None  # XGBoost Regressor
        self.tail_risk_model = None  # XGBoost Classifier (tail events)
        
        # Scalers
        self.scaler = StandardScaler()
        
        # Feature importance
        self.feature_names = None
        
        print("="*60)
        print("ML Models for Options Trading")
        print("="*60)
        
    def load_and_merge_data(self):
        """Load and merge all data sources"""
        print("\n📊 Loading data...")
        
        # Load NIFTY
        nifty = pd.read_csv(self.data_file)
        nifty['Date'] = pd.to_datetime(nifty['Date']).dt.tz_localize(None)
        nifty = nifty.rename(columns={'Close': 'Close_nifty'})
        
        # Load VIX
        vix = pd.read_csv(self.vix_file)
        vix['Date'] = pd.to_datetime(vix['Date']).dt.tz_localize(None)
        vix = vix.rename(columns={'Close': 'Close_vix'})
        
        # Merge NIFTY + VIX
        df = pd.merge(nifty[['Date', 'Close_nifty']], vix[['Date', 'Close_vix']], 
                     on='Date', how='inner')
        
        # Load macro data if available
        try:
            macro = pd.read_csv(self.macro_file)
            macro['Date'] = pd.to_datetime(macro['Date']).dt.tz_localize(None)
            df = pd.merge(df, macro, on='Date', how='left')
            print(f"  ✓ Loaded macro data ({len(macro)} records)")
        except:
            print("  ⚠️ Macro data not available")
        
        df = df.sort_values('Date').reset_index(drop=True)
        print(f"  ✓ Merged dataset: {len(df)} records")
        print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        
        self.df = df
        return df
    
    def create_features(self, df=None):
        """
        Create features for ML models
        
        Features include:
        - Price momentum (multiple timeframes)
        - Volatility indicators (realized, implied)
        - Technical indicators (RSI, MACD, Bollinger Bands)
        - Macro variables (VIX, currency, commodities)
        - Calendar effects (day of week, month)
        """
        if df is None:
            df = self.df.copy()
        else:
            df = df.copy()
        
        print("\n🔧 Engineering features...")
        
        # === Price Features ===
        # Returns (multiple horizons)
        for period in [1, 3, 5, 10, 21]:
            df[f'return_{period}d'] = df['Close_nifty'].pct_change(period) * 100
        
        # Momentum indicators
        df['momentum_5_21'] = df['return_5d'] - df['return_21d']
        df['momentum_10_21'] = df['return_10d'] - df['return_21d']
        
        # Moving averages
        for window in [5, 10, 21, 50, 100, 200]:
            df[f'sma_{window}'] = df['Close_nifty'].rolling(window).mean()
            df[f'price_to_sma_{window}'] = (df['Close_nifty'] / df[f'sma_{window}'] - 1) * 100
        
        # === Volatility Features ===
        # Realized volatility (multiple windows)
        for window in [5, 10, 21, 63]:
            df[f'realized_vol_{window}d'] = df['return_1d'].rolling(window).std() * np.sqrt(252)
        
        # Volatility term structure
        df['vol_term_5_21'] = df['realized_vol_5d'] - df['realized_vol_21d']
        df['vol_term_21_63'] = df['realized_vol_21d'] - df['realized_vol_63d']
        
        # VIX features
        df['vix_change_1d'] = df['Close_vix'].pct_change()
        df['vix_change_5d'] = df['Close_vix'].pct_change(5)
        df['vix_ma_21'] = df['Close_vix'].rolling(21).mean()
        df['vix_deviation'] = (df['Close_vix'] - df['vix_ma_21']) / df['vix_ma_21']
        df['vix_regime'] = (df['Close_vix'] > df['vix_ma_21']).astype(int)
        
        # === Technical Indicators ===
        # RSI (14-day)
        delta = df['Close_nifty'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['Close_nifty'].ewm(span=12).mean()
        ema_26 = df['Close_nifty'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['Close_nifty'].rolling(20).mean()
        bb_std = df['Close_nifty'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * bb_std
        df['bb_lower'] = df['bb_middle'] - 2 * bb_std
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['Close_nifty'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # === Risk Metrics ===
        # Drawdown
        df['cummax'] = df['Close_nifty'].cummax()
        df['drawdown'] = (df['Close_nifty'] / df['cummax'] - 1) * 100
        
        # Skewness and Kurtosis (21-day rolling)
        df['return_skew_21d'] = df['return_1d'].rolling(21).skew()
        df['return_kurt_21d'] = df['return_1d'].rolling(21).kurt()
        
        # === Macro Features (if available) ===
        if 'US_VIX' in df.columns:
            df['us_vix_change'] = df['US_VIX'].pct_change()
            df['vix_spread'] = df['Close_vix'] - df['US_VIX']
        
        if 'USDINR_vol' in df.columns:
            df['usdinr_vol_change'] = df['USDINR_vol'].pct_change()
        
        # === Calendar Features ===
        df['day_of_week'] = pd.to_datetime(df['Date']).dt.dayofweek
        df['month'] = pd.to_datetime(df['Date']).dt.month
        df['quarter'] = pd.to_datetime(df['Date']).dt.quarter
        
        # Expiry week effect (assume weekly expiries on Thursday)
        df['days_to_expiry'] = (3 - df['day_of_week']) % 7  # Days to next Thursday
        df['is_expiry_week'] = (df['days_to_expiry'] <= 2).astype(int)
        
        print(f"  ✓ Created {len([c for c in df.columns if c not in ['Date', 'Close_nifty', 'Close_vix']])} features")
        
        self.df_features = df
        return df
    
    def create_targets(self, df=None, horizon=22):
        """
        Create target variables for ML models
        
        Args:
            df: Features DataFrame
            horizon: Forward-looking period (default 22 trading days ~ 1 month)
        
        Returns:
            DataFrame with targets
        """
        if df is None:
            df = self.df_features.copy()
        else:
            df = df.copy()
        
        print(f"\n🎯 Creating targets (horizon={horizon} days)...")
        
        # Future price
        df['future_price'] = df['Close_nifty'].shift(-horizon)
        
        # Target 1: Direction (Binary Classification)
        df['future_return'] = ((df['future_price'] / df['Close_nifty']) - 1) * 100
        df['target_direction'] = (df['future_return'] > 0).astype(int)  # 1=up, 0=down
        
        # Target 2: Magnitude (Regression)
        df['target_magnitude'] = df['future_return'].abs()  # Absolute % move
        df['target_return'] = df['future_return']  # Signed return
        
        # Target 3: Tail Risk (Binary Classification for extreme moves)
        # Define tail as >2 standard deviations
        return_std = df['future_return'].std()
        df['target_tail_risk'] = (df['future_return'].abs() > 2 * return_std).astype(int)
        
        # Drop rows without future data
        df = df[df['future_price'].notna()].copy()
        
        print(f"  ✓ Direction: {df['target_direction'].mean()*100:.1f}% up, {(1-df['target_direction'].mean())*100:.1f}% down")
        print(f"  ✓ Magnitude: Mean={df['target_magnitude'].mean():.2f}%, Median={df['target_magnitude'].median():.2f}%")
        print(f"  ✓ Tail Risk: {df['target_tail_risk'].mean()*100:.1f}% extreme moves (>{2*return_std:.1f}%)")
        
        self.df_targets = df
        return df
    
    def prepare_training_data(self, test_size=0.2):
        """
        Prepare train/test splits using time-series split
        
        Args:
            test_size: Fraction for test set
        """
        df = self.df_targets.copy()
        
        # Select feature columns (exclude Date, targets, intermediate calculations)
        exclude_cols = ['Date', 'Close_nifty', 'Close_vix', 'future_price', 'future_return',
                       'target_direction', 'target_magnitude', 'target_return', 'target_tail_risk',
                       'cummax', 'sma_5', 'sma_10', 'sma_21', 'sma_50', 'sma_100', 'sma_200',
                       'bb_middle', 'bb_upper', 'bb_lower', 'vix_ma_21']
        
        feature_cols = [c for c in df.columns if c not in exclude_cols and not df[c].isna().all()]
        
        # Remove any remaining NaN
        df = df.dropna(subset=feature_cols)
        
        X = df[feature_cols]
        y_direction = df['target_direction']
        y_magnitude = df['target_magnitude']
        y_tail_risk = df['target_tail_risk']
        dates = df['Date']
        
        # Time-series split (no shuffling!)
        split_idx = int(len(X) * (1 - test_size))
        
        self.X_train = X.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        
        self.y_direction_train = y_direction.iloc[:split_idx]
        self.y_direction_test = y_direction.iloc[split_idx:]
        
        self.y_magnitude_train = y_magnitude.iloc[:split_idx]
        self.y_magnitude_test = y_magnitude.iloc[split_idx:]
        
        self.y_tail_train = y_tail_risk.iloc[:split_idx]
        self.y_tail_test = y_tail_risk.iloc[split_idx:]
        
        self.dates_train = dates.iloc[:split_idx]
        self.dates_test = dates.iloc[split_idx:]
        
        self.feature_names = feature_cols
        
        print(f"\n📊 Training Data Prepared")
        print(f"  Train: {len(self.X_train)} samples ({self.dates_train.min().date()} to {self.dates_train.max().date()})")
        print(f"  Test:  {len(self.X_test)} samples ({self.dates_test.min().date()} to {self.dates_test.max().date()})")
        print(f"  Features: {len(feature_cols)}")
        
        return self.X_train, self.X_test
    
    def train_direction_model(self):
        """
        Train XGBoost Classifier for 22-day direction prediction
        Used for: Strategy selection (condor vs directional)
        """
        print("\n" + "="*60)
        print("Training Direction Model (XGBoost Classifier)")
        print("="*60)
        
        # XGBoost parameters optimized for binary classification
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 5,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'tree_method': 'hist'
        }
        
        self.direction_model = xgb.XGBClassifier(**params)
        self.direction_model.fit(self.X_train, self.y_direction_train,
                                eval_set=[(self.X_test, self.y_direction_test)],
                                verbose=False)
        
        # Predictions
        y_pred_train = self.direction_model.predict(self.X_train)
        y_pred_test = self.direction_model.predict(self.X_test)
        
        # Probabilities
        y_prob_test = self.direction_model.predict_proba(self.X_test)[:, 1]
        
        # Metrics
        train_acc = accuracy_score(self.y_direction_train, y_pred_train)
        test_acc = accuracy_score(self.y_direction_test, y_pred_test)
        test_precision = precision_score(self.y_direction_test, y_pred_test)
        test_recall = recall_score(self.y_direction_test, y_pred_test)
        test_f1 = f1_score(self.y_direction_test, y_pred_test)
        
        print(f"\n📊 Direction Model Performance:")
        print(f"  Train Accuracy: {train_acc*100:.2f}%")
        print(f"  Test Accuracy:  {test_acc*100:.2f}%")
        print(f"  Precision:      {test_precision*100:.2f}%")
        print(f"  Recall:         {test_recall*100:.2f}%")
        print(f"  F1-Score:       {test_f1:.4f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.direction_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔝 Top 10 Features:")
        for idx, row in importance.head(10).iterrows():
            print(f"  {row['feature']:30s}: {row['importance']:.4f}")
        
        self.direction_importance = importance
        
        return self.direction_model
    
    def train_magnitude_model(self):
        """
        Train XGBoost Regressor for 22-day % move prediction
        Used for: Strike selection & spread width optimization
        """
        print("\n" + "="*60)
        print("Training Magnitude Model (XGBoost Regressor)")
        print("="*60)
        
        # XGBoost parameters for regression
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'tree_method': 'hist'
        }
        
        self.magnitude_model = xgb.XGBRegressor(**params)
        self.magnitude_model.fit(self.X_train, self.y_magnitude_train,
                                eval_set=[(self.X_test, self.y_magnitude_test)],
                                verbose=False)
        
        # Predictions
        y_pred_train = self.magnitude_model.predict(self.X_train)
        y_pred_test = self.magnitude_model.predict(self.X_test)
        
        # Metrics
        train_mae = mean_absolute_error(self.y_magnitude_train, y_pred_train)
        test_mae = mean_absolute_error(self.y_magnitude_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(self.y_magnitude_test, y_pred_test))
        test_r2 = r2_score(self.y_magnitude_test, y_pred_test)
        
        print(f"\n📊 Magnitude Model Performance:")
        print(f"  Train MAE: {train_mae:.2f}%")
        print(f"  Test MAE:  {test_mae:.2f}%")
        print(f"  Test RMSE: {test_rmse:.2f}%")
        print(f"  Test R²:   {test_r2:.4f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.magnitude_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔝 Top 10 Features:")
        for idx, row in importance.head(10).iterrows():
            print(f"  {row['feature']:30s}: {row['importance']:.4f}")
        
        self.magnitude_importance = importance
        
        return self.magnitude_model
    
    def train_tail_risk_model(self):
        """
        Train XGBoost Classifier for extreme move prediction (>2 SD)
        Used for: Position sizing & hedging decisions
        """
        print("\n" + "="*60)
        print("Training Tail Risk Model (XGBoost Classifier)")
        print("="*60)
        
        # Handle class imbalance with scale_pos_weight
        pos_ratio = (self.y_tail_train == 0).sum() / (self.y_tail_train == 1).sum()
        
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 4,
            'learning_rate': 0.03,
            'n_estimators': 300,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'scale_pos_weight': pos_ratio,
            'random_state': 42,
            'tree_method': 'hist'
        }
        
        self.tail_risk_model = xgb.XGBClassifier(**params)
        self.tail_risk_model.fit(self.X_train, self.y_tail_train,
                                eval_set=[(self.X_test, self.y_tail_test)],
                                verbose=False)
        
        # Predictions
        y_pred_train = self.tail_risk_model.predict(self.X_train)
        y_pred_test = self.tail_risk_model.predict(self.X_test)
        
        # Probabilities (key for position sizing!)
        y_prob_test = self.tail_risk_model.predict_proba(self.X_test)[:, 1]
        
        # Metrics
        train_acc = accuracy_score(self.y_tail_train, y_pred_train)
        test_acc = accuracy_score(self.y_tail_test, y_pred_test)
        test_precision = precision_score(self.y_tail_test, y_pred_test, zero_division=0)
        test_recall = recall_score(self.y_tail_test, y_pred_test, zero_division=0)
        test_f1 = f1_score(self.y_tail_test, y_pred_test, zero_division=0)
        
        print(f"\n📊 Tail Risk Model Performance:")
        print(f"  Train Accuracy: {train_acc*100:.2f}%")
        print(f"  Test Accuracy:  {test_acc*100:.2f}%")
        print(f"  Precision:      {test_precision*100:.2f}%")
        print(f"  Recall:         {test_recall*100:.2f}%")
        print(f"  F1-Score:       {test_f1:.4f}")
        
        # Tail risk statistics
        tail_prob_mean = y_prob_test.mean()
        tail_prob_75 = np.percentile(y_prob_test, 75)
        tail_prob_95 = np.percentile(y_prob_test, 95)
        
        print(f"\n⚠️  Tail Risk Probability Distribution:")
        print(f"  Mean:     {tail_prob_mean*100:.2f}%")
        print(f"  75th pct: {tail_prob_75*100:.2f}%")
        print(f"  95th pct: {tail_prob_95*100:.2f}%")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.tail_risk_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔝 Top 10 Features:")
        for idx, row in importance.head(10).iterrows():
            print(f"  {row['feature']:30s}: {row['importance']:.4f}")
        
        self.tail_importance = importance
        
        return self.tail_risk_model
    
    def save_models(self, prefix='models/'):
        """Save trained models to disk"""
        import os
        os.makedirs(prefix, exist_ok=True)
        
        with open(f'{prefix}direction_model.pkl', 'wb') as f:
            pickle.dump(self.direction_model, f)
        
        with open(f'{prefix}magnitude_model.pkl', 'wb') as f:
            pickle.dump(self.magnitude_model, f)
        
        with open(f'{prefix}tail_risk_model.pkl', 'wb') as f:
            pickle.dump(self.tail_risk_model, f)
        
        # Save feature names
        with open(f'{prefix}feature_names.pkl', 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        # Save feature importance DataFrames
        if hasattr(self, 'direction_importance'):
            with open(f'{prefix}direction_importance.pkl', 'wb') as f:
                pickle.dump(self.direction_importance, f)
        
        if hasattr(self, 'magnitude_importance'):
            with open(f'{prefix}magnitude_importance.pkl', 'wb') as f:
                pickle.dump(self.magnitude_importance, f)
        
        if hasattr(self, 'tail_importance'):
            with open(f'{prefix}tail_importance.pkl', 'wb') as f:
                pickle.dump(self.tail_importance, f)
        
        print(f"\n✅ Models saved to {prefix}")
    
    def load_models(self, prefix='models/'):
        """Load trained models from disk"""
        with open(f'{prefix}direction_model.pkl', 'rb') as f:
            self.direction_model = pickle.load(f)
        
        with open(f'{prefix}magnitude_model.pkl', 'rb') as f:
            self.magnitude_model = pickle.load(f)
        
        with open(f'{prefix}tail_risk_model.pkl', 'rb') as f:
            self.tail_risk_model = pickle.load(f)
        
        with open(f'{prefix}feature_names.pkl', 'rb') as f:
            self.feature_names = pickle.load(f)
        
        # Load feature importance DataFrames
        try:
            with open(f'{prefix}direction_importance.pkl', 'rb') as f:
                self.direction_importance = pickle.load(f)
        except FileNotFoundError:
            # Generate from model if not saved
            self.direction_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.direction_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        try:
            with open(f'{prefix}magnitude_importance.pkl', 'rb') as f:
                self.magnitude_importance = pickle.load(f)
        except FileNotFoundError:
            self.magnitude_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.magnitude_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        try:
            with open(f'{prefix}tail_importance.pkl', 'rb') as f:
                self.tail_importance = pickle.load(f)
        except FileNotFoundError:
            self.tail_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.tail_risk_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        print(f"\n✅ Models loaded from {prefix}")
    
    def predict(self, X):
        """
        Make predictions for new data
        
        Args:
            X: Features DataFrame
        
        Returns:
            dict with predictions from all models
        """
        predictions = {}
        
        # Direction
        predictions['direction_prob'] = self.direction_model.predict_proba(X)[:, 1]
        predictions['direction'] = self.direction_model.predict(X)
        
        # Magnitude
        predictions['magnitude'] = self.magnitude_model.predict(X)
        
        # Tail Risk
        predictions['tail_risk_prob'] = self.tail_risk_model.predict_proba(X)[:, 1]
        predictions['tail_risk'] = self.tail_risk_model.predict(X)
        
        return predictions


if __name__ == "__main__":
    # Full training pipeline
    print("Starting ML Model Training Pipeline...")
    
    # Initialize
    ml = OptionsMLModels()
    
    # Load data
    ml.load_and_merge_data()
    
    # Create features
    ml.create_features()
    
    # Create targets (22-day horizon for monthly expiry)
    ml.create_targets(horizon=22)
    
    # Prepare train/test splits
    ml.prepare_training_data(test_size=0.2)
    
    # Train all models
    ml.train_direction_model()
    ml.train_magnitude_model()
    ml.train_tail_risk_model()
    
    # Save models
    ml.save_models()
    
    print("\n" + "="*60)
    print("✅ All models trained and saved successfully!")
    print("="*60)
