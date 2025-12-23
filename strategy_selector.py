"""
ML-Based Options Strategy Selector
===================================

Uses trained XGBoost models to:
1. Select strategy type (Condor vs Directional)
2. Set strike distances dynamically
3. Determine position sizing based on tail risk

Author: Trading System
Date: December 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
from ml_models import OptionsMLModels


class MLStrategySelector:
    """
    Select and configure options strategies using ML predictions
    """
    
    def __init__(self, model_path='models/'):
        """
        Load trained ML models
        
        Args:
            model_path: Directory containing saved models
        """
        self.model_path = model_path
        self.ml = OptionsMLModels()
        self.ml.load_models(model_path)
        
        print("="*60)
        print("ML-Based Strategy Selector")
        print("="*60)
        print(f"✓ Models loaded from {model_path}")
    
    def prepare_current_features(self, nifty_file='nifty_history.csv', 
                                vix_file='india_vix_history.csv',
                                macro_file='macro_data.csv'):
        """
        Prepare features for current market conditions
        
        Returns:
            Latest feature vector for prediction
        """
        # Load data
        self.ml.load_and_merge_data()
        self.ml.create_features()
        
        # Get latest row (most recent data)
        latest_features = self.ml.df_features.iloc[-1:]
        
        # Select only model features
        feature_cols = self.ml.feature_names
        X_current = latest_features[feature_cols]
        
        self.current_features = X_current
        self.current_date = latest_features['Date'].iloc[0]
        self.current_nifty = latest_features['Close_nifty'].iloc[0]
        self.current_vix = latest_features['Close_vix'].iloc[0]
        
        return X_current
    
    def get_predictions(self, X=None):
        """
        Get predictions from all ML models
        
        Args:
            X: Feature vector (if None, uses current features)
        
        Returns:
            dict with predictions
        """
        if X is None:
            X = self.current_features
        
        predictions = self.ml.predict(X)
        
        # Format predictions
        # Note: direction_prob and tail_risk_prob come from model as 0-1 probabilities
        direction_prob = predictions['direction_prob'][0]  # 0-1 range
        tail_risk_prob = predictions['tail_risk_prob'][0]  # 0-1 range
        
        result = {
            'direction': 'UP' if predictions['direction'][0] == 1 else 'DOWN',
            'direction_prob': direction_prob * 100,  # Convert to percentage for display
            'confidence': abs(direction_prob - 0.5) * 200,  # 0-100% confidence
            'expected_move': predictions['magnitude'][0],
            'tail_risk_prob': tail_risk_prob * 100,  # Convert to percentage for display
            'predicted_nifty': None,
            'predicted_range_low': None,
            'predicted_range_high': None
        }
        
        # Calculate predicted NIFTY level (22 days forward)
        if result['direction'] == 'UP':
            result['predicted_nifty'] = self.current_nifty * (1 + result['expected_move']/100)
        else:
            result['predicted_nifty'] = self.current_nifty * (1 - result['expected_move']/100)
        
        # Predicted range (±1 std dev)
        result['predicted_range_low'] = self.current_nifty * (1 - result['expected_move']/100)
        result['predicted_range_high'] = self.current_nifty * (1 + result['expected_move']/100)
        
        return result
    
    def select_strategy(self, predictions=None, confidence_threshold=60):
        """
        Select strategy based on ML predictions
        
        Strategy Logic:
        - High confidence (>60%) + Low tail risk (<5%): Directional (Straddle/Strangle)
        - Low confidence (<40%): Iron Condor (neutral)
        - High tail risk (>10%): Avoid or hedge heavily
        
        Args:
            predictions: Prediction dict from get_predictions()
            confidence_threshold: Minimum confidence for directional trades
        
        Returns:
            dict with strategy recommendation
        """
        if predictions is None:
            predictions = self.get_predictions()
        
        confidence = predictions['confidence']
        tail_risk = predictions['tail_risk_prob']
        direction = predictions['direction']
        expected_move = predictions['expected_move']
        
        # Debug logging
        print(f"\n[STRATEGY DEBUG] Confidence: {confidence:.2f}%, Tail Risk: {tail_risk:.2f}%, Direction: {direction}, Expected Move: {expected_move:.2f}%")
        print(f"[STRATEGY DEBUG] Threshold: {confidence_threshold}%, Tail Risk Check: {tail_risk} <= 7 = {tail_risk <= 7}")
        print(f"[STRATEGY DEBUG] Confidence Check: {confidence} >= {confidence_threshold} = {confidence >= confidence_threshold}")
        print(f"[STRATEGY DEBUG] Large Move Check: {expected_move} >= 4.0 = {expected_move >= 4.0}")
        
        # Decision logic
        if tail_risk > 10:
            strategy_type = 'AVOID'
            reasoning = f"High tail risk ({tail_risk:.1f}%) - extreme move likely. Reduce exposure or hedge."
            print(f"[STRATEGY DEBUG] Selected: AVOID (tail_risk > 10)")
        
        elif (confidence >= confidence_threshold and tail_risk <= 7) or (expected_move >= 4.0 and confidence >= 50):
            # Directional strategy: High confidence OR large expected move
            if direction == 'UP':
                strategy_type = 'LONG_CALL'  # Or Bull Call Spread
            else:
                strategy_type = 'LONG_PUT'   # Or Bear Put Spread
            
            if expected_move >= 4.0:
                reasoning = f"Large expected move ({expected_move:.2f}%) {direction} with {confidence:.1f}% confidence. Go directional."
            else:
                reasoning = f"High confidence ({confidence:.1f}%) {direction} move with low tail risk. Go directional."
            print(f"[STRATEGY DEBUG] Selected: {strategy_type} (high confidence OR large move)")
        
        elif confidence < 40 and expected_move < 2.5:
            # Neutral strategy: Low confidence AND small move expected
            strategy_type = 'IRON_CONDOR'
            reasoning = f"Low confidence ({confidence:.1f}%) with limited move ({expected_move:.2f}%) - range-bound expected. Sell premium."
            print(f"[STRATEGY DEBUG] Selected: IRON_CONDOR (confidence < 40 AND small move)")
        
        else:
            # Moderate confidence or moderate move
            strategy_type = 'SHORT_STRANGLE'
            reasoning = f"Moderate confidence ({confidence:.1f}%) with {expected_move:.2f}% expected move. Sell strangle."
            print(f"[STRATEGY DEBUG] Selected: SHORT_STRANGLE (moderate confidence/move)")
        
        return {
            'strategy': strategy_type,
            'reasoning': reasoning,
            'confidence': confidence,
            'tail_risk': tail_risk,
            'direction': direction
        }
    
    def calculate_strike_distances(self, predictions=None):
        """
        Calculate optimal strike distances based on predicted magnitude
        
        Uses regression model output to set strikes that capture expected move
        
        Args:
            predictions: Prediction dict
        
        Returns:
            dict with strike recommendations
        """
        if predictions is None:
            predictions = self.get_predictions()
        
        expected_move_pct = predictions['expected_move']
        current_nifty = self.current_nifty
        
        # Convert % to points
        expected_move_points = current_nifty * (expected_move_pct / 100)
        
        # Round to nearest 50
        expected_move_rounded = round(expected_move_points / 50) * 50
        
        # Strategy-specific strike distances
        strikes = {}
        
        # For Iron Condor
        # Short strikes: Just outside expected range (0.8x move)
        # Long strikes: Further out for protection (1.5x move)
        strikes['ic_short_call'] = round(expected_move_rounded * 0.8 / 50) * 50
        strikes['ic_short_put'] = round(expected_move_rounded * 0.8 / 50) * 50
        strikes['ic_long_call'] = round(expected_move_rounded * 1.5 / 50) * 50
        strikes['ic_long_put'] = round(expected_move_rounded * 1.5 / 50) * 50
        
        # For Short Strangle
        # Strikes at 1x expected move
        strikes['strangle_call'] = round(expected_move_rounded * 1.0 / 50) * 50
        strikes['strangle_put'] = round(expected_move_rounded * 1.0 / 50) * 50
        
        # For Directional (Long Call/Put)
        # ATM or slightly OTM
        strikes['directional_distance'] = round(expected_move_rounded * 0.3 / 50) * 50
        
        # Spread width for credit/debit spreads
        strikes['spread_width'] = round(expected_move_rounded * 0.5 / 50) * 50
        
        return strikes
    
    def calculate_position_size(self, predictions=None, base_capital=100000, 
                               max_risk_pct=2.0):
        """
        Calculate position sizing using Kelly Criterion adjusted for tail risk
        
        Kelly = (p×b - q) / b
        where p = win prob, q = loss prob, b = win/loss ratio
        
        Adjusted by tail risk probability
        
        Args:
            predictions: Prediction dict
            base_capital: Trading capital
            max_risk_pct: Maximum risk per trade (%)
        
        Returns:
            dict with position sizing recommendations
        """
        if predictions is None:
            predictions = self.get_predictions()
        
        tail_risk = predictions['tail_risk_prob'] / 100  # Convert to decimal
        confidence = predictions['confidence'] / 100
        
        # Win probability (from direction model confidence)
        win_prob = 0.5 + (confidence / 2)  # Maps 0-100% confidence to 50-100% win prob
        loss_prob = 1 - win_prob
        
        # Win/loss ratio (assume 2:1 for credit strategies, 1:2 for debit)
        win_loss_ratio = 2.0  # Conservative estimate for premium selling
        
        # Kelly fraction
        kelly_fraction = (win_prob * win_loss_ratio - loss_prob) / win_loss_ratio
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        # Adjust for tail risk (reduce size if high tail risk)
        tail_risk_factor = 1 - (tail_risk * 2)  # 10% tail risk → 80% size
        tail_risk_factor = max(0.2, tail_risk_factor)  # Minimum 20% size
        
        adjusted_kelly = kelly_fraction * tail_risk_factor
        
        # Position size
        risk_amount = base_capital * (max_risk_pct / 100)
        position_size_kelly = base_capital * adjusted_kelly
        
        # Number of lots (NIFTY lot size = 25)
        lot_size = 25
        lots_kelly = int(position_size_kelly / (self.current_nifty * lot_size))
        lots_max_risk = int(risk_amount / (self.current_nifty * lot_size * 0.1))  # Assume 10% margin
        
        # Take minimum for safety
        recommended_lots = min(max(1, lots_kelly), lots_max_risk)
        
        return {
            'kelly_fraction': kelly_fraction * 100,
            'tail_adjusted_fraction': adjusted_kelly * 100,
            'position_size': position_size_kelly,
            'recommended_lots': recommended_lots,
            'max_loss_per_lot': risk_amount / recommended_lots if recommended_lots > 0 else 0,
            'tail_risk_factor': tail_risk_factor * 100
        }
    
    def generate_trading_plan(self, confidence_threshold=60, base_capital=100000):
        """
        Generate complete trading plan with strategy, strikes, and sizing
        
        Args:
            confidence_threshold: Confidence threshold for directional trades
            base_capital: Trading capital
        
        Returns:
            Complete trading plan dict
        """
        print("\n" + "="*60)
        print("ML-Based Trading Plan")
        print("="*60)
        print(f"Date: {self.current_date.date()}")
        print(f"NIFTY: {self.current_nifty:.2f}")
        print(f"VIX: {self.current_vix:.2f}")
        print("="*60)
        
        # Get predictions
        predictions = self.get_predictions()
        
        print(f"\n📊 Market Forecast (22-day horizon):")
        print(f"  Direction: {predictions['direction']}")
        print(f"  Confidence: {predictions['confidence']:.1f}%")
        print(f"  Expected Move: ±{predictions['expected_move']:.2f}%")
        print(f"  Predicted NIFTY: {predictions['predicted_nifty']:.2f}")
        print(f"  Predicted Range: {predictions['predicted_range_low']:.2f} - {predictions['predicted_range_high']:.2f}")
        print(f"  Tail Risk: {predictions['tail_risk_prob']:.2f}%")
        
        # Select strategy
        strategy_rec = self.select_strategy(predictions, confidence_threshold)
        
        print(f"\n🎯 Strategy Recommendation:")
        print(f"  Strategy: {strategy_rec['strategy']}")
        print(f"  Reasoning: {strategy_rec['reasoning']}")
        
        # Calculate strikes
        strikes = self.calculate_strike_distances(predictions)
        
        print(f"\n📍 Strike Selection:")
        if strategy_rec['strategy'] == 'IRON_CONDOR':
            print(f"  Short Call: ATM + {strikes['ic_short_call']}")
            print(f"  Long Call:  ATM + {strikes['ic_long_call']}")
            print(f"  Short Put:  ATM - {strikes['ic_short_put']}")
            print(f"  Long Put:   ATM - {strikes['ic_long_put']}")
        elif strategy_rec['strategy'] == 'SHORT_STRANGLE':
            print(f"  Short Call: ATM + {strikes['strangle_call']}")
            print(f"  Short Put:  ATM - {strikes['strangle_put']}")
        elif strategy_rec['strategy'] in ['LONG_CALL', 'LONG_PUT']:
            print(f"  Strike Distance: {strikes['directional_distance']} from ATM")
            print(f"  Spread Width: {strikes['spread_width']} (if using spread)")
        
        # Position sizing
        position = self.calculate_position_size(predictions, base_capital)
        
        print(f"\n💰 Position Sizing:")
        print(f"  Kelly Fraction: {position['kelly_fraction']:.2f}%")
        print(f"  Tail-Adjusted: {position['tail_adjusted_fraction']:.2f}%")
        print(f"  Position Size: ₹{position['position_size']:,.0f}")
        print(f"  Recommended Lots: {position['recommended_lots']}")
        print(f"  Max Loss/Lot: ₹{position['max_loss_per_lot']:,.0f}")
        
        print("\n" + "="*60)
        
        # Combine into trading plan
        trading_plan = {
            'date': self.current_date,
            'market_data': {
                'nifty': self.current_nifty,
                'vix': self.current_vix
            },
            'predictions': predictions,
            'strategy': strategy_rec,
            'strikes': strikes,
            'position': position
        }
        
        return trading_plan
    
    def backtest_strategy_selection(self, start_date='2023-01-01', end_date='2024-12-31'):
        """
        Backtest ML-based strategy selection vs static rules
        
        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
        
        Returns:
            Backtest results DataFrame
        """
        print("\n" + "="*60)
        print("Backtesting ML Strategy Selection")
        print("="*60)
        
        # Load full dataset
        self.ml.load_and_merge_data()
        self.ml.create_features()
        self.ml.create_targets(horizon=22)
        
        df = self.ml.df_targets.copy()
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        
        # Get predictions for all dates
        X = df[self.ml.feature_names]
        predictions = self.ml.predict(X)
        
        # Add predictions to dataframe
        df['pred_direction'] = predictions['direction']
        df['pred_prob'] = predictions['direction_prob']
        df['pred_magnitude'] = predictions['magnitude']
        df['pred_tail_risk'] = predictions['tail_risk_prob']
        
        # Calculate actual outcomes
        df['actual_direction'] = df['target_direction']
        df['actual_magnitude'] = df['target_magnitude']
        df['actual_tail'] = df['target_tail_risk']
        
        # ML strategy selection
        df['ml_confidence'] = abs(df['pred_prob'] - 0.5) * 200
        df['ml_strategy'] = 'IRON_CONDOR'  # Default
        
        # High confidence directional
        df.loc[(df['ml_confidence'] >= 60) & (df['pred_tail_risk'] < 0.05), 'ml_strategy'] = 'DIRECTIONAL'
        
        # Moderate confidence
        df.loc[(df['ml_confidence'] >= 40) & (df['ml_confidence'] < 60), 'ml_strategy'] = 'SHORT_STRANGLE'
        
        # High tail risk
        df.loc[df['pred_tail_risk'] > 0.10, 'ml_strategy'] = 'AVOID'
        
        # Static rule strategy (baseline)
        df['static_strategy'] = 'SHORT_STRANGLE'  # Always sell premium
        df.loc[df['Close_vix'] < 12, 'static_strategy'] = 'LONG_STRADDLE'
        df.loc[df['Close_vix'] > 18, 'static_strategy'] = 'IRON_CONDOR'
        
        # Calculate outcomes
        # Direction accuracy
        df['ml_correct'] = (df['pred_direction'] == df['actual_direction']).astype(int)
        
        # Magnitude error
        df['ml_mag_error'] = abs(df['pred_magnitude'] - df['actual_magnitude'])
        
        # Tail risk accuracy
        df['ml_tail_correct'] = (df['pred_tail_risk'] > 0.10) == (df['actual_tail'] == 1)
        
        # Summary statistics
        print(f"\nBacktest Period: {df['Date'].min().date()} to {df['Date'].max().date()}")
        print(f"Total Signals: {len(df)}")
        print(f"\n📊 ML Model Accuracy:")
        print(f"  Direction: {df['ml_correct'].mean()*100:.2f}%")
        print(f"  Magnitude MAE: {df['ml_mag_error'].mean():.2f}%")
        print(f"  Tail Risk Accuracy: {df['ml_tail_correct'].mean()*100:.2f}%")
        
        print(f"\n🎯 Strategy Distribution (ML):")
        print(df['ml_strategy'].value_counts())
        
        print(f"\n📈 Strategy Distribution (Static VIX Rules):")
        print(df['static_strategy'].value_counts())
        
        return df


if __name__ == "__main__":
    # Example usage
    selector = MLStrategySelector()
    
    # Prepare current features
    selector.prepare_current_features()
    
    # Generate trading plan
    plan = selector.generate_trading_plan(confidence_threshold=60, base_capital=500000)
    
    # Backtest
    backtest_results = selector.backtest_strategy_selection(start_date='2023-01-01')
