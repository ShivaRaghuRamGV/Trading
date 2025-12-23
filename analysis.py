"""
Advanced Trading Analysis Module
Comprehensive analysis for NIFTY 50 and INDIA VIX
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from scipy.signal import correlate


class TradingAnalyzer:
    def __init__(self, nifty_df, vix_df):
        """Initialize with NIFTY and VIX dataframes"""
        self.nifty_df = nifty_df.copy()
        self.vix_df = vix_df.copy()
        self.merged_df = None
        self._prepare_data()
    
    def _prepare_data(self):
        """Merge and prepare data"""
        # Merge NIFTY and VIX
        self.merged_df = pd.merge(
            self.nifty_df[['Date', 'Open', 'High', 'Low', 'Close']],
            self.vix_df[['Date', 'Close']],
            on='Date',
            suffixes=('_nifty', '_vix')
        )
        self.merged_df = self.merged_df.sort_values('Date').reset_index(drop=True)
        self.merged_df.columns = ['Date', 'Open', 'High', 'Low', 'Close_nifty', 'Close_vix']
        
        # Remove timezone info if present
        if hasattr(self.merged_df['Date'].dtype, 'tz') and self.merged_df['Date'].dtype.tz is not None:
            self.merged_df['Date'] = self.merged_df['Date'].dt.tz_localize(None)
    
    # ========================================
    # 1. NIFTY 50 RETURNS ANALYSIS
    # ========================================
    
    def calculate_returns(self):
        """Calculate daily, weekly, monthly returns (both simple and log)"""
        df = self.merged_df.copy()
        
        # Daily returns
        df['Daily_Return_Simple'] = df['Close_nifty'].pct_change() * 100
        df['Daily_Return_Log'] = np.log(df['Close_nifty'] / df['Close_nifty'].shift(1)) * 100
        
        # Weekly returns (resample to week-end)
        weekly = df.set_index('Date').resample('W').last()
        weekly['Weekly_Return_Simple'] = weekly['Close_nifty'].pct_change() * 100
        weekly['Weekly_Return_Log'] = np.log(weekly['Close_nifty'] / weekly['Close_nifty'].shift(1)) * 100
        
        # Monthly returns
        monthly = df.set_index('Date').resample('ME').last()
        monthly['Monthly_Return_Simple'] = monthly['Close_nifty'].pct_change() * 100
        monthly['Monthly_Return_Log'] = np.log(monthly['Close_nifty'] / monthly['Close_nifty'].shift(1)) * 100
        
        # Rolling means
        df['Return_MA_5'] = df['Daily_Return_Simple'].rolling(5).mean()
        df['Return_MA_21'] = df['Daily_Return_Simple'].rolling(21).mean()
        df['Return_MA_50'] = df['Daily_Return_Simple'].rolling(50).mean()
        
        return df, weekly, monthly
    
    def calculate_drawdowns(self):
        """Calculate drawdowns and max drawdown"""
        df = self.merged_df.copy()
        
        # Calculate cumulative max (running peak)
        df['Cumulative_Max'] = df['Close_nifty'].cummax()
        
        # Drawdown in percentage
        df['Drawdown'] = ((df['Close_nifty'] - df['Cumulative_Max']) / df['Cumulative_Max']) * 100
        
        # Max drawdown
        max_dd = df['Drawdown'].min()
        max_dd_date = df.loc[df['Drawdown'].idxmin(), 'Date']
        
        # Drawdown duration
        df['In_Drawdown'] = df['Drawdown'] < 0
        
        return df, max_dd, max_dd_date
    
    def label_trend(self, window=50):
        """Label market trend as bullish, bearish, or sideways"""
        df = self.merged_df.copy()
        
        # Calculate moving averages
        df['SMA_50'] = df['Close_nifty'].rolling(50).mean()
        df['SMA_200'] = df['Close_nifty'].rolling(200).mean()
        
        # Price momentum
        df['Price_Change_Pct'] = df['Close_nifty'].pct_change(window) * 100
        
        # Trend conditions
        conditions = [
            (df['Close_nifty'] > df['SMA_50']) & (df['SMA_50'] > df['SMA_200']) & (df['Price_Change_Pct'] > 5),
            (df['Close_nifty'] < df['SMA_50']) & (df['SMA_50'] < df['SMA_200']) & (df['Price_Change_Pct'] < -5),
        ]
        choices = ['Bullish', 'Bearish']
        df['Trend'] = np.select(conditions, choices, default='Sideways')
        
        return df
    
    # ========================================
    # 2. INDIA VIX ANALYSIS
    # ========================================
    
    def analyze_vix(self):
        """Comprehensive VIX analysis"""
        df = self.merged_df.copy()
        
        # Daily change
        df['VIX_Change'] = df['Close_vix'].diff()
        df['VIX_Pct_Change'] = df['Close_vix'].pct_change() * 100
        
        # Rolling statistics
        df['VIX_MA_5'] = df['Close_vix'].rolling(5).mean()
        df['VIX_MA_21'] = df['Close_vix'].rolling(21).mean()
        df['VIX_STD_21'] = df['Close_vix'].rolling(21).std()
        
        # VIX Regime classification
        conditions = [
            df['Close_vix'] < 12,
            (df['Close_vix'] >= 12) & (df['Close_vix'] < 18),
            (df['Close_vix'] >= 18) & (df['Close_vix'] < 25),
            df['Close_vix'] >= 25
        ]
        choices = ['Low', 'Normal', 'High', 'Panic']
        df['VIX_Regime'] = np.select(conditions, choices, default='Normal')
        
        # VIX Trend
        df['VIX_Trend'] = np.where(
            df['Close_vix'] > df['VIX_MA_21'], 'Rising',
            np.where(df['Close_vix'] < df['VIX_MA_21'], 'Falling', 'Flat')
        )
        
        # Alternative regime: Quantile-based
        try:
            df['VIX_Quantile_Regime'] = pd.qcut(
                df['Close_vix'], 
                q=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                duplicates='drop'
            )
        except ValueError:
            # If quantiles have duplicates, use simpler cut
            df['VIX_Quantile_Regime'] = pd.cut(
                df['Close_vix'],
                bins=[0, 12, 15, 18, 22, 100],
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
            )
        
        return df
    
    def get_vix_regime_stats(self, df):
        """Get statistics by VIX regime"""
        regime_stats = df.groupby('VIX_Regime').agg({
            'Daily_Return_Simple': ['mean', 'std', 'min', 'max'],
            'Close_vix': ['mean', 'count']
        }).round(3)
        
        return regime_stats
    
    # ========================================
    # 3. NIFTY-VIX CORRELATION & LEAD-LAG
    # ========================================
    
    def correlation_analysis(self):
        """Analyze correlation between NIFTY returns and VIX"""
        df = self.merged_df.copy()
        df['Daily_Return_Simple'] = df['Close_nifty'].pct_change() * 100
        df['VIX_Change'] = df['Close_vix'].diff()
        
        # Overall correlation
        overall_corr = df[['Daily_Return_Simple', 'VIX_Change']].corr().iloc[0, 1]
        
        # Rolling 30-day correlation
        df['Rolling_Corr_30'] = df['Daily_Return_Simple'].rolling(30).corr(df['VIX_Change'])
        
        # Rolling 60-day correlation
        df['Rolling_Corr_60'] = df['Daily_Return_Simple'].rolling(60).corr(df['VIX_Change'])
        
        return df, overall_corr
    
    def lead_lag_analysis(self, max_lag=10):
        """Cross-correlation to find lead-lag relationship"""
        df = self.merged_df.copy()
        df['Daily_Return_Simple'] = df['Close_nifty'].pct_change() * 100
        df['VIX_Change'] = df['Close_vix'].diff()
        
        # Remove NaN
        returns = df['Daily_Return_Simple'].dropna().values
        vix_change = df['VIX_Change'].dropna().values
        
        # Align lengths
        min_len = min(len(returns), len(vix_change))
        returns = returns[-min_len:]
        vix_change = vix_change[-min_len:]
        
        # Calculate cross-correlation
        cross_corr = np.correlate(returns - returns.mean(), vix_change - vix_change.mean(), mode='full')
        cross_corr = cross_corr / (len(returns) * returns.std() * vix_change.std())
        
        # Extract relevant lags
        middle = len(cross_corr) // 2
        lags = np.arange(-max_lag, max_lag + 1)
        corr_values = cross_corr[middle - max_lag:middle + max_lag + 1]
        
        lead_lag_df = pd.DataFrame({'Lag': lags, 'Correlation': corr_values})
        
        return lead_lag_df
    
    def granger_causality(self, max_lag=5):
        """Test if VIX predicts NIFTY direction (Granger causality)"""
        df = self.merged_df.copy()
        df['Daily_Return_Simple'] = df['Close_nifty'].pct_change() * 100
        df['VIX_Change'] = df['Close_vix'].diff()
        
        # Prepare data
        test_data = df[['Daily_Return_Simple', 'VIX_Change']].dropna()
        
        if len(test_data) < 100:
            return None
        
        try:
            # Test if VIX_Change Granger-causes Daily_Return_Simple
            gc_result = grangercausalitytests(test_data[['Daily_Return_Simple', 'VIX_Change']], max_lag, verbose=False)
            
            # Extract p-values
            p_values = {}
            for lag in range(1, max_lag + 1):
                p_values[lag] = gc_result[lag][0]['ssr_ftest'][1]  # F-test p-value
            
            return p_values
        except Exception as e:
            print(f"Granger causality test failed: {e}")
            return None
    
    # ========================================
    # 4. IMPLIED VS REALIZED VOLATILITY
    # ========================================
    
    def calculate_realized_volatility(self):
        """Calculate realized volatility for different windows"""
        df = self.merged_df.copy()
        df['Daily_Return_Log'] = np.log(df['Close_nifty'] / df['Close_nifty'].shift(1))
        
        # Annualization factor
        ann_factor = np.sqrt(252)
        
        # Realized volatility for different windows
        df['RV_5D'] = df['Daily_Return_Log'].rolling(5).std() * ann_factor * 100
        df['RV_10D'] = df['Daily_Return_Log'].rolling(10).std() * ann_factor * 100
        df['RV_21D'] = df['Daily_Return_Log'].rolling(21).std() * ann_factor * 100
        df['RV_30D'] = df['Daily_Return_Log'].rolling(30).std() * ann_factor * 100
        
        return df
    
    def calculate_iv_rv_spread(self):
        """Calculate IV-RV spread and z-scores"""
        df = self.calculate_realized_volatility()
        
        # IV-RV Spread (using 21-day RV as benchmark)
        df['IV_RV_Spread'] = df['Close_vix'] - df['RV_21D']
        
        # Z-score of spread
        df['IV_RV_ZScore'] = (
            df['IV_RV_Spread'] - df['IV_RV_Spread'].rolling(60).mean()
        ) / df['IV_RV_Spread'].rolling(60).std()
        
        # Trading signals
        df['IV_RV_Signal'] = np.where(
            df['IV_RV_ZScore'] > 1.5, 'Sell Options',
            np.where(df['IV_RV_ZScore'] < -1.5, 'Buy Options', 'Neutral')
        )
        
        return df
    
    # ========================================
    # 5. TAIL RISK & EXTREME MOVES
    # ========================================
    
    def tail_risk_analysis(self):
        """Analyze tail risk: kurtosis, skewness, extreme moves"""
        df = self.merged_df.copy()
        df['Daily_Return_Simple'] = df['Close_nifty'].pct_change() * 100
        df['VIX_Change'] = df['Close_vix'].diff()
        
        # Statistics
        nifty_stats = {
            'Mean': df['Daily_Return_Simple'].mean(),
            'Std': df['Daily_Return_Simple'].std(),
            'Skewness': stats.skew(df['Daily_Return_Simple'].dropna()),
            'Kurtosis': stats.kurtosis(df['Daily_Return_Simple'].dropna()),
            'Min': df['Daily_Return_Simple'].min(),
            'Max': df['Daily_Return_Simple'].max(),
            'P95': df['Daily_Return_Simple'].quantile(0.95),
            'P99': df['Daily_Return_Simple'].quantile(0.99),
            'P1': df['Daily_Return_Simple'].quantile(0.01),
            'P5': df['Daily_Return_Simple'].quantile(0.05)
        }
        
        vix_stats = {
            'Mean': df['VIX_Change'].mean(),
            'Std': df['VIX_Change'].std(),
            'Skewness': stats.skew(df['VIX_Change'].dropna()),
            'Kurtosis': stats.kurtosis(df['VIX_Change'].dropna()),
            'Min': df['VIX_Change'].min(),
            'Max': df['VIX_Change'].max()
        }
        
        # Extreme move frequency
        extreme_moves = {
            'Nifty_>2%': (df['Daily_Return_Simple'].abs() > 2).sum(),
            'Nifty_>3%': (df['Daily_Return_Simple'].abs() > 3).sum(),
            'VIX_>10%': (df['VIX_Change'].abs() > 10).sum(),
            'Total_Days': len(df)
        }
        
        return nifty_stats, vix_stats, extreme_moves
    
    # ========================================
    # 6. TRADING INSIGHTS GENERATION
    # ========================================
    
    def generate_trading_insights(self):
        """Generate actionable trading insights"""
        # Get all analyses combined
        df = self.merged_df.copy()
        
        # Add trend analysis
        df_trend = self.label_trend()
        df['Trend'] = df_trend['Trend']
        
        # Add VIX analysis
        df_vix = self.analyze_vix()
        df['VIX_Regime'] = df_vix['VIX_Regime']
        df['VIX_Trend'] = df_vix['VIX_Trend']
        
        # Get latest values
        latest = df.iloc[-1]
        
        insights = []
        
        # Market trend + VIX regime
        if 'Trend' in df.columns and latest['Trend'] == 'Bullish' and latest['VIX_Regime'] == 'Low':
            insights.append("📈 BULLISH + LOW VIX → Buy calls / Put spreads")
        elif 'Trend' in df.columns and latest['Trend'] == 'Sideways' and latest['VIX_Regime'] == 'Low':
            insights.append("↔️ SIDEWAYS + LOW VIX → Iron condors")
        elif latest['VIX_Regime'] == 'High':
            insights.append("⚠️ HIGH VIX → Option selling with hedges")
        elif latest['VIX_Trend'] == 'Rising':
            insights.append("📊 RISING VIX → Long straddles / strangles")
        
        # VIX regime specific
        if latest['VIX_Regime'] == 'Low' and latest['VIX_Trend'] == 'Falling':
            insights.append("✅ Low & Falling VIX → Buy debit spreads")
        elif latest['VIX_Regime'] == 'Low' and latest['VIX_Trend'] == 'Rising':
            insights.append("⚡ Low & Rising VIX → Long straddle entry")
        elif latest['VIX_Regime'] == 'High' and latest['VIX_Trend'] == 'Flat':
            insights.append("💰 High & Flat VIX → Short strangle opportunity")
        elif latest['VIX_Regime'] == 'Panic':
            insights.append("🛡️ EXTREME VIX → Hedge only, avoid new positions")
        
        return insights, latest
    
    def get_strategy_suggestions(self, df):
        """Get strategy suggestions based on current market state"""
        df = df.copy()
        
        # Calculate IV-RV spread if not already done
        if 'IV_RV_Spread' not in df.columns:
            df = self.calculate_iv_rv_spread()
        
        latest = df.iloc[-1]
        
        strategies = []
        
        # Based on IV-RV spread
        if 'IV_RV_ZScore' in df.columns and not pd.isna(latest['IV_RV_ZScore']):
            if latest['IV_RV_ZScore'] > 1.5:
                strategies.append({
                    'Strategy': 'Short Strangle/Iron Condor',
                    'Reason': f'IV significantly higher than RV (Z-score: {latest["IV_RV_ZScore"]:.2f})',
                    'Risk': 'High volatility expansion'
                })
            elif latest['IV_RV_ZScore'] < -1.5:
                strategies.append({
                    'Strategy': 'Long Straddle/Calendar Spread',
                    'Reason': f'IV significantly lower than RV (Z-score: {latest["IV_RV_ZScore"]:.2f})',
                    'Risk': 'Volatility contraction'
                })
        
        return pd.DataFrame(strategies)


def main():
    """Test the analysis module"""
    # Load data
    nifty_df = pd.read_csv('nifty_history.csv', parse_dates=['Date'])
    vix_df = pd.read_csv('india_vix_history.csv', parse_dates=['Date'])
    
    # Initialize analyzer
    analyzer = TradingAnalyzer(nifty_df, vix_df)
    
    print("=" * 60)
    print("Trading Analysis Complete")
    print("=" * 60)
    
    # Run some basic tests
    df_returns, weekly, monthly = analyzer.calculate_returns()
    print(f"\n✓ Calculated returns for {len(df_returns)} days")
    
    df_dd, max_dd, max_dd_date = analyzer.calculate_drawdowns()
    print(f"✓ Max Drawdown: {max_dd:.2f}% on {max_dd_date.date()}")
    
    insights, latest = analyzer.generate_trading_insights()
    print("\n📊 Current Trading Insights:")
    for insight in insights:
        print(f"  {insight}")


if __name__ == "__main__":
    main()
