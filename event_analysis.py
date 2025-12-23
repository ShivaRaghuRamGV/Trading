"""
Event-Based Analysis Module
Analyzes market behavior around key events: Expiry, RBI, Budget, etc.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import calendar


class EventAnalyzer:
    def __init__(self, merged_df):
        """Initialize with merged NIFTY-VIX dataframe"""
        self.df = merged_df.copy()
        self.df['Daily_Return'] = self.df['Close_nifty'].pct_change() * 100
        self.df['VIX_Change'] = self.df['Close_vix'].diff()
        
    def identify_expiry_weeks(self):
        """
        Identify monthly expiry weeks (last Thursday of each month)
        """
        df = self.df.copy()
        
        # Find last Thursday of each month
        def last_thursday(year, month):
            """Get last Thursday of the month"""
            # Get last day of month
            last_day = calendar.monthrange(year, month)[1]
            # Start from last day and go backwards
            for day in range(last_day, 0, -1):
                if calendar.weekday(year, month, day) == 3:  # 3 = Thursday
                    return datetime(year, month, day).date()
            return None
        
        # Generate all expiry dates
        years = range(df['Date'].dt.year.min(), df['Date'].dt.year.max() + 1)
        expiry_dates = []
        for year in years:
            for month in range(1, 13):
                expiry = last_thursday(year, month)
                if expiry:
                    expiry_dates.append(expiry)
        
        expiry_dates = pd.to_datetime(expiry_dates)
        
        # Mark expiry week (Thu-Wed before expiry)
        df['Is_Expiry_Week'] = False
        for expiry_date in expiry_dates:
            week_start = expiry_date - timedelta(days=6)
            week_end = expiry_date
            mask = (df['Date'] >= pd.Timestamp(week_start)) & (df['Date'] <= pd.Timestamp(week_end))
            df.loc[mask, 'Is_Expiry_Week'] = True
        
        # Mark expiry day
        df['Is_Expiry_Day'] = df['Date'].isin(expiry_dates)
        
        return df
    
    def analyze_expiry_behavior(self):
        """Analyze price and VIX behavior during expiry vs non-expiry weeks"""
        df = self.identify_expiry_weeks()
        
        # Stats by expiry week
        expiry_stats = df.groupby('Is_Expiry_Week').agg({
            'Daily_Return': ['mean', 'std', 'min', 'max'],
            'VIX_Change': ['mean', 'std'],
            'Close_vix': ['mean']
        }).round(3)
        
        # Expiry day stats
        expiry_day_stats = df[df['Is_Expiry_Day']].agg({
            'Daily_Return': ['mean', 'std', 'min', 'max'],
            'VIX_Change': ['mean', 'std'],
            'Close_vix': ['mean']
        }).round(3)
        
        # VIX decay after expiry
        vix_decay = self._calculate_vix_decay(df)
        
        return df, expiry_stats, expiry_day_stats, vix_decay
    
    def _calculate_vix_decay(self, df):
        """Calculate VIX decay pattern after expiry"""
        df = df[df['Is_Expiry_Day']].copy()
        
        decay_data = []
        for idx in df.index:
            if idx + 5 < len(self.df):  # Need 5 days after expiry
                vix_expiry = self.df.loc[idx, 'Close_vix']
                for days_after in range(1, 6):
                    if idx + days_after < len(self.df):
                        vix_after = self.df.loc[idx + days_after, 'Close_vix']
                        decay = ((vix_after - vix_expiry) / vix_expiry) * 100
                        decay_data.append({
                            'Days_After_Expiry': days_after,
                            'VIX_Decay_%': decay
                        })
        
        if decay_data:
            decay_df = pd.DataFrame(decay_data)
            avg_decay = decay_df.groupby('Days_After_Expiry')['VIX_Decay_%'].mean()
            return avg_decay
        return None
    
    def identify_event_days(self):
        """
        Identify major event days (simplified version)
        In production, you'd maintain a calendar of:
        - RBI policy dates
        - Budget dates
        - Major global events
        """
        df = self.df.copy()
        
        # Budget days (typically Feb 1st, but can vary)
        # Simplified: mark Feb 1 of each year as budget day
        df['Is_Budget_Day'] = (df['Date'].dt.month == 2) & (df['Date'].dt.day == 1)
        
        # RBI policy (typically bi-monthly: Feb, Apr, Jun, Aug, Oct, Dec)
        # Simplified: first week of these months
        rbi_months = [2, 4, 6, 8, 10, 12]
        df['Is_RBI_Week'] = (df['Date'].dt.month.isin(rbi_months)) & (df['Date'].dt.day <= 7)
        
        # Extreme volatility days (>2 std moves)
        std_threshold = df['Daily_Return'].std() * 2
        df['Is_Extreme_Day'] = df['Daily_Return'].abs() > std_threshold
        
        # VIX spikes: VIX jumps > 15% in a day or VIX > 75th percentile
        df['VIX_Change_Pct'] = df['Close_vix'].pct_change() * 100
        vix_threshold = df['Close_vix'].quantile(0.75)
        df['VIX_Spike'] = ((df['VIX_Change_Pct'] > 15) | (df['Close_vix'] > vix_threshold)).astype(int)
        
        return df
    
    def analyze_event_impact(self):
        """Analyze pre-event vs post-event behavior"""
        df = self.identify_event_days()
        
        results = {}
        
        # Budget day analysis
        if df['Is_Budget_Day'].sum() > 0:
            budget_days = df[df['Is_Budget_Day']].index
            
            pre_budget = []
            post_budget = []
            
            for idx in budget_days:
                # 5 days before
                if idx >= 5:
                    pre_vix = df.loc[idx-5:idx-1, 'Close_vix'].mean()
                    pre_budget.append(pre_vix)
                
                # 5 days after
                if idx + 5 < len(df):
                    post_vix = df.loc[idx+1:idx+5, 'Close_vix'].mean()
                    post_budget.append(post_vix)
            
            results['Budget'] = {
                'Pre_Event_Avg_VIX': np.mean(pre_budget) if pre_budget else None,
                'Post_Event_Avg_VIX': np.mean(post_budget) if post_budget else None,
                'Avg_Return_On_Day': df[df['Is_Budget_Day']]['Daily_Return'].mean()
            }
        
        # RBI week analysis
        if df['Is_RBI_Week'].sum() > 0:
            rbi_stats = df.groupby('Is_RBI_Week').agg({
                'Daily_Return': ['mean', 'std'],
                'Close_vix': ['mean']
            })
            results['RBI_Week'] = rbi_stats
        
        # Extreme days
        extreme_stats = df.groupby('Is_Extreme_Day').agg({
            'Daily_Return': ['mean', 'std', 'count'],
            'VIX_Change': ['mean', 'std']
        })
        results['Extreme_Days'] = extreme_stats
        
        return df, results
    
    def analyze_intraday_patterns(self):
        """
        Analyze opening gaps and intraday volatility
        Note: This works if we have Open/High/Low data
        """
        df = self.df.copy()
        
        if 'Open' in df.columns:
            # Opening gap
            df['Gap_%'] = ((df['Open'] - df['Close_nifty'].shift(1)) / df['Close_nifty'].shift(1)) * 100
            
            # Intraday range
            df['Intraday_Range_%'] = ((df['High'] - df['Low']) / df['Open']) * 100
            
            # Gap vs full day performance
            df['Gap_Direction'] = np.where(df['Gap_%'] > 0, 'Gap_Up', 
                                          np.where(df['Gap_%'] < 0, 'Gap_Down', 'Flat'))
            
            # Analyze gap behavior by VIX level
            df['VIX_Level'] = pd.cut(df['Close_vix'], 
                                     bins=[0, 12, 18, 25, 100],
                                     labels=['Low', 'Normal', 'High', 'Panic'])
            
            gap_analysis = df.groupby(['VIX_Level', 'Gap_Direction']).agg({
                'Gap_%': ['mean', 'count'],
                'Daily_Return': ['mean', 'std'],
                'Intraday_Range_%': 'mean'
            }).round(3)
            
            return df, gap_analysis
        
        return df, None
    
    def backtest_simple_strategy(self, strategy_type='short_strangle'):
        """
        Simple strategy backtesting
        
        Strategy types:
        - short_strangle: Sell when VIX is high
        - long_straddle: Buy when VIX is rising
        - iron_condor: Sell in low VIX sideways markets
        """
        df = self.df.copy()
        
        # Add VIX regime
        df['VIX_Regime'] = pd.cut(df['Close_vix'], 
                                  bins=[0, 12, 18, 25, 100],
                                  labels=['Low', 'Normal', 'High', 'Panic'])
        
        df['VIX_MA'] = df['Close_vix'].rolling(21).mean()
        df['VIX_Trend'] = np.where(df['Close_vix'] > df['VIX_MA'], 'Rising', 'Falling')
        
        # Calculate IV-RV spread
        df['RV_21'] = df['Daily_Return'].rolling(21).std() * np.sqrt(252)
        df['IV_RV_Spread'] = df['Close_vix'] - df['RV_21']
        
        # Strategy signals
        if strategy_type == 'short_strangle':
            # Short strangle: High VIX + Flat trend
            df['Signal'] = ((df['VIX_Regime'] == 'High') & 
                           (df['VIX_Trend'] == 'Falling')).astype(int)
            
            # Simplified P&L: assume profit from theta decay, loss from big moves
            df['Strategy_Return'] = np.where(
                df['Signal'] == 1,
                np.where(df['Daily_Return'].abs() < 1, 0.5, -2 * df['Daily_Return'].abs()),
                0
            )
        
        elif strategy_type == 'long_straddle':
            # Long straddle: Low VIX + Rising
            df['Signal'] = ((df['VIX_Regime'] == 'Low') & 
                           (df['VIX_Trend'] == 'Rising')).astype(int)
            
            # Profit from big moves
            df['Strategy_Return'] = np.where(
                df['Signal'] == 1,
                df['Daily_Return'].abs() - 0.5,  # Subtract theta decay
                0
            )
        
        elif strategy_type == 'iron_condor':
            # Iron condor: Low VIX + Sideways
            df['Price_MA'] = df['Close_nifty'].rolling(50).mean()
            df['Price_Trend'] = np.where(
                (df['Close_nifty'] - df['Price_MA']).abs() / df['Price_MA'] < 0.02,
                'Sideways', 'Trending'
            )
            
            df['Signal'] = ((df['VIX_Regime'] == 'Low') & 
                           (df['Price_Trend'] == 'Sideways')).astype(int)
            
            df['Strategy_Return'] = np.where(
                df['Signal'] == 1,
                np.where(df['Daily_Return'].abs() < 0.5, 0.3, -1.5 * df['Daily_Return'].abs()),
                0
            )
        
        # Calculate cumulative returns
        df['Cumulative_Strategy_Return'] = df['Strategy_Return'].cumsum()
        
        # Performance metrics
        total_trades = df['Signal'].sum()
        avg_return = df[df['Signal'] == 1]['Strategy_Return'].mean()
        win_rate = (df[df['Signal'] == 1]['Strategy_Return'] > 0).sum() / total_trades if total_trades > 0 else 0
        
        metrics = {
            'Strategy': strategy_type,
            'Total_Trades': total_trades,
            'Avg_Return_Per_Trade': avg_return,
            'Win_Rate_%': win_rate * 100,
            'Total_Return': df['Strategy_Return'].sum(),
            'Sharpe_Ratio': df['Strategy_Return'].mean() / df['Strategy_Return'].std() if df['Strategy_Return'].std() > 0 else 0
        }
        
        return df, metrics
    
    def backtest_with_real_options(self, strategy_type='short_strangle', **kwargs):
        """
        Backtest strategies using real historical options data
        
        Args:
            strategy_type: 'short_strangle' or 'long_straddle'
            **kwargs: Additional parameters passed to the backtester (including start_date, end_date)
        
        Returns:
            (trades_df, metrics_dict) or None if options data not available
        """
        try:
            from options_backtester import OptionsBacktester
            
            # Extract date range if provided
            start_date = kwargs.pop('start_date', None)
            end_date = kwargs.pop('end_date', None)
            
            # Initialize with NIFTY data and date filtering
            backtester = OptionsBacktester(self.df, start_date=start_date, end_date=end_date)
            
            if backtester.ce_data is None or backtester.pe_data is None:
                print("⚠️ Real options data not available. Using simplified strategy.")
                print("Run 'python options_data_fetcher.py' to download historical options data.")
                return None
            
            # Run backtest based on strategy type
            if strategy_type == 'short_strangle':
                trades_df = backtester.backtest_short_strangle(**kwargs)
            elif strategy_type == 'long_straddle':
                trades_df = backtester.backtest_long_straddle(**kwargs)
            else:
                print(f"❌ Unknown strategy type: {strategy_type}")
                return None
            
            if trades_df.empty:
                print(f"⚠️ No trades executed for {strategy_type}")
                return None
            
            # Calculate metrics
            metrics = backtester.calculate_metrics(trades_df)
            metrics['Strategy'] = strategy_type
            
            return trades_df, metrics
            
        except ImportError:
            print("⚠️ Options backtester module not found")
            return None
        except Exception as e:
            print(f"❌ Error in options backtesting: {e}")
            return None


def main():
    """Test event analysis"""
    # Load merged data (assuming it exists)
    try:
        nifty_df = pd.read_csv('nifty_history.csv', parse_dates=['Date'])
        vix_df = pd.read_csv('india_vix_history.csv', parse_dates=['Date'])
        
        merged_df = pd.merge(
            nifty_df[['Date', 'Open', 'High', 'Low', 'Close']],
            vix_df[['Date', 'Close']],
            on='Date',
            suffixes=('_nifty', '_vix')
        )
        merged_df.columns = ['Date', 'Open', 'High', 'Low', 'Close_nifty', 'Close_vix']
        
        analyzer = EventAnalyzer(merged_df)
        
        print("=" * 60)
        print("Event Analysis")
        print("=" * 60)
        
        # Expiry analysis
        df_expiry, expiry_stats, expiry_day_stats, vix_decay = analyzer.analyze_expiry_behavior()
        print(f"\n✓ Identified {df_expiry['Is_Expiry_Day'].sum()} expiry days")
        
        # Backtest simple strategy
        df_backtest, metrics = analyzer.backtest_simple_strategy('short_strangle')
        print(f"\n📊 Short Strangle Backtest:")
        print(f"  Total Trades: {metrics['Total_Trades']}")
        print(f"  Win Rate: {metrics['Win_Rate_%']:.2f}%")
        
    except FileNotFoundError:
        print("⚠ Data files not found. Run nse_data_fetcher.py first.")


if __name__ == "__main__":
    main()
