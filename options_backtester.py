"""
Options Strategy Backtester with Real Options Data
Uses actual historical options prices from nsepy
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import glob


class OptionsBacktester:
    def __init__(self, nifty_df, options_data_dir='nifty_option_excel', start_date=None, end_date=None):
        """
        Initialize backtester with NIFTY spot data and options data
        
        Args:
            nifty_df: DataFrame with NIFTY spot prices and VIX
            options_data_dir: Directory containing options Excel files
            start_date: Optional start date for filtering data
            end_date: Optional end date for filtering data
        """
        self.nifty_df = nifty_df.copy()
        self.options_data_dir = options_data_dir
        self.ce_data = None
        self.pe_data = None
        self.options_data = None  # Combined options data
        
        # Cache min/max dates to avoid recalculating
        self.ce_date_min = None
        self.ce_date_max = None
        self.pe_date_min = None
        self.pe_date_max = None
        
        # Try to load options data with date filtering
        self.load_options_data(start_date, end_date)
    
    def load_options_data(self, start_date=None, end_date=None):
        """
        Load Call and Put options data from Parquet files (much faster than Excel)
        
        Args:
            start_date: Optional start date to filter data
            end_date: Optional end date to filter data
        """
        if not os.path.exists(self.options_data_dir):
            print(f"⚠️ Options data directory '{self.options_data_dir}' not found.")
            print("Run options_data_fetcher.py to download historical options data.")
            return False
        
        # Try Parquet first (much faster), fall back to Excel
        parquet_files = glob.glob(f"{self.options_data_dir}/NIFTY_options_*.parquet")
        excel_files = glob.glob(f"{self.options_data_dir}/NIFTY_options_*.xlsx")
        
        # Filter files by year if date range specified
        if start_date and end_date:
            start_year = pd.Timestamp(start_date).year
            end_year = pd.Timestamp(end_date).year
            
            if parquet_files:
                parquet_files = [f for f in parquet_files 
                                if any(str(year) in f for year in range(start_year, end_year + 1))]
            if excel_files:
                excel_files = [f for f in excel_files 
                              if any(str(year) in f for year in range(start_year, end_year + 1))]
        
        if parquet_files:
            print(f"✓ Found {len(parquet_files)} Parquet files (fast loading)")
            files_to_load = parquet_files
            file_format = 'parquet'
        elif excel_files:
            print(f"✓ Found {len(excel_files)} Excel files (slow loading)")
            print("  Tip: Run convert_to_parquet.py for 10x faster loading")
            files_to_load = excel_files
            file_format = 'excel'
        else:
            print(f"⚠️ No data files found in '{self.options_data_dir}'")
            return False
        
        all_data = []
        for file in sorted(files_to_load):
            try:
                print(f"  Loading {os.path.basename(file)}...", end=' ')
                
                if file_format == 'parquet':
                    df = pd.read_parquet(file)
                else:
                    df = pd.read_excel(file, engine='openpyxl')
                
                all_data.append(df)
                print(f"✓ {len(df):,} rows")
            except Exception as e:
                print(f"⚠️ Error: {e}")
        
        if not all_data:
            print("❌ No data loaded from files")
            return False
        
        # Combine all years
        self.options_data = pd.concat(all_data, ignore_index=True)
        
        # Normalize column names (handle different formats)
        col_map = {}
        for col in self.options_data.columns:
            col_upper = col.upper()
            if 'DATE' in col_upper or 'TIMESTAMP' in col_upper:
                col_map[col] = 'Date'
            elif 'STRIKE' in col_upper:
                col_map[col] = 'Strike'
            elif 'EXPIRY' in col_upper:
                col_map[col] = 'Expiry'
            elif col_upper in ['OPTION_TYP', 'OPTION_TYPE']:
                col_map[col] = 'OptionType'
            elif col_upper == 'OPEN':
                col_map[col] = 'Open'
            elif col_upper == 'HIGH':
                col_map[col] = 'High'
            elif col_upper == 'LOW':
                col_map[col] = 'Low'
            elif col_upper == 'CLOSE':
                col_map[col] = 'Close'
            elif 'CONTRACTS' in col_upper or col_upper == 'VOLUME':
                col_map[col] = 'Volume'
            elif 'OPEN_INT' in col_upper or 'OPENINT' in col_upper:
                col_map[col] = 'Open Interest'
        
        self.options_data.rename(columns=col_map, inplace=True)
        
        # Convert dates
        self.options_data['Date'] = pd.to_datetime(self.options_data['Date'])
        self.options_data['Expiry'] = pd.to_datetime(self.options_data['Expiry'])
        
        # Filter by date range if specified
        if start_date and end_date:
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
            self.options_data = self.options_data[
                (self.options_data['Date'] >= start_ts) & 
                (self.options_data['Date'] <= end_ts)
            ]
            print(f"  Filtered to date range: {start_date} to {end_date}")
        
        # Split into CE and PE
        self.ce_data = self.options_data[self.options_data['OptionType'] == 'CE'].copy()
        self.pe_data = self.options_data[self.options_data['OptionType'] == 'PE'].copy()
        
        # Cache min/max dates for faster lookups
        self.ce_date_min = self.ce_data['Date'].min()
        self.ce_date_max = self.ce_data['Date'].max()
        self.pe_date_min = self.pe_data['Date'].min()
        self.pe_date_max = self.pe_data['Date'].max()
        
        print(f"✓ Loaded {len(self.options_data):,} total option records")
        print(f"  - Call options (CE): {len(self.ce_data):,}")
        print(f"  - Put options (PE): {len(self.pe_data):,}")
        print(f"  - Date range: {self.options_data['Date'].min()} to {self.options_data['Date'].max()}")
        
        return True
    
    def get_atm_strike(self, spot_price, strike_gap=50):
        """Get ATM strike price"""
        return round(spot_price / strike_gap) * strike_gap
    
    def get_option_price(self, date, strike, expiry, option_type='CE'):
        """
        Get option price for specific date, strike, and expiry
        
        Returns:
            Dict with price data or None if not found
        """
        data = self.ce_data if option_type == 'CE' else self.pe_data
        
        if data is None or data.empty:
            return None
        
        # Convert date to timestamp for comparison
        date_ts = pd.Timestamp(date)
        expiry_ts = pd.Timestamp(expiry)
        
        # Check if date is within available data range using cached values
        date_min = self.ce_date_min if option_type == 'CE' else self.pe_date_min
        date_max = self.ce_date_max if option_type == 'CE' else self.pe_date_max
        
        if date_ts < date_min or date_ts > date_max:
            return None
        
        # Filter for specific option
        try:
            mask = (
                (data['Date'] == date_ts) &
                (data['Strike'] == strike) &
                (data['Expiry'] == expiry_ts)
            )
            
            filtered = data[mask]
            
            if not filtered.empty:
                row = filtered.iloc[0]
                return {
                    'Close': row['Close'] if pd.notna(row['Close']) else 0,
                    'Open': row['Open'] if pd.notna(row['Open']) else 0,
                    'High': row['High'] if pd.notna(row['High']) else 0,
                    'Low': row['Low'] if pd.notna(row['Low']) else 0,
                    'Volume': row['Volume'] if 'Volume' in row and pd.notna(row['Volume']) else 0,
                    'Open Interest': row['Open Interest'] if 'Open Interest' in row and pd.notna(row['Open Interest']) else 0
                }
        except Exception as e:
            # Handle any filtering errors silently
            pass
        
        return None
    
    def backtest_short_strangle(self, vix_threshold=18, dte_entry=7, dte_exit=1, strike_distance=200):
        """
        Backtest Short Strangle strategy with real options data
        
        Strategy:
        - Enter when VIX > threshold and DTE = dte_entry
        - Sell OTM Call and OTM Put (e.g., ATM ± strike_distance)
        - Exit when DTE = dte_exit or stop loss hit
        
        Args:
            vix_threshold: Minimum VIX to enter trade
            dte_entry: Days to expiry when entering trade
            dte_exit: Days to expiry when exiting trade
            strike_distance: Distance from ATM for strikes (default 200)
        
        Returns:
            DataFrame with trade results
        """
        if self.ce_data is None or self.pe_data is None:
            print("❌ Options data not loaded. Cannot backtest.")
            return pd.DataFrame()
        
        # Get date range where we have options data
        options_start = max(self.ce_data['Date'].min(), self.pe_data['Date'].min())
        options_end = min(self.ce_data['Date'].max(), self.pe_data['Date'].max())
        
        print(f"  Options data available: {options_start.date()} to {options_end.date()}")
        
        trades = []
        
        # Get unique expiries more efficiently - don't filter first
        all_expiries = self.ce_data['Expiry'].unique()
        # Then filter the smaller list
        expiries = sorted([exp for exp in all_expiries 
                          if exp >= options_start and exp <= options_end])
        
        print(f"  Found {len(expiries)} expiries to test")
        
        for expiry in expiries:
            # Calculate entry and exit dates
            entry_date = expiry - timedelta(days=dte_entry)
            exit_date = expiry - timedelta(days=dte_exit)
            
            # Skip if dates are out of options data range
            if entry_date < options_start or exit_date > options_end:
                continue
            
            # Get NIFTY spot and VIX on entry date
            entry_data = self.nifty_df[
                (self.nifty_df['Date'] >= entry_date) & 
                (self.nifty_df['Date'] <= options_end)
            ].head(1)
            
            if entry_data.empty:
                continue
                
            entry_date = entry_data['Date'].iloc[0]
            spot_price = entry_data['Close_nifty'].iloc[0]
            vix = entry_data['Close_vix'].iloc[0]
            
            # Check VIX condition
            if vix < vix_threshold:
                continue
            
            # Determine strikes using user-provided distance
            atm_strike = self.get_atm_strike(spot_price)
            call_strike = atm_strike + strike_distance  # OTM Call
            put_strike = atm_strike - strike_distance   # OTM Put
            
            # Get entry prices
            call_entry = self.get_option_price(entry_date, call_strike, expiry, 'CE')
            put_entry = self.get_option_price(entry_date, put_strike, expiry, 'PE')
            
            if call_entry is None or put_entry is None:
                continue
            
            # Get exit prices
            exit_data = self.nifty_df[
                (self.nifty_df['Date'] >= exit_date) &
                (self.nifty_df['Date'] <= options_end)
            ].head(1)
            
            if exit_data.empty:
                continue
            
            actual_exit_date = exit_data['Date'].iloc[0]
            
            call_exit = self.get_option_price(actual_exit_date, call_strike, expiry, 'CE')
            put_exit = self.get_option_price(actual_exit_date, put_strike, expiry, 'PE')
            
            if call_exit is None or put_exit is None:
                continue
            
            # Calculate P&L (Short strangle: Sell both, so profit when premium decreases)
            call_pnl = (call_entry['Close'] - call_exit['Close']) * 75  # NIFTY lot size = 75
            put_pnl = (put_entry['Close'] - put_exit['Close']) * 75
            total_pnl = call_pnl + put_pnl
            
            premium_collected = (call_entry['Close'] + put_entry['Close']) * 75
            return_pct = (total_pnl / premium_collected) * 100 if premium_collected > 0 else 0
            
            trades.append({
                'Expiry': expiry,
                'Entry_Date': entry_date,
                'Exit_Date': actual_exit_date,
                'Spot_Entry': spot_price,
                'VIX_Entry': vix,
                'Call_Strike': call_strike,
                'Put_Strike': put_strike,
                'Call_Entry_Premium': call_entry['Close'],
                'Put_Entry_Premium': put_entry['Close'],
                'Call_Exit_Premium': call_exit['Close'],
                'Put_Exit_Premium': put_exit['Close'],
                'Premium_Collected': premium_collected,
                'Call_PnL': call_pnl,
                'Put_PnL': put_pnl,
                'Total_PnL': total_pnl,
                'Return_%': return_pct
            })
        
        if trades:
            df_trades = pd.DataFrame(trades)
            return df_trades
        
        return pd.DataFrame()
    
    def backtest_long_straddle(self, vix_threshold=12, dte_entry=7, dte_exit=1):
        """
        Backtest Long Straddle strategy with real options data
        
        Strategy:
        - Enter when VIX < threshold (expecting breakout)
        - Buy ATM Call and ATM Put
        - Exit when DTE = dte_exit
        """
        if self.ce_data is None or self.pe_data is None:
            print("❌ Options data not loaded. Cannot backtest.")
            return pd.DataFrame()
        
        # Get date range where we have options data
        options_start = max(self.ce_data['Date'].min(), self.pe_data['Date'].min())
        options_end = min(self.ce_data['Date'].max(), self.pe_data['Date'].max())
        
        trades = []
        
        # Get expiries within available data range (optimized)
        all_expiries = self.ce_data['Expiry'].unique()
        expiries = sorted([exp for exp in all_expiries 
                          if exp >= options_start and exp <= options_end])
        
        for expiry in expiries:
            entry_date = expiry - timedelta(days=dte_entry)
            exit_date = expiry - timedelta(days=dte_exit)
            
            # Skip if dates are out of options data range
            if entry_date < options_start or exit_date > options_end:
                continue
            
            entry_data = self.nifty_df[
                (self.nifty_df['Date'] >= entry_date) & 
                (self.nifty_df['Date'] <= options_end)
            ].head(1)
            
            if entry_data.empty:
                continue
            
            entry_date = entry_data['Date'].iloc[0]
            spot_price = entry_data['Close_nifty'].iloc[0]
            vix = entry_data['Close_vix'].iloc[0]
            
            if vix > vix_threshold:
                continue
            
            atm_strike = self.get_atm_strike(spot_price)
            
            call_entry = self.get_option_price(entry_date, atm_strike, expiry, 'CE')
            put_entry = self.get_option_price(entry_date, atm_strike, expiry, 'PE')
            
            if call_entry is None or put_entry is None:
                continue
            
            exit_data = self.nifty_df[
                (self.nifty_df['Date'] >= exit_date) &
                (self.nifty_df['Date'] <= options_end)
            ].head(1)
            
            if exit_data.empty:
                continue
            
            actual_exit_date = exit_data['Date'].iloc[0]
            
            call_exit = self.get_option_price(actual_exit_date, atm_strike, expiry, 'CE')
            put_exit = self.get_option_price(actual_exit_date, atm_strike, expiry, 'PE')
            
            if call_exit is None or put_exit is None:
                continue
            
            # Long straddle: Buy both, profit when premium increases
            call_pnl = (call_exit['Close'] - call_entry['Close']) * 75
            put_pnl = (put_exit['Close'] - put_entry['Close']) * 75
            total_pnl = call_pnl + put_pnl
            
            cost = (call_entry['Close'] + put_entry['Close']) * 75
            return_pct = (total_pnl / cost) * 100 if cost > 0 else 0
            
            trades.append({
                'Expiry': expiry,
                'Entry_Date': entry_date,
                'Exit_Date': actual_exit_date,
                'Spot_Entry': spot_price,
                'VIX_Entry': vix,
                'ATM_Strike': atm_strike,
                'Call_Entry_Premium': call_entry['Close'],
                'Put_Entry_Premium': put_entry['Close'],
                'Call_Exit_Premium': call_exit['Close'],
                'Put_Exit_Premium': put_exit['Close'],
                'Cost': cost,
                'Call_PnL': call_pnl,
                'Put_PnL': put_pnl,
                'Total_PnL': total_pnl,
                'Return_%': return_pct
            })
        
        if trades:
            return pd.DataFrame(trades)
        
        return pd.DataFrame()
    
    def calculate_metrics(self, trades_df):
        """Calculate performance metrics from trades DataFrame"""
        if trades_df.empty:
            return {}
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['Total_PnL'] > 0])
        losing_trades = len(trades_df[trades_df['Total_PnL'] < 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = trades_df['Total_PnL'].sum()
        avg_pnl = trades_df['Total_PnL'].mean()
        
        avg_win = trades_df[trades_df['Total_PnL'] > 0]['Total_PnL'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['Total_PnL'] < 0]['Total_PnL'].mean() if losing_trades > 0 else 0
        
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Sharpe-like ratio
        returns = trades_df['Return_%']
        sharpe = (returns.mean() / returns.std()) if returns.std() > 0 else 0
        
        # Max drawdown
        cumulative_pnl = trades_df['Total_PnL'].cumsum()
        running_max = cumulative_pnl.cummax()
        drawdown = cumulative_pnl - running_max
        max_drawdown = drawdown.min()
        
        return {
            'Total_Trades': total_trades,
            'Winning_Trades': winning_trades,
            'Losing_Trades': losing_trades,
            'Win_Rate_%': round(win_rate, 2),
            'Total_PnL': round(total_pnl, 2),
            'Avg_PnL': round(avg_pnl, 2),
            'Avg_Win': round(avg_win, 2),
            'Avg_Loss': round(avg_loss, 2),
            'Profit_Factor': round(profit_factor, 2),
            'Sharpe_Ratio': round(sharpe, 2),
            'Max_Drawdown': round(max_drawdown, 2),
            'Avg_Return_%': round(returns.mean(), 2),
            'Std_Return_%': round(returns.std(), 2)
        }


# Example usage
if __name__ == "__main__":
    # Load NIFTY data
    nifty_df = pd.read_csv('nifty_data.csv')
    nifty_df['Date'] = pd.to_datetime(nifty_df['Date'])
    
    # Initialize backtester
    backtester = OptionsBacktester(nifty_df)
    
    # Backtest strategies
    print("\n" + "="*60)
    print("Backtesting Short Strangle Strategy")
    print("="*60)
    
    strangle_trades = backtester.backtest_short_strangle(vix_threshold=18, dte_entry=7, dte_exit=1)
    
    if not strangle_trades.empty:
        print(f"\n✓ Completed {len(strangle_trades)} trades")
        print("\nSample trades:")
        print(strangle_trades.head())
        
        metrics = backtester.calculate_metrics(strangle_trades)
        print("\nPerformance Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")
    else:
        print("❌ No trades executed")
    
    print("\n" + "="*60)
    print("Backtesting Long Straddle Strategy")
    print("="*60)
    
    straddle_trades = backtester.backtest_long_straddle(vix_threshold=12, dte_entry=7, dte_exit=1)
    
    if not straddle_trades.empty:
        print(f"\n✓ Completed {len(straddle_trades)} trades")
        print("\nSample trades:")
        print(straddle_trades.head())
        
        metrics = backtester.calculate_metrics(straddle_trades)
        print("\nPerformance Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")
    else:
        print("❌ No trades executed")
