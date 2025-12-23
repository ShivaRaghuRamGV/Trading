"""
NSE Data Fetcher for NIFTY 50 and INDIA VIX
Downloads historical data for the last 10 years using yfinance
Automatically updates to the previous trading day
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import os
import sys

# Set console encoding to UTF-8 to handle unicode characters
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        pass

class NSEDataFetcher:
    def __init__(self):
        self.nifty_symbol = '^NSEI'
        self.vix_symbol = '^INDIAVIX'
        self.nifty_file = 'nifty_history.csv'
        self.vix_file = 'india_vix_history.csv'
        
    def get_last_date_in_file(self, filename):
        """Get the last date from existing CSV file"""
        if not os.path.exists(filename):
            return None
        
        try:
            df = pd.read_csv(filename)
            if df.empty:
                return None
            df['Date'] = pd.to_datetime(df['Date'])
            return df['Date'].max()
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            return None
    
    def get_previous_trading_day(self):
        """Get the previous trading day (yesterday, or Friday if today is Monday)"""
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        
        # If yesterday was Saturday or Sunday, go back to Friday
        while yesterday.weekday() >= 5:  # 5=Saturday, 6=Sunday
            yesterday = yesterday - timedelta(days=1)
        
        return yesterday
        
    def fetch_nifty_data(self, start_date=None, end_date=None, update_mode=False):
        """
        Fetch NIFTY 50 historical data using yfinance
        
        Args:
            start_date: Start date for fetch (if None, will use last date in file or 10 years ago)
            end_date: End date for fetch (if None, will use previous trading day)
            update_mode: If True, will append to existing file instead of replacing
        """
        # Determine end_date (previous trading day)
        if end_date is None:
            end_date = self.get_previous_trading_day()
        
        # Determine start_date
        if start_date is None:
            if update_mode:
                last_date = self.get_last_date_in_file(self.nifty_file)
                if last_date:
                    # Start from day after last date
                    start_date = (last_date + timedelta(days=1)).date()
                else:
                    # No existing file, fetch last 10 years
                    start_date = (datetime.now() - timedelta(days=3650)).date()
            else:
                start_date = (datetime.now() - timedelta(days=3650)).date()
        
        # Convert to string for yfinance
        start_str = start_date.strftime('%Y-%m-%d') if isinstance(start_date, (datetime, type(end_date))) else start_date
        end_str = end_date.strftime('%Y-%m-%d') if isinstance(end_date, (datetime, type(end_date))) else end_date
        
        # Check if we're already up to date
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
            
        if start_date > end_date:
            print(f"✓ NIFTY data is already up to date (last date: {start_date - timedelta(days=1)})")
            return None
        
        try:
            print(f"Fetching NIFTY 50 data from {start_str} to {end_str}...")
            ticker = yf.Ticker(self.nifty_symbol)
            df = ticker.history(start=start_str, end=(datetime.strptime(end_str, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d'))
            
            if df.empty:
                print("No new NIFTY data found")
                return None
            
            # Reset index to get Date as a column
            df = df.reset_index()
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'Date': 'Date',
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close'
            })
            
            # Select only needed columns
            df = df[['Date', 'Open', 'High', 'Low', 'Close']].copy()
            
            # Ensure Date is datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Sort by date
            df = df.sort_values('Date').reset_index(drop=True)
            
            print(f"✓ Fetched {len(df)} NIFTY records")
            print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
            print(f"  Latest Close: {df['Close'].iloc[-1]:.2f}")
            
            return df
                
        except Exception as e:
            print(f"Error fetching NIFTY data: {e}")
            return None
    
    def fetch_vix_data(self, start_date=None, end_date=None, update_mode=False):
        """
        Fetch INDIA VIX historical data using yfinance
        
        Args:
            start_date: Start date for fetch (if None, will use last date in file or 10 years ago)
            end_date: End date for fetch (if None, will use previous trading day)
            update_mode: If True, will append to existing file instead of replacing
        """
        # Determine end_date (previous trading day)
        if end_date is None:
            end_date = self.get_previous_trading_day()
        
        # Determine start_date
        if start_date is None:
            if update_mode:
                last_date = self.get_last_date_in_file(self.vix_file)
                if last_date:
                    # Start from day after last date
                    start_date = (last_date + timedelta(days=1)).date()
                else:
                    # No existing file, fetch last 10 years
                    start_date = (datetime.now() - timedelta(days=3650)).date()
            else:
                start_date = (datetime.now() - timedelta(days=3650)).date()
        
        # Convert to string for yfinance
        start_str = start_date.strftime('%Y-%m-%d') if isinstance(start_date, (datetime, type(end_date))) else start_date
        end_str = end_date.strftime('%Y-%m-%d') if isinstance(end_date, (datetime, type(end_date))) else end_date
        
        # Check if we're already up to date
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
            
        if start_date > end_date:
            print(f"✓ VIX data is already up to date (last date: {start_date - timedelta(days=1)})")
            return None
        
        try:
            print(f"Fetching INDIA VIX data from {start_str} to {end_str}...")
            ticker = yf.Ticker(self.vix_symbol)
            df = ticker.history(start=start_str, end=(datetime.strptime(end_str, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d'))
            
            if df.empty:
                print("No new VIX data found")
                return None
            
            # Reset index to get Date as a column
            df = df.reset_index()
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'Date': 'Date',
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close'
            })
            
            # Select only needed columns
            df = df[['Date', 'Open', 'High', 'Low', 'Close']].copy()
            
            # Ensure Date is datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Sort by date
            df = df.sort_values('Date').reset_index(drop=True)
            
            print(f"✓ Fetched {len(df)} VIX records")
            print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
            print(f"  Latest Close: {df['Close'].iloc[-1]:.2f}")
            
            return df
                
        except Exception as e:
            print(f"Error fetching VIX data: {e}")
            return None
    
    def save_data(self, nifty_df, vix_df, update_mode=False):
        """
        Save data to CSV files
        
        Args:
            nifty_df: New NIFTY data to save
            vix_df: New VIX data to save
            update_mode: If True, append to existing files; if False, overwrite
        """
        if nifty_df is not None:
            if update_mode and os.path.exists(self.nifty_file):
                # Read existing data
                existing_df = pd.read_csv(self.nifty_file)
                existing_df['Date'] = pd.to_datetime(existing_df['Date'])
                
                # Ensure new data has datetime
                nifty_df['Date'] = pd.to_datetime(nifty_df['Date'])
                
                # Concatenate and remove duplicates
                combined_df = pd.concat([existing_df, nifty_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['Date'], keep='last')
                combined_df = combined_df.sort_values('Date').reset_index(drop=True)
                
                combined_df.to_csv(self.nifty_file, index=False)
                print(f"✓ Updated {self.nifty_file} (added {len(nifty_df)} new records, total: {len(combined_df)})")
            else:
                nifty_df.to_csv(self.nifty_file, index=False)
                print(f"✓ Saved {self.nifty_file} ({len(nifty_df)} records)")
        
        if vix_df is not None:
            if update_mode and os.path.exists(self.vix_file):
                # Read existing data
                existing_df = pd.read_csv(self.vix_file)
                existing_df['Date'] = pd.to_datetime(existing_df['Date'])
                
                # Ensure new data has datetime
                vix_df['Date'] = pd.to_datetime(vix_df['Date'])
                
                # Concatenate and remove duplicates
                combined_df = pd.concat([existing_df, vix_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['Date'], keep='last')
                combined_df = combined_df.sort_values('Date').reset_index(drop=True)
                
                combined_df.to_csv(self.vix_file, index=False)
                print(f"✓ Updated {self.vix_file} (added {len(vix_df)} new records, total: {len(combined_df)})")
            else:
                vix_df.to_csv(self.vix_file, index=False)
                print(f"✓ Saved {self.vix_file} ({len(vix_df)} records)")


def generate_sample_data():
    """
    Generate sample data if NSE fetch fails
    This creates realistic synthetic data for testing
    """
    print("\nGenerating sample data for testing...")
    
    # Generate dates for last 10 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3650)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.dayofweek < 5]  # Remove weekends
    
    # Generate NIFTY data - end at realistic current value ~26150
    np.random.seed(42)
    nifty_current = 23500  # Target realistic NIFTY value
    nifty_start = 7500     # Starting value 10 years ago
    
    # Create realistic price path
    num_days = len(dates)
    trend = np.linspace(nifty_start, nifty_current, num_days)
    noise = np.random.normal(0, 120, num_days).cumsum()
    nifty_close = trend + noise * 0.15
    
    # Add some volatility clusters and market events
    for i in range(5):
        crash_day = np.random.randint(100, num_days - 100)
        nifty_close[crash_day:crash_day+20] *= 0.95  # 5% correction
    
    nifty_df = pd.DataFrame({
        'Date': dates,
        'Close': nifty_close
    })
    
    # Generate OHLC from Close
    nifty_df['Open'] = nifty_df['Close'].shift(1) * (1 + np.random.normal(0, 0.003, len(dates)))
    nifty_df['High'] = nifty_df[['Open', 'Close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.005, len(dates))))
    nifty_df['Low'] = nifty_df[['Open', 'Close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.005, len(dates))))
    
    # Fill first row
    nifty_df.loc[0, 'Open'] = nifty_df.loc[0, 'Close'] * 0.998
    
    # Generate VIX data (inverse correlation with NIFTY) - end at realistic current value ~11
    vix_current = 11  # Current realistic VIX value
    vix_mean = 15     # Long-term average
    
    # Calculate NIFTY returns for VIX generation
    nifty_returns = nifty_df['Close'].pct_change().fillna(0)
    
    # VIX tends to spike when NIFTY falls
    vix_base = np.full(num_days, vix_mean)
    vix_from_returns = -nifty_returns * 100 * 5  # Negative correlation
    vix_noise = np.random.normal(0, 2, num_days)
    vix_ar = np.zeros(num_days)
    
    # Add autoregressive component (VIX is mean-reverting)
    for i in range(1, num_days):
        vix_ar[i] = 0.95 * vix_ar[i-1] + vix_from_returns.iloc[i] + vix_noise[i]
    
    vix_close = vix_base + vix_ar
    
    # Adjust to end near current value
    vix_adjustment = vix_current - vix_close[-1]
    vix_close = vix_close + np.linspace(0, vix_adjustment, num_days)
    
    # Clip VIX to reasonable range (8 to 50)
    vix_close = np.clip(vix_close, 8, 50)
    
    vix_df = pd.DataFrame({
        'Date': dates,
        'Close': vix_close
    })
    
    # Generate OHLC for VIX
    vix_df['Open'] = vix_df['Close'].shift(1) * (1 + np.random.normal(0, 0.02, len(dates)))
    vix_df['High'] = vix_df[['Open', 'Close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.03, len(dates))))
    vix_df['Low'] = vix_df[['Open', 'Close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.03, len(dates))))
    
    # Fill first row
    vix_df.loc[0, 'Open'] = vix_df.loc[0, 'Close'] * 1.02
    
    print(f"✓ Generated {len(nifty_df)} NIFTY records")
    print(f"  NIFTY range: {nifty_df['Close'].min():.2f} to {nifty_df['Close'].max():.2f}")
    print(f"  Current NIFTY: {nifty_df['Close'].iloc[-1]:.2f}")
    print(f"✓ Generated {len(vix_df)} VIX records")
    print(f"  VIX range: {vix_df['Close'].min():.2f} to {vix_df['Close'].max():.2f}")
    print(f"  Current VIX: {vix_df['Close'].iloc[-1]:.2f}")
    
    return nifty_df, vix_df


def main():
    """Main function to fetch and save data"""
    print("=" * 60)
    print("NSE Data Fetcher - NIFTY 50 & INDIA VIX (via yfinance)")
    print("Auto-Update Mode: Updates to previous trading day")
    print("=" * 60)
    
    fetcher = NSEDataFetcher()
    
    # Check if files exist to determine if this is an update or initial fetch
    nifty_exists = os.path.exists(fetcher.nifty_file)
    vix_exists = os.path.exists(fetcher.vix_file)
    
    if nifty_exists or vix_exists:
        print(f"\nExisting data found. Running in UPDATE mode...")
        update_mode = True
        
        if nifty_exists:
            last_nifty = fetcher.get_last_date_in_file(fetcher.nifty_file)
            if last_nifty:
                print(f"  Last NIFTY date: {last_nifty.date()}")
        
        if vix_exists:
            last_vix = fetcher.get_last_date_in_file(fetcher.vix_file)
            if last_vix:
                print(f"  Last VIX date: {last_vix.date()}")
        
        target_date = fetcher.get_previous_trading_day()
        print(f"  Updating to: {target_date}")
    else:
        print(f"\nNo existing data. Running INITIAL FETCH (last 10 years)...")
        update_mode = False
    
    # Try to fetch real data from Yahoo Finance
    print("\nFetching data from Yahoo Finance...")
    nifty_df = fetcher.fetch_nifty_data(update_mode=update_mode)
    vix_df = fetcher.fetch_vix_data(update_mode=update_mode)
    
    # If both are None and we're updating, it means we're already up to date
    if update_mode and nifty_df is None and vix_df is None:
        print("\n✓ All data is already up to date!")
        print("=" * 60)
        return
    
    # If fetch fails on initial fetch, use sample data
    if not update_mode and (nifty_df is None or vix_df is None or len(nifty_df) < 100 or len(vix_df) < 100):
        print("\n⚠ Yahoo Finance fetch incomplete or failed. Using sample data for demonstration.")
        nifty_df, vix_df = generate_sample_data()
    
    # Save data
    if nifty_df is not None or vix_df is not None:
        print("\nSaving data to CSV files...")
        fetcher.save_data(nifty_df, vix_df, update_mode=update_mode)
    
    # Print summary
    print("\n" + "=" * 60)
    print("✓ Data fetch complete!")
    
    # Read final files to show totals
    if os.path.exists(fetcher.nifty_file):
        final_nifty = pd.read_csv(fetcher.nifty_file)
        final_nifty['Date'] = pd.to_datetime(final_nifty['Date'])
        print(f"  NIFTY: {len(final_nifty)} total records")
        print(f"    Range: {final_nifty['Date'].min().date()} to {final_nifty['Date'].max().date()}")
        print(f"    Latest Close: {final_nifty['Close'].iloc[-1]:.2f}")
    
    if os.path.exists(fetcher.vix_file):
        final_vix = pd.read_csv(fetcher.vix_file)
        final_vix['Date'] = pd.to_datetime(final_vix['Date'])
        print(f"  VIX: {len(final_vix)} total records")
        print(f"    Range: {final_vix['Date'].min().date()} to {final_vix['Date'].max().date()}")
        print(f"    Latest Close: {final_vix['Close'].iloc[-1]:.2f}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
