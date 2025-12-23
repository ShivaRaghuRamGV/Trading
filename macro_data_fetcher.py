"""
Fetch macroeconomic data for VIX forecasting (Tier 2)
Downloads US VIX, USD/INR, Crude Oil using yfinance
Automatically updates to the previous trading day
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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

def get_last_date_in_file(filename='macro_data.csv'):
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

def get_previous_trading_day():
    """Get the previous trading day (yesterday, or Friday if today is Monday)"""
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    
    # If yesterday was Saturday or Sunday, go back to Friday
    while yesterday.weekday() >= 5:  # 5=Saturday, 6=Sunday
        yesterday = yesterday - timedelta(days=1)
    
    return yesterday

def fetch_macro_data(start_date='2015-01-01', end_date=None, update_mode=False):
    """
    Fetch macro variables from Yahoo Finance
    
    Args:
        start_date: Start date for data download (or None to auto-detect from existing file)
        end_date: End date (default: previous trading day)
        update_mode: If True, will append to existing file instead of replacing
    
    Returns:
        DataFrame with macro variables
    """
    # Determine end_date (previous trading day)
    if end_date is None:
        end_date = get_previous_trading_day()
    
    # Auto-detect update mode
    if update_mode or os.path.exists('macro_data.csv'):
        last_date = get_last_date_in_file('macro_data.csv')
        if last_date:
            update_mode = True
            start_date = (last_date + timedelta(days=1)).date()
            
            # Check if already up to date
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
            
            if start_date > end_date:
                print("="*60)
                print("Macroeconomic Data Update")
                print("="*60)
                print(f"✓ Macro data is already up to date (last date: {last_date.date()})")
                print("="*60)
                return pd.read_csv('macro_data.csv')
    
    # Convert to string for yfinance
    start_str = start_date.strftime('%Y-%m-%d') if isinstance(start_date, (datetime, type(end_date))) else start_date
    end_str = end_date.strftime('%Y-%m-%d') if isinstance(end_date, (datetime, type(end_date))) else end_date
    
    print("="*60)
    print("Fetching Macroeconomic Data")
    if update_mode:
        print("UPDATE MODE: Appending new data")
    print("="*60)
    print(f"Date range: {start_str} to {end_str}\n")
    
    macro_data = {}
    
    # 1. US VIX (Global fear gauge)
    print("Downloading US VIX (^VIX)...")
    try:
        us_vix = yf.download('^VIX', start=start_str, end=end_str, progress=False)
        if not us_vix.empty:
            macro_data['US_VIX'] = us_vix['Close']
            print(f"   ✓ Downloaded {len(us_vix)} records")
        else:
            print("   ⚠️ No data retrieved")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 2. USD/INR (Currency volatility affects India VIX)
    print("Downloading USD/INR (USDINR=X)...")
    try:
        usdinr = yf.download('USDINR=X', start=start_str, end=end_str, progress=False)
        if not usdinr.empty:
            macro_data['USDINR'] = usdinr['Close']
            print(f"   ✓ Downloaded {len(usdinr)} records")
        else:
            print("   ⚠️ No data retrieved")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 3. Crude Oil (India imports 80%+ oil, affects macro stability)
    print("Downloading Crude Oil (CL=F)...")
    try:
        crude = yf.download('CL=F', start=start_str, end=end_str, progress=False)
        if not crude.empty:
            macro_data['Crude_Oil'] = crude['Close']
            print(f"   ✓ Downloaded {len(crude)} records")
        else:
            print("   ⚠️ No data retrieved")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 4. US 10-Year Treasury Yield (Risk-free rate, affects global risk appetite)
    print("Downloading US 10Y Treasury (^TNX)...")
    try:
        tnx = yf.download('^TNX', start=start_str, end=end_str, progress=False)
        if not tnx.empty:
            macro_data['US_10Y'] = tnx['Close']
            print(f"   ✓ Downloaded {len(tnx)} records")
        else:
            print("   ⚠️ No data retrieved")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 5. S&P 500 (Global equity benchmark)
    print("Downloading S&P 500 (^GSPC)...")
    try:
        sp500 = yf.download('^GSPC', start=start_str, end=end_str, progress=False, timeout=60)
        if not sp500.empty:
            macro_data['SP500'] = sp500['Close']
            print(f"   ✓ Downloaded {len(sp500)} records")
        else:
            print("   ⚠️ No data retrieved")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Combine into single DataFrame
    if macro_data:
        # Find common dates across all series
        common_dates = None
        for series in macro_data.values():
            if common_dates is None:
                common_dates = set(series.index)
            else:
                common_dates = common_dates.intersection(set(series.index))
        
        common_dates = sorted(list(common_dates))
        
        # Align all series to common dates
        aligned_data = {}
        for key, series in macro_data.items():
            aligned_data[key] = series.loc[common_dates].values.flatten()
        
        df_macro = pd.DataFrame(aligned_data, index=common_dates)
        df_macro.index.name = 'Date'
        df_macro = df_macro.reset_index()
        
        # Calculate derived features
        if 'US_VIX' in df_macro.columns:
            df_macro['US_VIX_change'] = df_macro['US_VIX'].pct_change()
        
        if 'USDINR' in df_macro.columns:
            df_macro['USDINR_change'] = df_macro['USDINR'].pct_change()
            df_macro['USDINR_vol'] = df_macro['USDINR_change'].rolling(21).std() * 100
        
        if 'Crude_Oil' in df_macro.columns:
            df_macro['Crude_change'] = df_macro['Crude_Oil'].pct_change()
            df_macro['Crude_vol'] = df_macro['Crude_change'].rolling(21).std() * 100
        
        if 'SP500' in df_macro.columns:
            df_macro['SP500_return'] = df_macro['SP500'].pct_change() * 100
            df_macro['SP500_vol'] = df_macro['SP500_return'].rolling(21).std() * np.sqrt(252)
        
        # If update mode, merge with existing data
        if update_mode and os.path.exists('macro_data.csv'):
            existing_df = pd.read_csv('macro_data.csv')
            existing_df['Date'] = pd.to_datetime(existing_df['Date'])
            df_macro['Date'] = pd.to_datetime(df_macro['Date'])
            
            # Concatenate and remove duplicates
            combined_df = pd.concat([existing_df, df_macro], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['Date'], keep='last')
            combined_df = combined_df.sort_values('Date').reset_index(drop=True)
            
            df_macro = combined_df
        
        # Save to CSV
        output_file = 'macro_data.csv'
        df_macro.to_csv(output_file, index=False)
        
        print("\n" + "="*60)
        if update_mode:
            print(f"✓ Macro data updated in {output_file}")
        else:
            print(f"✓ Macro data saved to {output_file}")
        print(f"  Total records: {len(df_macro)}")
        print(f"  Date range: {df_macro['Date'].min()} to {df_macro['Date'].max()}")
        print(f"  Variables: {', '.join([c for c in df_macro.columns if c != 'Date'])}")
        print("="*60)
        
        return df_macro
    else:
        print("\n❌ No macro data retrieved")
        return pd.DataFrame()


def analyze_correlations(india_vix_file='india_vix_history.csv', 
                         macro_file='macro_data.csv'):
    """
    Analyze correlations between India VIX and macro variables
    
    Args:
        india_vix_file: Path to India VIX CSV
        macro_file: Path to macro data CSV
    
    Returns:
        Correlation analysis results
    """
    print("\n" + "="*60)
    print("Correlation Analysis: India VIX vs Macro Variables")
    print("="*60)
    
    # Load data
    india_vix = pd.read_csv(india_vix_file)
    india_vix['Date'] = pd.to_datetime(india_vix['Date']).dt.tz_localize(None)
    india_vix = india_vix.rename(columns={'Close': 'India_VIX'})
    
    macro_data = pd.read_csv(macro_file)
    macro_data['Date'] = pd.to_datetime(macro_data['Date']).dt.tz_localize(None)
    
    # Merge
    df = pd.merge(india_vix[['Date', 'India_VIX']], macro_data, on='Date', how='inner')
    
    print(f"\nMerged dataset: {len(df)} records")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}\n")
    
    # Calculate correlations
    correlations = {}
    
    india_vix_series = df['India_VIX']
    
    for col in df.columns:
        if col not in ['Date', 'India_VIX'] and df[col].notna().sum() > 100:
            corr = india_vix_series.corr(df[col])
            correlations[col] = corr
    
    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print("Correlations with India VIX:")
    print("-" * 60)
    for var, corr in sorted_corr:
        strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
        direction = "Positive" if corr > 0 else "Negative"
        print(f"{var:25s}: {corr:+.4f}  ({strength} {direction})")
    
    print("-" * 60)
    
    # Rolling correlations for key variables
    if 'US_VIX' in df.columns:
        df['India_US_VIX_corr_63d'] = df['India_VIX'].rolling(63).corr(df['US_VIX'])
        print(f"\nRolling 63-day correlation (India VIX vs US VIX):")
        print(f"  Mean: {df['India_US_VIX_corr_63d'].mean():.4f}")
        print(f"  Min:  {df['India_US_VIX_corr_63d'].min():.4f}")
        print(f"  Max:  {df['India_US_VIX_corr_63d'].max():.4f}")
    
    return df, sorted_corr


if __name__ == "__main__":
    # Fetch macro data (auto-detects update mode)
    df_macro = fetch_macro_data(update_mode=True)
    
    # Analyze correlations
    if not df_macro.empty and len(df_macro) > 100:
        try:
            df_merged, correlations = analyze_correlations()
        except Exception as e:
            print(f"\n⚠️ Correlation analysis skipped: {e}")
