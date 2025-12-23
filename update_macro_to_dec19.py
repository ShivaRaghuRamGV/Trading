"""
Quick script to update macro data from Dec 11 to Dec 19, 2025
"""
import yfinance as yf
import pandas as pd
from datetime import datetime

# Load existing macro data
df_existing = pd.read_csv('macro_data.csv')
print(f'Current macro data ends at: {df_existing["Date"].iloc[-1]}')
print(f'Total existing records: {len(df_existing)}')

# Fetch data from Dec 11 to Dec 19, 2025
tickers = {
    'US_VIX': '^VIX',
    'USDINR': 'USDINR=X', 
    'Crude': 'CL=F',
    'US_10Y': '^TNX',
    'SP500': '^GSPC'
}

start = '2025-12-11'
end = '2025-12-20'

print(f'\nFetching data from {start} to {end}...')

data_dict = {}
for name, ticker in tickers.items():
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if not df.empty:
            data_dict[name] = df['Close']
            print(f'✓ {name}: {len(df)} days fetched')
    except Exception as e:
        print(f'✗ {name}: {e}')

# Merge all data
if data_dict:
    df_new = pd.DataFrame(data_dict)
    df_new.index.name = 'Date'
    df_new = df_new.reset_index()
    df_new['Date'] = pd.to_datetime(df_new['Date']).dt.strftime('%Y-%m-%d')
    
    print(f'\nNew data fetched:\n{df_new.to_string()}')
    
    # Calculate volatilities and changes (using rolling window)
    for col in ['US_VIX', 'USDINR', 'Crude', 'SP500']:
        if col in df_new.columns:
            df_new[f'{col}_change'] = df_new[col].pct_change()
            df_new[f'{col}_vol'] = df_new[col].rolling(5, min_periods=1).std()
    
    # Append to existing data
    # Match column order
    expected_cols = df_existing.columns.tolist()
    
    # Ensure all columns exist in new data
    for col in expected_cols:
        if col not in df_new.columns:
            df_new[col] = None
    
    # Reorder columns to match
    df_new = df_new[expected_cols]
    
    # Append
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    
    # Remove duplicates
    df_combined = df_combined.drop_duplicates(subset=['Date'], keep='last')
    
    # Save
    df_combined.to_csv('macro_data.csv', index=False)
    print(f'\n✓ Updated macro_data.csv')
    print(f'  Total records now: {len(df_combined)}')
    print(f'  Latest date: {df_combined["Date"].iloc[-1]}')
else:
    print('No new data fetched')
