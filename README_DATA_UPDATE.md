# Auto-Update Data Fetcher

## Overview
The NSE data fetcher has been updated to automatically fetch the latest market data up to the previous trading day.

## What Changed

### 1. **NSE Data Fetcher** (`nse_data_fetcher.py`)
- **Auto-Update Mode**: Automatically detects existing data files and only fetches missing days
- **Previous Trading Day Logic**: Automatically calculates the last trading day (skips weekends)
- **Smart Incremental Updates**: Only downloads data from the last date in the file to yesterday
- **Merge & Deduplicate**: Appends new data to existing files and removes duplicates

### 2. **Macro Data Fetcher** (`macro_data_fetcher.py`)
- **Auto-Update Mode**: Same auto-update logic as NSE fetcher
- **Incremental Downloads**: Only fetches missing dates for all macro variables
- **Correlation Analysis**: Runs automatically after updates (if enough data)

### 3. **Master Updater** (`update_all_data.py`)
- **One-Command Update**: Runs both NSE and macro data fetchers in sequence
- **Status Summary**: Shows success/failure for each data source
- **Next Steps Guidance**: Tells you what to do after updating

## How to Use

### Daily Data Update (Recommended)
Run this single command every day to keep all data current:

```powershell
python update_all_data.py
```

This will:
1. Update NIFTY 50 historical data to yesterday
2. Update India VIX historical data to yesterday
3. Update all macro variables (US VIX, USD/INR, Crude Oil, S&P 500, US 10Y) to yesterday
4. Show a summary of what was updated

### Individual Updates (If Needed)

**Update only NIFTY & VIX:**
```powershell
python nse_data_fetcher.py
```

**Update only Macro Data:**
```powershell
python macro_data_fetcher.py
```

## How It Works

### Auto-Detection Logic

1. **First Run (No existing files)**:
   - Downloads last 10 years of data
   - Creates new CSV files

2. **Subsequent Runs (Files exist)**:
   - Reads last date from existing file
   - Calculates previous trading day
   - Only downloads data between last date + 1 and yesterday
   - Appends to existing file and removes duplicates

### Example Scenarios

**Scenario 1: Running on Monday (Dec 12, 2025)**
- Last file date: Dec 9, 2025
- Previous trading day: Dec 11, 2025 (Friday, since Sat/Sun are skipped)
- Updates: Dec 10 and Dec 11

**Scenario 2: Already up to date**
- Last file date: Dec 11, 2025
- Previous trading day: Dec 11, 2025
- Result: "✓ All data is already up to date!" (no downloads)

**Scenario 3: Missed a week**
- Last file date: Dec 4, 2025
- Previous trading day: Dec 11, 2025
- Updates: Dec 5, 6, 9, 10, 11 (skips Dec 7-8 weekend)

## Data Files Updated

The following CSV files are automatically updated:

1. **nifty_history.csv**: NIFTY 50 OHLC data
2. **india_vix_history.csv**: India VIX OHLC data
3. **macro_data.csv**: US VIX, USD/INR, Crude Oil, S&P 500, US 10Y Treasury

## Automation Tips

### Windows Task Scheduler
Create a scheduled task to run daily at 6 PM:

```powershell
# Create task (run as Administrator)
$action = New-ScheduledTaskAction -Execute "C:\Users\USER\trading\.venv\Scripts\python.exe" -Argument "C:\Users\USER\trading\update_all_data.py" -WorkingDirectory "C:\Users\USER\trading"
$trigger = New-ScheduledTaskTrigger -Daily -At 6PM
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable
Register-ScheduledTask -TaskName "Update Trading Data" -Action $action -Trigger $trigger -Settings $settings
```

### Manual Daily Routine
1. Open PowerShell in `C:\Users\USER\trading`
2. Run: `python update_all_data.py`
3. Wait ~30 seconds for completion
4. Run dashboards: `python forecast_dashboard.py` and `python dashboard.py`

## Troubleshooting

### "No new data found"
- **Cause**: Yahoo Finance doesn't have data for that date yet (markets closed, holiday, or data delay)
- **Solution**: Wait a few hours and try again, or check if it was a market holiday

### "Already up to date"
- **Cause**: You already ran the update today
- **Solution**: Normal behavior, no action needed

### Network Errors
- **Cause**: Internet connection issues or Yahoo Finance API problems
- **Solution**: Check internet connection and try again later

### Unicode Encoding Errors (Windows)
- **Cause**: Console doesn't support UTF-8 characters
- **Solution**: Scripts now auto-configure UTF-8 encoding, but may see garbled symbols (won't affect data)

## Next Steps After Update

1. **Start Forecast Dashboard**:
   ```powershell
   python forecast_dashboard.py
   ```
   Opens at http://localhost:8051

2. **Start Main Dashboard**:
   ```powershell
   python dashboard.py
   ```
   Opens at http://localhost:8050

3. **Check Latest Forecast**:
   - Go to forecast dashboard → "🤖 ML Strategy Selector" tab
   - See latest prediction for tomorrow
   - Review recommended strategy

## Technical Details

### Date Calculation
```python
def get_previous_trading_day():
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    
    # Skip weekends
    while yesterday.weekday() >= 5:  # 5=Sat, 6=Sun
        yesterday = yesterday - timedelta(days=1)
    
    return yesterday
```

### Merge Logic
```python
# Read existing data
existing_df = pd.read_csv('nifty_history.csv')

# Fetch new data
new_df = fetch_data(start=last_date+1, end=yesterday)

# Merge and deduplicate
combined_df = pd.concat([existing_df, new_df])
combined_df = combined_df.drop_duplicates(subset=['Date'], keep='last')
combined_df = combined_df.sort_values('Date')
```

## Benefits

✅ **No Manual Date Entry**: Automatically calculates what dates to fetch
✅ **Efficient**: Only downloads missing data, not entire history
✅ **Safe**: Never overwrites existing data, only appends
✅ **Idempotent**: Running multiple times doesn't cause issues
✅ **Fast**: Typical update takes 10-30 seconds for 1-2 days of data
✅ **Robust**: Handles weekends, holidays, and data gaps automatically

## Version History

- **v2.0 (2025-12-12)**: Added auto-update functionality
- **v1.0 (2025-12-09)**: Initial manual date entry version
