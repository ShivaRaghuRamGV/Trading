# 🚀 HOW TO RUN THE TRADING DASHBOARD

## Quick Start (Easiest Method)

### Option 1: Automated Setup (PowerShell)
```powershell
.\run_dashboard.ps1
```
This single command will:
1. Check Python installation
2. Install all dependencies
3. Fetch data from NSE
4. Launch the dashboard

### Option 2: Manual Setup (Step-by-Step)

#### Step 1: Install Dependencies
```powershell
pip install -r requirements.txt
```

#### Step 2: Fetch Data
```powershell
python nse_data_fetcher.py
```

Expected output:
```
============================================================
NSE Data Fetcher - NIFTY 50 & INDIA VIX
============================================================

Attempting to fetch data from NSE...
✓ Fetched 2500+ NIFTY records
✓ Fetched 2500+ VIX records

Saving data to CSV files...
✓ Saved nifty_history.csv
✓ Saved india_vix_history.csv

============================================================
✓ Data fetch complete!
  NIFTY records: 2547
  VIX records: 2547
  Date range: 2015-12-06 to 2025-12-06
============================================================
```

#### Step 3: Run Dashboard
```powershell
python dashboard.py
```

Expected output:
```
============================================================
Loading Trading Dashboard...
============================================================
✓ Loaded 2547 NIFTY records
✓ Loaded 2547 VIX records
Preparing dashboard data...
✓ Data preparation complete

============================================================
🚀 Dashboard ready!
Opening at http://localhost:8050
============================================================
Dash is running on http://127.0.0.1:8050/

 * Serving Flask app 'dashboard'
 * Debug mode: on
```

#### Step 4: Open Browser
Navigate to: **http://localhost:8050**

---

## Running Individual Components

### Test Data Fetcher Only
```powershell
python nse_data_fetcher.py
```

### Test Analysis Module
```powershell
python analysis.py
```

### Test Event Analysis
```powershell
python event_analysis.py
```

### Run Original Simple Dashboard
```powershell
python trading.py
```

---

## Command to Run Python Files

The command to run any Python file in the terminal is:
```powershell
python <filename>.py
```

Examples:
```powershell
python nse_data_fetcher.py
python dashboard.py
python analysis.py
python event_analysis.py
python trading.py
```

---

## Troubleshooting

### Problem: "python is not recognized"
**Solution:** Python is not installed or not in PATH
```powershell
# Try with python3
python3 dashboard.py

# Or find Python path
where python
```

### Problem: "No module named 'pandas'"
**Solution:** Dependencies not installed
```powershell
pip install -r requirements.txt
```

### Problem: "FileNotFoundError: nifty_history.csv"
**Solution:** Data files not generated
```powershell
# Run data fetcher first
python nse_data_fetcher.py
```

### Problem: Port 8050 already in use
**Solution:** Change port or kill existing process
```powershell
# Option 1: Kill existing process
netstat -ano | findstr :8050
taskkill /PID <process_id> /F

# Option 2: Use different port (modify dashboard.py)
# Change: dashboard.run(port=8051)
```

### Problem: Dashboard loads but shows errors
**Solution:** Check data integrity
```powershell
# Re-fetch data
python nse_data_fetcher.py

# Check if CSV files exist and have data
Get-Content nifty_history.csv -Head 5
Get-Content india_vix_history.csv -Head 5
```

---

## Understanding the Output

### When you run `dashboard.py`, you should see:

1. **Data Loading**
   ```
   Loading Trading Dashboard...
   ✓ Loaded 2547 NIFTY records
   ✓ Loaded 2547 VIX records
   ```

2. **Analysis Preparation**
   ```
   Preparing dashboard data...
   ✓ Data preparation complete
   ```

3. **Server Start**
   ```
   🚀 Dashboard ready!
   Opening at http://localhost:8050
   Dash is running on http://127.0.0.1:8050/
   ```

4. **Keep Running**
   - The terminal will show request logs
   - Dashboard updates in real-time
   - Press `Ctrl+C` to stop

### Browser View:
- Dashboard with multiple tabs
- Interactive charts (hover, zoom, pan)
- Real-time metric cards
- Strategy performance tables

---

## File Structure After Setup

```
trading/
│
├── nse_data_fetcher.py       # Data fetching script
├── analysis.py                # Analysis functions
├── event_analysis.py          # Event & strategy analysis
├── dashboard.py               # Main dashboard ⭐
├── trading.py                 # Original simple dashboard
├── run_dashboard.ps1          # Quick start script
├── requirements.txt           # Dependencies
├── README.md                  # Full documentation
├── HOWTO_RUN.md              # This file
│
├── nifty_history.csv          # Generated data
└── india_vix_history.csv      # Generated data
```

---

## Common Workflows

### First Time Setup
```powershell
pip install -r requirements.txt
python nse_data_fetcher.py
python dashboard.py
```

### Daily Usage (Data Already Fetched)
```powershell
python dashboard.py
```

### Update Data and Run
```powershell
python nse_data_fetcher.py
python dashboard.py
```

### Test Individual Module
```powershell
python analysis.py
```

---

## Pro Tips

### 1. Run in Background
```powershell
Start-Process python -ArgumentList "dashboard.py" -NoNewWindow
```

### 2. Auto-open Browser
Add to `dashboard.py` before `dashboard.run()`:
```python
import webbrowser
webbrowser.open('http://localhost:8050')
```

### 3. Check Logs
Dashboard shows real-time logs in terminal:
- Request URLs
- Callback executions
- Error messages

### 4. Stop Dashboard
Press `Ctrl+C` in the terminal window

---

## Quick Reference Card

| Task | Command |
|------|---------|
| Install dependencies | `pip install -r requirements.txt` |
| Fetch data | `python nse_data_fetcher.py` |
| Run dashboard | `python dashboard.py` |
| Run original app | `python trading.py` |
| Test analysis | `python analysis.py` |
| Quick setup | `.\run_dashboard.ps1` |
| Stop server | `Ctrl+C` |

---

## What Each File Does

| File | Purpose | When to Run |
|------|---------|-------------|
| `nse_data_fetcher.py` | Downloads NIFTY & VIX data from NSE | First time, or to update data |
| `analysis.py` | Core analysis functions | Imported by dashboard |
| `event_analysis.py` | Event & strategy analysis | Imported by dashboard |
| `dashboard.py` | **Main interactive dashboard** | **This is what you run!** |
| `trading.py` | Simple January analysis dashboard | Alternative simpler version |
| `run_dashboard.ps1` | Automated setup script | First time setup |

---

## Success Checklist

Before running dashboard, ensure:
- [x] Python 3.8+ installed
- [x] Dependencies installed (`pip install -r requirements.txt`)
- [x] Data files exist (`nifty_history.csv`, `india_vix_history.csv`)
- [x] Port 8050 is available
- [x] No errors in data fetch

---

## Getting Help

1. **Check Console Output**: Error messages are descriptive
2. **Verify Data Files**: Ensure CSVs are generated
3. **Test Modules Individually**: Run each .py file separately
4. **Check Dependencies**: `pip list | findstr pandas`

---

**Ready to Trade! 📈**

*Most common command you'll use:*
```powershell
python dashboard.py
```
