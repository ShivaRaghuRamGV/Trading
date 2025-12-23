# Quick Start Script for Trading Dashboard
# This script will set up everything automatically

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "  NIFTY 50 & INDIA VIX Trading Dashboard - Quick Setup" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Python
Write-Host "[Step 1/4] Checking Python installation..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ Python found: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "  ✗ Python not found! Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

# Step 2: Install dependencies
Write-Host ""
Write-Host "[Step 2/4] Installing required packages..." -ForegroundColor Yellow
pip install -r requirements.txt --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ All packages installed successfully" -ForegroundColor Green
} else {
    Write-Host "  ✗ Failed to install packages" -ForegroundColor Red
    exit 1
}

# Step 3: Fetch data
Write-Host ""
Write-Host "[Step 3/4] Fetching NIFTY & VIX data from NSE..." -ForegroundColor Yellow
python nse_data_fetcher.py
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ Data fetched successfully" -ForegroundColor Green
} else {
    Write-Host "  ⚠ Data fetch encountered issues (may have used sample data)" -ForegroundColor Yellow
}

# Step 4: Launch dashboard
Write-Host ""
Write-Host "[Step 4/4] Launching dashboard..." -ForegroundColor Yellow
Write-Host ""
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "  🚀 Dashboard starting at http://localhost:8050" -ForegroundColor Green
Write-Host "  📊 Press Ctrl+C to stop the dashboard" -ForegroundColor Yellow
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

Start-Sleep -Seconds 2
python dashboard.py
