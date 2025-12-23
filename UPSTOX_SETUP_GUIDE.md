# Upstox Integration Setup Guide

## ✅ Upstox API Integration Complete!

Your Greek Regime Flip Model now supports **Upstox API** for real-time NIFTY option chain data.

---

## 🔑 How to Set Up Upstox Access Token

### Option 1: Using Config File (Recommended)

1. **Get Your Upstox Access Token:**
   - Go to [Upstox Developer Console](https://api.upstox.com/)
   - Login with your Upstox credentials
   - Create an app (if you haven't already)
   - Generate an access token
   
2. **Configure the Token:**
   - Copy `upstox_config_template.txt` to `upstox_config.txt`
   - Open `upstox_config.txt`
   - Replace `YOUR_UPSTOX_ACCESS_TOKEN_HERE` with your actual token
   - Save the file

3. **Run the Dashboard:**
   ```powershell
   python greek_regime_flip_live.py
   ```

### Option 2: Using Environment Variable

Set the environment variable before running:
```powershell
$env:UPSTOX_ACCESS_TOKEN="your_token_here"
python greek_regime_flip_live.py
```

---

## 📊 Data Source Priority

The dashboard will try to fetch data in this order:

1. **Upstox API** (if token is configured)
2. **Historical Data** (fallback from parquet files)

---

## 🎯 Features

✅ Real-time NIFTY option chain from Upstox
✅ Live spot price
✅ Greeks data (Delta, Gamma, Vega, Theta)
✅ Implied Volatility (IV)
✅ Open Interest (OI) and Volume
✅ Bid/Ask prices
✅ All expiry dates

---

## 🔧 Troubleshooting

**"Upstox access token not found"**
- Make sure you've created `upstox_config.txt` with your token
- Or set the `UPSTOX_ACCESS_TOKEN` environment variable

**"Upstox API error"**
- Check if your access token is valid
- Tokens usually expire - generate a new one from Upstox console
- Check your internet connection

**Dashboard falls back to historical data**
- This is normal if Upstox API fails
- Historical data is loaded from `nifty_option_excel/` folder
- It's a backup to keep the dashboard working

---

## 📝 Notes

- Upstox access tokens expire periodically - you'll need to regenerate them
- The API has rate limits - don't spam the fetch button
- Minimum 2-second interval between requests is enforced

---

## 🚀 Next Steps

After setting up your token:
1. Run the dashboard
2. Click "FETCH LIVE DATA"
3. Watch for "✅ Upstox API" status message
4. Start analyzing with real-time data!

---

**Need help?** Check Upstox API documentation: https://upstox.com/developer/api-documentation/
