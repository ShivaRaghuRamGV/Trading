import os
import io
import zipfile
import requests
import pandas as pd
from datetime import date, timedelta

# ========= CONFIG =========
SYMBOL = "NIFTY"
START_DATE = date(2015, 1, 1)   # change as needed
END_DATE   = date(2024, 12, 31) # change as needed
OUT_DIR = "nifty_option_excel"  # output folder
os.makedirs(OUT_DIR, exist_ok=True)

# Excel limit is ~1,048,576 rows. One year of NIFTY options is usually < 1M.
# Script will create one Excel file per year: NIFTY_options_YYYY.xlsx

# ========= NSE URL HELPERS =========

def fo_bhavcopy_url_old(d):
    """
    Old archives URL (pre-UDiFF) e.g.
    https://archives.nseindia.com/content/historical/DERIVATIVES/2022/AUG/fo12AUG2022bhav.csv.zip
    """
    mon_str = d.strftime("%b").upper()  # JAN, FEB, ...
    return (
        f"https://archives.nseindia.com/content/historical/DERIVATIVES/"
        f"{d.year}/{mon_str}/fo{d.day:02d}{mon_str}{d.year}bhav.csv.zip"
    )

def fo_bhavcopy_url_new(d):
    """
    New UDiFF-style FO Bhavcopy (post July 2024).
    Pattern seen in All-Reports page, e.g.
    BhavCopy_NSE_FO_0_0_0_20251205_F_0000.csv.zip
    """
    ymd = d.strftime("%Y%m%d")
    return (
        "https://nsearchives.nseindia.com/content/fo/"
        f"BhavCopy_NSE_FO_0_0_0_{ymd}_F_0000.csv.zip"
    )

# Date when NSE switched to UDiFF bhavcopy for derivatives (approx)
UDIFF_SWITCH_DATE = date(2024, 7, 8)

def get_fo_bhavcopy_url(d):
    """Choose old vs new URL based on date."""
    if d < UDIFF_SWITCH_DATE:
        return fo_bhavcopy_url_old(d)
    else:
        return fo_bhavcopy_url_new(d)

# ========= CORE DOWNLOADER =========

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; NIFTY-options-downloader/1.0)"
})

def download_nifty_options_for_day(d):
    """
    Download FO bhavcopy for date d, filter to NIFTY index options,
    and return a DataFrame. Returns None if file not found / weekend / holiday.
    """
    url = get_fo_bhavcopy_url(d)
    try:
        resp = session.get(url, timeout=30)
        if resp.status_code != 200:
            # Probably holiday/weekend or file not yet available - this is normal
            return None
    except requests.RequestException as e:
        print(f"✗ {d}: Network error – skipping")
        return None

    try:
        # Read zip
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            # Typically there is only one CSV inside
            csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(f)
    except Exception as e:
        print(f"✗ {d}: Failed to read data – skipping")
        return None

    # Different formats (old vs UDiFF) might have slightly different column names,
    # so we normalise the key ones we care about.

    cols = {c.upper(): c for c in df.columns}  # map upper->actual

    def col(name):
        return cols.get(name, None)

    # Try to identify option rows for NIFTY index options
    instr_col = col("INSTRUMENT")
    sym_col   = col("SYMBOL")
    opt_type  = col("OPTION_TYP") or col("OPTION_TYPE")
    strike    = col("STRIKE_PR") or col("STRIKE_PRICE")

    if not instr_col or not sym_col:
        print(f"{d}: missing INSTRUMENT/SYMBOL columns – skipping")
        return None

    df = df[(df[instr_col] == "OPTIDX") & (df[sym_col] == SYMBOL)]

    if df.empty:
        print(f"{d}: no NIFTY options rows – skipping")
        return None

    # Add a DATE column if not already present
    date_col = col("TIMESTAMP") or col("DATE")
    if date_col:
        # standardise to datetime
        df["DATE"] = pd.to_datetime(df[date_col])
    else:
        df["DATE"] = pd.to_datetime(d)

    # Keep only some useful columns (and anything else you like)
    keep_cols = []
    for name in [
        instr_col, sym_col, opt_type, strike,
        col("EXPIRY_DT") or col("EXPIRY_DATE"),
        col("OPEN"), col("HIGH"), col("LOW"),
        col("CLOSE"), col("SETTLE_PR") or col("SETTLE_PRICE"),
        col("CONTRACTS"), col("OPEN_INT"),
        "DATE"
    ]:
        if name and name not in keep_cols:
            keep_cols.append(name)

    return df[keep_cols]


# ========= MAIN LOOP – YEAR BY YEAR TO EXCEL =========

current = START_DATE
total_days_processed = 0
total_trading_days = 0

while current.year <= END_DATE.year:
    year = current.year
    year_start = date(year, 1, 1)
    year_end = date(year, 12, 31)
    if year == START_DATE.year:
        year_start = START_DATE
    if year == END_DATE.year:
        year_end = END_DATE

    print(f"\n{'='*60}")
    print(f"Processing year {year} ({year_start} to {year_end})")
    print(f"{'='*60}")
    dfs_year = []
    
    trading_days_year = 0
    total_days_year = 0

    d = year_start
    while d <= year_end:
        total_days_year += 1
        total_days_processed += 1
        
        # Skip weekends to reduce unnecessary requests
        if d.weekday() >= 5:  # Saturday=5, Sunday=6
            d += timedelta(days=1)
            continue
        
        df_day = download_nifty_options_for_day(d)
        if df_day is not None:
            dfs_year.append(df_day)
            trading_days_year += 1
            total_trading_days += 1
            print(f"✓ {d}: Downloaded {len(df_day)} rows")
        d += timedelta(days=1)

    if dfs_year:
        year_df = pd.concat(dfs_year, ignore_index=True)
        out_path = os.path.join(OUT_DIR, f"NIFTY_options_{year}.xlsx")
        year_df.to_excel(out_path, index=False)
        print(f"\n{'='*60}")
        print(f"✓ Year {year} complete: {trading_days_year} trading days")
        print(f"  Saved {len(year_df):,} rows to {out_path}")
        print(f"{'='*60}")
    else:
        print(f"\n⚠ No data for year {year}, nothing saved.")

    current = date(year + 1, 1, 1)

print("\n" + "="*60)
print("Download Complete!")
print(f"Total trading days with data: {total_trading_days}")
print(f"Files saved to: {OUT_DIR}/")
print("="*60)
