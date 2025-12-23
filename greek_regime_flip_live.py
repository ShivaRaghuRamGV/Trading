"""
THE GREEK REGIME FLIP MODEL (GRFM) - LIVE NSE DATA
Greeks-Driven Entry–Exit System for NIFTY Options with Real Option Chain

Based on: Greek_Regime_Flip_Model_Entry_Exit_System.pdf
Data Source: NSE Live Option Chain API
"""

import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import requests
import warnings
warnings.filterwarnings('ignore')

app = dash.Dash(__name__)
app.title = "Greek Regime Flip Model - Live NSE"


# ============== SECURITY VALIDATORS ==============
class SecurityValidator:
    """Validate and sanitize all external inputs"""
    
    @staticmethod
    def validate_numeric(value, min_val, max_val, name="value"):
        """Validate numeric input is within acceptable bounds"""
        try:
            num = float(value)
            if not (min_val <= num <= max_val):
                raise ValueError(f"{name} must be between {min_val} and {max_val}")
            if not np.isfinite(num):
                raise ValueError(f"{name} must be a finite number")
            return num
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid {name}: {e}")
    
    @staticmethod
    def validate_string(value, max_length=50, allowed_chars=None):
        """Validate and sanitize string inputs"""
        if not isinstance(value, str):
            return str(value)[:max_length]
        
        # Remove any script tags or suspicious patterns
        sanitized = value.replace('<', '').replace('>', '').replace('script', '').replace('javascript:', '')
        sanitized = sanitized[:max_length]
        
        if allowed_chars:
            sanitized = ''.join(c for c in sanitized if c in allowed_chars)
        
        return sanitized
    
    @staticmethod
    def validate_date(date_str):
        """Validate date format"""
        try:
            datetime.strptime(date_str, '%d-%b-%Y')
            return True
        except:
            return False
    
    @staticmethod
    def validate_api_response(data):
        """Validate API response structure"""
        if not isinstance(data, dict):
            raise ValueError("API response must be a dictionary")
        if 'records' not in data:
            raise ValueError("Invalid API response structure")
        return True


class NSEOptionChain:
    """Fetch real-time NIFTY option chain from NSE with security controls"""
    
    # Whitelist for allowed URLs
    ALLOWED_DOMAINS = ['www.nseindia.com', 'nsearchives.nseindia.com']
    MAX_RETRIES = 3
    REQUEST_TIMEOUT = 10
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive'
        }
        self.session = requests.Session()
        self.base_url = 'https://www.nseindia.com'
        self.validator = SecurityValidator()
        self.last_fetch_time = None
        self.min_fetch_interval = 1  # Minimum 1 second between requests
        
    def _validate_url(self, url):
        """Validate URL is from allowed domain"""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.netloc not in self.ALLOWED_DOMAINS:
            raise ValueError(f"URL domain {parsed.netloc} not in allowed list")
        if parsed.scheme != 'https':
            raise ValueError("Only HTTPS connections allowed")
        return True
    
    def _rate_limit_check(self):
        """Prevent excessive API requests"""
        if self.last_fetch_time:
            elapsed = (datetime.now() - self.last_fetch_time).total_seconds()
            if elapsed < self.min_fetch_interval:
                raise ValueError(f"Rate limit: Wait {self.min_fetch_interval - elapsed:.1f}s")
        self.last_fetch_time = datetime.now()
    
    def fetch_option_chain(self):
        """Fetch live NIFTY option chain from NSE with security checks"""
        print("\n" + "="*70)
        print("ATTEMPTING NSE API FETCH")
        print("="*70)
        
        try:
            # Rate limiting
            print("[1/5] Checking rate limit...")
            self._rate_limit_check()
            print("✓ Rate limit OK")
            
            # Validate URLs
            print("[2/5] Validating URLs...")
            self._validate_url(self.base_url)
            print(f"✓ Base URL validated: {self.base_url}")
            
            # First visit NSE homepage to get cookies
            print("[3/5] Fetching cookies from NSE homepage...")
            response = self.session.get(self.base_url, headers=self.headers, 
                                       timeout=self.REQUEST_TIMEOUT, verify=True)
            print(f"✓ Homepage response: {response.status_code}")
            print(f"  Cookies received: {len(self.session.cookies)} cookies")
            
            # Fetch option chain
            url = f'{self.base_url}/api/option-chain-indices?symbol=NIFTY'
            print(f"[4/5] Fetching option chain from: {url}")
            self._validate_url(url)
            
            response = self.session.get(url, headers=self.headers, 
                                       timeout=self.REQUEST_TIMEOUT, verify=True)
            
            print(f"[5/5] Option chain API response: {response.status_code}")
            print(f"  Response size: {len(response.content)} bytes")
            
            if response.status_code == 200:
                # Validate response size (prevent memory attacks)
                if len(response.content) > 10 * 1024 * 1024:  # 10MB limit
                    raise ValueError("Response too large")
                
                print("  Parsing JSON...")
                data = response.json()
                
                # Debug: Print response structure
                print(f"\n✓ NSE API JSON STRUCTURE:")
                print(f"  Top-level keys: {list(data.keys())}")
                print(f"  Data type: {type(data)}")
                
                if 'records' in data:
                    print(f"  ✓ 'records' key found")
                    print(f"  Records keys: {list(data['records'].keys())}")
                    if 'data' in data['records']:
                        print(f"  ✓ 'data' key found with {len(data['records']['data'])} items")
                    else:
                        print(f"  ❌ No 'data' key in records!")
                else:
                    print(f"  ❌ NO 'records' KEY FOUND!")
                    print(f"  Full response (first 1000 chars): {str(data)[:1000]}")
                
                # Validate response structure
                print("\n  Validating API response structure...")
                self.validator.validate_api_response(data)
                print("  ✓ Validation passed")
                
                print("\n  Parsing option chain data...")
                result = self.parse_option_chain(data)
                if result:
                    print(f"  ✓ Successfully parsed option chain")
                else:
                    print(f"  ❌ Parse returned None")
                return result
            else:
                print(f"\n❌ NSE API returned HTTP status {response.status_code}")
                print(f"  Response text (first 500 chars): {response.text[:500]}")
                return None
                
        except ValueError as e:
            print(f"\n❌ VALIDATION ERROR: {e}")
            import traceback
            traceback.print_exc()
            return None
        except requests.exceptions.SSLError as e:
            print(f"\n❌ SSL ERROR: {e}")
            return None
        except requests.exceptions.Timeout as e:
            print(f"\n❌ TIMEOUT ERROR: {e}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"\n❌ REQUEST ERROR: {e}")
            return None
        except Exception as e:
            print(f"\n❌ UNEXPECTED ERROR: {e}")
            print(f"  Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return None
            return None
    
    def parse_option_chain(self, data):
        """Parse NSE option chain JSON to DataFrame with input validation"""
        try:
            records = data['records']
            spot = records['underlyingValue']
            
            # Validate spot price
            spot = self.validator.validate_numeric(spot, 10000, 100000, "spot price")
            
            option_data = []
            max_records = 5000  # Prevent memory exhaustion
            
            for idx, item in enumerate(records['data']):
                if idx >= max_records:
                    print(f"⚠️ Truncated to {max_records} records")
                    break
                
                exp_date = self.validator.validate_string(item['expiryDate'], 20, 
                                                         '0123456789-AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpRrSsTtUuVvWwXxYyZz')
                strike = self.validator.validate_numeric(item['strikePrice'], 0, 200000, "strike")
                
                # Validate date format
                if not self.validator.validate_date(exp_date):
                    continue
                
                # Call options
                if 'CE' in item:
                    ce = item['CE']
                    try:
                        premium = self.validator.validate_numeric(ce.get('lastPrice', 0), 0, 50000, "premium")
                        iv = self.validator.validate_numeric(ce.get('impliedVolatility', 0), 0, 200, "IV")
                        volume = max(0, int(ce.get('totalTradedVolume', 0)))
                        oi = max(0, int(ce.get('openInterest', 0)))
                        
                        option_data.append({
                            'Expiry': exp_date,
                            'Strike': strike,
                            'Type': 'CE',
                            'Premium': premium,
                            'IV': iv,
                            'Delta': ce.get('delta', np.nan),
                            'Gamma': ce.get('gamma', np.nan),
                            'Vega': ce.get('vega', np.nan),
                            'Theta': ce.get('theta', np.nan),
                            'Volume': min(volume, 1000000000),  # Cap at 1B
                            'OI': min(oi, 1000000000),
                            'Bid': self.validator.validate_numeric(ce.get('bidprice', 0), 0, 50000, "bid"),
                            'Ask': self.validator.validate_numeric(ce.get('askPrice', 0), 0, 50000, "ask")
                        })
                    except ValueError:
                        continue  # Skip invalid records
                
                # Put options
                if 'PE' in item:
                    pe = item['PE']
                    try:
                        premium = self.validator.validate_numeric(pe.get('lastPrice', 0), 0, 50000, "premium")
                        iv = self.validator.validate_numeric(pe.get('impliedVolatility', 0), 0, 200, "IV")
                        volume = max(0, int(pe.get('totalTradedVolume', 0)))
                        oi = max(0, int(pe.get('openInterest', 0)))
                        
                        option_data.append({
                            'Expiry': exp_date,
                            'Strike': strike,
                            'Type': 'PE',
                            'Premium': premium,
                            'IV': iv,
                            'Delta': pe.get('delta', np.nan),
                            'Gamma': pe.get('gamma', np.nan),
                            'Vega': pe.get('vega', np.nan),
                            'Theta': pe.get('theta', np.nan),
                            'Volume': min(volume, 1000000000),
                            'OI': min(oi, 1000000000),
                            'Bid': self.validator.validate_numeric(pe.get('bidprice', 0), 0, 50000, "bid"),
                            'Ask': self.validator.validate_numeric(pe.get('askPrice', 0), 0, 50000, "ask")
                        })
                    except ValueError:
                        continue  # Skip invalid records
            
            if not option_data:
                raise ValueError("No valid option data found")
            
            df = pd.DataFrame(option_data)
            
            # Validate DataFrame
            if len(df) == 0:
                raise ValueError("Empty DataFrame")
            if len(df) > max_records:
                df = df.head(max_records)
            
            return df, spot
            
        except Exception as e:
            print(f"⚠️ Parse error: {e}")
            return None


class UpstoxOptionChain:
    """Fetch real-time NIFTY option chain from Upstox API"""
    
    def __init__(self, access_token=None):
        """
        Initialize Upstox API client
        Args:
            access_token: Upstox API access token (optional, can be set later)
        """
        self.access_token = access_token
        self.base_url = 'https://api.upstox.com/v2'
        self.session = requests.Session()
        self.validator = SecurityValidator()
        
    def set_access_token(self, token):
        """Set the Upstox access token"""
        self.access_token = token
        
    def fetch_option_chain(self, symbol='NIFTY', expiry_date=None):
        """
        Fetch NIFTY option chain from Upstox
        Args:
            symbol: Index symbol (default: NIFTY)
            expiry_date: Expiry date in YYYY-MM-DD format (optional, uses nearest if not provided)
        Returns:
            Tuple of (DataFrame, spot_price) or None on error
        """
        if not self.access_token:
            print("❌ Upstox access token not set. Please configure in config file or environment.")
            return None
            
        print(f"\n{'='*70}", flush=True)
        print(f"FETCHING FROM UPSTOX API - {datetime.now()}", flush=True)
        print(f"{'='*70}", flush=True)
        
        try:
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {self.access_token}'
            }
            
            # Step 1: Get NIFTY spot price
            print("[1/3] Fetching NIFTY spot price...", flush=True)
            spot_url = f'{self.base_url}/market-quote/quotes'
            spot_params = {'instrument_key': 'NSE_INDEX|Nifty 50'}
            
            response = self.session.get(spot_url, headers=headers, params=spot_params, timeout=5)
            
            if response.status_code == 200:
                spot_data = response.json()
                spot_price = spot_data['data']['NSE_INDEX:Nifty 50']['last_price']
                print(f"✓ NIFTY Spot: ₹{spot_price:,.2f}", flush=True)
            else:
                print(f"⚠️ Spot price fetch failed: {response.status_code}", flush=True)
                print(f"Response: {response.text[:200]}", flush=True)
                spot_price = 24000  # Fallback
            
            # Step 2: Get option chain
            print("[2/3] Fetching option chain...", flush=True)
            
            # Upstox uses instrument keys format: NSE_FO|NIFTY{EXPIRY}{STRIKE}{CE/PE}
            # We'll need to construct this or use their option chain endpoint
            
            # Get option chain data
            option_url = f'{self.base_url}/option/chain'
            option_params = {
                'instrument_key': 'NSE_INDEX|Nifty 50',
                'expiry_date': expiry_date if expiry_date else ''  # Will use nearest expiry if empty
            }
            
            response = self.session.get(option_url, headers=headers, params=option_params, timeout=8)
            
            if response.status_code != 200:
                print(f"❌ Option chain fetch failed: {response.status_code}", flush=True)
                print(f"Response: {response.text[:500]}", flush=True)
                return None
            
            data = response.json()
            print(f"✓ Option chain data received", flush=True)
            
            # Step 3: Parse option chain
            print("[3/3] Parsing option chain...")
            return self.parse_upstox_option_chain(data, spot_price)
            
        except requests.exceptions.Timeout:
            print("❌ Upstox API timeout")
            return None
        except Exception as e:
            print(f"❌ Upstox API error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def parse_upstox_option_chain(self, data, spot_price):
        """Parse Upstox option chain response to DataFrame"""
        try:
            option_data = []
            
            if 'data' not in data:
                print("❌ No data in Upstox response")
                return None
            
            chain_data = data['data']
            
            for item in chain_data:
                try:
                    strike = float(item.get('strike_price', 0))
                    expiry = item.get('expiry', '')
                    
                    # Call option
                    if 'call_options' in item:
                        ce = item['call_options']['market_data']
                        option_data.append({
                            'Expiry': expiry,
                            'Strike': strike,
                            'Type': 'CE',
                            'Premium': float(ce.get('ltp', 0)),
                            'IV': float(item['call_options'].get('iv', 0)) if 'iv' in item.get('call_options', {}) else np.nan,
                            'Delta': float(item['call_options'].get('delta', 0)) if 'delta' in item.get('call_options', {}) else np.nan,
                            'Gamma': float(item['call_options'].get('gamma', 0)) if 'gamma' in item.get('call_options', {}) else np.nan,
                            'Vega': float(item['call_options'].get('vega', 0)) if 'vega' in item.get('call_options', {}) else np.nan,
                            'Theta': float(item['call_options'].get('theta', 0)) if 'theta' in item.get('call_options', {}) else np.nan,
                            'Volume': int(ce.get('volume', 0)),
                            'OI': int(ce.get('oi', 0)),
                            'Bid': float(ce.get('bid_price', 0)),
                            'Ask': float(ce.get('ask_price', 0))
                        })
                    
                    # Put option
                    if 'put_options' in item:
                        pe = item['put_options']['market_data']
                        option_data.append({
                            'Expiry': expiry,
                            'Strike': strike,
                            'Type': 'PE',
                            'Premium': float(pe.get('ltp', 0)),
                            'IV': float(item['put_options'].get('iv', 0)) if 'iv' in item.get('put_options', {}) else np.nan,
                            'Delta': float(item['put_options'].get('delta', 0)) if 'delta' in item.get('put_options', {}) else np.nan,
                            'Gamma': float(item['put_options'].get('gamma', 0)) if 'gamma' in item.get('put_options', {}) else np.nan,
                            'Vega': float(item['put_options'].get('vega', 0)) if 'vega' in item.get('put_options', {}) else np.nan,
                            'Theta': float(item['put_options'].get('theta', 0)) if 'theta' in item.get('put_options', {}) else np.nan,
                            'Volume': int(pe.get('volume', 0)),
                            'OI': int(pe.get('oi', 0)),
                            'Bid': float(pe.get('bid_price', 0)),
                            'Ask': float(pe.get('ask_price', 0))
                        })
                        
                except Exception as e:
                    continue  # Skip invalid records
            
            if not option_data:
                print("❌ No valid option data parsed")
                return None
            
            df = pd.DataFrame(option_data)
            print(f"✓ Parsed {len(df)} option contracts")
            
            return df, spot_price
            
        except Exception as e:
            print(f"❌ Parse error: {e}")
            import traceback
            traceback.print_exc()
            return None


def load_daily_option_chain_csv(expiry_date, trading_date):
    """
    Load daily option chain CSV file with actual IV values.
    Filename format: option-chain-ED-NIFTY-{expiry_date}_{trading_date}.csv
    
    Args:
        expiry_date: Format like "30-Dec-2025"
        trading_date: Format like "19 Dec 2025"
    
    Returns:
        DataFrame with columns: STRIKE, Type, IV, LTP, VOLUME, OI, CHNG_IN_OI, etc.
    """
    try:
        import os
        filename = f"nifty_option_excel/option-chain-ED-NIFTY-{expiry_date}_{trading_date}.csv"
        
        if not os.path.exists(filename):
            print(f"⚠️ Daily chain file not found: {filename}")
            return None
        
        # Read CSV with skiprows to handle header
        df = pd.read_csv(filename, skiprows=1)
        
        # Extract STRIKE column
        df['STRIKE'] = pd.to_numeric(df['STRIKE'].astype(str).str.replace(',', ''), errors='coerce')
        
        # Extract CALL data (columns before STRIKE)
        calls = df[['OI', 'CHNG IN OI', 'VOLUME', 'IV', 'LTP', 'CHNG', 'BID QTY', 'BID', 'ASK', 'ASK QTY', 'STRIKE']].copy()
        calls.columns = ['OI', 'CHNG_IN_OI', 'VOLUME', 'IV', 'LTP', 'CHNG', 'BID_QTY', 'BID', 'ASK', 'ASK_QTY', 'Strike']
        calls['Type'] = 'CE'
        
        # Extract PUT data (columns after STRIKE)
        puts = df[['STRIKE', 'BID QTY.1', 'BID.1', 'ASK.1', 'ASK QTY.1', 'CHNG.1', 'LTP.1', 'IV.1', 'VOLUME.1', 'CHNG IN OI.1', 'OI.1']].copy()
        puts.columns = ['Strike', 'BID_QTY', 'BID', 'ASK', 'ASK_QTY', 'CHNG', 'LTP', 'IV', 'VOLUME', 'CHNG_IN_OI', 'OI']
        puts['Type'] = 'PE'
        
        # Combine
        combined = pd.concat([calls, puts], ignore_index=True)
        
        # Clean numeric fields
        for col in ['IV', 'LTP', 'Strike', 'VOLUME', 'OI', 'CHNG_IN_OI']:
            combined[col] = pd.to_numeric(combined[col].astype(str).str.replace(',', ''), errors='coerce')
        
        # Rename LTP to Premium for consistency
        combined = combined.rename(columns={'LTP': 'Premium'})
        
        # Remove rows with missing STRIKE or IV
        combined = combined.dropna(subset=['Strike', 'IV'])
        
        print(f"✓ Loaded {len(combined)} options from {filename} (IV range: {combined['IV'].min():.1f}%-{combined['IV'].max():.1f}%)")
        return combined
        
    except Exception as e:
        print(f"❌ Error loading daily chain CSV: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def calculate_iv_change_per_strike(df_current, expiry_date, current_trading_date, previous_trading_date):
    """
    Calculate IV change per strike by comparing current day to previous day.
    
    Args:
        df_current: Current day's option data
        expiry_date: Option expiry date
        current_trading_date: Today's trading date
        previous_trading_date: Previous trading date for comparison
    
    Returns:
        DataFrame with added 'IV_Change_%' column
    """
    try:
        # Load previous day's data
        df_prev = load_daily_option_chain_csv(expiry_date, previous_trading_date)
        
        if df_prev is None:
            print(f"⚠️ Previous day data not available, IV change set to 0")
            df_current['IV_Change_%'] = 0.0
            return df_current
        
        # Merge on Strike and Type to match options
        df_merged = df_current.merge(
            df_prev[['Strike', 'Type', 'IV']],
            on=['Strike', 'Type'],
            how='left',
            suffixes=('', '_prev')
        )
        
        # Calculate IV change percentage
        df_merged['IV_Change_%'] = (
            (df_merged['IV'] - df_merged['IV_prev']) / df_merged['IV_prev'] * 100
        ).fillna(0)
        
        # Drop the previous IV column
        df_merged = df_merged.drop(columns=['IV_prev'])
        
        print(f"✓ IV changes calculated (range: {df_merged['IV_Change_%'].min():.1f}% to {df_merged['IV_Change_%'].max():.1f}%)")
        return df_merged
        
    except Exception as e:
        print(f"⚠️ Error calculating IV change: {str(e)}")
        df_current['IV_Change_%'] = 0.0
        return df_current

def load_fallback_option_data():
    """Load historical option data from all parquet files"""
    try:
        import glob
        import os
        
        # Find all parquet files
        parquet_files = glob.glob('nifty_option_excel/NIFTY_options_*.parquet')
        if not parquet_files:
            return None
        
        print(f"📁 Loading data from {len(parquet_files)} parquet files...")
        
        # Load and combine all parquet files
        all_dfs = []
        for pf in parquet_files:
            df_temp = pd.read_parquet(pf)
            # Convert DATE column to datetime if it exists
            if 'DATE' in df_temp.columns:
                df_temp['DATE'] = pd.to_datetime(df_temp['DATE'], errors='coerce')
            elif 'Date' in df_temp.columns:
                df_temp['Date'] = pd.to_datetime(df_temp['Date'], errors='coerce')
            all_dfs.append(df_temp)
        
        df = pd.concat(all_dfs, ignore_index=True)
        print(f"📊 Combined {len(df)} total records from all files")
        
        # Get most recent date in the combined data
        if 'DATE' in df.columns:
            latest_date = df['DATE'].max()
            df_latest = df[df['DATE'] == latest_date].copy()
            print(f"📅 Using latest date: {latest_date}")
        elif 'Date' in df.columns:
            latest_date = df['Date'].max()
            df_latest = df[df['Date'] == latest_date].copy()
            print(f"📅 Using latest date: {latest_date}")
        else:
            # If no date column, use all data
            df_latest = df.copy()
            print("⚠️ No date column found, using all data")
        
        # Rename columns to match NSE format
        column_mapping = {
            'Strike Price': 'Strike',
            'STRIKE_PR': 'Strike',
            'Option Type': 'Type',
            'OPTION_TYP': 'Type',
            'Close': 'Premium',
            'CLOSE': 'Premium',
            'Open Interest': 'OI',
            'OPEN_INT': 'OI',
            'Volume': 'Volume',
            'CONTRACTS': 'Volume',
            'Expiry Date': 'Expiry',
            'EXPIRY_DT': 'Expiry',
            'DATE': 'Date'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df_latest.columns:
                df_latest.rename(columns={old_col: new_col}, inplace=True)
        
        # Convert numeric columns to proper types
        numeric_columns = ['Strike', 'Premium', 'OI', 'Volume']
        for col in numeric_columns:
            if col in df_latest.columns:
                df_latest[col] = pd.to_numeric(df_latest[col], errors='coerce').fillna(0)
        
        # Add required columns if missing
        if 'IV' not in df_latest.columns:
            print("📊 Calculating Implied Volatility from option premiums...")
            # Calculate IV from premiums using Black-Scholes
            df_latest['IV'] = 20.0  # Initial default
            
            # We'll calculate IV after we know the spot price and expiry
            # For now, set a placeholder that will be updated in process_chain()
        if 'Delta' not in df_latest.columns:
            df_latest['Delta'] = np.nan
        if 'Gamma' not in df_latest.columns:
            df_latest['Gamma'] = np.nan
        if 'Vega' not in df_latest.columns:
            df_latest['Vega'] = np.nan
        if 'Theta' not in df_latest.columns:
            df_latest['Theta'] = np.nan
        if 'Bid' not in df_latest.columns:
            df_latest['Bid'] = df_latest.get('Premium', 0) * 0.98
        if 'Ask' not in df_latest.columns:
            df_latest['Ask'] = df_latest.get('Premium', 0) * 1.02
        
        # Ensure Type column has CE/PE
        if 'Type' in df_latest.columns:
            df_latest['Type'] = df_latest['Type'].replace({'Call': 'CE', 'Put': 'PE', 'call': 'CE', 'put': 'PE'})
        
        # Get spot price (use ATM strike as approximation)
        if 'Strike' in df_latest.columns and len(df_latest) > 0:
            strikes = df_latest['Strike'].unique()
            spot = np.median(strikes)
        else:
            spot = 24000  # Default
        
        # Filter to keep only relevant strikes (ATM ± 1500)
        if 'Strike' in df_latest.columns:
            df_latest = df_latest[
                (df_latest['Strike'] >= spot - 1500) & 
                (df_latest['Strike'] <= spot + 1500)
            ].copy()
        
        print(f"✅ Loaded {len(df_latest)} option records, NIFTY Spot: ₹{spot:,.0f}")
        return df_latest, spot
        
    except Exception as e:
        print(f"❌ Fallback data load failed: {e}")
        import traceback
        traceback.print_exc()
        return None


class GreekRegimeFlipModel:
    """
    Greek Regime Flip Model for systematic options trading
    Uses real NSE option chain data with calculated Greeks
    """
    
    def __init__(self, spot, r=0.07):
        self.spot = spot
        self.r = r
        self.ema_alpha = 0.4  # EMA smoothing factor (0.3-0.5)
        self.rolling_window = 4  # 3-5 day rolling window
        self.weights_cache = {}  # Cache for endogenous weights per expiry
        self.default_weights = {
            'DELTA': 0.45,
            'GAMMA': 0.40,
            'VEGA': 0.30,
            'THETA': -0.35
        }
    
    def calculate_iv(self, option_price, K, T, otype='call', initial_guess=0.20):
        """Calculate Implied Volatility using Newton-Raphson method"""
        if option_price <= 0 or T <= 0 or K <= 0:
            return 0.20  # Default fallback
        
        sigma = initial_guess
        max_iterations = 100
        tolerance = 1e-5
        
        for i in range(max_iterations):
            try:
                # Calculate option price with current sigma
                greeks = self.bs_greeks(K, T, sigma, otype)
                price_diff = greeks['price'] - option_price
                
                # If close enough, return
                if abs(price_diff) < tolerance:
                    return sigma
                
                # Newton-Raphson: sigma_new = sigma - f(sigma)/f'(sigma)
                # f'(sigma) = vega
                vega = greeks['vega']
                if vega < 1e-10:  # Avoid division by zero
                    return sigma
                
                sigma = sigma - price_diff / (vega * 100)  # vega is per 1% move
                
                # Keep sigma in reasonable bounds
                sigma = max(0.01, min(sigma, 3.0))
                
            except:
                return 0.20  # Fallback on error
        
        return sigma
        
    def bs_greeks(self, K, T, sigma, otype='call'):
        """Calculate Black-Scholes Greeks"""
        S, r = self.spot, self.r
        if T <= 0 or sigma <= 0 or K <= 0:
            return {'price': 0, 'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0}
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if otype == 'call':
            price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            delta = norm.cdf(d1)
            theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2))/365
        else:
            price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
            delta = -norm.cdf(-d1)
            theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2))/365
        
        gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
        vega = S*norm.pdf(d1)*np.sqrt(T)/100
        
        return {'price': price, 'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta}
    
    def calculate_dte(self, expiry_str):
        """Calculate days to expiry from expiry date string"""
        try:
            expiry = datetime.strptime(expiry_str, '%d-%b-%Y')
            dte = (expiry - datetime.now()).days
            return max(1, dte)
        except:
            return 7  # Default to 7 days if parsing fails
    
    def enrich_greeks(self, df):
        """Calculate IV from premiums if not already present, then calculate Greeks"""
        enriched = []
        actual_iv_count = 0
        calculated_iv_count = 0
        
        for idx, row in df.iterrows():
            dte = self.calculate_dte(row['Expiry'])
            T = dte / 365
            otype = 'call' if row['Type'] == 'CE' else 'put'
            
            # Convert row to dict for easier manipulation
            row_dict = row.to_dict()
            
            # Use actual IV if available (from daily CSV), otherwise calculate from premium
            if 'IV' in row_dict and pd.notna(row_dict['IV']) and row_dict['IV'] > 0:
                # Already have actual IV from CSV file, use it
                sigma = row_dict['IV'] / 100
                actual_iv_count += 1
            elif 'Premium' in row_dict and row_dict['Premium'] > 0 and row_dict['Strike'] > 0 and T > 0:
                # Calculate IV from premium using Newton-Raphson
                calculated_iv = self.calculate_iv(row_dict['Premium'], row_dict['Strike'], T, otype)
                row_dict['IV'] = calculated_iv * 100  # Store as percentage
                sigma = calculated_iv
                calculated_iv_count += 1
            else:
                row_dict['IV'] = 20.0  # Fallback default
                sigma = 0.20
            
            # Ensure IV_Change_% exists
            if 'IV_Change_%' not in row_dict or pd.isna(row_dict['IV_Change_%']):
                row_dict['IV_Change_%'] = 0.0
            
            # Calculate Greeks using the IV (actual or calculated)
            if pd.isna(row_dict.get('Delta')) or pd.isna(row_dict.get('Gamma')):
                greeks = self.bs_greeks(row_dict['Strike'], T, sigma, otype)
                row_dict['Delta'] = greeks['delta']
                row_dict['Gamma'] = greeks['gamma']
                row_dict['Vega'] = greeks['vega']
                row_dict['Theta'] = greeks['theta']
            
            row_dict['DTE'] = dte
            enriched.append(row_dict)
        
        if actual_iv_count > 0:
            print(f"  ✓ {actual_iv_count} options with actual IV, {calculated_iv_count} with calculated IV")
        
        return pd.DataFrame(enriched)
    
    def normalize_greeks(self, greeks, iv, option_price):
        """
        Step 3: Normalize Greeks for comparison
        """
        delta_n = abs(greeks['Delta'])
        gamma_n = greeks['Gamma'] * self.spot
        vega_n = greeks['Vega'] / max(iv/100, 0.01)
        theta_n = abs(greeks['Theta']) / max(option_price, 0.01) if option_price > 0.01 else 0
        
        return {
            'delta_n': delta_n,
            'gamma_n': gamma_n,
            'vega_n': vega_n,
            'theta_n': theta_n
        }
    
    def calculate_endogenous_weights(self, df_history, expiry):
        """
        Calculate dynamic weights based on historical Greek behavior
        Uses 3-5 day rolling window with EMA smoothing
        """
        # Filter for specific expiry
        if 'Expiry' in df_history.columns:
            df_exp = df_history[df_history['Expiry'] == expiry].copy()
        else:
            df_exp = df_history.copy()
        
        # Need at least rolling_window days of data
        if 'DATE' in df_exp.columns or 'Date' in df_exp.columns:
            date_col = 'DATE' if 'DATE' in df_exp.columns else 'Date'
            unique_dates = df_exp[date_col].unique()
            
            if len(unique_dates) < self.rolling_window:
                return self.default_weights  # Not enough history
            
            # Get last rolling_window days
            recent_dates = sorted(unique_dates)[-self.rolling_window:]
            df_recent = df_exp[df_exp[date_col].isin(recent_dates)]
        else:
            df_recent = df_exp.tail(200)  # Fallback: last 200 records
        
        if len(df_recent) < 10:  # Need minimum data
            return self.default_weights
        
        # Calculate variance/volatility of each normalized Greek
        # Higher variance = more important for regime detection
        greek_vars = {}
        
        for _, row in df_recent.iterrows():
            if pd.isna(row.get('Delta')) or pd.isna(row.get('Premium')):
                continue
                
            normalized = self.normalize_greeks(
                {'Delta': row['Delta'], 'Gamma': row['Gamma'],
                 'Vega': row['Vega'], 'Theta': row['Theta']},
                row.get('IV', 20), row['Premium']
            )
            
            for greek in ['delta_n', 'gamma_n', 'vega_n', 'theta_n']:
                if greek not in greek_vars:
                    greek_vars[greek] = []
                greek_vars[greek].append(normalized[greek])
        
        # Calculate coefficient of variation (std/mean) for each Greek
        # This shows relative importance
        cv_scores = {}
        for greek, values in greek_vars.items():
            if len(values) > 0:
                mean_val = np.mean(np.abs(values))
                std_val = np.std(values)
                cv_scores[greek] = std_val / max(mean_val, 0.01)
        
        # Convert to weights (normalize so they sum to ~1.0)
        total_cv = sum(cv_scores.values())
        if total_cv > 0:
            raw_weights = {
                'DELTA': cv_scores.get('delta_n', 0.45) / total_cv,
                'GAMMA': cv_scores.get('gamma_n', 0.40) / total_cv,
                'VEGA': cv_scores.get('vega_n', 0.30) / total_cv,
                'THETA': -cv_scores.get('theta_n', 0.35) / total_cv  # Negative for theta
            }
            
            # Apply EMA smoothing with cached weights
            cache_key = f"{expiry}"
            if cache_key in self.weights_cache:
                old_weights = self.weights_cache[cache_key]
                # EMA: new_weight = alpha * new + (1-alpha) * old
                smoothed_weights = {
                    greek: self.ema_alpha * raw_weights[greek] + (1 - self.ema_alpha) * old_weights[greek]
                    for greek in raw_weights
                }
            else:
                smoothed_weights = raw_weights
            
            # Cache for next iteration
            self.weights_cache[cache_key] = smoothed_weights
            return smoothed_weights
        
        return self.default_weights
    
    def calculate_gds(self, normalized, weights=None):
        """
        Step 4: Greek Dominance Score (GDS) with Dynamic Weights
        Uses endogenous weights calculated from historical Greek behavior
        """
        if weights is None:
            weights = self.default_weights
        
        gds = {
            'DELTA': weights['DELTA'] * normalized['delta_n'],
            'GAMMA': weights['GAMMA'] * normalized['gamma_n'],
            'VEGA': weights['VEGA'] * normalized['vega_n'],
            'THETA': weights['THETA'] * normalized['theta_n']
        }
        
        dominant = max(gds, key=gds.get)
        dominant_value = gds[dominant]
        
        return {
            'gds_scores': gds,
            'dominant_greek': dominant,
            'dominant_value': dominant_value,
            'weights_used': weights  # Track which weights were used
        }
    
    def classify_regime(self, gds_scores, iv_change, spot_change, gamma_percentile):
        """Step 5: Market Regime Classification"""
        threshold = 0.15
        
        if gds_scores['GAMMA'] > threshold and abs(iv_change) < 2.0:
            regime = 'GAMMA-DRIVEN'
            conf = min(95, 70 + gamma_percentile/3)
        elif gds_scores['VEGA'] > threshold and iv_change > 1.2:
            regime = 'VEGA-DRIVEN'
            conf = min(95, 60 + abs(iv_change)*5)
        elif gds_scores['THETA'] > threshold and abs(spot_change) < 0.5:
            regime = 'THETA-DRIVEN'
            conf = min(95, 70 + (1-abs(spot_change))*20)
        else:
            regime = 'DELTA-DRIVEN'
            conf = min(95, 50 + abs(spot_change)*10)
        
        return {
            'regime': regime,
            'confidence': conf,
            'iv_change': iv_change,
            'spot_change': spot_change,
            'gamma_percentile': gamma_percentile
        }
    
    def process_chain(self, df, df_history=None):
        """Process option chain with GDS calculations using dynamic weights"""
        df = self.enrich_greeks(df)
        
        # Calculate dynamic weights per expiry if historical data available
        expiry_weights = {}
        if df_history is not None and len(df_history) > 0:
            unique_expiries = df['Expiry'].unique() if 'Expiry' in df.columns else [None]
            for expiry in unique_expiries:
                expiry_weights[expiry] = self.calculate_endogenous_weights(df_history, expiry)
        
        results = []
        for idx, row in df.iterrows():
            normalized = self.normalize_greeks(
                {'Delta': row['Delta'], 'Gamma': row['Gamma'], 
                 'Vega': row['Vega'], 'Theta': row['Theta']},
                row['IV'], row['Premium']
            )
            
            # Use expiry-specific weights if available
            expiry = row.get('Expiry', None)
            weights = expiry_weights.get(expiry, self.default_weights)
            
            gds = self.calculate_gds(normalized, weights)
            
            theta_hourly = (abs(row['Theta']) / row['Premium'] * 100) if row['Premium'] > 0.01 else 0
            
            results.append({
                **row.to_dict(),
                'Moneyness_%': ((row['Strike'] - self.spot) / self.spot) * 100,
                'Delta_n': normalized['delta_n'],
                'Gamma_n': normalized['gamma_n'],
                'Vega_n': normalized['vega_n'],
                'Theta_n': normalized['theta_n'],
                'GDS_DELTA': gds['gds_scores']['DELTA'],
                'GDS_GAMMA': gds['gds_scores']['GAMMA'],
                'GDS_VEGA': gds['gds_scores']['VEGA'],
                'GDS_THETA': gds['gds_scores']['THETA'],
                'Dominant': gds['dominant_greek'],
                'Dominant_Value': gds['dominant_value'],
                'Theta_Hourly_%': theta_hourly
            })
        
        return pd.DataFrame(results)


# Load historical data
try:
    nifty = pd.read_csv('nifty_history.csv')
    vix = pd.read_csv('india_vix_history.csv')
    SPOT_PREV = nifty['Close'].iloc[-2] if len(nifty) > 1 else 25800
    VIX_PREV = vix['Close'].iloc[-2] if len(vix) > 1 else 10.8
    print(f"✓ Historical data loaded")
except:
    SPOT_PREV, VIX_PREV = 25800, 10.8
    print(f"⚠️ Using default historical values")


app.layout = html.Div([
    html.Div([
        html.H1("🎯 GREEK REGIME FLIP MODEL", style={'color':'white','margin':0}),
        html.P("NIFTY Option Chain Analysis with Greeks-Based Regime Classification", 
               style={'color':'#ecf0f1','fontSize':16,'marginTop':5})
    ], style={'background':'linear-gradient(135deg, #667eea 0%, #764ba2 100%)','padding':25,'marginBottom':25}),
    
    html.Div([
        html.Div([
            html.Label('Capital (₹):', style={'fontWeight':'bold','display':'block','marginBottom':5}),
            dcc.Input(id='capital', type='number', value=500000, step=10000, 
                     min=10000, max=100000000,
                     style={'width':'100%','padding':8})
        ], style={'flex':1,'marginRight':15}),
        
        html.Div([
            html.Label('Select Expiry:', style={'fontWeight':'bold','display':'block','marginBottom':5}),
            dcc.Dropdown(id='expiry-select', options=[], value=None, 
                        style={'width':'100%'})
        ], style={'flex':2,'marginRight':15}),
        
        html.Div([
            html.Label('Live NIFTY Spot:', style={'fontWeight':'bold','display':'block','marginBottom':5,'color':'#059669'}),
            dcc.Input(id='live-nifty', type='number', value=None, placeholder='Enter live NIFTY (optional)',
                     step=0.01, min=0,
                     style={'width':'100%','padding':8,'border':'2px solid #10b981'})
        ], style={'flex':1,'marginRight':15}),
        
        html.Div([
            html.Label('Live India VIX:', style={'fontWeight':'bold','display':'block','marginBottom':5,'color':'#7c3aed'}),
            dcc.Input(id='live-vix', type='number', value=None, placeholder='Enter live VIX (optional)',
                     step=0.01, min=0,
                     style={'width':'100%','padding':8,'border':'2px solid #a855f7'})
        ], style={'flex':1,'marginRight':15}),
        
        html.Div([
            html.Label('Trading Date:', style={'fontWeight':'bold','display':'block','marginBottom':5,'color':'#f59e0b'}),
            dcc.Input(id='trading-date', type='text', value='19 Dec 2025', placeholder='19 Dec 2025',
                     style={'width':'100%','padding':8,'border':'2px solid #f59e0b'})
        ], style={'flex':1,'marginRight':15}),
        
        html.Div([
            html.Label('\u00a0', style={'display':'block','marginBottom':5}),
            html.Button('� LOAD OPTION DATA', id='fetch-btn', n_clicks=0,
                       style={'width':'100%','padding':10,'background':'#10b981','color':'white',
                             'border':'none','borderRadius':5,'fontWeight':'bold','fontSize':16,
                             'cursor':'pointer'})
        ], style={'flex':1})
    ], style={'display':'flex','padding':'20px 30px','background':'#f8fafc','borderRadius':10,'marginBottom':25}),
    
    html.Div(id='status-msg', style={'padding':'0 30px','marginBottom':15}),
    html.Div(id='regime-box', style={'marginBottom':25}),
    
    dcc.Store(id='chain-data'),
    dcc.Store(id='spot-data'),
    
    dcc.Tabs([
        dcc.Tab(label='📊 Entry Signals', children=[
            html.Div(id='entry-signals', style={'padding':25})
        ]),
        
        dcc.Tab(label='📈 GDS Analysis', children=[
            html.Div([
                dcc.Graph(id='gds-chart'),
                html.Div([
                    html.Div([dcc.Graph(id='greek-heatmap-ce')], style={'width':'48%','display':'inline-block'}),
                    html.Div([dcc.Graph(id='greek-heatmap-pe')], style={'width':'48%','display':'inline-block','marginLeft':'4%'})
                ])
            ], style={'padding':20})
        ]),
        
        dcc.Tab(label='📋 Full Chain', children=[
            html.Div(id='full-chain', style={'padding':25})
        ])
    ])
])


@app.callback(
    [Output('chain-data','data'),
     Output('spot-data','data'),
     Output('expiry-select','options'),
     Output('expiry-select','value'),
     Output('status-msg','children')],
    [Input('fetch-btn','n_clicks')]
)
def fetch_live_data(n):
    if n == 0:
        return None, None, [], None, None
    
    # Validate n_clicks is reasonable (prevent abuse)
    if n > 1000:
        return None, None, [], None, html.Div("❌ Too many requests. Please refresh page.", 
                                               style={'color':'#ef4444','fontWeight':'bold'})
    
    print(f"\n{'='*70}")
    print(f"FETCH BUTTON CLICKED - Loading data from parquet files")
    print(f"{'='*70}\n")
    
    # Load data from parquet files
    try:
        result = load_fallback_option_data()
        if result is None:
            error_msg = html.Div([
                html.H3("❌ No Data Available", style={'color':'#ef4444'}),
                html.P("Could not find option data files."),
                html.P("Make sure you have parquet files in the nifty_option_excel/ folder.", 
                       style={'fontStyle':'italic'})
            ], style={'color':'#ef4444','padding':'20px', 'backgroundColor':'#fee2e2', 
                     'borderRadius':'8px', 'border':'2px solid #ef4444'})
            return None, None, [], None, error_msg
        
        df, spot = result
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        error_msg = html.Div(f"❌ Error: {str(e)}", 
                            style={'color':'#ef4444','fontWeight':'bold','padding':'10px',
                                  'backgroundColor':'#fee2e2','borderRadius':'5px'})
        return None, None, [], None, error_msg
    
    if df is None or len(df) == 0:
        return None, None, [], None, html.Div("❌ No option chain data available", 
                                               style={'color':'#ef4444','fontWeight':'bold'})
    
    # Get unique expiries
    expiries = sorted(df['Expiry'].unique()) if 'Expiry' in df.columns else []
    expiry_options = [{'label': exp, 'value': exp} for exp in expiries]
    default_expiry = expiries[0] if expiries else None
    
    # Create status message
    status = html.Div(
        f"✅ Option Data Loaded: NIFTY ₹{spot:,.0f} | {len(df)} options | {len(expiries)} expiries", 
        style={'color':'#10b981','fontWeight':'bold','padding':'10px',
               'backgroundColor':'#d1fae5','borderRadius':'5px'}
    )
    
    print(f"✅ Data loaded successfully: {len(df)} options across {len(expiries)} expiries")
    
    return df.to_dict('records'), spot, expiry_options, default_expiry, status


@app.callback(
    [Output('regime-box','children'),
     Output('entry-signals','children'),
     Output('gds-chart','figure'),
     Output('greek-heatmap-ce','figure'),
     Output('greek-heatmap-pe','figure'),
     Output('full-chain','children')],
    [Input('expiry-select','value')],
    [State('chain-data','data'), State('spot-data','data'), State('capital','value'),
     State('live-nifty','value'), State('live-vix','value'), State('trading-date','value')]
)
def analyze(expiry, chain_data, spot, capital, live_nifty, live_vix, trading_date):
    # Use live inputs if provided
    if live_nifty is not None and live_nifty > 0:
        spot = live_nifty
        print(f"🔴 Using LIVE NIFTY input: ₹{spot:,.2f}")
    
    if live_vix is not None and live_vix > 0:
        global VIX_PREV
        VIX_PREV = live_vix
        print(f"🔴 Using LIVE VIX input: {live_vix:.2f}")
    
    # Input validation
    if chain_data is None or spot is None or expiry is None:
        empty = go.Figure()
        empty.update_layout(title="Fetch live data to begin")
        return None, None, empty, empty, empty, None
    
    # Validate capital
    try:
        validator = SecurityValidator()
        capital = validator.validate_numeric(capital, 10000, 100000000, "capital")
    except ValueError:
        capital = 500000  # Default fallback
    
    # Validate expiry string
    try:
        expiry = validator.validate_string(expiry, 20, '0123456789-AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpRrSsTtUuVvWwXxYyZz')
    except:
        empty = go.Figure()
        empty.update_layout(title="Invalid expiry format")
        return None, None, empty, empty, empty, None
    
    # Convert back to DataFrame
    try:
        df_all = pd.DataFrame(chain_data)
        if len(df_all) == 0 or len(df_all) > 10000:
            raise ValueError("Invalid data size")
    except Exception as e:
        empty = go.Figure()
        empty.update_layout(title=f"Data validation failed: {str(e)[:50]}")
        return None, None, empty, empty, empty, None
    df = df_all[df_all['Expiry'] == expiry].copy()
    
    if len(df) == 0:
        empty = go.Figure()
        empty.update_layout(title="No data for selected expiry")
        return None, None, empty, empty, empty, None
    
    # Try to load actual IV data from daily CSV file
    if trading_date:
        try:
            from datetime import datetime
            # Convert expiry to filename format (e.g., "30-Dec-2025")
            expiry_dt = pd.to_datetime(expiry)
            expiry_formatted = expiry_dt.strftime("%d-%b-%Y")
            
            # Load current day's data with actual IV
            df_actual_iv = load_daily_option_chain_csv(expiry_formatted, trading_date)
            
            if df_actual_iv is not None:
                print(f"✓ Using actual IV from daily CSV for {trading_date}")
                
                # Instead of merging, use CSV as primary data source
                # Keep columns from parquet for compatibility, add from CSV
                df = df.merge(
                    df_actual_iv[['Strike', 'Type', 'IV', 'Premium', 'VOLUME', 'OI', 'CHNG_IN_OI']],
                    on=['Strike', 'Type'],
                    how='inner',  # Only keep strikes that exist in both
                    suffixes=('_parquet', '')
                )
                
                # Drop parquet columns we replaced
                cols_to_drop = [c for c in df.columns if c.endswith('_parquet')]
                if cols_to_drop:
                    df = df.drop(columns=cols_to_drop)
                
                # Ensure IV_Change_% exists
                if 'IV_Change_%' not in df.columns:
                    df['IV_Change_%'] = 0.0
                    
                print(f"  ✓ Loaded {len(df)} options with actual IV (range: {df['IV'].min():.1f}%-{df['IV'].max():.1f}%)")
            else:
                print(f"⚠️ Daily CSV not found, will calculate IV from premiums")
                if 'IV_Change_%' not in df.columns:
                    df['IV_Change_%'] = 0.0
        except Exception as e:
            print(f"⚠️ Error loading daily IV data: {str(e)}")
            import traceback
            traceback.print_exc()
            if 'IV_Change_%' not in df.columns:
                df['IV_Change_%'] = 0.0
    else:
        # No trading date provided, ensure column exists
        if 'IV_Change_%' not in df.columns:
            df['IV_Change_%'] = 0.0
    
    # Initialize model
    model = GreekRegimeFlipModel(spot)
    
    # Pass historical data for dynamic weight calculation
    df = model.process_chain(df, df_history=df_all)
    
    # Calculate regime
    spot_change = ((spot - SPOT_PREV) / SPOT_PREV) * 100
    avg_iv = df['IV'].mean()
    iv_change = ((avg_iv - VIX_PREV) / VIX_PREV) * 100
    
    # Get ATM option
    atm_idx = df['Moneyness_%'].abs().idxmin()
    atm = df.loc[atm_idx]
    
    gamma_pct = 75  # Simplified
    
    normalized = model.normalize_greeks(
        {'Delta': atm['Delta'], 'Gamma': atm['Gamma'], 
         'Vega': atm['Vega'], 'Theta': atm['Theta']},
        avg_iv, atm['Premium']
    )
    
    # Calculate GDS with dynamic weights
    expiry_weights = model.calculate_endogenous_weights(df_all, expiry)
    gds = model.calculate_gds(normalized, expiry_weights)
    regime_data = model.classify_regime(gds['gds_scores'], iv_change, spot_change, gamma_pct)
    
    # Regime Box
    regime_colors = {
        'DELTA-DRIVEN': '#3b82f6',
        'GAMMA-DRIVEN': '#ef4444',
        'VEGA-DRIVEN': '#a855f7',
        'THETA-DRIVEN': '#10b981'
    }
    
    regime_icons = {
        'DELTA-DRIVEN': '📈',
        'GAMMA-DRIVEN': '⚡',
        'VEGA-DRIVEN': '🌊',
        'THETA-DRIVEN': '⏰'
    }
    
    regime = regime_data['regime']
    
    # Dynamic Weights Info Box
    weights_info = html.Div([
        html.H4("📊 Dynamic Weights (EMA Smoothed)", style={'color':'#1f2937','marginBottom':10}),
        html.Div([
            html.Div([
                html.Span("Δ Delta: ", style={'fontWeight':'bold'}),
                html.Span(f"{expiry_weights['DELTA']:.3f}")
            ], style={'marginBottom':5}),
            html.Div([
                html.Span("Γ Gamma: ", style={'fontWeight':'bold'}),
                html.Span(f"{expiry_weights['GAMMA']:.3f}")
            ], style={'marginBottom':5}),
            html.Div([
                html.Span("ν Vega: ", style={'fontWeight':'bold'}),
                html.Span(f"{expiry_weights['VEGA']:.3f}")
            ], style={'marginBottom':5}),
            html.Div([
                html.Span("Θ Theta: ", style={'fontWeight':'bold'}),
                html.Span(f"{expiry_weights['THETA']:.3f}")
            ])
        ], style={'fontSize':14,'color':'#4b5563'})
    ], style={'padding':15,'backgroundColor':'#f3f4f6','borderRadius':8,'marginBottom':20,'border':'2px solid #d1d5db'})
    
    # Create regime box first
    data_source_badge = html.Span(
        "🔴 LIVE DATA" if live_nifty is not None else "📁 HISTORICAL",
        style={'backgroundColor':'#10b981' if live_nifty is not None else '#f59e0b',
               'color':'white','padding':'5px 12px','borderRadius':'15px',
               'fontSize':12,'fontWeight':'bold','marginLeft':10}
    )
    
    regime_box = html.Div([
        html.Div([
            html.Div([
                html.H2(f"{regime_icons[regime]} {regime}", style={'color':'white','marginBottom':10,'display':'inline-block'}),
                data_source_badge
            ]),
            html.P(f"Confidence: {regime_data['confidence']:.1f}%", style={'color':'#ecf0f1','fontSize':20}),
            html.Div([
                html.P(f"NIFTY: ₹{spot:,.0f} ({spot_change:+.2f}%)", style={'color':'white','marginBottom':5}),
                html.P(f"Avg IV: {avg_iv:.2f}% ({iv_change:+.2f}%)", style={'color':'white','marginBottom':5}),
                html.P(f"Dominant: {gds['dominant_greek']}", style={'color':'#fbbf24','fontSize':18,'fontWeight':'bold'}),
                html.P(f"Expiry: {expiry}", style={'color':'#ecf0f1','fontSize':14})
            ])
        ], style={'padding':20,'background':regime_colors[regime],'borderRadius':10})
    ])
    
    # Combine weights info and regime box
    regime_with_weights = html.Div([
        weights_info,
        regime_box
    ], style={'padding':'0 30px'})
    
    # Entry Signals (filter ATM ±2%)
    entry_opts = df[df['Moneyness_%'].abs() <= 2.0].copy()
    entry_cards = []
    
    for idx, row in entry_opts.iterrows():
        if row['Volume'] > 0:  # Only liquid options
            signal_info = html.Div([
                html.H4(f"₹{row['Strike']} {row['Type']}", style={'marginBottom':10}),
                html.Div([
                    html.Div([
                        html.P('Premium', style={'color':'#64748b','fontSize':12}),
                        html.P(f"₹{row['Premium']:.2f}", style={'fontSize':16,'fontWeight':'bold'})
                    ], style={'flex':1}),
                    html.Div([
                        html.P('IV', style={'color':'#64748b','fontSize':12}),
                        html.P(f"{row['IV']:.1f}%", style={'fontSize':16,'fontWeight':'bold'})
                    ], style={'flex':1}),
                    html.Div([
                        html.P('Volume', style={'color':'#64748b','fontSize':12}),
                        html.P(f"{int(row['Volume']):,}", style={'fontSize':16,'fontWeight':'bold'})
                    ], style={'flex':1}),
                    html.Div([
                        html.P('Dominant', style={'color':'#64748b','fontSize':12}),
                        html.P(row['Dominant'], style={'fontSize':16,'fontWeight':'bold','color':'#f59e0b'})
                    ], style={'flex':1})
                ], style={'display':'flex','gap':15})
            ], style={'padding':15,'background':'#f8fafc','borderRadius':8,'marginBottom':15,'border':'2px solid #e2e8f0'})
            
            entry_cards.append(signal_info)
    
    if not entry_cards:
        entry_cards = [html.P("No liquid ATM options found", style={'textAlign':'center','color':'#64748b','padding':40})]
    
    entry_section = html.Div(entry_cards)
    
    # Charts
    ce = df[df['Type']=='CE'].sort_values('Strike')
    pe = df[df['Type']=='PE'].sort_values('Strike')
    
    fig_gds = make_subplots(rows=2, cols=2,
                           subplot_titles=('CE - GDS by Greek', 'PE - GDS by Greek',
                                         'CE - Premium', 'PE - Premium'),
                           vertical_spacing=0.15)
    
    for greek, color in [('GDS_DELTA','#3b82f6'), ('GDS_GAMMA','#ef4444'), 
                         ('GDS_VEGA','#a855f7'), ('GDS_THETA','#10b981')]:
        fig_gds.add_trace(go.Bar(x=ce['Strike'], y=ce[greek], name=greek.replace('GDS_',''),
                                marker_color=color, legendgroup='gds'), row=1, col=1)
        fig_gds.add_trace(go.Bar(x=pe['Strike'], y=pe[greek], name=greek.replace('GDS_',''),
                                marker_color=color, showlegend=False, legendgroup='gds'), row=1, col=2)
    
    fig_gds.add_trace(go.Scatter(x=ce['Strike'], y=ce['Premium'], mode='lines+markers',
                                name='Premium', line=dict(color='#10b981',width=3)), row=2, col=1)
    fig_gds.add_trace(go.Scatter(x=pe['Strike'], y=pe['Premium'], mode='lines+markers',
                                name='Premium', line=dict(color='#ef4444',width=3), showlegend=False), row=2, col=2)
    
    fig_gds.add_vline(x=spot, line_dash="dash", line_color="black", annotation_text="ATM")
    fig_gds.update_layout(height=700, title=f'Live Option Chain Analysis - {expiry}', barmode='stack')
    fig_gds.update_xaxes(title_text="Strike")
    fig_gds.update_yaxes(title_text="GDS")
    
    # Heatmaps
    def create_heatmap(data, title):
        cols = ['Delta_n', 'Gamma_n', 'Vega_n', 'Theta_n']
        z = data[cols].T.values
        return go.Figure(data=go.Heatmap(
            z=z, x=data['Strike'].values, y=cols,
            colorscale='RdYlGn', text=np.round(z, 3),
            texttemplate='%{text}', textfont={"size":9}
        )).update_layout(title=title, height=300, xaxis_title='Strike')
    
    hm_ce = create_heatmap(ce, 'CE - Normalized Greeks')
    hm_pe = create_heatmap(pe, 'PE - Normalized Greeks')
    
    # Full Chain
    display_cols = ['Strike','Type','Premium','IV','IV_Change_%','Delta','Gamma','Vega','Theta',
                   'Volume','OI','Dominant','Dominant_Value','Moneyness_%']
    
    df_display = df[display_cols].copy()
    for c in ['Premium','IV','IV_Change_%','Delta','Gamma','Vega','Theta','Dominant_Value','Moneyness_%']:
        if c in df_display.columns:
            df_display[c] = df_display[c].round(2)
    
    full_chain_table = dash_table.DataTable(
        data=df_display.to_dict('records'),
        columns=[{'name': c, 'id': c} for c in display_cols],
        style_cell={'textAlign':'center','padding':'10px','fontSize':12},
        style_header={'background':'#1e293b','color':'white','fontWeight':'bold'},
        style_data_conditional=[
            {'if':{'filter_query':'{Dominant}="DELTA"'},'backgroundColor':'#dbeafe'},
            {'if':{'filter_query':'{Dominant}="GAMMA"'},'backgroundColor':'#fee2e2'},
            {'if':{'filter_query':'{Dominant}="VEGA"'},'backgroundColor':'#f3e8ff'},
            {'if':{'filter_query':'{Dominant}="THETA"'},'backgroundColor':'#d1fae5'}
        ],
        filter_action='native',
        sort_action='native',
        page_size=30
    )
    
    return regime_with_weights, entry_section, fig_gds, hm_ce, hm_pe, full_chain_table


if __name__ == '__main__':
    print("="*70)
    print("GREEK REGIME FLIP MODEL - LIVE NSE DATA")
    print("Real-time NIFTY Option Chain with Greeks Analysis")
    print("="*70)
    print("Dashboard: http://127.0.0.1:8055")
    print("="*70)
    app.run(host='127.0.0.1', port=8055, debug=True)
