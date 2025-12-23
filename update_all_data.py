"""
Master script to update all market data to the latest available date
Run this daily to keep all data files current
"""

import subprocess
import sys
from datetime import datetime

def run_script(script_name, description):
    """Run a Python script and report results"""
    print("\n" + "="*70)
    print(f"Running: {description}")
    print("="*70)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"⚠️ Warning: {script_name} exited with code {result.returncode}")
            if result.stderr:
                print(f"Error output: {result.stderr}")
            return False
        
        return True
        
    except subprocess.TimeoutExpired:
        print(f"❌ Error: {script_name} timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False


def main():
    """Update all data sources"""
    print("\n" + "="*70)
    print("MASTER DATA UPDATER")
    print(f"   Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    results = {}
    
    # 1. Update NIFTY and VIX data
    results['NIFTY & VIX'] = run_script(
        'nse_data_fetcher.py',
        'Updating NIFTY 50 & India VIX Historical Data'
    )
    
    # 2. Update macro data
    results['Macro Data'] = run_script(
        'macro_data_fetcher.py',
        'Updating Macroeconomic Variables (US VIX, USD/INR, Crude, etc.)'
    )
    
    # Print summary
    print("\n" + "="*70)
    print("UPDATE SUMMARY")
    print("="*70)
    
    all_success = True
    for data_type, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {data_type:20s}: {status}")
        if not success:
            all_success = False
    
    print("="*70)
    
    if all_success:
        print("All data sources updated successfully!")
        print("\nNext steps:")
        print("   1. Run forecast_dashboard.py to see latest forecasts")
        print("   2. Run dashboard.py for options backtesting")
        print("   3. Check ML Strategy Selector for latest recommendations")
    else:
        print("⚠️ Some data sources failed to update. Check output above.")
    
    print("="*70)
    
    return all_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
