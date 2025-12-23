"""Convert Excel files to Parquet format for faster loading"""
import pandas as pd
import glob
import os
from tqdm import tqdm

def convert_excel_to_parquet():
    """Convert all Excel files to Parquet format"""
    excel_files = sorted(glob.glob('nifty_option_excel/NIFTY_options_*.xlsx'))
    
    if not excel_files:
        print("No Excel files found")
        return
    
    print(f"Converting {len(excel_files)} Excel files to Parquet...")
    
    for file in tqdm(excel_files, desc="Converting"):
        try:
            # Read Excel
            df = pd.read_excel(file, engine='openpyxl')
            
            # Create parquet filename
            parquet_file = file.replace('.xlsx', '.parquet')
            
            # Save as parquet (much faster to read)
            df.to_parquet(parquet_file, index=False, compression='snappy')
            
            file_size_excel = os.path.getsize(file) / (1024*1024)
            file_size_parquet = os.path.getsize(parquet_file) / (1024*1024)
            
            print(f"✓ {os.path.basename(file)}: {file_size_excel:.1f}MB → {file_size_parquet:.1f}MB")
            
        except Exception as e:
            print(f"⚠️ Error converting {file}: {e}")
    
    print("\n✓ Conversion complete!")
    print("Parquet files are 5-10x faster to load than Excel")

if __name__ == '__main__':
    convert_excel_to_parquet()
