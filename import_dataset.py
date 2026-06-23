import os
import sys
import subprocess
import zipfile
import shutil
import urllib.request

# Ensure terminal streams support UTF-8 or handle encoding errors gracefully on Windows
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        try:
            stream.reconfigure(encoding='utf-8')
        except Exception:
            try:
                stream.reconfigure(errors='replace')
            except Exception:
                pass

# 1. Ensure required packages are installed
required_packages = ["pandas", "tqdm"]
missing_packages = []
for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        missing_packages.append(pkg)

if missing_packages:
    print(f"Installing missing packages: {', '.join(missing_packages)}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        # Refresh sys.path and importlib cache so that new packages are immediately importable
        import site
        import importlib
        importlib.invalidate_caches()
        if hasattr(site, "getusersitepackages"):
            user_site = site.getusersitepackages()
            if user_site not in sys.path:
                site.addsitedir(user_site)
    except Exception as e:
        print(f"\n Error installing packages: {e}")
        print(f"\n Please install them manually: pip install {' '.join(missing_packages)}")
        sys.exit(1)

import pandas as pd
from tqdm import tqdm

class TqdmUpTo(tqdm):
    """Provides progress bar updates for urllib retrieves."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(base_dir, "individual_household_electric_power_consumption.zip")
    txt_path = os.path.join(base_dir, "household_power_consumption.txt")
    data_dir = os.path.join(base_dir, "data")
    
    url = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
    
    # 2. Cleanup existing data folder if it exists
    if os.path.exists(data_dir):
        print(f"\n🧹 Removing existing data directory: {data_dir}")
        shutil.rmtree(data_dir)

    # 3. Download the ZIP file directly
    print(f"\n⬇️ Downloading dataset from: {url}")
    try:
        with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading") as t:
            urllib.request.urlretrieve(url, zip_path, reporthook=t.update_to)
        print("✓ Download complete")
    except Exception as e:
        print(f"\n❌ Failed to download dataset: {e}")
        sys.exit(1)

    # 4. Extract the TXT file from ZIP
    print(f"\n📤 Extracting dataset file...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # We look for the txt file in the zip
            txt_filename = None
            for name in zip_ref.namelist():
                if name.endswith('.txt'):
                    txt_filename = name
                    break
            
            if not txt_filename:
                raise FileNotFoundError("Could not find .txt file inside the ZIP archive.")
                
            # Extract only the txt file
            zip_ref.extract(txt_filename, path=base_dir)
            # If it extracted to a different name, make sure we know the exact path
            txt_path = os.path.join(base_dir, txt_filename)
        print("✓ Extraction complete")
    except Exception as e:
        print(f"\n❌ Failed to extract zip file: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        sys.exit(1)

    # 5. Preprocess and aggregate dataset
    print("\n🛠️ Preprocessing and aggregating dataset to daily level (this may take a moment)...")
    try:
        # Load the raw semicolon-separated text file
        df = pd.read_csv(txt_path, sep=';', low_memory=False)
        
        # Convert active power to numeric (coercing invalid entries like '?' to NaN)
        df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
        
        # Parse dates from raw DD/MM/YYYY format
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        
        # Group by Date and sum the power consumption
        daily = df.groupby('Date')['Global_active_power'].sum().reset_index()
        daily.columns = ['date', 'power_consumption']
        
        # Apply identical feature engineering matching the original dataset
        daily['year'] = daily['date'].dt.year
        daily['semester'] = (daily['date'].dt.month > 6).astype(int) + 1
        daily['quarter'] = daily['date'].dt.quarter
        daily['day_in_week'] = daily['date'].dt.day_name().str[:3]
        daily['week_in_year'] = (daily['date'].dt.dayofyear - 1) // 7 + 1
        daily['day_in_year'] = daily['date'].dt.dayofyear
        daily['month'] = daily['date'].dt.month
        
        # Format date as M/D/YYYY string to match the original layout
        daily['date_str'] = daily['date'].apply(lambda dt: f"{dt.month}/{dt.day}/{dt.year}")
        
        # Select and order columns
        cols = ['date_str', 'power_consumption', 'year', 'semester', 'quarter', 'day_in_week', 'week_in_year', 'day_in_year', 'month']
        daily_final = daily[cols].rename(columns={'date_str': 'date'})
        
        # Replicate train/test date splits
        train_df = daily_final[daily['date'] <= '2010-03-31']
        test_df = daily_final[(daily['date'] >= '2010-04-01') & (daily['date'] <= '2010-11-26')]
        
        # Create target directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Save splits
        train_path = os.path.join(data_dir, 'df_train.csv')
        test_path = os.path.join(data_dir, 'df_test.csv')
        
        print(f"💾 Saving training set ({len(train_df)} rows) to: {train_path}")
        train_df.to_csv(train_path, index=False)
        
        print(f"💾 Saving testing set ({len(test_df)} rows) to: {test_path}")
        test_df.to_csv(test_path, index=False)
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        sys.exit(1)
    finally:
        # Cleanup temporary downloaded/extracted files
        print("🧹 Cleaning up temporary files...")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        if os.path.exists(txt_path):
            os.remove(txt_path)
        print("✓ Cleanup complete")
        
    print("✅ Dataset auto-import completed successfully!")

if __name__ == "__main__":
    main()
