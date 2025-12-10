#!/usr/bin/env python3
"""
Download King County Assessor Data (Optional - for data updates)

NOTE: The compressed data files (EXTR_ResBldg.csv.gz and EXTR_Parcel.csv.gz)
are already included in the repository. You only need this script if you want
to download fresh data from King County's open data portal.

This script downloads the required CSV files for the V4.1 address lookup feature.

Required files:
- EXTR_ResBldg.csv (~147MB) - Residential building details
- EXTR_Parcel.csv (~60MB) - Parcel/lot information

Usage:
    python scripts/download_kc_data.py

The data will be downloaded to: data/king_county/
"""

import os
import sys
import zipfile
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: requests library required. Install with: pip install requests")
    sys.exit(1)


# King County Open Data Portal URLs
# These are the official assessment data exports
KC_DATA_URL = "https://aqua.kingcounty.gov/extranet/assessor/Real%20Property%20Sales.zip"

# Direct download links for specific files (more reliable)
EXTR_FILES = {
    "EXTR_ResBldg.csv": "https://aqua.kingcounty.gov/extranet/assessor/Residential%20Building.zip",
    "EXTR_Parcel.csv": "https://aqua.kingcounty.gov/extranet/assessor/Parcel.zip",
}

TARGET_DIR = Path("references/King_County_Assessment_data_ALL")


def download_file(url: str, dest_path: Path) -> bool:
    """Download a file with progress indicator."""
    print(f"Downloading: {url}")
    print(f"  -> {dest_path}")
    
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    print(f"\r  Progress: {pct:.1f}% ({downloaded // 1024 // 1024}MB)", end="")
        
        print(f"\n  Done: {dest_path.stat().st_size // 1024 // 1024}MB")
        return True
        
    except requests.RequestException as e:
        print(f"\n  Error: {e}")
        return False


def extract_zip(zip_path: Path, target_dir: Path) -> bool:
    """Extract a zip file to the target directory."""
    print(f"Extracting: {zip_path}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(target_dir)
        print(f"  Extracted to: {target_dir}")
        return True
    except zipfile.BadZipFile as e:
        print(f"  Error: {e}")
        return False


def main():
    print("=" * 60)
    print("King County Assessor Data Downloader")
    print("For V4.1 Address Lookup Feature")
    print("=" * 60)
    print()
    
    # Create target directory
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {TARGET_DIR.absolute()}")
    print()
    
    # Check if files already exist
    existing = []
    for filename in EXTR_FILES.keys():
        if (TARGET_DIR / filename).exists():
            existing.append(filename)
    
    if existing:
        print(f"Already downloaded: {', '.join(existing)}")
        response = input("Re-download? (y/N): ").strip().lower()
        if response != 'y':
            print("Keeping existing files.")
            if len(existing) == len(EXTR_FILES):
                print("\nAll required files present. Ready to use!")
                return
    
    print()
    print("Downloading required files from King County Open Data Portal...")
    print("This may take a few minutes depending on your connection.")
    print()
    
    success = True
    for filename, url in EXTR_FILES.items():
        zip_path = TARGET_DIR / f"{filename}.zip"
        csv_path = TARGET_DIR / filename
        
        # Download the zip
        if not download_file(url, zip_path):
            print(f"Failed to download {filename}")
            success = False
            continue
        
        # Extract the zip
        if not extract_zip(zip_path, TARGET_DIR):
            print(f"Failed to extract {filename}")
            success = False
            continue
        
        # Clean up zip
        zip_path.unlink()
        
        # Verify CSV exists
        if csv_path.exists():
            print(f"  Verified: {csv_path.name} ({csv_path.stat().st_size // 1024 // 1024}MB)")
        else:
            # Check for extracted file with different name
            extracted = list(TARGET_DIR.glob("*.csv"))
            print(f"  Note: Extracted files: {[f.name for f in extracted]}")
    
    print()
    if success:
        print("=" * 60)
        print("Download complete!")
        print()
        print("You can now use the address lookup feature:")
        print()
        print('  python scripts/compare_by_address.py "1523 15th Ave S" Seattle 98144')
        print()
        print("=" * 60)
    else:
        print("Some downloads failed. Please try again or download manually from:")
        print("  https://info.kingcounty.gov/assessor/DataDownload/default.aspx")


if __name__ == "__main__":
    main()
