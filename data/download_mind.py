"""
MIND-Small Dataset Downloader
Downloads official Microsoft News Dataset (MIND-Small) for training and validation.
"""

import os
import zipfile
import urllib.request
import sys

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

URLS = {
    "train": "https://mind201910.blob.core.windows.net/mind/MINDsmall_train.zip",
    "dev": "https://mind201910.blob.core.windows.net/mind/MINDsmall_dev.zip"
}

def download_and_extract(split_name, url):
    dest_folder = os.path.join(DATA_DIR, split_name)
    zip_path = os.path.join(DATA_DIR, f"MINDsmall_{split_name}.zip")
    
    os.makedirs(dest_folder, exist_ok=True)
    
    # Check if files already extracted
    news_file = os.path.join(dest_folder, "news.tsv")
    behaviors_file = os.path.join(dest_folder, "behaviors.tsv")
    if os.path.exists(news_file) and os.path.exists(behaviors_file):
        print(f"[{split_name.upper()}] Files already exist in {dest_folder}. Skipping download.")
        return

    print(f"[{split_name.upper()}] Downloading from {url}...")
    
    def reporthook(blocknum, blocksize, totalsize):
        read = blocknum * blocksize
        if totalsize > 0:
            percent = min(100, read * 100 // totalsize)
            sys.stdout.write(f"\rDownloading {split_name}: {percent}% ({read // (1024*1024)} MB / {totalsize // (1024*1024)} MB)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, zip_path, reporthook=reporthook)
        print(f"\n[{split_name.upper()}] Extracting to {dest_folder}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_folder)
        os.remove(zip_path)
        print(f"[{split_name.upper()}] Extraction completed successfully!")
    except Exception as e:
        print(f"\n[ERROR] Failed to download {split_name}: {e}")
        print("Note: If network access is blocked, run 'python data/sample_data.py' to generate a demo dataset instantly.")

if __name__ == "__main__":
    print("=" * 60)
    print("MIND-Small Dataset Downloader")
    print("=" * 60)
    for split, split_url in URLS.items():
        download_and_extract(split, split_url)
    print("\nDataset preparation finished!")

