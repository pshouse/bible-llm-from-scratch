"""
Download public domain English Bible translations for LLM training.

This script downloads multiple public domain Bible translations and saves them
as text files for use in training a language model.
"""

import urllib.request
import os
import zipfile
import io

# Create a data directory for Bible texts
data_dir = "bible_data"
os.makedirs(data_dir, exist_ok=True)

print("Downloading public domain English Bible translations...")

# 1. King James Version (KJV) from Project Gutenberg
print("\n1. Downloading King James Version (KJV)...")
kjv_url = "https://www.gutenberg.org/cache/epub/10/pg10.txt"
kjv_path = os.path.join(data_dir, "kjv.txt")
try:
    urllib.request.urlretrieve(kjv_url, kjv_path)
    print(f"   [OK] KJV downloaded to {kjv_path}")
    # Get file size
    size_kb = os.path.getsize(kjv_path) / 1024
    print(f"   Size: {size_kb:.1f} KB")
except Exception as e:
    print(f"   [ERROR] Error downloading KJV: {e}")

# 2. World English Bible (WEB) from eBible.org
print("\n2. Downloading World English Bible (WEB)...")
web_url = "https://ebible.org/Scriptures/eng-web_readaloud.txt"
web_path = os.path.join(data_dir, "web.txt")
try:
    urllib.request.urlretrieve(web_url, web_path)
    print(f"   [OK] WEB downloaded to {web_path}")
    size_kb = os.path.getsize(web_path) / 1024
    print(f"   Size: {size_kb:.1f} KB")
except Exception as e:
    print(f"   [ERROR] Error downloading WEB: {e}")
    # Try alternative URL
    print("   Trying alternative URL...")
    try:
        web_url_alt = "https://ebible.org/Scriptures/engwebp_readaloud.txt"
        urllib.request.urlretrieve(web_url_alt, web_path)
        print(f"   [OK] WEB downloaded to {web_path}")
        size_kb = os.path.getsize(web_path) / 1024
        print(f"   Size: {size_kb:.1f} KB")
    except Exception as e2:
        print(f"   [ERROR] Error with alternative URL: {e2}")

# 3. American Standard Version (ASV) from Project Gutenberg
print("\n3. Downloading American Standard Version (ASV)...")
asv_url = "https://www.gutenberg.org/cache/epub/8294/pg8294.txt"
asv_path = os.path.join(data_dir, "asv.txt")
try:
    urllib.request.urlretrieve(asv_url, asv_path)
    print(f"   [OK] ASV downloaded to {asv_path}")
    size_kb = os.path.getsize(asv_path) / 1024
    print(f"   Size: {size_kb:.1f} KB")
except Exception as e:
    print(f"   [ERROR] Error downloading ASV: {e}")

# 4. Young's Literal Translation (YLT) from Project Gutenberg
print("\n4. Downloading Young's Literal Translation (YLT)...")
ylt_url = "https://www.gutenberg.org/cache/epub/8801/pg8801.txt"
ylt_path = os.path.join(data_dir, "ylt.txt")
try:
    urllib.request.urlretrieve(ylt_url, ylt_path)
    print(f"   [OK] YLT downloaded to {ylt_path}")
    size_kb = os.path.getsize(ylt_path) / 1024
    print(f"   Size: {size_kb:.1f} KB")
except Exception as e:
    print(f"   [ERROR] Error downloading YLT: {e}")

# 5. Darby Bible from Project Gutenberg
print("\n5. Downloading Darby Bible...")
darby_url = "https://www.gutenberg.org/cache/epub/8765/pg8765.txt"
darby_path = os.path.join(data_dir, "darby.txt")
try:
    urllib.request.urlretrieve(darby_url, darby_path)
    print(f"   [OK] Darby Bible downloaded to {darby_path}")
    size_kb = os.path.getsize(darby_path) / 1024
    print(f"   Size: {size_kb:.1f} KB")
except Exception as e:
    print(f"   [ERROR] Error downloading Darby Bible: {e}")

# 6. Webster Bible from Project Gutenberg
print("\n6. Downloading Webster Bible...")
webster_url = "https://www.gutenberg.org/cache/epub/8812/pg8812.txt"
webster_path = os.path.join(data_dir, "webster.txt")
try:
    urllib.request.urlretrieve(webster_url, webster_path)
    print(f"   [OK] Webster Bible downloaded to {webster_path}")
    size_kb = os.path.getsize(webster_path) / 1024
    print(f"   Size: {size_kb:.1f} KB")
except Exception as e:
    print(f"   [ERROR] Error downloading Webster Bible: {e}")

print("\n" + "="*60)
print("Download Summary")
print("="*60)

# List all downloaded files
downloaded_files = []
total_size = 0

for filename in os.listdir(data_dir):
    if filename.endswith(".txt"):
        filepath = os.path.join(data_dir, filename)
        size_kb = os.path.getsize(filepath) / 1024
        total_size += size_kb
        downloaded_files.append((filename, size_kb))
        print(f"  {filename}: {size_kb:.1f} KB")

print(f"\nTotal files: {len(downloaded_files)}")
print(f"Total size: {total_size:.1f} KB ({total_size/1024:.1f} MB)")
print(f"\nAll Bible texts saved in: {data_dir}/")
