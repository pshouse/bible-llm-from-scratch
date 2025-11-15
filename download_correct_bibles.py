"""
Download correct Bible translations with verified Project Gutenberg IDs.
"""

import urllib.request
import os

data_dir = "bible_data"
os.makedirs(data_dir, exist_ok=True)

print("Downloading correct Bible translations...")

# Rename the incorrectly labeled ASV file (it's actually WEB)
asv_file = os.path.join(data_dir, "asv.txt")
if os.path.exists(asv_file):
    web_file = os.path.join(data_dir, "web.txt")
    os.rename(asv_file, web_file)
    print("[OK] Renamed asv.txt to web.txt (it was actually WEB)")

# Delete the incorrect YLT file (it's Drum-Taps by Walt Whitman)
ylt_file = os.path.join(data_dir, "ylt.txt")
if os.path.exists(ylt_file):
    os.remove(ylt_file)
    print("[OK] Removed incorrect ylt.txt file")

# Download Douay-Rheims Bible
print("\n1. Downloading Douay-Rheims Bible...")
douay_url = "https://www.gutenberg.org/cache/epub/1581/pg1581.txt"
douay_path = os.path.join(data_dir, "douay_rheims.txt")
try:
    urllib.request.urlretrieve(douay_url, douay_path)
    print(f"   [OK] Douay-Rheims downloaded to {douay_path}")
    size_kb = os.path.getsize(douay_path) / 1024
    print(f"   Size: {size_kb:.1f} KB")
except Exception as e:
    print(f"   [ERROR] Error downloading Douay-Rheims: {e}")

# Download Tyndale New Testament (historic)
print("\n2. Downloading Tyndale New Testament...")
tyndale_url = "https://www.gutenberg.org/cache/epub/10553/pg10553.txt"
tyndale_path = os.path.join(data_dir, "tyndale_nt.txt")
try:
    urllib.request.urlretrieve(tyndale_url, tyndale_path)
    print(f"   [OK] Tyndale NT downloaded to {tyndale_path}")
    size_kb = os.path.getsize(tyndale_path) / 1024
    print(f"   Size: {size_kb:.1f} KB")
except Exception as e:
    print(f"   [ERROR] Error downloading Tyndale: {e}")

# Try alternative source for additional Bible
print("\n3. Attempting additional downloads...")

# Check if we can download from other Project Gutenberg variants
# Webster's Bible
webster_url = "https://www.gutenberg.org/cache/epub/8294/pg8294.txt"
webster_alt_path = os.path.join(data_dir, "webster.txt")
if not os.path.exists(os.path.join(data_dir, "web.txt")):
    try:
        urllib.request.urlretrieve(webster_url, webster_alt_path)
        print(f"   [OK] Additional translation downloaded")
        size_kb = os.path.getsize(webster_alt_path) / 1024
        print(f"   Size: {size_kb:.1f} KB")
    except Exception as e:
        print(f"   [INFO] No additional downloads available")

print("\n" + "="*60)
print("Final Bible Collection Summary")
print("="*60)

total_size = 0
file_count = 0

for filename in sorted(os.listdir(data_dir)):
    if filename.endswith(".txt"):
        filepath = os.path.join(data_dir, filename)
        size_kb = os.path.getsize(filepath) / 1024
        total_size += size_kb
        file_count += 1
        print(f"  {filename}: {size_kb:.1f} KB")

print(f"\nTotal translations: {file_count}")
print(f"Total size: {total_size:.1f} KB ({total_size/1024:.1f} MB)")
print(f"\nAll Bible texts saved in: {data_dir}/")
