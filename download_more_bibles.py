"""
Download additional public domain Bible translations.
"""

import io
import json
import os
import urllib.request
import zipfile

data_dir = "bible_data"
os.makedirs(data_dir, exist_ok=True)


def download_text(url, destination, description):
    print(f"\nDownloading {description}...")
    try:
        urllib.request.urlretrieve(url, destination)
        size_kb = os.path.getsize(destination) / 1024
        print(f"   [OK] Saved to {destination} ({size_kb:.1f} KB)")
    except Exception as exc:
        print(f"   [ERROR] Could not download {description}: {exc}")


def download_bbe(destination):
    print("\nDownloading Bible in Basic English (BBE)...")
    url = "https://eBible.org/Scriptures/engBBE_readaloud.zip"
    try:
        data = urllib.request.urlopen(url).read()
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            chapter_files = [name for name in archive.namelist() if name.endswith("_read.txt")]
            chapter_files.sort()

            with open(destination, "w", encoding="utf-8") as outfile:
                for name in chapter_files:
                    with archive.open(name) as handle:
                        text = handle.read().decode("utf-8").strip()
                        if text:
                            outfile.write(text)
                            outfile.write("\n\n")

        size_kb = os.path.getsize(destination) / 1024
        print(f"   [OK] Processed BBE chapters into {destination} ({size_kb:.1f} KB)")
    except Exception as exc:
        print(f"   [ERROR] Could not process BBE: {exc}")


print("Downloading additional Bible translations...")

# Existing WEB download (JSON aggregation) retained for completeness
print("\nDownloading World English Bible (WEB) JSON corpus...")
try:
    base_url = "https://raw.githubusercontent.com/TehShrike/world-english-bible/master/"
    books_url = base_url + "books.json"
    with urllib.request.urlopen(books_url) as response:
        books_data = json.loads(response.read().decode("utf-8"))

    web_text = []
    book_count = 0

    for book_info in books_data:
        book_id = book_info["id"]
        book_name = book_info["name"]
        try:
            book_url = base_url + f"{book_id}.json"
            with urllib.request.urlopen(book_url) as response:
                book_data = json.loads(response.read().decode("utf-8"))

            web_text.append(f"\n\n{'='*60}\n{book_name}\n{'='*60}\n\n")
            for chapter in book_data.get("chapters", []):
                chapter_num = chapter.get("chapter", 1)
                web_text.append(f"\nChapter {chapter_num}\n")
                for verse in chapter.get("verses", []):
                    verse_num = verse.get("verse", 1)
                    verse_text = verse.get("text", "")
                    web_text.append(f"{verse_num}. {verse_text}\n")
            book_count += 1
            print(f"   Downloaded: {book_name}")
        except Exception as exc:
            print(f"   [ERROR] Could not download {book_name}: {exc}")

    web_path = os.path.join(data_dir, "web.txt")
    with open(web_path, "w", encoding="utf-8") as handle:
        handle.write("".join(web_text))

    size_kb = os.path.getsize(web_path) / 1024
    print(f"\n   [OK] WEB combined into {web_path} ({size_kb:.1f} KB)")
except Exception as exc:
    print(f"   [WARN] Failed to download WEB JSON corpus: {exc}")
    download_text(
        "https://openbible.com/textfiles/web.txt",
        os.path.join(data_dir, "web.txt"),
        "World English Bible (WEB)"
    )


openbible_sources = {
    "asv.txt": ("https://openbible.com/textfiles/asv.txt", "American Standard Version (ASV)"),
    "ylt.txt": ("https://openbible.com/textfiles/ylt.txt", "Young's Literal Translation (YLT)"),
    "darby.txt": ("https://openbible.com/textfiles/dbt.txt", "Darby Bible Translation"),
    "webster.txt": ("https://openbible.com/textfiles/wbt.txt", "Webster's Bible Translation"),
}

for filename, (source_url, label) in openbible_sources.items():
    destination_path = os.path.join(data_dir, filename)
    download_text(source_url, destination_path, label)

download_bbe(os.path.join(data_dir, "bbe.txt"))

print("\n" + "=" * 60)
print("Download Summary")
print("=" * 60)

for filename in sorted(os.listdir(data_dir)):
    if filename.endswith(".txt"):
        filepath = os.path.join(data_dir, filename)
        size_kb = os.path.getsize(filepath) / 1024
        print(f"  {filename}: {size_kb:.1f} KB")

print(f"\nAll Bible texts in: {data_dir}/")
