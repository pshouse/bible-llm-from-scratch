"""
Download additional Bible translations from GitHub and other sources.
"""

import urllib.request
import os
import json

data_dir = "bible_data"
os.makedirs(data_dir, exist_ok=True)

print("Downloading additional Bible translations...")

# World English Bible from GitHub (TehShrike repo - JSON format)
print("\n1. Downloading World English Bible from GitHub...")
try:
    # This repo has the WEB in JSON format, we'll download all books and combine them
    base_url = "https://raw.githubusercontent.com/TehShrike/world-english-bible/master/"

    # Get the list of books from the repo structure
    # We'll download a few key files to build the complete text
    books_url = base_url + "books.json"

    print("   Downloading books index...")
    with urllib.request.urlopen(books_url) as response:
        books_data = json.loads(response.read().decode('utf-8'))

    web_text = []
    book_count = 0

    for book_info in books_data:
        book_id = book_info['id']
        book_name = book_info['name']

        try:
            # Download each book's JSON file
            book_url = base_url + f"{book_id}.json"
            with urllib.request.urlopen(book_url) as response:
                book_data = json.loads(response.read().decode('utf-8'))

            # Extract text from the book
            web_text.append(f"\n\n{'='*60}\n{book_name}\n{'='*60}\n\n")

            for chapter in book_data.get('chapters', []):
                chapter_num = chapter.get('chapter', 1)
                web_text.append(f"\nChapter {chapter_num}\n")

                for verse in chapter.get('verses', []):
                    verse_num = verse.get('verse', 1)
                    verse_text = verse.get('text', '')
                    web_text.append(f"{verse_num}. {verse_text}\n")

            book_count += 1
            print(f"   Downloaded: {book_name}")

        except Exception as e:
            print(f"   [ERROR] Could not download {book_name}: {e}")

    # Save the combined text
    web_path = os.path.join(data_dir, "web.txt")
    with open(web_path, 'w', encoding='utf-8') as f:
        f.write(''.join(web_text))

    size_kb = os.path.getsize(web_path) / 1024
    print(f"\n   [OK] WEB downloaded ({book_count} books) to {web_path}")
    print(f"   Size: {size_kb:.1f} KB")

except Exception as e:
    print(f"   [ERROR] Failed to download WEB: {e}")

# Try to download Berean Standard Bible (released into public domain in 2023)
print("\n2. Attempting to download Berean Standard Bible...")
print("   Note: May not be available as plain text file")

# Download list
print("\n" + "="*60)
print("Download Summary")
print("="*60)

for filename in sorted(os.listdir(data_dir)):
    if filename.endswith(".txt"):
        filepath = os.path.join(data_dir, filename)
        size_kb = os.path.getsize(filepath) / 1024
        print(f"  {filename}: {size_kb:.1f} KB")

print(f"\nAll Bible texts in: {data_dir}/")
