"""
Prepare Bible corpus for LLM training by cleaning and combining translations.

This script:
1. Removes Project Gutenberg headers and footers
2. Cleans up formatting
3. Combines all Bible translations into a single training corpus
4. Creates individual cleaned versions
"""

import os
import re

data_dir = "bible_data"
output_dir = "bible_corpus"
os.makedirs(output_dir, exist_ok=True)

def clean_gutenberg_text(text):
    """Remove Project Gutenberg headers and footers."""
    # Find the start of actual content
    start_patterns = [
        r'\*\*\* START OF .* PROJECT GUTENBERG .*\*\*\*',
        r'\*\*\*START OF .* PROJECT GUTENBERG .*\*\*\*',
    ]

    for pattern in start_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            text = text[match.end():]
            break

    # Find the end of actual content
    end_patterns = [
        r'\*\*\* END OF .* PROJECT GUTENBERG .*\*\*\*',
        r'\*\*\*END OF .* PROJECT GUTENBERG .*\*\*\*',
    ]

    for pattern in end_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            text = text[:match.start()]
            break

    return text.strip()

def clean_bible_text(text):
    """Clean and normalize Bible text."""
    # Remove BOM if present
    text = text.replace('\ufeff', '')

    # Normalize line endings
    text = text.replace('\r\n', '\n')

    # Remove excessive blank lines (more than 2 consecutive)
    text = re.sub(r'\n{4,}', '\n\n\n', text)

    return text.strip()

print("Preparing Bible corpus for LLM training...")
print("="*60)

combined_corpus = []
stats = {}

# Process each Bible translation
bible_files = {
    'kjv.txt': 'King James Version (KJV)',
    'web.txt': 'World English Bible (WEB)',
    'douay_rheims.txt': 'Douay-Rheims Bible',
    'tyndale_nt.txt': 'Tyndale New Testament',
    'asv.txt': 'American Standard Version (ASV)',
    'ylt.txt': "Young's Literal Translation (YLT)",
    'darby.txt': 'Darby Bible Translation',
    'webster.txt': "Webster's Bible Translation",
    'bbe.txt': 'Bible in Basic English (BBE)'
}

for filename, description in bible_files.items():
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        print(f"[SKIP] {description}: File not found")
        continue

    print(f"\nProcessing {description}...")

    # Read the file
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # Clean the text
    cleaned_text = clean_gutenberg_text(raw_text)
    cleaned_text = clean_bible_text(cleaned_text)

    # Save individual cleaned version
    clean_filename = filename.replace('.txt', '_cleaned.txt')
    clean_filepath = os.path.join(output_dir, clean_filename)
    with open(clean_filepath, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)

    # Add to combined corpus with separator
    combined_corpus.append(f"\n\n{'='*60}\n")
    combined_corpus.append(f"{description}\n")
    combined_corpus.append(f"{'='*60}\n\n")
    combined_corpus.append(cleaned_text)
    combined_corpus.append("\n\n")

    # Calculate stats
    char_count = len(cleaned_text)
    word_count = len(cleaned_text.split())
    line_count = len(cleaned_text.split('\n'))

    stats[description] = {
        'chars': char_count,
        'words': word_count,
        'lines': line_count,
        'size_kb': char_count / 1024
    }

    print(f"  Characters: {char_count:,}")
    print(f"  Words: {word_count:,}")
    print(f"  Lines: {line_count:,}")
    print(f"  Size: {char_count/1024:.1f} KB")

# Save combined corpus
print(f"\n{'='*60}")
print("Creating combined corpus...")
combined_text = ''.join(combined_corpus)
combined_filepath = os.path.join(output_dir, 'bible_combined.txt')

with open(combined_filepath, 'w', encoding='utf-8') as f:
    f.write(combined_text)

total_chars = len(combined_text)
total_words = len(combined_text.split())
total_lines = len(combined_text.split('\n'))

print(f"  Combined file: {combined_filepath}")
print(f"  Total characters: {total_chars:,}")
print(f"  Total words: {total_words:,}")
print(f"  Total lines: {total_lines:,}")
print(f"  Total size: {total_chars/1024:.1f} KB ({total_chars/1024/1024:.1f} MB)")

# Summary
print(f"\n{'='*60}")
print("SUMMARY")
print("="*60)
print(f"\nBible Translations Processed: {len(stats)}")
for desc, stat in stats.items():
    print(f"\n{desc}:")
    print(f"  {stat['words']:,} words | {stat['size_kb']:.1f} KB")

print(f"\nOutput directory: {output_dir}/")
print(f"\nFiles created:")
print(f"  - bible_combined.txt (all translations combined)")
for filename in bible_files.keys():
    clean_filename = filename.replace('.txt', '_cleaned.txt')
    if os.path.exists(os.path.join(output_dir, clean_filename)):
        print(f"  - {clean_filename}")

print(f"\n{'='*60}")
print("Bible corpus ready for LLM training!")
print("="*60)
