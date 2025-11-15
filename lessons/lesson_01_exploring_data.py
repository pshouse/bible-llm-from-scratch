"""
LESSON 1: Exploring the Bible Corpus
======================================

In this lesson, you'll learn:
- What our training data looks like
- How to load and inspect text data
- Basic text statistics
- Why data exploration matters

Time: 15-20 minutes
"""

print("="*60)
print("LESSON 1: Exploring the Bible Corpus")
print("="*60)

# ============================================================================
# CONCEPT: What is a Corpus?
# ============================================================================
print("\n📚 CONCEPT: What is a Corpus?")
print("-" * 60)
print("""
A "corpus" is a large collection of text used for training language models.
Our corpus contains 4 public domain English Bible translations:
- King James Version (KJV) - 1611
- World English Bible (WEB) - Modern
- Douay-Rheims - Catholic translation
- Tyndale New Testament - First printed English NT

Think of it as the "textbook" our AI will study to learn English!
""")

input("Press Enter to continue...")

# ============================================================================
# STEP 1: Load the Corpus
# ============================================================================
print("\n📖 STEP 1: Loading the Corpus")
print("-" * 60)

import os

# Check if corpus exists
corpus_file = "bible_corpus/bible_combined.txt"

if not os.path.exists(corpus_file):
    print("❌ Corpus not found!")
    print("   Please run: uv run python prepare_bible_corpus.py")
    exit(1)

print("Loading corpus from:", corpus_file)

# Load the text
with open(corpus_file, "r", encoding="utf-8") as f:
    corpus_text = f.read()

print("✅ Corpus loaded successfully!")

input("\nPress Enter to see the statistics...")

# ============================================================================
# STEP 2: Basic Statistics
# ============================================================================
print("\n📊 STEP 2: Corpus Statistics")
print("-" * 60)

# Calculate statistics
total_chars = len(corpus_text)
total_words = len(corpus_text.split())
total_lines = len(corpus_text.split('\n'))

# Find unique words (case-insensitive)
words = corpus_text.lower().split()
unique_words = len(set(words))

print(f"Total characters: {total_chars:,}")
print(f"Total words:      {total_words:,}")
print(f"Total lines:      {total_lines:,}")
print(f"Unique words:     {unique_words:,}")
print(f"Size:             {total_chars / 1024 / 1024:.2f} MB")

print("\n💡 What this means:")
print("   - More text = more for the model to learn from")
print("   - Unique words = vocabulary the model will understand")
print(f"   - Average words per line: {total_words / total_lines:.1f}")

input("\nPress Enter to see a sample of the text...")

# ============================================================================
# STEP 3: Inspecting the Text
# ============================================================================
print("\n🔍 STEP 3: Sample Text")
print("-" * 60)

# Show a sample from the beginning (after the header)
sample_start = 5000  # Skip the header
sample_length = 500

sample = corpus_text[sample_start:sample_start + sample_length]
print(sample)
print("\n... (showing 500 characters)")

input("\nPress Enter to see word frequency analysis...")

# ============================================================================
# STEP 4: Word Frequency Analysis
# ============================================================================
print("\n📈 STEP 4: Most Common Words")
print("-" * 60)

from collections import Counter

# Count word frequencies
word_counts = Counter(words)
most_common = word_counts.most_common(20)

print("Top 20 most frequent words:\n")
for i, (word, count) in enumerate(most_common, 1):
    percentage = (count / total_words) * 100
    print(f"{i:2d}. '{word:12s}' - {count:6,} times ({percentage:.2f}%)")

print("\n💡 Observations:")
print("   - Common words like 'the', 'and', 'of' appear frequently")
print("   - These are called 'stop words' in NLP")
print("   - The model will learn these patterns")

input("\nPress Enter to continue...")

# ============================================================================
# STEP 5: Understanding Why This Matters
# ============================================================================
print("\n🎯 STEP 5: Why Data Exploration Matters")
print("-" * 60)
print("""
Before training any model, we need to understand our data:

1. DATA SIZE: Our 3M words is enough for a small model, but large
   models like GPT-4 use trillions of words!

2. VOCABULARY: Our {unique_words:,} unique words will become the model's
   vocabulary (with tokenization, we'll reduce this further).

3. PATTERNS: By seeing the most common words, we know what the
   model will learn first.

4. QUALITY: Exploring the text helps us spot issues like encoding
   problems, formatting errors, or unexpected content.
""".format(unique_words=unique_words))

input("\nPress Enter to continue...")

# ============================================================================
# EXPERIMENT: Try It Yourself!
# ============================================================================
print("\n🧪 EXPERIMENT TIME!")
print("-" * 60)
print("""
Now it's your turn! Try these experiments:

1. Change `sample_start` to 50000 and see different text
2. Find the top 50 words instead of 20
3. Calculate the average word length
4. Find how many times "God" or "Lord" appears

Modify this file and run it again!
""")

# ============================================================================
# QUIZ: Test Your Understanding
# ============================================================================
print("\n❓ QUIZ: Test Your Understanding")
print("-" * 60)
print("""
Answer these questions (think about them, answers below):

Q1: Why do we need a large corpus for training a language model?
Q2: What are "unique words" and why do they matter?
Q3: Why are common words like "the" important for the model?

Answers:
A1: A larger corpus provides more examples for the model to learn
    language patterns, grammar, and context.

A2: Unique words represent the vocabulary. The model needs to see
    each word multiple times to learn how to use it correctly.

A3: Common words appear in many contexts, helping the model learn
    sentence structure and how words relate to each other.
""")

# ============================================================================
# SUMMARY & NEXT STEPS
# ============================================================================
print("\n" + "="*60)
print("✅ LESSON 1 COMPLETE!")
print("="*60)
print("""
What you learned:
- How to load and inspect a text corpus
- Calculate basic statistics about text data
- Analyze word frequency
- Why data exploration is crucial before training

Next Lesson: Tokenization - Converting Text to Numbers
""")
print("="*60)

print("\nRun: uv run python lessons/lesson_02_tokenization.py")
