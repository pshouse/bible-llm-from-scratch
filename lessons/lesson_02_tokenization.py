"""
LESSON 2: Tokenization - Converting Text to Numbers
===================================================

In this lesson, you'll learn:
- Why computers need numbers instead of text
- Different tokenization strategies
- How GPT-2 tokenization works
- Building training examples

Time: 20-25 minutes
"""

print("="*60)
print("LESSON 2: Tokenization")
print("="*60)

# ============================================================================
# CONCEPT: Why Tokenization?
# ============================================================================
print("\n🤔 CONCEPT: Why Can't Computers Read Text Directly?")
print("-" * 60)
print("""
Neural networks are mathematical models - they only understand numbers!

To train on text, we need to:
1. Break text into small pieces (tokens)
2. Assign each unique token a number (token ID)
3. Convert text → numbers → feed to model
4. Convert numbers → text ← get model output

Example:
  Text:     "In the beginning"
  Tokens:   ["In", " the", " beginning"]
  Token IDs: [818, 262, 3726]

Think of it like translating English to math language!
""")

input("Press Enter to continue...")

# ============================================================================
# STEP 1: Simple Word Tokenization
# ============================================================================
print("\n📝 STEP 1: Simple Word Tokenization")
print("-" * 60)

text = "In the beginning God created the heaven and the earth."
print(f"Original text: {text}")
print()

# Method 1: Split by spaces
simple_tokens = text.split()
print("Method 1 - Split by spaces:")
print(f"Tokens: {simple_tokens}")
print(f"Number of tokens: {len(simple_tokens)}")
print()

print("⚠️  Problem: Punctuation is attached to words!")
print("   'earth.' is different from 'earth'")

input("\nPress Enter to see a better method...")

# Method 2: Using regex to handle punctuation
import re

better_tokens = re.findall(r'\w+|[^\w\s]', text)
print("\nMethod 2 - Regex to separate punctuation:")
print(f"Tokens: {better_tokens}")
print(f"Number of tokens: {len(better_tokens)}")
print()

print("✅ Better! Now 'earth' and '.' are separate tokens")

input("\nPress Enter to see the real challenge...")

# ============================================================================
# STEP 2: The Vocabulary Problem
# ============================================================================
print("\n🎯 STEP 2: The Vocabulary Problem")
print("-" * 60)

# Load a sample from our corpus
import os
corpus_file = "../bible_corpus/bible_combined.txt"

if os.path.exists(corpus_file):
    with open(corpus_file, "r", encoding="utf-8") as f:
        sample_text = f.read()[:10000]  # First 10,000 characters

    words = re.findall(r'\w+', sample_text.lower())
    unique_words = set(words)

    print(f"In just 10,000 characters:")
    print(f"  - Total words: {len(words):,}")
    print(f"  - Unique words: {len(unique_words):,}")
    print()

    print("💡 The Problem:")
    print("   Our full corpus has ~3 million words")
    print("   Word-based tokenization would create a HUGE vocabulary")
    print("   Larger vocabulary = more parameters = slower training")
    print()
    print("   We need a smarter approach!")

input("\nPress Enter to learn about BPE tokenization...")

# ============================================================================
# STEP 3: Byte Pair Encoding (BPE) - The GPT-2 Way
# ============================================================================
print("\n🚀 STEP 3: BPE Tokenization (What GPT-2 Uses)")
print("-" * 60)
print("""
Byte Pair Encoding (BPE) is smarter:

Instead of whole words, it uses:
- Common character combinations
- Whole words if frequent enough
- Individual characters for rare words

Example:
  Word "beginning" might tokenize as:
  - "begin" + "ning" (if these are common subwords)
  - Or "be" + "gin" + "ning"

Benefits:
✅ Smaller vocabulary (50,257 tokens for GPT-2 vs millions of words)
✅ Handles unknown words by breaking them into known parts
✅ More efficient training
""")

input("\nPress Enter to see it in action...")

# ============================================================================
# STEP 4: Using tiktoken (GPT-2's Tokenizer)
# ============================================================================
print("\n⚙️  STEP 4: GPT-2 Tokenizer in Action")
print("-" * 60)

import tiktoken

# Initialize GPT-2 tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
print(f"Loaded GPT-2 tokenizer")
print(f"Vocabulary size: {tokenizer.n_vocab:,} tokens")
print()

# Tokenize some examples
examples = [
    "In the beginning",
    "God created the heaven and the earth",
    "supercalifragilisticexpialidocious",  # Unknown word
]

print("Let's tokenize some examples:\n")

for text in examples:
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)

    print(f"Text:      '{text}'")
    print(f"Token IDs: {tokens}")
    print(f"# tokens:  {len(tokens)}")
    print(f"Decoded:   '{decoded}'")
    print()

print("💡 Notice:")
print("   - Common phrases use fewer tokens")
print("   - Rare words are split into multiple tokens")
print("   - Decode(encode(text)) returns the original text!")

input("\nPress Enter to tokenize our full corpus...")

# ============================================================================
# STEP 5: Tokenizing the Full Corpus
# ============================================================================
print("\n📊 STEP 5: Tokenizing the Bible Corpus")
print("-" * 60)

if os.path.exists(corpus_file):
    print("Loading full corpus...")
    with open(corpus_file, "r", encoding="utf-8") as f:
        full_text = f.read()

    print(f"Corpus size: {len(full_text):,} characters")
    print()
    print("Tokenizing... (this may take a moment)")

    # Tokenize the entire corpus
    token_ids = tokenizer.encode(full_text)

    print("✅ Tokenization complete!")
    print()
    print(f"Results:")
    print(f"  - Total tokens: {len(token_ids):,}")
    print(f"  - Characters:   {len(full_text):,}")
    print(f"  - Ratio:        {len(full_text) / len(token_ids):.2f} chars/token")
    print()

    # Show token distribution
    print("Token frequency analysis:")
    from collections import Counter
    token_counts = Counter(token_ids)
    most_common = token_counts.most_common(10)

    print("\nTop 10 most frequent tokens:")
    for i, (token_id, count) in enumerate(most_common, 1):
        token_text = tokenizer.decode([token_id])
        percentage = (count / len(token_ids)) * 100
        print(f"{i:2d}. ID {token_id:5d} '{token_text:10s}' - {count:6,} times ({percentage:.2f}%)")

input("\nPress Enter to learn about creating training examples...")

# ============================================================================
# STEP 6: Creating Training Examples
# ============================================================================
print("\n🎓 STEP 6: From Tokens to Training Examples")
print("-" * 60)
print("""
Language models learn to predict the NEXT token given previous tokens.

We create training examples using a "sliding window":

Example with window size 4:
  Tokens: [1, 2, 3, 4, 5, 6, 7, 8, 9]

  Training example 1:
    Input:  [1, 2, 3, 4]  →  Target: [2, 3, 4, 5]

  Training example 2:
    Input:  [2, 3, 4, 5]  →  Target: [3, 4, 5, 6]

  Training example 3:
    Input:  [3, 4, 5, 6]  →  Target: [4, 5, 6, 7]

The model learns: "Given [1,2,3,4], predict [2,3,4,5]"
""")

input("\nPress Enter to see this in action...")

# Demo with actual Bible text
if os.path.exists(corpus_file):
    print("\n📖 Real Example from Bible Corpus:")
    print("-" * 60)

    # Get a small sample
    sample_tokens = token_ids[1000:1010]

    print("Token IDs:", sample_tokens)
    print("Text:", tokenizer.decode(sample_tokens))
    print()

    # Create training examples
    window_size = 4
    print(f"Creating training examples (window size = {window_size}):\n")

    for i in range(len(sample_tokens) - window_size):
        input_seq = sample_tokens[i:i + window_size]
        target_seq = sample_tokens[i + 1:i + window_size + 1]

        input_text = tokenizer.decode(input_seq)
        target_text = tokenizer.decode(target_seq)

        print(f"Example {i + 1}:")
        print(f"  Input:  {input_seq} → '{input_text}'")
        print(f"  Target: {target_seq} → '{target_text}'")
        print()

input("\nPress Enter to continue...")

# ============================================================================
# STEP 7: Calculate Dataset Size
# ============================================================================
print("\n📈 STEP 7: How Much Training Data Do We Have?")
print("-" * 60)

if os.path.exists(corpus_file):
    context_window = 256  # Typical for small models
    stride = 128  # 50% overlap

    num_examples = (len(token_ids) - context_window) // stride

    print(f"Configuration:")
    print(f"  - Context window: {context_window} tokens")
    print(f"  - Stride: {stride} tokens")
    print(f"  - Total tokens: {len(token_ids):,}")
    print()
    print(f"Training examples: {num_examples:,}")
    print()

    batch_size = 8
    num_batches = num_examples // batch_size
    print(f"With batch size {batch_size}:")
    print(f"  - Batches per epoch: {num_batches:,}")
    print(f"  - This is how many training steps per epoch!")

input("\nPress Enter to continue...")

# ============================================================================
# EXPERIMENT: Try It Yourself!
# ============================================================================
print("\n🧪 EXPERIMENT TIME!")
print("-" * 60)
print("""
Try these experiments:

1. Tokenize your own sentences:
   text = "Your sentence here"
   tokens = tokenizer.encode(text)
   print(tokens)
   print(tokenizer.decode(tokens))

2. Try different window sizes (4, 8, 16)
   See how it affects the number of training examples

3. Find the token ID for the word "God"
   Compare it to "god" (lowercase)

Add your code below and re-run!
""")

# Your experiments here:


input("\nPress Enter for the quiz...")

# ============================================================================
# QUIZ: Test Your Understanding
# ============================================================================
print("\n❓ QUIZ: Test Your Understanding")
print("-" * 60)
print("""
Q1: Why do we use BPE instead of word-based tokenization?
Q2: What is the difference between input and target in training?
Q3: Why do we use a stride when creating training examples?

Think about your answers, then press Enter for solutions...
""")
input()

print("""
Answers:

A1: BPE creates a smaller, more manageable vocabulary while handling
    unknown words by breaking them into known subword pieces.

A2: The input is what we give to the model, the target is what we
    want it to predict. The target is shifted by one token.

A3: Stride controls overlap between examples. A stride of 128 with
    window 256 means 50% overlap, giving us more training examples
    and helping the model see contexts in different positions.
""")

input("\nPress Enter to finish...")

# ============================================================================
# SUMMARY & NEXT STEPS
# ============================================================================
print("\n" + "="*60)
print("✅ LESSON 2 COMPLETE!")
print("="*60)
print("""
What you learned:
- Why we need to convert text to numbers
- Different tokenization strategies
- How BPE (Byte Pair Encoding) works
- Creating training examples with sliding windows
- Calculating dataset size

Key Concepts:
- Token: A piece of text (word, subword, or character)
- Token ID: The number representing a token
- Context Window: How many tokens the model sees at once
- Stride: Overlap between training examples

Next Lesson: Embeddings - Representing Words as Vectors
""")
print("="*60)

print("\nRun: uv run python lessons/lesson_03_embeddings.py")
