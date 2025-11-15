"""
Test Bible corpus tokenization without PyTorch dependencies.
This verifies the data pipeline is working correctly.
"""

import tiktoken
import os

print("="*60)
print("BIBLE CORPUS TOKENIZATION TEST")
print("="*60)

# Load the Bible corpus
CORPUS_FILE = "bible_corpus/bible_combined.txt"
print(f"\nLoading Bible corpus from: {CORPUS_FILE}")

with open(CORPUS_FILE, "r", encoding="utf-8") as f:
    bible_text = f.read()

print(f"  Total characters: {len(bible_text):,}")
print(f"  Total words: {len(bible_text.split()):,}")
print(f"  Size: {len(bible_text)/1024/1024:.2f} MB")

# Initialize tokenizer
print("\nInitializing GPT-2 tokenizer...")
tokenizer = tiktoken.get_encoding("gpt2")
print(f"  Vocabulary size: {tokenizer.n_vocab:,}")

# Tokenize the corpus
print("\nTokenizing corpus...")
enc_text = tokenizer.encode(bible_text)
print(f"  Total tokens: {len(enc_text):,}")

# Calculate tokens per word ratio
tokens_per_word = len(enc_text) / len(bible_text.split())
print(f"  Tokens per word: {tokens_per_word:.2f}")

# Test encoding/decoding with a sample
print("\nTesting encoding/decoding...")
sample_text = bible_text[10000:10500]  # Get a 500 character sample
sample_encoded = tokenizer.encode(sample_text)
sample_decoded = tokenizer.decode(sample_encoded)

print(f"  Original sample (first 200 chars):")
print(f"  {sample_text[:200]}...")
print(f"\n  Encoded to {len(sample_encoded)} tokens")
print(f"  Decoded sample (first 200 chars):")
print(f"  {sample_decoded[:200]}...")

# Simulate creating training examples
MAX_LENGTH = 256
STRIDE = 128

print(f"\nSimulating training data creation...")
print(f"  Context window: {MAX_LENGTH} tokens")
print(f"  Stride: {STRIDE} tokens")

num_examples = (len(enc_text) - MAX_LENGTH) // STRIDE
print(f"  Total training examples: {num_examples:,}")

# Calculate batch information
BATCH_SIZE = 8
num_batches = num_examples // BATCH_SIZE
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Batches per epoch: {num_batches:,}")

# Show some example training pairs
print(f"\nExample training sequences:")
for i in range(3):
    start_idx = i * 1000
    end_idx = start_idx + 10
    input_seq = enc_text[start_idx:end_idx]
    target_seq = enc_text[start_idx + 1:end_idx + 1]

    print(f"\n  Example {i+1}:")
    print(f"    Input tokens:  {input_seq}")
    print(f"    Target tokens: {target_seq}")
    print(f"    Input text:  {tokenizer.decode(input_seq)[:80]}...")

print("\n" + "="*60)
print("TOKENIZATION TEST COMPLETE")
print("="*60)
print(f"\nCorpus Statistics:")
print(f"  Words: {len(bible_text.split()):,}")
print(f"  Tokens: {len(enc_text):,}")
print(f"  Training examples: {num_examples:,}")
print(f"  Batches: {num_batches:,}")
print("\nThe Bible corpus is ready for LLM training!")
print("="*60)
