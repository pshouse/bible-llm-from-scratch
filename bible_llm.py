"""
Bible LLM - Training a Language Model on Public Domain English Bible Translations

This script adapts the LLM architecture from the_verdict.py to train on
a corpus of public domain English Bible translations.

Corpus: ~3 million words from KJV, WEB, Douay-Rheims, and Tyndale translations
"""

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
import os

print("="*60)
print("BIBLE LLM - Language Model Training on Biblical Texts")
print("="*60)

# Configuration
CORPUS_FILE = "bible_corpus/bible_combined.txt"
BATCH_SIZE = 8
MAX_LENGTH = 256  # Context window size
STRIDE = 128      # Overlap between sequences
EMBEDDING_DIM = 256
NUM_EPOCHS = 1    # Start with 1 epoch for initial testing

# Load the Bible corpus
print(f"\nLoading Bible corpus from: {CORPUS_FILE}")
if not os.path.exists(CORPUS_FILE):
    print(f"ERROR: Corpus file not found. Please run prepare_bible_corpus.py first.")
    exit(1)

with open(CORPUS_FILE, "r", encoding="utf-8") as f:
    bible_text = f.read()

print(f"  Total characters: {len(bible_text):,}")
print(f"  Total words: {len(bible_text.split()):,}")
print(f"  Size: {len(bible_text)/1024/1024:.2f} MB")

# Initialize tokenizer (GPT-2 tokenizer)
print("\nInitializing tokenizer...")
tokenizer = tiktoken.get_encoding("gpt2")

# Tokenize the entire corpus
print("Tokenizing corpus...")
enc_text = tokenizer.encode(bible_text)
print(f"  Total tokens: {len(enc_text):,}")

# Dataset class (same as GPTDatasetV1)
class BibleDataset(Dataset):
    def __init__(self, token_ids, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# Create dataset
print(f"\nCreating dataset...")
print(f"  Context window (max_length): {MAX_LENGTH}")
print(f"  Stride: {STRIDE}")

dataset = BibleDataset(enc_text, MAX_LENGTH, STRIDE)
print(f"  Total training examples: {len(dataset):,}")

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=0
)
print(f"  Batches per epoch: {len(dataloader):,}")

# Get a sample batch to test
print("\nTesting data pipeline...")
data_iter = iter(dataloader)
sample_inputs, sample_targets = next(data_iter)
print(f"  Sample batch shape: {sample_inputs.shape}")
print(f"  Input IDs shape: {sample_inputs.shape}")
print(f"  Target IDs shape: {sample_targets.shape}")

# Decode a sample to verify
sample_text = tokenizer.decode(sample_inputs[0].tolist())
print(f"\n  Sample input text (first 200 chars):")
print(f"  {sample_text[:200]}...")

# Create embeddings
print("\nInitializing embedding layers...")
vocab_size = tokenizer.n_vocab
print(f"  Vocabulary size: {vocab_size:,}")
print(f"  Embedding dimension: {EMBEDDING_DIM}")

# Token embeddings
token_embedding_layer = torch.nn.Embedding(vocab_size, EMBEDDING_DIM)

# Positional embeddings
context_length = MAX_LENGTH
pos_embedding_layer = torch.nn.Embedding(context_length, EMBEDDING_DIM)

# Test embeddings on sample batch
print("\nTesting embeddings...")
token_embeddings = token_embedding_layer(sample_inputs)
pos_embeddings = pos_embedding_layer(torch.arange(MAX_LENGTH))
input_embeddings = token_embeddings + pos_embeddings

print(f"  Token embeddings shape: {token_embeddings.shape}")
print(f"  Position embeddings shape: {pos_embeddings.shape}")
print(f"  Combined embeddings shape: {input_embeddings.shape}")
print(f"  Expected shape: (batch_size={BATCH_SIZE}, max_length={MAX_LENGTH}, embedding_dim={EMBEDDING_DIM})")

# Multi-Head Attention (from the_verdict.py)
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec

# Test Multi-Head Attention
print("\nTesting Multi-Head Attention...")
d_in = EMBEDDING_DIM
d_out = EMBEDDING_DIM
num_heads = 4
dropout_rate = 0.1

torch.manual_seed(123)
mha = MultiHeadAttention(d_in, d_out, context_length, dropout_rate, num_heads)
context_vecs = mha(input_embeddings)
print(f"  Multi-head attention output shape: {context_vecs.shape}")
print(f"  Expected: (batch_size={BATCH_SIZE}, max_length={MAX_LENGTH}, d_out={d_out})")

print("\n" + "="*60)
print("BIBLE LLM INITIALIZATION COMPLETE")
print("="*60)
print(f"\nDataset Statistics:")
print(f"  Total tokens: {len(enc_text):,}")
print(f"  Training examples: {len(dataset):,}")
print(f"  Batches: {len(dataloader):,}")
print(f"  Context window: {MAX_LENGTH} tokens")
print(f"  Batch size: {BATCH_SIZE}")
print(f"\nModel Configuration:")
print(f"  Vocabulary size: {vocab_size:,}")
print(f"  Embedding dimension: {EMBEDDING_DIM}")
print(f"  Number of attention heads: {num_heads}")
print(f"  Dropout rate: {dropout_rate}")

print("\n" + "="*60)
print("Ready for model training!")
print("="*60)
print("\nNext steps:")
print("  1. Define full transformer model architecture")
print("  2. Define loss function (CrossEntropyLoss)")
print("  3. Define optimizer (AdamW)")
print("  4. Implement training loop")
print("  5. Train the model")
print("  6. Generate text samples from trained model")
