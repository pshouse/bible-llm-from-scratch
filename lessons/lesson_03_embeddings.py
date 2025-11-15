"""
LESSON 3: Embeddings - Representing Words as Vectors
====================================================

In this lesson, you'll learn:
- Why we need embeddings (token IDs aren't enough!)
- What vectors are and how they represent meaning
- Token embeddings vs positional embeddings
- How to combine them for model input

Time: 20-25 minutes
"""

print("="*60)
print("LESSON 3: Embeddings")
print("="*60)

# ============================================================================
# CONCEPT: The Problem with Token IDs
# ============================================================================
print("\n🤔 CONCEPT: Why Aren't Token IDs Enough?")
print("-" * 60)
print("""
Remember from Lesson 2: We convert text to token IDs
  "In the beginning" → [818, 262, 3726]

But there's a problem! Token IDs are just labels.

  ID 818 ("In")  vs  ID 819 (some other token)

The difference is just 1, but these might be completely unrelated words!
The model can't learn relationships from arbitrary numbers.

What we need: A way to represent MEANING, not just labels.
""")

input("Press Enter to see the solution...")

# ============================================================================
# STEP 1: Understanding Vectors
# ============================================================================
print("\n📊 STEP 1: Enter the Vector!")
print("-" * 60)
print("""
A vector is just a list of numbers, like coordinates on a map.

Instead of:  Token ID = 818

We use:      Token Vector = [0.2, -0.5, 0.8, 0.1, ...]

Example with 4 dimensions:
  "king"    → [0.5, 0.7, 0.1, -0.3]
  "queen"   → [0.5, 0.6, 0.1,  0.8]  ← Similar to "king"!
  "apple"   → [-0.2, 0.1, 0.9,  0.0] ← Very different!

Notice: "king" and "queen" have similar numbers - they're close in meaning!
This is how the model understands relationships.
""")

input("Press Enter to create embeddings...")

# ============================================================================
# STEP 2: Creating Token Embeddings
# ============================================================================
print("\n⚙️  STEP 2: Token Embeddings in PyTorch")
print("-" * 60)

import torch
import torch.nn as nn

# Configuration
vocab_size = 50257  # GPT-2 vocabulary size
embedding_dim = 256  # How many numbers represent each token

print(f"Configuration:")
print(f"  Vocabulary size: {vocab_size:,} tokens")
print(f"  Embedding dimension: {embedding_dim}")
print()

# Create embedding layer
torch.manual_seed(123)  # For reproducibility
token_embedding = nn.Embedding(vocab_size, embedding_dim)

print(f"Created embedding layer!")
print(f"  Shape: ({vocab_size:,} x {embedding_dim})")
print(f"  This is a lookup table with {vocab_size:,} vectors")
print()

# Example: Convert token IDs to embeddings
token_ids = torch.tensor([818, 262, 3726])  # "In the beginning"
print(f"Token IDs: {token_ids.tolist()}")

embeddings = token_embedding(token_ids)
print(f"Embeddings shape: {embeddings.shape}")
print(f"  ({len(token_ids)} tokens, each as a {embedding_dim}-dimensional vector)")
print()

print("First token's embedding (first 10 values):")
print(embeddings[0, :10])
print()

print("💡 Key Insight:")
print("   Each token ID → unique {}-dimensional vector".format(embedding_dim))
print("   The model will LEARN these vectors during training!")
print("   Initially random, but they'll capture meaning over time.")

input("\nPress Enter to understand positional embeddings...")

# ============================================================================
# STEP 3: The Position Problem
# ============================================================================
print("\n📍 STEP 3: The Position Problem")
print("-" * 60)
print("""
Consider these sentences:
  1. "The cat sat on the mat"
  2. "The mat sat on the cat"

Same words, different meanings! Position matters!

But our token embeddings don't know about position.
The token "cat" has the same embedding regardless of where it appears.

Solution: POSITIONAL EMBEDDINGS
  - Each position (0, 1, 2, 3, ...) gets its own vector
  - We ADD positional embeddings to token embeddings
  - Now the model knows both WHAT the token is AND WHERE it is!
""")

input("Press Enter to create positional embeddings...")

# ============================================================================
# STEP 4: Creating Positional Embeddings
# ============================================================================
print("\n🎯 STEP 4: Positional Embeddings")
print("-" * 60)

context_length = 256  # Maximum sequence length
pos_embedding = nn.Embedding(context_length, embedding_dim)

print(f"Created positional embedding layer!")
print(f"  Max positions: {context_length}")
print(f"  Embedding dimension: {embedding_dim}")
print()

# Create position indices
positions = torch.arange(len(token_ids))
print(f"Positions: {positions.tolist()}")

# Get positional embeddings
pos_embeddings = pos_embedding(positions)
print(f"Positional embeddings shape: {pos_embeddings.shape}")
print()

print("Position 0 embedding (first 10 values):")
print(pos_embeddings[0, :10])
print()
print("Position 1 embedding (first 10 values):")
print(pos_embeddings[1, :10])
print()

print("💡 Notice: Each position has different values!")

input("\nPress Enter to combine them...")

# ============================================================================
# STEP 5: Combining Token + Positional Embeddings
# ============================================================================
print("\n➕ STEP 5: The Magic Combination")
print("-" * 60)

# Combine them (simple addition!)
combined_embeddings = embeddings + pos_embeddings

print("Token embeddings shape:      ", embeddings.shape)
print("Positional embeddings shape: ", pos_embeddings.shape)
print("Combined embeddings shape:   ", combined_embeddings.shape)
print()

print("Example - First token:")
print(f"  Token embedding (first 5):    {embeddings[0, :5]}")
print(f"  + Position embedding (first 5): {pos_embeddings[0, :5]}")
print(f"  = Combined (first 5):         {combined_embeddings[0, :5]}")
print()

print("✅ This combined embedding goes into the transformer!")
print("   It contains both token identity AND position information.")

input("\nPress Enter to see a full batch example...")

# ============================================================================
# STEP 6: Processing a Batch
# ============================================================================
print("\n📦 STEP 6: Processing Batches")
print("-" * 60)

# Simulate a batch of sequences
batch_size = 4
sequence_length = 8

# Random token IDs (in practice, from our corpus)
torch.manual_seed(456)
batch_token_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))

print(f"Batch of token IDs:")
print(f"  Shape: {batch_token_ids.shape}")
print(f"  ({batch_size} sequences, each with {sequence_length} tokens)")
print()
print("First sequence:", batch_token_ids[0].tolist())
print()

# Get token embeddings for the batch
batch_token_embeddings = token_embedding(batch_token_ids)
print(f"Token embeddings shape: {batch_token_embeddings.shape}")
print(f"  ({batch_size} sequences, {sequence_length} tokens, {embedding_dim} dimensions)")
print()

# Get positional embeddings
positions = torch.arange(sequence_length)
batch_pos_embeddings = pos_embedding(positions)
print(f"Positional embeddings shape: {batch_pos_embeddings.shape}")
print()

# Combine (broadcasting handles the batch dimension)
batch_combined = batch_token_embeddings + batch_pos_embeddings
print(f"Combined embeddings shape: {batch_combined.shape}")
print()

print("🎉 This is the input to our transformer model!")

input("\nPress Enter to visualize relationships...")

# ============================================================================
# STEP 7: Visualizing Embedding Relationships
# ============================================================================
print("\n🔍 STEP 7: Measuring Similarity")
print("-" * 60)
print("""
We can measure how similar two vectors are using "cosine similarity":
  - Values near 1.0  = very similar
  - Values near 0.0  = unrelated
  - Values near -1.0 = opposite

Let's see if similar tokens have similar embeddings (before training):
""")

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

# Encode some related words
words = ["God", "Lord", "Jesus", "heaven", "earth", "apple"]
word_ids = [tokenizer.encode(word)[0] for word in words]

print(f"Words: {words}")
print(f"Token IDs: {word_ids}")
print()

# Get embeddings
word_embeddings = token_embedding(torch.tensor(word_ids))

# Calculate cosine similarity
from torch.nn.functional import cosine_similarity

print("Cosine Similarities (before training - random):\n")
print("        ", "  ".join(f"{w:8s}" for w in words))
print("-" * 60)

for i, word1 in enumerate(words):
    row = f"{word1:8s}"
    for j, word2 in enumerate(words):
        if i == j:
            sim = 1.0
        else:
            sim = cosine_similarity(
                word_embeddings[i].unsqueeze(0),
                word_embeddings[j].unsqueeze(0)
            ).item()
        row += f"  {sim:6.3f}  "
    print(row)

print()
print("💡 Currently random! But after training:")
print("   - Religious words (God, Lord, Jesus) will cluster together")
print("   - Place words (heaven, earth) will be similar")
print("   - Unrelated words (apple) will be distant")

input("\nPress Enter to understand how embeddings are learned...")

# ============================================================================
# STEP 8: How Embeddings Learn
# ============================================================================
print("\n🧠 STEP 8: How Embeddings Learn Meaning")
print("-" * 60)
print("""
Embeddings start RANDOM, but during training:

1. The model tries to predict next tokens
2. When it's wrong, backpropagation adjusts the embeddings
3. Words that appear in similar contexts get similar embeddings

Example during training:
  "God created the heaven" → predict "and"
  "Lord made the earth"    → predict "and"

The model learns: "God" and "Lord" are used similarly!
Their embeddings become closer.

This is called "distributional semantics":
  "You shall know a word by the company it keeps"
  - John Rupert Firth, 1957
""")

input("\nPress Enter for experiments...")

# ============================================================================
# EXPERIMENT: Try It Yourself!
# ============================================================================
print("\n🧪 EXPERIMENT TIME!")
print("-" * 60)
print("""
Try these experiments:

1. Change embedding_dim to 128 or 512
   How does this affect the embedding table size?

2. Create embeddings for your own words:
   word = "beginning"
   word_id = tokenizer.encode(word)[0]
   emb = token_embedding(torch.tensor([word_id]))
   print(emb.shape)

3. Try different sequence lengths
   What happens with sequences longer than context_length?

4. Calculate similarity between other word pairs

Add your experiments below:
""")

# Your experiments here:


input("\nPress Enter for the quiz...")

# ============================================================================
# QUIZ: Test Your Understanding
# ============================================================================
print("\n❓ QUIZ: Test Your Understanding")
print("-" * 60)
print("""
Q1: Why do we use embeddings instead of token IDs directly?
Q2: What is the difference between token and positional embeddings?
Q3: How do embeddings learn meaning during training?
Q4: What does embedding_dim represent?

Think about your answers...
""")
input("Press Enter for solutions...")

print("""
Answers:

A1: Token IDs are arbitrary labels. Embeddings are vectors that can
    represent relationships and meaning. Similar tokens get similar
    embeddings, which helps the model learn patterns.

A2: Token embeddings represent WHAT the token is.
    Positional embeddings represent WHERE it is in the sequence.
    We add them together to get both pieces of information.

A3: Embeddings start random. During training, when the model makes
    predictions, backpropagation adjusts the embeddings. Words used
    in similar contexts gradually get similar embedding vectors.

A4: embedding_dim is how many numbers we use to represent each token.
    Larger = more capacity to capture nuances, but more parameters.
    GPT-2 uses 768, GPT-3 uses 12,288!
""")

input("\nPress Enter to finish...")

# ============================================================================
# SUMMARY & NEXT STEPS
# ============================================================================
print("\n" + "="*60)
print("✅ LESSON 3 COMPLETE!")
print("="*60)
print("""
What you learned:
- Why embeddings are crucial (IDs → meaningful vectors)
- Token embeddings: What the token is
- Positional embeddings: Where the token is
- Combining them for model input
- How embeddings learn during training
- Measuring similarity with cosine similarity

Key Formulas:
  input_embedding = token_embedding + positional_embedding

Key Parameters:
  - vocab_size: 50,257 (GPT-2)
  - embedding_dim: 256 (our model), 768 (GPT-2), 12,288 (GPT-3)
  - context_length: 256 (our model), 1024 (GPT-2)

Next Lesson: Attention Mechanism - How Models Focus
""")
print("="*60)

print("\nRun: uv run python lessons/lesson_04_attention.py")
