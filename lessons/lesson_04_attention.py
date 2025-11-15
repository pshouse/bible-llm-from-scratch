"""
LESSON 4: Attention Mechanism - How Models Focus
================================================

In this lesson, you'll learn:
- What attention is and why it's revolutionary
- How attention weights are calculated
- Self-attention vs other types
- Implementing scaled dot-product attention

Time: 25-30 minutes
"""

print("="*60)
print("LESSON 4: Attention Mechanism")
print("="*60)

# ============================================================================
# CONCEPT: The Attention Revolution
# ============================================================================
print("\n🚀 CONCEPT: The Attention Revolution")
print("-" * 60)
print("""
Imagine reading: "The animal didn't cross the street because IT was too tired."

What does "IT" refer to?
  → The animal! (not the street)

How did you know? You paid ATTENTION to the context!

This is what attention mechanisms do in neural networks:
  - Look at ALL words in a sentence
  - Decide which words are important for understanding each word
  - Focus more on relevant context

Before attention (RNNs): Process words sequentially, forget earlier context
After attention (Transformers): See all words at once, focus on what matters!

"Attention is All You Need" - Vaswani et al., 2017
""")

input("Press Enter to see how it works...")

# ============================================================================
# STEP 1: A Simple Example
# ============================================================================
print("\n📝 STEP 1: Understanding with a Simple Example")
print("-" * 60)

import torch
import torch.nn as nn

# Simple example: embeddings for a short sequence
print("Sequence: 'In the beginning'")
print()

# Simulate embeddings (normally these come from our embedding layer)
torch.manual_seed(123)
embeddings = torch.tensor([
    [0.43, 0.15, 0.89],  # "In"
    [0.55, 0.87, 0.66],  # "the"
    [0.57, 0.85, 0.64],  # "beginning"
])

print("Token embeddings (3 tokens, 3 dimensions):")
print(embeddings)
print(f"Shape: {embeddings.shape}")
print()

# Let's compute attention for the second word "the"
query = embeddings[1]  # The word we're focusing on
print(f"Query (word 'the'): {query}")
print()

print("💡 The Goal:")
print("   For the word 'the', how much should we pay attention")
print("   to each word in the sequence (including itself)?")

input("\nPress Enter to calculate attention scores...")

# ============================================================================
# STEP 2: Computing Attention Scores
# ============================================================================
print("\n🎯 STEP 2: Computing Attention Scores")
print("-" * 60)
print("""
Step 1: Calculate "similarity" between query and all words
  - Use DOT PRODUCT (multiply corresponding values and sum)
  - Higher dot product = more similar = more important!
""")

# Calculate attention scores using dot product
attention_scores = torch.zeros(len(embeddings))

for i, embedding in enumerate(embeddings):
    # Dot product: sum of element-wise multiplication
    score = torch.dot(query, embedding)
    attention_scores[i] = score
    print(f"Score with token {i}: {score:.4f}")

print()
print(f"Attention scores: {attention_scores}")
print()

print("💡 Interpretation:")
print(f"   Token 1 ('the') has highest score with itself: {attention_scores[1]:.4f}")
print(f"   Tokens are similar to nearby tokens")

input("\nPress Enter to normalize the scores...")

# ============================================================================
# STEP 3: Softmax - Converting Scores to Weights
# ============================================================================
print("\n📊 STEP 3: Softmax Normalization")
print("-" * 60)
print("""
Problem: Scores can be any value (-∞ to +∞)
Solution: Use SOFTMAX to convert to probabilities (0 to 1, sum to 1)

Softmax formula:
  attention_weight_i = exp(score_i) / sum(exp(all_scores))
""")

# Apply softmax
attention_weights = torch.softmax(attention_scores, dim=0)

print("After softmax:")
for i, weight in enumerate(attention_weights):
    print(f"  Token {i}: {weight:.4f} ({weight*100:.1f}%)")

print()
print(f"Sum of weights: {attention_weights.sum():.4f} (should be 1.0)")
print()

print("💡 Now we have a probability distribution!")
print("   These weights tell us how much to focus on each word.")

input("\nPress Enter to see the final step...")

# ============================================================================
# STEP 4: Computing the Context Vector
# ============================================================================
print("\n🎨 STEP 4: Creating the Context Vector")
print("-" * 60)
print("""
Final step: Create a weighted sum of all embeddings
  - Multiply each embedding by its attention weight
  - Sum them up
  - Result: A new vector that represents "the" in context!
""")

context_vector = torch.zeros(embeddings.shape[1])

print("Computing weighted sum:")
for i, (embedding, weight) in enumerate(zip(embeddings, attention_weights)):
    weighted = embedding * weight
    context_vector += weighted
    print(f"Token {i}: {embedding} × {weight:.4f}")

print()
print(f"Context vector: {context_vector}")
print()

print("💡 This context vector now represents 'the' with awareness")
print("   of its surroundings ('In' and 'beginning')!")

input("\nPress Enter to see the complete picture...")

# ============================================================================
# STEP 5: Self-Attention for All Tokens
# ============================================================================
print("\n🔄 STEP 5: Self-Attention for ALL Tokens")
print("-" * 60)
print("""
We just computed attention for ONE token ('the').
In practice, we do this for EVERY token simultaneously!

This is called SELF-ATTENTION:
  - Each token attends to all tokens (including itself)
  - Creates a context-aware representation for each position
""")

# Compute attention for all tokens
print("Computing full self-attention matrix:\n")

# Attention scores matrix
attention_matrix = embeddings @ embeddings.T  # Matrix multiplication
print("Attention scores (before softmax):")
print(attention_matrix)
print()

# Apply softmax to each row
attention_weights_full = torch.softmax(attention_matrix, dim=1)
print("Attention weights (after softmax):")
print(attention_weights_full)
print()

print("Reading this matrix:")
print(f"  Row 0: How much token 0 ('In') attends to each token")
print(f"  Row 1: How much token 1 ('the') attends to each token")
print(f"  Row 2: How much token 2 ('beginning') attends to each token")
print()

# Verify each row sums to 1
row_sums = attention_weights_full.sum(dim=1)
print(f"Row sums (should all be 1.0): {row_sums}")

input("\nPress Enter to compute context vectors for all...")

# Create context vectors for all tokens
context_vectors = attention_weights_full @ embeddings
print("\nContext vectors:")
print(context_vectors)
print(f"Shape: {context_vectors.shape}")
print()

print("✅ Now each token has a context-aware representation!")

input("\nPress Enter to learn about Query, Key, Value...")

# ============================================================================
# STEP 6: Query, Key, Value - The Full Picture
# ============================================================================
print("\n🔑 STEP 6: Queries, Keys, and Values")
print("-" * 60)
print("""
What we just did was simplified. In real transformers:

Instead of using embeddings directly, we project them into 3 spaces:

  QUERY  (Q): "What am I looking for?"
  KEY    (K): "What do I contain?"
  VALUE  (V): "What information do I have?"

Think of it like a library:
  - Query: "I need books about dragons"
  - Key: Book titles/categories
  - Value: The actual book content

Process:
  1. Compare Query with all Keys → attention scores
  2. Softmax → attention weights
  3. Weighted sum of Values → context vector
""")

input("Press Enter to implement Q, K, V...")

# ============================================================================
# STEP 7: Implementing Q, K, V Attention
# ============================================================================
print("\n⚙️  STEP 7: Query-Key-Value Attention")
print("-" * 60)

d_in = 3   # Input dimension
d_out = 2  # Output dimension

# Create projection matrices
torch.manual_seed(123)
W_query = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

print(f"Projection matrices:")
print(f"  W_query: {W_query.shape}")
print(f"  W_key:   {W_key.shape}")
print(f"  W_value: {W_value.shape}")
print()

# Project embeddings
queries = embeddings @ W_query
keys = embeddings @ W_key
values = embeddings @ W_value

print(f"Queries: {queries.shape}")
print(f"Keys:    {keys.shape}")
print(f"Values:  {values.shape}")
print()

# Compute attention scores
scores = queries @ keys.T
print("Attention scores (Q × K^T):")
print(scores)
print()

# Scaled dot-product attention (divide by sqrt(d_k))
d_k = keys.shape[-1]
scaled_scores = scores / (d_k ** 0.5)
print(f"Scaled by √{d_k} = {d_k**0.5:.2f}:")
print(scaled_scores)
print()

print("💡 Why scale?")
print("   Large dot products → very large softmax values → gradient problems")
print("   Scaling keeps values in a reasonable range")

# Attention weights
weights = torch.softmax(scaled_scores, dim=1)
print("\nAttention weights:")
print(weights)
print()

# Context vectors (weighted sum of values)
context = weights @ values
print("Final context vectors:")
print(context)
print(f"Shape: {context.shape}")

input("\nPress Enter to understand causal masking...")

# ============================================================================
# STEP 8: Causal Masking - Only Look at the Past
# ============================================================================
print("\n🔒 STEP 8: Causal Masking (For Language Models)")
print("-" * 60)
print("""
In language modeling, we predict NEXT words.
We can't look into the future!

When processing "In the beginning":
  - "In" can only see: "In"
  - "the" can only see: "In", "the"
  - "beginning" can only see: "In", "the", "beginning"

We achieve this with a MASK:
""")

# Create mask (upper triangular)
seq_len = len(embeddings)
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
print("Mask matrix (1 = blocked, 0 = allowed):")
print(mask)
print()

# Apply mask by setting blocked positions to -inf
masked_scores = scaled_scores.masked_fill(mask.bool(), float('-inf'))
print("Scores after masking:")
print(masked_scores)
print()

# Softmax (exp(-inf) = 0, so blocked positions get 0 weight)
masked_weights = torch.softmax(masked_scores, dim=1)
print("Attention weights after masking:")
print(masked_weights)
print()

print("💡 Notice:")
print("   - Row 0: Only attends to position 0")
print("   - Row 1: Only attends to positions 0, 1")
print("   - Row 2: Attends to all (0, 1, 2)")
print("   This is CAUSAL (autoregressive) attention!")

input("\nPress Enter for experiments...")

# ============================================================================
# EXPERIMENT: Try It Yourself!
# ============================================================================
print("\n🧪 EXPERIMENT TIME!")
print("-" * 60)
print("""
Try these experiments:

1. Change d_out to different values (4, 8, 16)
   How does this affect the computation?

2. Create your own embeddings for a different sentence:
   embeddings = torch.randn(5, 3)  # 5 words, 3 dimensions

3. Visualize attention weights as a heatmap
   (Hint: darker = more attention)

4. Compute attention WITHOUT scaling
   Compare the softmax outputs - what changes?

Add your experiments below:
""")

# Your experiments here:


input("\nPress Enter for the quiz...")

# ============================================================================
# QUIZ: Test Your Understanding
# ============================================================================
print("\n❓ QUIZ")
print("-" * 60)
print("""
Q1: What does attention allow the model to do?
Q2: Why do we use softmax on attention scores?
Q3: What are Query, Key, and Value? Explain the analogy.
Q4: What is causal masking and why do we need it?
Q5: Why do we scale attention scores by √d_k?

Think about your answers...
""")
input("Press Enter for solutions...")

print("""
Answers:

A1: Attention allows the model to look at ALL tokens in a sequence
    and decide which ones are most relevant for understanding each
    token. It's like selective focus.

A2: Softmax converts arbitrary scores into a probability distribution
    (values 0-1 that sum to 1). This gives us weights for combining
    the value vectors.

A3: Query = "What am I looking for?"
    Key = "What do I contain?"
    Value = "What information do I provide?"
    Like searching a library: query is your question, keys are book
    titles, values are the actual content.

A4: Causal masking prevents tokens from attending to future positions.
    In language modeling, we predict next tokens, so we can't "peek"
    at the answer. The mask ensures we only see the past.

A5: Without scaling, large d_k causes large dot products, leading to
    extreme softmax values (near 0 or 1). This causes vanishing
    gradients. Scaling by √d_k keeps values in a good range.
""")

input("\nPress Enter to finish...")

# ============================================================================
# SUMMARY & NEXT STEPS
# ============================================================================
print("\n" + "="*60)
print("✅ LESSON 4 COMPLETE!")
print("="*60)
print("""
What you learned:
- Attention mechanism: selective focus on relevant context
- Computing attention: dot product → softmax → weighted sum
- Query-Key-Value formulation
- Scaled dot-product attention
- Causal masking for autoregressive generation

Key Formula:
  Attention(Q, K, V) = softmax(QK^T / √d_k) × V

Key Concepts:
  - Self-attention: tokens attend to each other
  - Attention weights: how much to focus on each token
  - Context vector: weighted combination of values
  - Causal mask: prevent seeing the future

Next Lesson: Multi-Head Attention - Parallel Understanding
""")
print("="*60)

print("\nRun: uv run python lessons/lesson_05_multihead_attention.py")
