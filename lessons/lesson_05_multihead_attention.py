"""
LESSON 5: Multi-Head Attention - Parallel Understanding
=======================================================

In this lesson, you'll learn:
- Why single attention isn't enough
- How multiple heads work in parallel
- Implementing multi-head attention
- Understanding what different heads learn

Time: 20-25 minutes
"""

print("="*60)
print("LESSON 5: Multi-Head Attention")
print("="*60)

# ============================================================================
# CONCEPT: Why Multiple Heads?
# ============================================================================
print("\n🤔 CONCEPT: The Limitation of Single-Head Attention")
print("-" * 60)
print("""
With single-head attention, the model has ONE way to focus on context.

But language is complex! For the word "bank":
  - One perspective: Is it a financial institution?
  - Another perspective: Is it a river bank?
  - Another: What's its grammatical role?

Multi-head attention = Multiple perspectives simultaneously!

Each "head" learns to focus on different types of relationships:
  - Head 1: Syntactic patterns (subject-verb agreement)
  - Head 2: Semantic meaning (word associations)
  - Head 3: Long-range dependencies
  - Head 4: Positional patterns

It's like having multiple experts analyzing the same text!
""")

input("Press Enter to see how it works...")

# ============================================================================
# STEP 1: Single Head vs Multiple Heads
# ============================================================================
print("\n🔄 STEP 1: From One Head to Many")
print("-" * 60)

import torch
import torch.nn as nn

# Example input
torch.manual_seed(123)
batch_size = 2
seq_len = 4
d_model = 8  # Model dimension

# Simulate token embeddings
x = torch.randn(batch_size, seq_len, d_model)
print(f"Input shape: {x.shape}")
print(f"  {batch_size} sequences, {seq_len} tokens each, {d_model} dimensions")
print()

# Single-head configuration
d_k = d_model  # Key/Query dimension
d_v = d_model  # Value dimension

print("SINGLE-HEAD ATTENTION:")
print(f"  d_model = {d_model}")
print(f"  Creates {d_k}-dim queries, keys, values")
print(f"  Output: {d_model} dimensions")
print()

# Multi-head configuration
num_heads = 4
d_k_per_head = d_model // num_heads
d_v_per_head = d_model // num_heads

print("MULTI-HEAD ATTENTION:")
print(f"  d_model = {d_model}")
print(f"  num_heads = {num_heads}")
print(f"  Each head: {d_k_per_head}-dim queries, keys, values")
print(f"  All heads combined: {num_heads} × {d_k_per_head} = {d_model} dimensions")
print()

print("💡 Key Insight:")
print("   Instead of one 8-dimensional attention,")
print("   we have four 2-dimensional attentions running in parallel!")

input("\nPress Enter to implement multi-head attention...")

# ============================================================================
# STEP 2: Implementing Multi-Head Attention
# ============================================================================
print("\n⚙️  STEP 2: Building Multi-Head Attention")
print("-" * 60)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, context_length, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Single projections for all heads (more efficient!)
        self.W_query = nn.Linear(d_model, d_model, bias=False)
        self.W_key = nn.Linear(d_model, d_model, bias=False)
        self.W_value = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        # 1. Project to Q, K, V
        Q = self.W_query(x)  # (batch, seq, d_model)
        K = self.W_key(x)
        V = self.W_value(x)

        # 2. Split into multiple heads
        # Reshape: (batch, seq, d_model) → (batch, seq, num_heads, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 3. Transpose for attention computation
        # (batch, seq, num_heads, head_dim) → (batch, num_heads, seq, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 4. Compute attention for all heads in parallel
        scores = Q @ K.transpose(-2, -1)  # (batch, heads, seq, seq)
        scores = scores / (self.head_dim ** 0.5)  # Scale

        # 5. Apply causal mask
        scores = scores.masked_fill(
            self.mask[:seq_len, :seq_len].bool(), float('-inf')
        )

        # 6. Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 7. Apply attention to values
        out = attn_weights @ V  # (batch, heads, seq, head_dim)

        # 8. Concatenate heads
        # (batch, heads, seq, head_dim) → (batch, seq, heads, head_dim)
        out = out.transpose(1, 2)

        # → (batch, seq, d_model)
        out = out.contiguous().view(batch_size, seq_len, self.d_model)

        # 9. Final projection
        out = self.out_proj(out)

        return out

print("✅ Multi-Head Attention class defined!")
print("\nKey steps:")
print("  1. Project input to Q, K, V")
print("  2. Split into multiple heads")
print("  3. Compute attention for each head in parallel")
print("  4. Concatenate head outputs")
print("  5. Final projection")

input("\nPress Enter to test it...")

# ============================================================================
# STEP 3: Testing Multi-Head Attention
# ============================================================================
print("\n🧪 STEP 3: Testing Multi-Head Attention")
print("-" * 60)

# Create model
context_length = 10
mha = MultiHeadAttention(
    d_model=d_model,
    num_heads=num_heads,
    context_length=context_length,
    dropout=0.1
)

print(f"Created MultiHeadAttention:")
print(f"  d_model: {d_model}")
print(f"  num_heads: {num_heads}")
print(f"  head_dim: {d_model // num_heads}")
print()

# Test forward pass
output = mha(x)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {output.shape}")
print()

print("✅ Shape preserved! (batch, seq, d_model) → (batch, seq, d_model)")
print("   Each token now has context from all others (via all heads)")

input("\nPress Enter to visualize what different heads learn...")

# ============================================================================
# STEP 4: Visualizing Different Heads
# ============================================================================
print("\n👁️  STEP 4: What Do Different Heads Learn?")
print("-" * 60)
print("""
Let's run the attention and see what each head focuses on.

We'll create a simple example and examine the attention patterns.
""")

# Create a clearer example
torch.manual_seed(42)
test_input = torch.randn(1, 6, d_model)  # 1 sequence, 6 tokens

# Get attention weights (we'll modify the class to return them)
class MultiHeadAttentionWithWeights(MultiHeadAttention):
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        Q = self.W_query(x)
        K = self.W_key(x)
        V = self.W_value(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(
            self.mask[:seq_len, :seq_len].bool(), float('-inf')
        )

        attn_weights = torch.softmax(scores, dim=-1)
        out = attn_weights @ V

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)

        return out, attn_weights

mha_with_weights = MultiHeadAttentionWithWeights(
    d_model=d_model,
    num_heads=num_heads,
    context_length=context_length,
    dropout=0.0  # No dropout for visualization
)

output, weights = mha_with_weights(test_input)

print(f"Attention weights shape: {weights.shape}")
print(f"  (batch=1, heads={num_heads}, seq=6, seq=6)")
print()

# Show attention patterns for each head
for head in range(num_heads):
    print(f"Head {head} attention pattern:")
    print(weights[0, head].detach().numpy().round(3))
    print()

print("💡 Observations:")
print("   - Each head has different attention patterns!")
print("   - Some heads focus locally (nearby tokens)")
print("   - Some heads focus more broadly")
print("   - The mask ensures causal attention (lower triangular)")

input("\nPress Enter to understand the benefits...")

# ============================================================================
# STEP 5: Benefits of Multi-Head Attention
# ============================================================================
print("\n🎯 STEP 5: Why Multi-Head Attention Works")
print("-" * 60)
print("""
1. DIVERSE REPRESENTATIONS
   Each head can learn different types of relationships:
   - Syntactic (grammar)
   - Semantic (meaning)
   - Positional (location in sequence)

2. INCREASED CAPACITY
   Multiple heads = more parameters = more learning capacity
   Without increasing dimension of individual heads

3. PARALLEL COMPUTATION
   All heads computed simultaneously (on GPU)
   Very efficient!

4. ROBUSTNESS
   If one head fails to learn useful patterns, others can compensate

Real example from GPT-2:
  - 12 layers
  - 12 heads per layer
  - 144 attention heads total!

Research shows different heads specialize:
  - Some track subject-verb agreement
  - Some handle coreference (pronoun resolution)
  - Some learn positional patterns
""")

input("\nPress Enter to compare parameters...")

# ============================================================================
# STEP 6: Parameter Count Comparison
# ============================================================================
print("\n📊 STEP 6: Parameter Analysis")
print("-" * 60)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Single head (hypothetical)
single_head = nn.Linear(d_model, d_model)
single_head_params = count_parameters(single_head) * 3  # Q, K, V

# Multi-head attention
mha_params = count_parameters(mha)

print(f"Configuration: d_model={d_model}, num_heads={num_heads}")
print()
print(f"Single-head parameters (Q+K+V):     {single_head_params:,}")
print(f"Multi-head attention parameters:    {mha_params:,}")
print()

print("Parameters breakdown:")
print(f"  W_query: {d_model} × {d_model} = {d_model**2:,}")
print(f"  W_key:   {d_model} × {d_model} = {d_model**2:,}")
print(f"  W_value: {d_model} × {d_model} = {d_model**2:,}")
print(f"  out_proj: {d_model} × {d_model} = {d_model**2:,}")
print(f"  Total: {mha_params:,}")
print()

print("💡 Multi-head doesn't add many more parameters,")
print("   but provides much more expressiveness!")

input("\nPress Enter for experiments...")

# ============================================================================
# EXPERIMENT: Try It Yourself!
# ============================================================================
print("\n🧪 EXPERIMENT TIME!")
print("-" * 60)
print("""
Try these experiments:

1. Change num_heads to 2, 8, or 16
   How does it affect the head_dim?
   What happens if d_model isn't divisible by num_heads?

2. Create a longer sequence (seq_len=20)
   Visualize the attention patterns

3. Try different d_model values (16, 32, 64)
   Calculate the parameter count

4. Set dropout=0.5
   Run multiple forward passes - outputs differ?

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
Q1: Why use multiple attention heads instead of one larger head?
Q2: How are the head outputs combined?
Q3: What does head_dim = d_model / num_heads mean?
Q4: Why can multi-head attention be computed efficiently?
Q5: What happens to the output shape after multi-head attention?

Think about your answers...
""")
input("Press Enter for solutions...")

print("""
Answers:

A1: Multiple heads allow the model to attend to different aspects
    simultaneously - syntax, semantics, position, etc. One large
    head can only learn one attention pattern.

A2: Each head produces output of dimension head_dim. These are
    concatenated to get d_model dimensions, then passed through
    a final linear projection (out_proj).

A3: We split the model dimension across heads. If d_model=512 and
    num_heads=8, each head works with 64 dimensions. This keeps
    total computation similar to single-head.

A4: All heads are computed in parallel using matrix operations.
    The "split into heads" is just a reshape, not actual separate
    computations. GPUs handle this very efficiently.

A5: Input shape (batch, seq, d_model) → Output shape (batch, seq, d_model).
    Shape is preserved! Each position gets d_model-dimensional output
    that incorporates information from all heads.
""")

input("\nPress Enter to finish...")

# ============================================================================
# SUMMARY & NEXT STEPS
# ============================================================================
print("\n" + "="*60)
print("✅ LESSON 5 COMPLETE!")
print("="*60)
print("""
What you learned:
- Why multiple attention heads are better than one
- How multi-head attention splits and combines information
- Implementing efficient multi-head attention
- Understanding attention patterns across heads
- Parameter efficiency of multi-head design

Key Formula:
  MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O
  where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

Key Architecture:
  - d_model: Total embedding dimension
  - num_heads: Number of parallel attention heads
  - head_dim = d_model / num_heads
  - Each head: scaled dot-product attention
  - Combine: concatenate + project

Next Lesson: Feed-Forward Networks - Processing Information
""")
print("="*60)

print("\nRun: uv run python lessons/lesson_06_feedforward.py")
