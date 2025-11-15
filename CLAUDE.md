# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project has two main components:

1. **Learning Foundation** - Educational code following the "LLMs-from-scratch" repository by Sebastian Raschka, implementing fundamental LLM components (tokenization, embeddings, attention mechanisms)

2. **Bible LLM** - A practical application building an LLM trained on public domain English Bible translations, including 4 complete translations (~3 million words, 4.78 million tokens)

## Development Environment

**Python Version**: 3.10+

**Package Manager**: This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable dependency management.

**Setup**:
```bash
# Install dependencies and create .venv automatically
uv sync

# Or run scripts directly (uv handles the environment)
uv run python <script>.py
```

**Required Dependencies**:
- `torch>=2.0.0` - PyTorch deep learning framework
- `tiktoken>=0.5.0` - OpenAI's BPE tokenizer library

**Optional Dependencies** (dev):
- `numpy>=1.24.0` - Numerical computing
- `matplotlib>=3.7.0` - Plotting
- `tqdm>=4.65.0` - Progress bars

**Installation** (if not using uv):
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Code

The main code file is `the_verdict.py`, which can be executed as a learning script:

```bash
uv run python the_verdict.py
```

This script downloads sample text data (`the-verdict.txt`) and demonstrates various LLM concepts sequentially.

## Code Architecture

The codebase is structured as a single educational script (`the_verdict.py`) that demonstrates LLM concepts in progression:

### 1. Tokenization (lines 1-102)
- **SimpleTokenizerV1** (lines 34-51): Basic tokenizer using regex splitting
- **SimpleTokenizerV2** (lines 72-92): Enhanced tokenizer with unknown token handling and special tokens (`<|endoftext|>`, `<|unk|>`)
- Uses regex pattern `r'([,.:;?_!"()\']|--|\s)'` for text preprocessing
- Integration with tiktoken's GPT-2 tokenizer for production-quality tokenization

### 2. Data Loading (lines 143-198)
- **GPTDatasetV1** (lines 145-162): PyTorch Dataset for creating training examples
  - Implements sliding window approach with configurable stride
  - Converts text into overlapping input-target pairs for next-token prediction
  - `max_length`: context window size
  - `stride`: step size between windows
- **create_dataloader_v1** (lines 164-177): Factory function for creating DataLoaders
  - Handles batching, shuffling, and multiprocessing
  - Uses tiktoken GPT-2 encoding by default

### 3. Embeddings (lines 200-228)
- **Token Embeddings**: Maps token IDs to dense vectors (vocab_size=50257, output_dim=256)
- **Positional Embeddings**: Adds position information to token embeddings
- Combined embeddings: `input_embeddings = token_embeddings + pos_embeddings`

### 4. Attention Mechanisms (lines 230-546)
The code implements progressively complex attention mechanisms:

- **Simplified Attention** (lines 231-292): Demonstrates core attention concepts
  - Dot product attention scores
  - Softmax normalization
  - Context vector computation via weighted sum

- **Self-Attention v1** (lines 328-348): Basic self-attention with query/key/value projections
  - Manual parameter matrices using `nn.Parameter`
  - Scaled dot-product attention (divides by sqrt(d_k))

- **Self-Attention v2** (lines 350-369): Refactored version using `nn.Linear` layers
  - Supports optional QKV bias
  - More production-ready implementation

- **CausalAttention** (lines 406-442): Implements masked self-attention for autoregressive generation
  - Upper triangular mask prevents attending to future tokens
  - Registers mask as buffer (non-trainable state)
  - Includes dropout for regularization
  - Handles batched inputs (batch_size, num_tokens, d_in)

- **MultiHeadAttentionWrapper** (lines 444-467): Demonstrates multi-head attention concept
  - Runs multiple attention heads in parallel using `nn.ModuleList`
  - Concatenates outputs from all heads

- **MultiHeadAttention** (lines 469-521): Efficient multi-head attention implementation
  - Single projection matrices for all heads (more efficient than wrapper)
  - Reshapes to (batch, num_heads, num_tokens, head_dim)
  - Includes output projection layer
  - Production-ready implementation matching GPT architecture

### Key Implementation Details

**Causal Masking**: The upper triangular mask ensures autoregressive property:
```python
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
```

**Scaled Dot-Product**: Attention scores are scaled by sqrt(d_k) to prevent gradient saturation:
```python
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
```

**Multi-Head Mechanism**: Splits embedding dimension across heads for diverse attention patterns:
```python
head_dim = d_out // num_heads
# Reshape: (batch, num_tokens, d_out) -> (batch, num_tokens, num_heads, head_dim)
# Transpose: (batch, num_heads, num_tokens, head_dim)
```

## Data Files

- `the-verdict.txt`: Sample text data (Edith Wharton short story) downloaded from the LLMs-from-scratch repository
- Auto-downloaded on first run via `urllib.request.urlretrieve()`

## Bible LLM Project

### Corpus Information
The project uses 4 public domain English Bible translations:
- **King James Version (KJV)** - 821K words, 4.2 MB
- **World English Bible (WEB)** - 799K words, 4.6 MB
- **Douay-Rheims Bible** - 1.07M words, 5.6 MB
- **Tyndale New Testament** - 346K words, 1.8 MB

**Total:** 3,035,129 words | 4,783,526 tokens | 37,369 training examples

### Directory Structure
- `bible_data/` - Raw Bible downloads from Project Gutenberg
- `bible_corpus/` - Cleaned, combined corpus ready for training
  - `bible_combined.txt` - All 4 translations combined (15.8 MB)
  - Individual cleaned translation files

### Key Scripts

**Data Collection:**
```bash
uv run python download_bibles.py              # Download from Project Gutenberg
uv run python download_correct_bibles.py      # Additional translations
uv run python prepare_bible_corpus.py         # Clean and combine
```

**Training Pipeline:**
```bash
uv run python test_bible_tokenization.py     # Test data pipeline (no PyTorch)
uv run python bible_llm.py                    # Full LLM initialization & training
```

### Bible LLM Architecture
- Tokenizer: GPT-2 BPE (50,257 vocab)
- Context window: 256 tokens
- Embedding dim: 256
- Attention heads: 4
- Batch size: 8
- Training examples: 37,369 (with 128 token stride)

### Important Implementation Notes
- `bible_llm.py` requires PyTorch, which may need Visual C++ Redistributables on Windows
- `test_bible_tokenization.py` provides tokenization verification without PyTorch dependencies
- The corpus uses sliding window approach with 50% overlap (stride=128, window=256)
- All Bible texts are properly cleaned of Project Gutenberg headers/footers

## Development Notes

- This is a sequential learning script meant to be run top-to-bottom
- Code demonstrates concepts progressively, building from simple to complex
- Many intermediate variables and examples are printed for educational purposes
- No formal testing framework - this is a learning/experimentation environment
- The code follows the structure from "Build a Large Language Model (From Scratch)" by Sebastian Raschka
