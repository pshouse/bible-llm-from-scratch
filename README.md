# Build an LLM from Scratch - Bible Edition

A project to build a Large Language Model (LLM) from scratch, trained on public domain English Bible translations.

## Project Overview

This repository contains:
1. **Learning code** from "LLMs-from-scratch" tutorial (original `the_verdict.py`)
2. **Bible corpus collection** - Multiple public domain English Bible translations
3. **Bible LLM implementation** - Adapted LLM architecture for training on Biblical texts

## Corpus Statistics

The Bible corpus includes 4 public domain English translations:

| Translation | Words | Size | Description |
|------------|-------|------|-------------|
| King James Version (KJV) | 821,496 | 4.2 MB | Classic 1611 translation |
| World English Bible (WEB) | 799,219 | 4.6 MB | Modern public domain translation |
| Douay-Rheims Bible | 1,067,908 | 5.6 MB | Catholic translation (1582-1609) |
| Tyndale New Testament | 346,485 | 1.8 MB | Historic first printed English NT |

**Total Corpus:**
- **3,035,129 words**
- **16.6 million characters**
- **4.78 million tokens** (GPT-2 tokenizer)
- **37,369 training examples** (with context window of 256 tokens)
- **15.8 MB** of training data

## Quick Start

### 1. Setup Environment

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable dependency management.

```bash
# Install uv (if not already installed)
# Windows (PowerShell):
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies (creates .venv and installs packages)
uv sync

# Or install just the core dependencies
uv pip install -r requirements.txt
```

**Alternative (traditional pip/venv):**
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download and Prepare Bible Corpus

```bash
# Download Bible translations from Project Gutenberg
uv run python download_bibles.py

# Download additional translations
uv run python download_correct_bibles.py

# Clean and combine all translations
uv run python prepare_bible_corpus.py
```

This creates:
- `bible_data/` - Raw downloaded Bible texts
- `bible_corpus/` - Cleaned and combined corpus ready for training

### 3. Test Tokenization

```bash
# Verify the data pipeline works correctly
uv run python test_bible_tokenization.py
```

### 4. Run Bible LLM (requires PyTorch)

```bash
# Note: May require Visual C++ Redistributables on Windows
uv run python bible_llm.py
```

## Project Structure

```
build-a-llm/
├── bible_data/                  # Raw Bible downloads
│   ├── kjv.txt
│   ├── web.txt
│   ├── douay_rheims.txt
│   └── tyndale_nt.txt
│
├── bible_corpus/                # Cleaned training data
│   ├── bible_combined.txt       # All translations combined
│   ├── kjv_cleaned.txt
│   ├── web_cleaned.txt
│   ├── douay_rheims_cleaned.txt
│   └── tyndale_nt_cleaned.txt
│
├── download_bibles.py           # Download Bibles from Project Gutenberg
├── download_correct_bibles.py   # Additional Bible downloads
├── prepare_bible_corpus.py      # Clean and combine corpus
├── test_bible_tokenization.py   # Test data pipeline
├── bible_llm.py                 # Main LLM training script
│
├── the_verdict.py               # Original learning code
└── the-verdict.txt              # Sample data for learning
```

## Model Architecture

The Bible LLM uses a GPT-style transformer architecture:

- **Tokenizer:** GPT-2 BPE tokenizer (50,257 vocab size)
- **Context Window:** 256 tokens
- **Embedding Dimension:** 256
- **Multi-Head Attention:** 4 heads
- **Dropout:** 0.1

Key components:
1. **Token Embeddings** - Maps tokens to dense vectors
2. **Positional Embeddings** - Adds position information
3. **Multi-Head Causal Attention** - Masked self-attention for autoregressive generation
4. **Feed-Forward Networks** - (to be implemented)
5. **Layer Normalization** - (to be implemented)

## Training Configuration

- **Batch Size:** 8
- **Context Window:** 256 tokens
- **Stride:** 128 tokens (50% overlap)
- **Training Examples:** 37,369
- **Batches per Epoch:** 4,671

## Data Pipeline

```python
# 1. Load and tokenize corpus
tokenizer = tiktoken.get_encoding("gpt2")
tokens = tokenizer.encode(bible_text)  # 4.78M tokens

# 2. Create sliding window dataset
dataset = BibleDataset(tokens, max_length=256, stride=128)

# 3. Create data loader
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 4. Each batch contains:
#    - input_ids: [batch_size, max_length] = [8, 256]
#    - target_ids: [batch_size, max_length] = [8, 256]
```

## Next Steps

To complete the LLM implementation:

1. **Define full transformer model architecture**
   - Stack multiple transformer blocks
   - Add feed-forward networks
   - Add layer normalization
   - Add output projection to vocabulary

2. **Define training components**
   - Loss function: CrossEntropyLoss
   - Optimizer: AdamW
   - Learning rate scheduler

3. **Implement training loop**
   - Forward pass
   - Backward pass
   - Gradient updates
   - Validation

4. **Train the model**
   - Multiple epochs
   - Checkpoint saving
   - Loss tracking

5. **Generate text**
   - Implement sampling strategies
   - Temperature scaling
   - Top-k/top-p sampling

## Bible Translations Sources

All Bible translations are in the public domain:

- **King James Version (KJV)** - Project Gutenberg #10
- **World English Bible (WEB)** - Project Gutenberg #8294
- **Douay-Rheims Bible** - Project Gutenberg #1581
- **Tyndale New Testament** - Project Gutenberg #10553

## Dependencies

```
torch>=2.0.0          # Deep learning framework
tiktoken>=0.5.0       # OpenAI's tokenizer library
```

## System Requirements

- Python 3.10 or higher
- 8+ GB RAM (for tokenization and data loading)
- GPU recommended for training (CPU training will be slow)
- Windows users may need Visual C++ Redistributables for PyTorch

## Learning Resources

This project follows "Build a Large Language Model (From Scratch)" by Sebastian Raschka:
- GitHub: https://github.com/rasbt/LLMs-from-scratch
- Original learning code in `the_verdict.py`

## License

- **Code:** MIT License (or similar open source)
- **Bible Texts:** Public Domain (no copyright restrictions)

## Acknowledgments

- Sebastian Raschka for the "LLMs-from-scratch" tutorial
- Project Gutenberg for hosting public domain Bible translations
- OpenAI for the tiktoken library
