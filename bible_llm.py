"""
Bible LLM - Training a Language Model on Public Domain English Bible Translations
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken
import os

# --- 1. Configuration ---
# Data Configuration
CORPUS_FILE = "bible_corpus/bible_combined.txt"
# Model Configuration
BATCH_SIZE = 8
MAX_LENGTH = 256
EMBEDDING_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT_RATE = 0.1
# Training Configuration
NUM_EPOCHS = 1
LEARNING_RATE = 0.0004
WEIGHT_DECAY = 0.1
TRAIN_BATCHES = 10  # Limit training to 10 batches for a quick test run

# --- 2. Data Preparation ---
def prepare_data():
    print("="*60)
    print("BIBLE LLM - Data Preparation")
    print("="*60)

    if not os.path.exists(CORPUS_FILE):
        print(f"ERROR: Corpus file not found. Please run prepare_bible_corpus.py first.")
        exit(1)

    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        bible_text = f.read()

    tokenizer = tiktoken.get_encoding("gpt2")
    enc_text = tokenizer.encode(bible_text)

    dataset = BibleDataset(enc_text, MAX_LENGTH, stride=128)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    print(f"Total tokens: {len(enc_text):,}")
    print(f"Training examples: {len(dataset):,}")
    return dataloader, tokenizer

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

# --- 3. Model Definition ---
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys, queries, values = self.W_key(x), self.W_query(x), self.W_value(x)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout_rate, context_length):
        super().__init__()
        self.attention = MultiHeadAttention(emb_dim, emb_dim, context_length, dropout_rate, num_heads)
        self.ff = nn.Sequential(nn.Linear(emb_dim, 4 * emb_dim), nn.GELU(), nn.Linear(4 * emb_dim, emb_dim))
        self.norm1, self.norm2 = nn.LayerNorm(emb_dim), nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + shortcut
        return x

class GPTModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, context_length, num_heads, num_layers, dropout_rate):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(context_length, emb_dim)
        self.drop_emb = nn.Dropout(dropout_rate)
        self.trf_blocks = nn.Sequential(*[TransformerBlock(emb_dim, num_heads, dropout_rate, context_length) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(emb_dim)
        self.out_head = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, in_idx):
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(in_idx.shape[1], device=in_idx.device))
        x = self.drop_emb(tok_embeds + pos_embeds)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)

# --- 4. Training ---
def train(model, dataloader, optimizer):
    print("\n" + "="*60)
    print("Starting model training...")
    print("="*60)
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for i, (input_ids, target_ids) in enumerate(dataloader):
            if i >= TRAIN_BATCHES: break
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Batch {i+1}/{TRAIN_BATCHES}, Loss: {loss.item():.4f}")
        print(f"\nEpoch {epoch+1} Summary: Average Loss: {total_loss / (i+1):.4f}")
    print("\nTraining complete!")

# --- 5. Generation ---
def generate_text(model, tokenizer, context, max_new_tokens=50):
    print("\n" + "="*60)
    print("Generating text sample...")
    print("="*60)
    model.eval()
    input_ids = torch.tensor(tokenizer.encode(context)).unsqueeze(0)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids)
        next_token_id = torch.multinomial(torch.softmax(logits[:, -1, :], dim=-1), num_samples=1)
        input_ids = torch.cat((input_ids, next_token_id), dim=1)

    generated_text = tokenizer.decode(input_ids[0].tolist())
    print(f"Start context: \"{context}\"")
    print(f"Generated text: \"{generated_text}\"")

# --- Main Execution ---
if __name__ == "__main__":
    dataloader, tokenizer = prepare_data()

    model = GPTModel(tokenizer.n_vocab, EMBEDDING_DIM, MAX_LENGTH, NUM_HEADS, NUM_LAYERS, DROPOUT_RATE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train(model, dataloader, optimizer)
    generate_text(model, tokenizer, "In the beginning,")
