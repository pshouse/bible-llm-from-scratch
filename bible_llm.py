"""
Bible LLM - Training a Language Model on Public Domain English Bible Translations

End-to-end transformer training pipeline on the combined corpus.
"""

import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

import tiktoken

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


SEPARATOR = "=" * 60
CORPUS_FILE = "bible_corpus/bible_combined.txt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainingConfig:
    vocab_size: int
    context_length: int = 256
    stride: int = 128
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    batch_size: int = 16
    max_steps: int = 200
    eval_interval: int = 50
    eval_steps: int = 10
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    val_split: float = 0.01
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compile: bool = False
    checkpoint_dir: str = "checkpoints"
    sample_prompt: str = "In the beginning God created"
    sample_tokens: int = 160


def apply_env_overrides(config: TrainingConfig) -> None:
    overrides = {
        "MAX_STEPS": ("max_steps", int),
        "EVAL_INTERVAL": ("eval_interval", int),
        "EVAL_STEPS": ("eval_steps", int),
        "BATCH_SIZE": ("batch_size", int),
        "LEARNING_RATE": ("lr", float),
        "LR": ("lr", float),
        "D_MODEL": ("d_model", int),
        "N_LAYERS": ("n_layers", int),
        "N_HEADS": ("n_heads", int),
        "DROPOUT": ("dropout", float),
        "DEVICE": ("device", str),
        "STRIDE": ("stride", int),
    }
    for env_key, (attr, cast) in overrides.items():
        value = os.getenv(env_key)
        if value is not None:
            setattr(config, attr, cast(value))


class BibleDataset(Dataset):
    def __init__(self, token_ids: Iterable[int], max_length: int, stride: int):
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.max_length = max_length
        self.stride = stride
        n = self.tokens.size(0)
        if n <= max_length:
            self.num_chunks = 0
        else:
            self.num_chunks = ((n - max_length - 1) // stride) + 1
            self.num_chunks = max(0, self.num_chunks)

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, idx: int):
        if idx >= self.num_chunks:
            raise IndexError(idx)
        start = idx * self.stride
        end = start + self.max_length
        inputs = self.tokens[start:end]
        targets = self.tokens[start + 1:end + 1]
        return inputs, targets


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model must divide by n_heads"
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.in_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.context_length, config.context_length))
        self.register_buffer("mask", mask.view(1, 1, config.context_length, config.context_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, steps, width = x.shape
        qkv = self.in_proj(x).chunk(3, dim=-1)
        q = qkv[0].view(bsz, steps, self.n_heads, self.head_dim).transpose(1, 2)
        k = qkv[1].view(bsz, steps, self.n_heads, self.head_dim).transpose(1, 2)
        v = qkv[2].view(bsz, steps, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.mask[:, :, :steps, :steps]
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, steps, width)
        y = self.out_proj(y)
        return self.resid_drop(y)


class FeedForward(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.context_length, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        bsz, steps = idx.shape
        if steps > self.config.context_length:
            idx = idx[:, -self.config.context_length:]
            steps = idx.size(1)
            if targets is not None:
                targets = targets[:, -self.config.context_length:]

        positions = torch.arange(0, steps, device=idx.device)
        x = self.token_embed(idx) + self.pos_embed(positions)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                threshold = values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx


@torch.no_grad()
def evaluate(model: GPTLanguageModel, data_loader: DataLoader, device: torch.device, max_steps: int) -> float:
    model.eval()
    losses: list[float] = []
    iterator = iter(data_loader)
    steps_to_run = min(max_steps, len(data_loader))
    for _ in range(steps_to_run):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(data_loader)
            batch = next(iterator)
        inputs, targets = (tensor.to(device) for tensor in batch)
        _, loss = model(inputs, targets)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def create_dataloaders(tokens: list[int], config: TrainingConfig):
    dataset = BibleDataset(tokens, config.context_length, config.stride)
    total = len(dataset)
    if total == 0:
        raise ValueError("Corpus too small for configured window/stride.")

    val_size = max(1, int(total * config.val_split))
    if val_size >= total:
        val_size = max(1, total // 10)
    train_size = total - val_size
    if train_size <= 0:
        raise ValueError("Validation split consumes entire dataset.")

    generator = torch.Generator().manual_seed(config.seed)
    if train_size < config.batch_size:
        raise ValueError("Training dataset smaller than batch_size; reduce batch size or adjust stride.")

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)
    return dataset, train_loader, val_loader


def train_model(
    model: GPTLanguageModel,
    config: TrainingConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> GPTLanguageModel:
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    ckpt_path = None
    if config.checkpoint_dir:
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(config.checkpoint_dir, "bible_gpt.pt")

    model.train()
    train_iter = iter(train_loader)
    progress = tqdm(range(1, config.max_steps + 1), total=config.max_steps, desc="training")

    for step in progress:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        inputs, targets = (tensor.to(device) for tensor in batch)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        if step % config.eval_interval == 0 or step == 1 or step == config.max_steps:
            val_loss = evaluate(model, val_loader, device, config.eval_steps)
            if hasattr(progress, "set_postfix"):
                progress.set_postfix({"train_loss": loss.item(), "val_loss": val_loss})
            else:
                print(f"step {step}/{config.max_steps} train_loss {loss.item():.4f} val_loss {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if ckpt_path:
                    torch.save({"model_state_dict": best_state, "config": asdict(config)}, ckpt_path)

    if best_state:
        model.load_state_dict(best_state)
    return model.to(device)


def main() -> None:
    print(SEPARATOR)
    print("BIBLE LLM - Transformer Training Pipeline")
    print(SEPARATOR)

    if not os.path.exists(CORPUS_FILE):
        raise FileNotFoundError(
            f"Corpus file not found: {CORPUS_FILE}. Run prepare_bible_corpus.py first."
        )

    with open(CORPUS_FILE, "r", encoding="utf-8") as handle:
        bible_text = handle.read()

    print(
        f"Corpus stats: {len(bible_text):,} characters | {len(bible_text.split()):,} words | "
        f"{len(bible_text) / (1024 * 1024):.2f} MB"
    )

    tokenizer = tiktoken.get_encoding("gpt2")
    enc_text = tokenizer.encode(bible_text)
    print(f"Total tokens: {len(enc_text):,}")

    config = TrainingConfig(vocab_size=tokenizer.n_vocab)
    apply_env_overrides(config)
    set_seed(config.seed)

    print("Training configuration:")
    print(f"  device = {config.device}")
    print(f"  context_length = {config.context_length}")
    print(f"  stride = {config.stride}")
    print(f"  batch_size = {config.batch_size}")
    print(f"  d_model = {config.d_model}, n_layers = {config.n_layers}, n_heads = {config.n_heads}")
    print(f"  max_steps = {config.max_steps}, eval_interval = {config.eval_interval}, eval_steps = {config.eval_steps}")
    print(f"  learning_rate = {config.lr}")

    dataset, train_loader, val_loader = create_dataloaders(enc_text, config)
    print(
        f"Dataset windows: {len(dataset):,} | train batches: {len(train_loader):,} | "
        f"val batches: {len(val_loader):,}"
    )

    model = GPTLanguageModel(config)
    if config.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    model = train_model(model, config, train_loader, val_loader)

    device = next(model.parameters()).device
    prompt_tokens = tokenizer.encode(config.sample_prompt)
    if not prompt_tokens:
        prompt_tokens = [getattr(tokenizer, "eot_token", 50256)]
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    generated = model.generate(input_ids, config.sample_tokens, temperature=1.0, top_k=50)
    generated_text = tokenizer.decode(generated[0].cpu().tolist())

    print(SEPARATOR)
    print("Sample generation:")
    print(generated_text)
    print(SEPARATOR)


if __name__ == "__main__":
    main()
