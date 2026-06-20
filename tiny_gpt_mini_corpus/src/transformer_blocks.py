import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """Satu head self-attention dengan causal masking."""

    def __init__(self, n_embd, head_size, block_size, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, time_steps, channels = x.shape

        k = self.key(x)
        q = self.query(x)

        weights = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        weights = weights.masked_fill(self.tril[:time_steps, :time_steps] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        v = self.value(x)
        out = weights @ v
        return out


class MultiHeadAttention(nn.Module):
    """Beberapa head attention yang dijalankan paralel."""

    def __init__(self, n_embd, num_heads, head_size, block_size, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Lapisan feed-forward sederhana setelah attention."""

    def __init__(self, n_embd, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Blok Transformer Decoder: attention + feed-forward."""

    def __init__(self, n_embd, n_head, block_size, dropout=0.2):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
