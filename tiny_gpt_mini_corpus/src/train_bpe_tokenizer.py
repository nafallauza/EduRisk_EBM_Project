from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

from transformer_blocks import Block


# =========================
# Konfigurasi model
# =========================
torch.manual_seed(42)

BASE_DIR = Path(__file__).resolve().parents[1]
CORPUS_PATH = BASE_DIR / "corpus" / "corpus_edurisk.txt"
OUTPUT_PATH = BASE_DIR / "output" / "hasil_bpe_tokenizer.txt"
TOKENIZER_PREFIX = BASE_DIR / "output" / "edurisk_bpe"

block_size = 16
batch_size = 16
n_embd = 64
n_head = 4
n_layer = 3
dropout = 0.2
learning_rate = 1e-3
max_iters = 800
eval_interval = 100
bpe_vocab_size = 200
device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# BPE Tokenizer dengan SentencePiece
# =========================
spm.SentencePieceTrainer.Train(
    input=str(CORPUS_PATH),
    model_prefix=str(TOKENIZER_PREFIX),
    vocab_size=bpe_vocab_size,
    model_type="bpe",
    character_coverage=1.0,
    hard_vocab_limit=False
)

sp = spm.SentencePieceProcessor()
sp.load(str(TOKENIZER_PREFIX) + ".model")

text = CORPUS_PATH.read_text(encoding="utf-8").lower()
ids = sp.encode(text, out_type=int)

data = torch.tensor(ids, dtype=torch.long)
vocab_size = sp.get_piece_size()

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    source_data = train_data if split == "train" else val_data
    ix = torch.randint(len(source_data) - block_size, (batch_size,))
    x = torch.stack([source_data[i:i + block_size] for i in ix])
    y = torch.stack([source_data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    losses = {}
    for split in ["train", "val"]:
        loss_values = []
        for _ in range(10):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            loss_values.append(loss.item())
        losses[split] = sum(loss_values) / len(loss_values)
    model.train()
    return losses


# =========================
# Model TinyGPT
# =========================
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            Block(n_embd=n_embd, n_head=n_head, block_size=block_size, dropout=dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        batch, time = idx.shape

        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(time, device=device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            batch, time, channels = logits.shape
            logits = logits.view(batch * time, channels)
            targets = targets.view(batch * time)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = TinyGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

log_lines = []
log_lines.append("EKSPERIMEN BPE TOKENIZATION / SENTENCEPIECE")
log_lines.append("=" * 50)
log_lines.append(f"Jumlah token BPE          : {len(ids)}")
log_lines.append(f"Vocabulary size           : {vocab_size}")
log_lines.append(f"Block size                : {block_size}")
log_lines.append(f"Embedding dimension       : {n_embd}")
log_lines.append(f"Jumlah head attention     : {n_head}")
log_lines.append(f"Jumlah layer transformer  : {n_layer}")
log_lines.append(f"Learning rate             : {learning_rate}")
log_lines.append(f"Training device           : {device}")
log_lines.append("")

print("Mulai training BPE Tokenization...")

for step in range(max_iters + 1):
    if step % eval_interval == 0:
        losses = estimate_loss(model)
        line = f"Step {step:4d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}"
        print(line)
        log_lines.append(line)

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

prompt = "risiko akademik mahasiswa"
context = torch.tensor([sp.encode(prompt, out_type=int)], dtype=torch.long, device=device)
generated_ids = model.generate(context, max_new_tokens=120)[0].tolist()
generated_text = sp.decode(generated_ids)

log_lines.append("")
log_lines.append("PROMPT:")
log_lines.append(prompt)
log_lines.append("")
log_lines.append("HASIL GENERATE:")
log_lines.append(generated_text)
log_lines.append("")
log_lines.append("ANALISIS SINGKAT:")
log_lines.append(
    "BPE Tokenization memecah teks menjadi subword. "
    "Pendekatan ini lebih fleksibel daripada word-level karena dapat membentuk kata dari potongan token yang lebih kecil. "
    "Vocabulary dapat dikontrol melalui vocab_size, sehingga lebih efisien untuk corpus yang memiliki banyak variasi kata."
)

OUTPUT_PATH.write_text("\n".join(log_lines), encoding="utf-8")

print("")
print("Training selesai.")
print(f"Hasil disimpan ke: {OUTPUT_PATH}")
