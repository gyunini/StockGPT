import torch
import torch.nn as nn
from torch.nn import functional as F
import wandb
import os

# hyperparameters
batch_size = 64 
block_size = 256 # maximum context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 # validation 200개만 평가
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

# wandb 초기화 (프로젝트 이름과 설정 값을 기록)
wandb.init(project="stockGPT", entity="gyunini", config={
    "batch_size": batch_size,
    "block_size": block_size,
    "max_iters": max_iters,
    "eval_interval": eval_interval,
    "learning_rate": learning_rate,
    "eval_iters": eval_iters,
    "n_embd": n_embd,
    "n_head": n_head,
    "n_layer": n_layer,
    "dropout": dropout,
})

# 로그와 체크포인트를 저장할 디렉토리 생성
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # 학습과 상관없는 parameter 
        self.dropout = nn.Dropout(dropout) # 0.2

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        # compute attention scores 계산 ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        # masking 처리 (autoregressive decoding)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        # attention coefficients 계산
        wei = F.softmax(wei, dim=-1) # (B, T, T) 
        wei = self.dropout(wei)
        # value와의 weighted sum 계산
        v = self.value(x) # (B,T,head_size)
        out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj_layer = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # 마지막 차원으로 concat: (B, T, head_size * num_head) (원복됨)
        out = self.dropout(self.proj_layer(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    
    def __init__(self, n_embd):
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
    """ Transformer block: multihead attention + feed forward로 구성된 Block 정의 """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) # head 개수만큼 나눠줌, 추후 각 head에서 attention value가 합쳐져서 원복
        self.ffwd = FeedFoward(n_embd)
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.layer_norm1(x)) # Pre-Norm Formulation: 최근에는 앞에 layer norm 적용함
        x = x + self.ffwd(self.layer_norm2(x))
        return x

# super simple bigram model
class GPT1(nn.Module):

    def __init__(self):
        super().__init__()
        # 각 토큰 위치에 해당하는 곳의 row를 lookup하는 table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.layer_norm_final = nn.LayerNorm(n_embd) # final layer norm
        # 다시 vocab_size 만큼의 선형 변환을 통해 logit을 구하기 위함
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx ,targets : (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # 0 ~ T-1 텐서 생성 (T, C)
        x = tok_emb + pos_emb # broadcasting 일어남: (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.layer_norm_final(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            current_idx = idx[:, -block_size:] # 최신 context만 crop
            # get the predictions
            logits, loss = self(current_idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPT1()
m = model.to(device)
print('Total Model Parameters:', round(sum(p.numel() for p in m.parameters())/1e6, 1), 'M')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # wandb에 손실 기록
        wandb.log({
            "step": step,
            "train_loss": losses['train'],
            "val_loss": losses['val']
        })

        # 500 step 혹은 마지막 step 체크포인트 저장
        if step > 0 and (step % 1000 == 0 or step == max_iters - 1):
            checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step,
                'val_loss': losses['val']
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at step {step} to {checkpoint_path}")
            wandb.save(checkpoint_path)  # wandb 서버에도 체크포인트 파일 저장

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))