import torch
import torch.nn as nn
from torch.nn import functional as F
import wandb
import os
import pandas as pd
import glob
import random

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
vocab_size = 402 # 0 ~ 401 까지의 Index
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

def load_return_tokens_from_file(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    
    if 'ReturnToken' not in df.columns:
        print(f"File {file_path} does not have 'ReturnToken' column. Skipping.")
        return None

    tokens = df['ReturnToken'].dropna().tolist()
    return torch.tensor(tokens, dtype=torch.long)

data_dir = "./data"
csv_files = glob.glob(os.path.join(data_dir, "*_with_returns_tokens.csv"))

train_data_list = []
for file_path in csv_files:
    tokens_tensor = load_return_tokens_from_file(file_path)
    if tokens_tensor is not None and len(tokens_tensor) > block_size:
        train_data_list.append(tokens_tensor)

train_data_list[:5]
print(f"총 {len(train_data_list)}개의 파일에서 ReturnToken 시퀀스를 로드했습니다.")

# train_data_list를 무작위로 섞은 후, train/val split (예: 80% train, 20% val)
train_val_ratio = 0.8
random.shuffle(train_data_list)
n_train = int(len(train_data_list) * train_val_ratio)
train_data = train_data_list[:n_train]
val_data = train_data_list[n_train:]

print(f"Train 파일 수: {len(train_data)}, Val 파일 수: {len(val_data)}")

def get_batch(split='train'):
    """
    각 배치 샘플은 한 파일 내에서 block_size 길이의 입력(x)와 그 바로 다음 토큰(y)로 구성
    파일 간의 토큰이 섞이지 않음
    """
    if split == 'train':
        data_list = train_data
    else:
        data_list = val_data
    
    xs, ys = [], []
    for _ in range(batch_size):
        # block_size+1 이상의 길이를 가진 파일 중에서 랜덤 선택
        valid_files = [d for d in data_list if len(d) > block_size]

        if not valid_files:
            raise ValueError("block_size보다 긴 ReturnToken 시퀀스가 있는 파일이 없습니다.")

        data = random.choice(valid_files)
        start_idx = random.randint(0, len(data) - block_size - 1)
        x = data[start_idx : start_idx + block_size]
        y = data[start_idx + 1 : start_idx + block_size + 1]
        xs.append(x)
        ys.append(y)
    
    # 배치 차원으로 스택
    x_batch = torch.stack(xs)  # (batch_size, block_size)
    y_batch = torch.stack(ys)  # (batch_size, block_size)
    return x_batch.to(device), y_batch.to(device)

xb, yb = get_batch('train')

def decode_return(token):
    """
    단일 토큰 인덱스(token: 0 ~ 401)를 해당하는 수익률(소수 형태)로 디코딩(복원)
    
    변환 규칙:
      - token == 0   -> -10000 basis points (-100%)
      - token == 401 -> +10000 basis points (+100%)
      - token 1 ~ 400: 해당 구간의 대표값(중간값)은
                       -10000 + (token - 1) * 50 + 25
                       basis point 단위이며, 이를 10000으로 나누어 소수 형태로 반환
    """
    if token == 0:
        r_bp = -10000
    elif token == 401:
        r_bp = 10000
    else:
        r_bp = -10000 + (token - 1) * 50 + 25
    return r_bp / 100.0 # 다시 % 변환

def decode(token_list):
    """
    token_list: 정수 토큰의 리스트 (예: [196, 200, 200, 210, 210])
    
    각 토큰을 decode_return 함수를 사용해 디코딩하고,
    디코딩된 수익률의 리스트(%)를 반환.
    """
    return [round(decode_return(token), 1) for token in token_list] # 소수점 반올림

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
            # wandb.save(checkpoint_path)  # wandb 서버에도 체크포인트 파일 저장

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
result = decode(m.generate(context, max_new_tokens=256)[0].tolist()) # 256개 생성
print('output 토큰 길이: ', len(result), '\n', result)
wandb.finish()