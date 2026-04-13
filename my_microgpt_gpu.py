"""
PyTorch 版本的 GPT 实现，使用官方 Transformer 模块
支持 GPU 加速，基于 my_microgpt.py 的架构
"""

import os
import math
import random
import urllib.request
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================
# 准备数据集
input_fname = "./input.txt"
checkpoint_file = "./models/microgpt_torch.pt"

if not os.path.exists(input_fname):
    url = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
    print(f"downloading {url} to {input_fname}")
    urllib.request.urlretrieve(url, input_fname)

docs = [line.strip() for line in open(input_fname) if line.strip()]
random.seed(int(time.time()))
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# =====================================
# 分词器
uchars = sorted(set("".join(docs)))
BOS = len(uchars)
PAD = len(uchars) + 1
vocab_size = len(uchars) + 2
print(f"vocab size: {vocab_size}")

# =====================================
# 模型超参数
n_layer = 4
n_embd = 256
block_size = 16
n_head = 8

# =====================================
# PyTorch 模型定义
class TorchGPT(nn.Module):
    """使用 PyTorch 内置 TransformerEncoder 的 GPT 模型"""
    def __init__(self, vocab_size, n_embd=256, block_size=16, n_head=8, n_layer=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd

        # 词嵌入和位置编码
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        # TransformerEncoder 使用内置 MultiheadAttention + FFN
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4 * n_embd,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.final_norm = nn.LayerNorm(n_embd)

        # 输出层
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # 权重绑定
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

        # 缓存最大长度的 causal mask
        self.register_buffer(
            "causal_mask",
            nn.Transformer.generate_square_subsequent_mask(block_size),
            persistent=False,
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, token_ids, key_padding_mask=None):
        """
        token_ids: (batch, seq_len)
        返回: logits (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = token_ids.shape
        if seq_len > self.block_size:
            raise ValueError(f"seq_len ({seq_len}) > block_size ({self.block_size})")

        # 位置编码
        positions = torch.arange(seq_len, device=token_ids.device)
        pos_emb = self.position_embedding(positions)

        # Token 嵌入
        tok_emb = self.token_embedding(token_ids)
        x = tok_emb + pos_emb

        # Causal mask: (seq_len, seq_len) with -inf upper triangle
        causal_mask = self.causal_mask[:seq_len, :seq_len]

        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        x = self.final_norm(x)

        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, device, max_len=20, temperature=0.5):
        """生成名字"""
        self.eval()
        token_ids = torch.tensor([BOS], device=device).unsqueeze(0)

        for _ in range(max_len):
            if token_ids.size(1) >= self.block_size:
                break

            logits = self.forward(token_ids)
            logits = logits[:, -1, :] / temperature
            logits[:, PAD] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs.squeeze(0), 1).item()

            if next_id == BOS:
                break
            if next_id >= len(uchars):
                next_id = BOS

            token_ids = torch.cat([token_ids, torch.tensor([[next_id]], device=device)], dim=1)

        result = []
        for tid in token_ids[0].tolist()[1:]:
            if tid == BOS:
                break
            result.append(uchars[tid] if tid < len(uchars) else '')

        return ''.join(result)


# =====================================
# 训练
def train(num_steps=2000, lr=0.001, batch_size=128):
    """训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    model = TorchGPT(
        vocab_size=vocab_size,
        n_embd=n_embd,
        block_size=block_size,
        n_head=n_head,
        n_layer=n_layer
    ).to(device)

    print(f"num params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None  # 混合精度加速

    # 预计算序列
    all_tokens = []
    for doc in docs:
        tokens = [BOS] + [uchars.index(c) for c in doc] + [BOS]
        all_tokens.append(tokens)
    num_docs = len(all_tokens)

    for step in range(num_steps):
        model.train()

        # 收集 batch（随机采样）
        batch_indices = [random.randrange(num_docs) for _ in range(batch_size)]
        batch_seqs = [all_tokens[i] for i in batch_indices]

        # 计算 batch 内最大长度，并截断到 block_size
        max_len = max(len(seq) for seq in batch_seqs)
        max_len = min(max_len, block_size)

        # Padding 到 max_len
        batch_tokens = []
        for seq in batch_seqs:
            seq = seq[:max_len]
            if len(seq) < max_len:
                seq = seq + [PAD] * (max_len - len(seq))
            batch_tokens.append(seq)

        tokens_tensor = torch.tensor(batch_tokens, device=device)  # (batch, max_len)
        input_ids = tokens_tensor[:, :-1]  # 输入：当前 token
        targets = tokens_tensor[:, 1:]    # 目标：下一个 token

        # key_padding_mask: True 表示该位置是 padding
        key_padding_mask = input_ids.eq(PAD)

        # targets 中 padding 位置忽略
        targets = targets.masked_fill(targets.eq(PAD), PAD)

        # 前向 + 混合精度
        if use_amp:
            with torch.amp.autocast('cuda'):
                logits = model(input_ids, key_padding_mask=key_padding_mask)
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    targets.reshape(-1),
                    ignore_index=PAD,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids, key_padding_mask=key_padding_mask)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1),
                ignore_index=PAD,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        interval = 1 if (step + 1) <= 10 else (10 if (step + 1) <= 100 else (100 if (step + 1) <= 1000 else 500))
        if (step + 1) % interval == 0:
            print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.item():.4f} | lr {scheduler.get_last_lr()[0]:.6f}")

    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'uchars': uchars,
        'vocab_size': vocab_size,
        'BOS': BOS,
        'PAD': PAD,
    }, checkpoint_file)
    print(f"Model saved to {checkpoint_file}")


# =====================================
# 推理
def infer(num_samples=20, temperature=0.5):
    """推理：生成名字"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TorchGPT(
        vocab_size=vocab_size,
        n_embd=n_embd,
        block_size=block_size,
        n_head=n_head,
        n_layer=n_layer
    ).to(device)

    if os.path.exists(checkpoint_file):
        ckpt = torch.load(checkpoint_file, map_location=device, weights_only=False)
        if ckpt.get('vocab_size') != vocab_size:
            print("Checkpoint vocab_size mismatch, retraining...")
            train()
            print()
        else:
            model.load_state_dict(ckpt['model'])
            print(f"Loaded model from {checkpoint_file}")
    else:
        print("No checkpoint found, training first...")
        train()
        print()

    print("\n--- PyTorch inference (hallucinated names) ---\n")
    for i in range(num_samples):
        name = model.generate(device, max_len=20, temperature=temperature)
        print(f"sample {i+1:2d}: {name}")


# =====================================
# 命令行入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="microgpt_gpu - PyTorch GPU 加速版 GPT")
    parser.add_argument('--mode', '-m', choices=['train', 'infer', 'help'], default='infer',
                        help='模式: train(训练) / infer(推理) / help(帮助)')
    parser.add_argument('--steps', '-s', type=int, default=2000,
                        help='训练步数 (默认 2000)')
    parser.add_argument('--samples', type=int, default=20,
                        help='推理样本数 (默认 20)')
    parser.add_argument('--temp', '-t', type=float, default=0.5,
                        help='采样温度 (默认 0.5)')
    parser.add_argument('--batch', '-b', type=int, default=128,
                        help='训练 batch size (默认 128)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率 (默认 0.001)')
    args = parser.parse_args()

    if args.mode == 'help':
        parser.print_help()
        print("\n示例:")
        print("  python my_microgpt_gpu.py --mode train -s 10000 -b 128 --lr 0.001")
        print("  python my_microgpt_gpu.py --mode infer")
    elif args.mode == 'train':
        train(args.steps, lr=args.lr, batch_size=args.batch)
    elif args.mode == 'infer':
        infer(args.samples, args.temp)
