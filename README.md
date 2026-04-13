# MicroGPT

一个极简的 GPT 实现，用于教学演示。支持纯 Python 手写版本和 PyTorch GPU 加速版本。

## 文件说明

| 文件 | 说明 |
|------|------|
| `my_microgpt.py` | 纯 Python 版本，无外部依赖，适合学习反向传播原理 |
| `my_microgpt_gpu.py` | PyTorch 版本，支持 GPU 加速，适合训练更大模型 |

## 核心组件

```
数据集 → 分词器 → 模型架构 → 训练循环 → 推理
```

1. **自动微分 (Value 类)** - 手动实现反向传播，支持 add/mul/pow/log/exp/relu 等操作
2. **分词器** - 基于字符级 tokenization，BOS 特殊符号标记序列开始
3. **模型架构**
   - Token Embedding + Position Embedding
   - RMSNorm 归一化
   - Multi-Head Self-Attention
   - MLP 前馈网络（扩展 4 倍维度）
4. **训练** - Adam 优化器 + 余弦学习率衰减
5. **推理** - 温度采样生成

## 使用方法

### 纯 Python 版本

```bash
python my_microgpt.py --mode train         # 训练
python my_microgpt.py --mode infer         # 推理
python my_microgpt.py --mode train -s 500  # 指定步数
```

### PyTorch GPU 版本

```bash
# 安装 PyTorch (需要 CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu121

python my_microgpt_gpu.py --mode train         # 训练 (自动使用 GPU)
python my_microgpt_gpu.py --mode infer         # 推理
python my_microgpt_gpu.py --mode train -s 5000 # 训练 5000 步
```

## CLI 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-m, --mode` | train / infer / help | infer |
| `-s, --steps` | 训练步数 | 1000 (Python) / 2000 (GPU) |
| `--samples` | 推理样本数 | 20 |
| `-t, --temp` | 采样温度 | 0.5 |

## 模型文件

- `input.txt` - 训练数据集（自动下载 names.txt）
- `models/` - 模型检查点目录

| 文件 | 说明 |
|------|------|
| `models/microgpt.checkpoint` | Python 版本模型 |
| `models/microgpt_torch.pt` | PyTorch 版本模型 |

## 架构参数

### Python 版本

| 参数 | 值 | 说明 |
|------|-----|------|
| n_layer | 1 | Transformer 层数 |
| n_embd | 16 | 嵌入维度 |
| n_head | 4 | 注意力头数 |
| 参数量 | ~26K | - |

### PyTorch GPU 版本

| 参数 | 值 | 说明 |
|------|-----|------|
| n_layer | 4 | Transformer 层数 |
| n_embd | 256 | 嵌入维度 |
| n_head | 8 | 注意力头数 |
| 参数量 | ~21M | - |

## 参考

- 原始实现: https://gist.github.com/8627fe009c40f57531cb18360106ce95
- 数据集来源: [makemore](https://github.com/karpathy/makemore)
- PyTorch: https://pytorch.org
