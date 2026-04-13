# MicroGPT

一个极简的 GPT 实现，用于教学演示。纯 Python实现，无外部依赖。

## 文件说明

| 文件 | 来源 | 说明 |
|------|------|------|
| `microgpt.py` | [原始 Gist](https://gist.github.com/8627fe009c40f57531cb18360106ce95) | @karpathy 原始实现 |
| `my_microgpt.py` | 改写练习 | 在原始基础上添加 CLI 支持和中文注释 |

## 核心组件

```
数据集 → 分词器 → 模型架构 → 训练循环 → 推理
```

1. **自动微分 (Value 类)** - 手动实现反向传播，支持 add/mul/pow/log/exp/relu 等操作
2. **分词器** - 基于字符级 tokenization，BOS 特殊符号标记序列开始
3. **模型架构**
   - Token Embedding + Position Embedding
   - RMSNorm 归一化
   - Multi-Head Self-Attention（4 头 × 4 维）
   - MLP 前馈网络（扩展 4 倍维度）
4. **训练** - Adam 优化器 + 线性学习率衰减
5. **推理** - 温度采样生成

## 使用方法

```bash
# 默认推理（无模型时自动训练）
python my_microgpt.py

# 帮助
python my_microgpt.py --help

# 训练
python my_microgpt.py --mode train
python my_microgpt.py --mode train -s 500   # 指定步数

# 推理
python my_microgpt.py --mode infer
python my_microgpt.py --mode infer -s 10 -t 0.8  # 样本数、温度
```

## CLI 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-m, --mode` | train / infer / help | infer |
| `-s, --steps` | 训练步数 | 1000 |
| `--samples` | 推理样本数 | 20 |
| `-t, --temp` | 采样温度 | 0.5 |

## 模型文件

- `input.txt` - 训练数据集（自动下载 names.txt）
- `model.checkpoint` - 训练保存的模型参数

## 架构参数

| 参数 | 值 | 说明 |
|------|-----|------|
| n_layer | 1 | Transformer 层数 |
| n_embd | 16 | 嵌入维度 |
| block_size | 16 | 上下文窗口 |
| n_head | 4 | 注意力头数 |
| head_dim | 4 | 每头维度 |
| 参数量 | ~26K | - |

## 参考

- 原始实现: https://gist.github.com/8627fe009c40f57531cb18360106ce95
- 数据集来源: [makemore](https://github.com/karpathy/makemore)
