"""
这是一个非常简化的GPT实现，主要用于教学目的。它包含了以下几个部分：
1. 定义模型参数和状态字典
2. 定义模型架构，包括线性变换、softmax、rmsnorm和GPT前向函数
3. 训练循环，使用简单的随机梯度下降优化器
"""

import os
import math
import random
import urllib.request

random.seed(11)
input_fname = "./input.txt"


# =====================================
# 准备数据集
if not os.path.exists(input_fname):
    url = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
    print(f"downloading {url} to {input_fname}")
    urllib.request.urlretrieve(url, input_fname)  # 下载数据集
docs = [
    line.strip() for line in open(input_fname) if line.strip()
]  # 读取数据集并去除空行
random.shuffle(docs)  # 打乱数据集
print(f"num docs: {len(docs)}")

# =====================================
# 分词器
uchars = sorted(set("".join(docs)))  # 获取数据集中所有唯一字符
BOS = len(uchars)  # 定义特殊的开始符号 BOS(beginning of sequence) 的 token id
vocab_size = len(uchars) + 1  # 词汇表大小，包含所有唯一字符和 BOS
print(f"vocab size: {vocab_size}")

# =====================================
# 自动微分
class Value:
    """
    封装了 add, mul, pow, log, exp, relu 等操作，并且在 backward() 中实现了反向传播算法。
    梯度：
        表示对结果的影响率， 即导数， 以 a = x + y 为例
        正向传播, 记录局部梯度
            ∂a/∂x = a._local_grads[0] = 1,  表示 a 对 x 的梯度， 即 x 对 a 的影响率
        反向传播, 链式法则累积全局梯度
            a.grad = ∂loss/∂a, 表示 loss 对 a 的梯度, 即 a 对 loss 的影响率
            x.grad = ∂loss/∂x = ∂loss/∂a * ∂a/∂x = a.grad * a._local_grads[0] = a.grad * 1 = a.grad,  表示 loss 对 x 的梯度, 即 x 对 loss 的影响率
            即: child.grad = self.grad * self._local_grads
    """
    __slots__ = ("data", "grad", "_children", "_local_grads")  # 优化内存使用

    # 初始化函数，接受数据值、子节点和局部梯度
    def __init__(self, data, children=(), local_grads=()):
        self.data = data  # 该节点的标量值，在前向传播中计算得到
        self.grad = 0  # 该节点的梯度，在反向传播中计算得到
        self._children = children  # 该节点在计算图中的子节点
        self._local_grads = local_grads  # 该节点相对于子节点的局部梯度

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)  # 确保 other 是一个 Value 对象
        return Value(self.data + other.data, (self, other), (1, 1))  # 加法时 局部梯度为 1

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)  # 确保 other 是一个 Value 对象
        return Value( self.data * other.data, (self, other), (other.data, self.data))  # 乘法时 局部梯度分别为另一个值的 data

    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data ** (other - 1),))  # 幂运算时 局部梯度为 other * self.data ** (other - 1)

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))  # 对数运算时 局部梯度为 1 / self.data

    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))  # 指数运算时 局部梯度为 math.exp(self.data)

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))  # ReLU 运算时 局部梯度为 self.data > 0 的布尔值
    
    def __neg__(self):
        return self * -1  # 负号运算时 局部梯度为 -1

    def __radd__(self, other):
        return self + other  # 反向加法调用正向加法

    def __sub__(self, other):
        return self + (-other)  # 减法运算可以通过加上负数实现

    def __rsub__(self, other):
        return other + (-self)  # 反向减法调用正向加法

    def __rmul__(self, other):
        return self * other  # 反向乘法调用正向乘法

    def __truediv__(self, other):
        return self * other ** -1  # 除法运算可以通过乘以另一个值的负幂实现

    def __rturediv__(self, other):
        return other * self ** -1  # 反向除法调用正向乘法

    def backward(self):
        topo = []  # 记录所有计算顺序
        visited = set()  # 记录已访问的节点
        def build_topo(v):
            if v not in visited:
                visited.add(v) 
                for child in v._children:  
                    build_topo(child)  # 先处理子节点
                topo.append(v)  # 后处理当前节点
        build_topo(self)  # 从当前节点开始构建拓扑排序
        self.grad = 1  # 初始化当前节点的梯度为 1
        for v in reversed(topo):  # 反向遍历拓扑排序列表
            for child, local_grad in zip(v._children, v._local_grads):  # 遍历当前节点的子节点
                child.grad += local_grad * v.grad  # 更新子节点的梯度, 根据上面公式  child.grad = self.grad * self._local_grads

# =====================================
# 模型参数
n_layer = 1                         # 神经网络的 隐藏层 数量
n_embd = 16                         # 每个 token 的 特征维度， 这里用 16 个维度的特征; 真实 GPT 模型中这个值通常是 768-128K 维
block_size = 16                     # 注意力窗口的最大上下文长度（注意：最长的名字是15个字符）
n_head = 4                          # 注意力头的数量, 同时用 4个 “视角” 去看上下文， 捕捉不同类型的关系
head_dim = n_embd // n_head         # 每个头 关注的 特征维度 数量

# 生成矩阵， nout 行, nin 列, 每个元素是一个 Value 对象，值服从均值为 0, 标准差为 std 的高斯分布
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]  

# 储存 模型权重参数 的字典
state_dict = {
    'wte': matrix(vocab_size, n_embd),      # 词嵌入矩阵, [27, 16],  表示 27 个 token 的 16 维度特征向量
    'wpe': matrix(block_size, n_embd),      # 位置编码矩阵, [16, 16],  表示 16 个 字符位置 的 16 维度特征向量
    'lm_head': matrix(vocab_size, n_embd)   # 输出预测矩阵, [27, 16],  把 最后一个隐藏层(16) 投影回 词汇表大小(27), 用于预测下一个 token 的概率分布
}

"""
生成每个 layer 的数据
每个 layer 包含:
    attn_wq, attn_wk, attn_wv: 注意力机制中计算 Query, Key, Value 的权重矩阵
    attn_wo: 注意力输出的投影矩阵
    mpl_fc1: 前馈网络的第一层权重矩阵（扩展维度）
    mpl_fc2: 前馈网络的第二层权重矩阵（压缩回原维度）

    注意力机制三剑客: Query, Key, Value
        输入向量 x
            Q = Wq * x -> Query, 我在找什么
            K = Wk * x -> Key, 我代表什么
            V = Wv * x -> Value, 我要传递什么信息

    注意力计算
        Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
        - Q * K^T 计算 Query 和 Key 之间的相似度，得到注意力权重
        - 除以 sqrt(d_k) 是为了稳定梯度，防止数值过大
        - softmax 将权重归一化为概率分布
        - 最后乘以 V 得到加权的输出向量，表示根据注意力权重聚合的信息

    MLP 多层感知机
        mlp_fc1:[64,16]  -> 非线性激活(ReLU) -> mlp_fc2:[16,64]
"""
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)    # Wq, Query, 查询
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)    # Wk, Key, 键
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)    # Wv, Value, 值
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)    # Output 投影， 将多头注意力拼接的的结果，投影回原来的维度
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)  # 前馈网络第一层（扩展维度）
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)  # 前馈网络第二层（压缩回原维度）

params = [ p for max in state_dict.values() for row in max for p in row ] # 将所有参数矩阵中的 Value 对象展平为一个列表
print(f"num params: {len(params)}")

# =====================================
# 架构


# 计算 W * x, 其中 W 是一个矩阵， x 是一个向量， 输出是一个向量
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]      # 计算 计算W 中的每行 wo 与输入向量 x 的点积，得到输出向量的每个元素


# 计算 logits 的 softmax， 输出概率分布
def softmax(logits):
    max_value = max(val.data for val in logits)         # 计算 最大值 
    exps = [(val - max_value).exp() for val in logits]  # 计算 exp， 减去 max_value 避免计算时溢出
    total = sum(exps)                                  # 计算 指数值的总和
    return [e / total for e in exps]                   # 将每个指数值除以总和，得到 softmax 输出的概率分布

# Root Mean Square Normalization（均方根归一化）
#   RMSNorm(x) = x / RMS(x)
#   其中 RMS(x) = sqrt(1/n * Σxi²)
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)              # 计算 每个元素的 平方和， 然后除以元素数量，得到均方
    scale = (ms + 1e-5) ** -0.5                         # 计算 sqrt(ms + 1e-5) 的倒数， 加上一个小常数 1e-5 来避免除以零
    return [xi * scale for xi in x]                     # 将输入向量 x 的每个元素乘以 scale，得到归一化后的输出向量

# 模型本身
def gpt( token_id, pos_id, keys, values ):
    tok_emb = state_dict['wte'][token_id]           # 根据 token_id 获取 特征向量, 即 token embedding
    pos_emb = state_dict['wpe'][pos_id]             # 根据 pos_id 获取 特征向量, 即 position embedding
    x = [t + p for t, p in zip(tok_emb, pos_emb)]   # 将 token embedding 和 position embedding 相加，得到输入向量 x
    x = rmsnorm(x)                                  # 对输入向量 x 进行 RMSNorm 归一化

    for li in range(n_layer):
        # 1) Multi-head Attention block
        x_residual = x                              # 保存残差连接的输入
        x = rmsnorm(x)                              # 对输入向量 x 进行 RMSNorm 归一化
        q = linear(x, state_dict[f'layer{li}.attn_wq'])  # 计算 Query 向量
        k = linear(x, state_dict[f'layer{li}.attn_wk'])  # 计算 Key 向量
        v = linear(x, state_dict[f'layer{li}.attn_wv'])  # 计算 Value 向量
        keys[li].append(k)                          # 将 Key 向量添加到 keys 列表中，供后续注意力计算使用
        values[li].append(v)                        # 将 Value 向量添加到 values 列表中，供后续注意力计算使用
        x_attn = []                                 # 存储多头注意力的输出
        for h in range(n_head):
            hs = h * head_dim                               # 计算当前头的起始维度索引
            q_h = q[hs:hs+head_dim]                         # 获取当前头的 Query 向量
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]   # 获取当前头的 Key 向量列表
            v_h = [vi[hs:hs+head_dim] for vi in values[li]] # 获取当前头的 Value 向量列表
            # 对每个 t in len(keys) 和 j in len(head_dim),  计算 q_h[j] * k_h[t][j] 的乘积，并将结果除以 head_dim 的平方根，得到注意力 logits
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]  # 计算注意力 logits
            attn_weights = softmax(attn_logits)  # 对注意力 logits 进行 softmax，得到注意力权重
            # 对每个 j in len(head_dim) 和 t in len(v_h), 计算 attn_weights[t] * v_h[t][j] 的乘积，并将结果求和，得到当前头的输出向量
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]  # 计算当前头的输出向量
            x_attn.extend( head_out)  # 将当前头的输出向量添加到 x_attn 列表中
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])  # 将多头注意力的输出向量通过线性变换投影回原来的维度
        x = [a + b for a, b in zip(x, x_residual)]  # 将投影后的向量与残差连接的输入相加，得到注意力块的输出
        # 2) MLP block
        x_residual = x                              # 保存残差连接的输入
        x = rmsnorm(x)                              # 对输入向量 x 进行 RMSNorm 归一化
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])  # 前馈网络第一层线性变换
        x = [xi.relu() for xi in x]                # 前馈网络第一层的非线性激活函数 ReLU
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])  # 前馈网络第二层线性变换
        x = [a + b for a, b in zip(x, x_residual)]  # 将前馈网络的输出与残差连接的输入相加，得到当前层的输出

    logits = linear(x, state_dict['lm_head'])  # 将最后一层的输出向量通过线性变换投影到词汇表大小的维度，得到 logits
    return logits  # 返回 logits，表示预测下一个 token 的概率分布

# =====================================
# 循环训练
learning_rate = 0.01                # 学习率, 控制参数更新的步长
beta1 = 0.85                        # Adam 优化器的第一个动量衰减率，控制一阶矩估计的指数衰减
beta2 = 0.99                        # Adam 优化器的第二个动量衰减率，控制二阶矩估计的指数衰减
eps_adam = 1e-8                     # Adam 优化器的数值稳定性常数，防止除以零
m = [0.0] * len(params)             # Adam 优化器的一阶矩缓冲区，初始化为全零，模拟惯性, 之前向右走， 现在也倾向于向右走
v = [0.0] * len(params)             # Adam 优化器的二阶矩缓冲区，初始化为全零，梯度大的少走， 梯度的的多走

num_steps = 1000 # 训练步骤数
for step in range(num_steps):

    # 获取样本
    doc = docs[ step % len(docs) ]  # 获取当前训练样本
    tokens = [BOS] + [uchars.index(c) for c in doc]  # 将文本转换为 token id 列表，前面加上 BOS 标记
    n = min(block_size, len(tokens) - 1)  # 计算当前样本的有效长度，不能超过 block_size

    # 向前传播
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]  # 初始化每层的 keys 和 values 列表
    losses = []  # 存储每个位置的损失
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]  # 获取当前输入 token id 和目标 token id
        logits = gpt(token_id, pos_id, keys, values)  # 前向传播，得到 logits
        probs = softmax(logits)  # 对 logits 进行 softmax，得到预测的概率分布
        loss_t = -probs[target_id].log()  # 计算当前时间步的损失，使用交叉熵损失
        losses.append(loss_t)  # 将当前时间步的损失添加到 losses 列表中
    loss = sum(losses) / n  # 计算当前样本的平均损失

    # 反向传播，计算所有参数的梯度
    loss.backward()  

    # Adam 优化器更新参数
    lr_t = learning_rate * (1 - step / num_steps)  # 学习率衰减，随着训练步骤增加，学习率逐渐减小
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad  # 更新一阶矩估计
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2  # 更新二阶矩估计
        m_hat = m[i] / (1 - beta1 ** (step + 1))  # 计算一阶矩的偏差修正
        v_hat = v[i] / (1 - beta2 ** (step + 1))  # 计算二阶矩的偏差修正
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)  # 更新参数，使用 Adam 的更新规则
        p.grad = 0  # 清零梯度，为下一次迭代做准备

    print( f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")  # 打印当前训练步骤和损失值