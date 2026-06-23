# PPTI Demo: 从 Transformer 骨架到 TinyBERT Smoke Encoder

本文档记录 PPTI（private/protected Transformer inference）在 HPMPC/PIGEON 上的第一阶段工程实现过程：先构建可运行的 Transformer 安全推理骨架，再扩展为 TinyBERT 风格的 4 层多头 encoder smoke test，并验证 CPU/CUDA 两条路径。

## 1. 当前目标

本阶段目标不是立即跑真实 TinyBERT 权重，而是先证明以下链路在 HPMPC 中可以端到端跑通：

```text
secret input embeddings
  -> Q/K/V GEMM
  -> multi-head attention score GEMM
  -> row-wise attention Softmax
  -> attention value GEMM
  -> output projection GEMM
  -> residual + LayerNorm
  -> FFN GEMM
  -> GELU approximation
  -> FFN output GEMM
  -> residual + LayerNorm
  -> reveal smoke output
```

这个 smoke 版本默认使用很小的维度，便于快速编译、运行和定位协议错误：

```text
seq_len    = 4
hidden     = 8
num_heads  = 2
head_dim   = 4
num_layers = 4
ffn_hidden = 16
```

这不是 TinyBERT 的真实维度，而是 TinyBERT-style 的协议骨架。真实 TinyBERT 维度将在权重导出和加载路径完成后启用。

## 2. 代码演进过程

### 2.1 接入 HPMPC function dispatch

在 `protocol_executer.hpp` 中加入新的 function id：

```cpp
#elif FUNCTION_IDENTIFIER == 87
#include "programs/transformer.hpp"
```

之后可以通过 `FUNCTION_IDENTIFIER=87` 编译运行 PPTI demo。

### 2.2 第一版 Transformer smoke block

第一版 `programs/transformer.hpp` 实现了单层 Transformer block：

```text
X
 -> Q = X * WQ
 -> K = X * WK
 -> V = X * WV
 -> scores = Q * K^T / sqrt(hidden)
 -> probs = softmax(scores)
 -> context = probs * V
 -> attn_out = context * WO
 -> residual + LayerNorm
 -> FFN + GELU approximation
 -> residual + LayerNorm
```

这一版验证了最关键的协议拼接点：

- `prepare_GEMM` 可以承载 Transformer 中的矩阵乘。
- attention Softmax 的插入位置明确。
- LayerNorm 可以用算术分享上的均值、方差、Newton reciprocal sqrt 实现。
- GELU 可以先用低阶多项式近似占位。

### 2.3 Softmax 路径完善

注意：PIGEON 原始函数表中虽然出现了 Softmax，但 CNN 分类头中的 Softmax 路径更接近最后一层分类/argmax 使用场景，不适合直接作为 attention Softmax。

因此在 `programs/transformer.hpp` 中实现了 attention 专用的 row-wise Softmax：

```text
scores row
  -> subtract row max
  -> exp(x) ~= 1 + x + 0.5x^2
  -> row_sum
  -> reciprocal(row_sum) by Newton iteration
  -> normalize
```

当前函数为：

```cpp
secure_rowwise_softmax_poly(scores, rows, cols)
```

这是一个 smoke 级近似，后续需要替换为更精确的 MPC-friendly exp/Softmax 设计。

### 2.4 扩展为 TinyBERT-style encoder

当前版本将单层 Transformer block 扩展为 TinyBERT-style encoder stack：

```text
for layer in 0..3:
  MultiHeadSelfAttention
  Add + LayerNorm
  FFN
  Add + LayerNorm
```

新增关键函数：

```cpp
secure_multi_head_attention(...)
secure_tinybert_encoder_layer(...)
```

多头注意力执行流程：

```text
Q = X * WQ
K = X * WK
V = X * WV

for each head h:
  Q_h, K_h, V_h = slice(Q, K, V)
  scores_h = Q_h * K_h^T / sqrt(head_dim)
  probs_h = rowwise_softmax(scores_h)
  context_h = probs_h * V_h

context = concat(context_0, ..., context_h)
out = context * WO
```

默认 smoke 维度通过宏控制：

```cpp
PPTI_SEQ_LEN
PPTI_HIDDEN
PPTI_NUM_HEADS
PPTI_NUM_LAYERS
PPTI_FFN_HIDDEN
```

如果未显式传入宏，则使用：

```text
4 tokens, hidden 8, 2 heads, 4 layers, ffn 16
```

## 3. 当前文件结构

核心文件：

```text
protocol_executer.hpp
programs/transformer.hpp
scripts/ppti_reference.py
docs/PPTI_WORKFLOW.md
StudyNote/PPTI-Demo.md
```

各文件作用：

| 文件 | 作用 |
| --- | --- |
| `protocol_executer.hpp` | 将 `FUNCTION_IDENTIFIER=87` 分发到 PPTI Transformer 程序 |
| `programs/transformer.hpp` | HPMPC 上的 TinyBERT-style 安全推理 smoke 实现 |
| `scripts/ppti_reference.py` | 无依赖明文参考，用同样的 dummy 输入、权重和近似函数复刻 C++ 逻辑 |
| `docs/PPTI_WORKFLOW.md` | 开发流程、测试命令、后续路线 |
| `StudyNote/PPTI-Demo.md` | 本实验说明文档 |

## 4. 明文参考测试

先运行明文参考，确认 TinyBERT smoke 拓扑没有维度错误：

```sh
cd /home/user/hpmpc
python3 scripts/ppti_reference.py
```

当前输出：

```text
ppti_reference_out0=-1.33777436
ppti_reference_shape=4x8
ppti_reference_topology=layers:4 heads:2 hidden:8 seq:4
```

如需打印中间矩阵：

```sh
python3 scripts/ppti_reference.py --dump
```

`--dump` 会输出：

- input
- layer0 head0 scores
- layer0 head0 probs
- each layer output
- final output

后续调试 C++ reveal trace 时，可用这些值做逐层对齐。

## 5. CPU 三方协议测试

编译 CPU 版本：

```sh
cd /home/user/hpmpc
make -j PARTY=all FUNCTION_IDENTIFIER=87 PROTOCOL=5 DATTYPE=64 BITLENGTH=64 FRACTIONAL=14 USE_CUDA_GEMM=0
```

运行三方本地协议：

```sh
scripts/run.sh -p all -n 3
```

当前 TinyBERT smoke CPU 结果：

```text
PPTI TinyBERT: sharing input embeddings...
PPTI TinyBERT: encoder layer...
PPTI TinyBERT: encoder layer...
PPTI TinyBERT: encoder layer...
PPTI TinyBERT: encoder layer...
PPTI TinyBERT smoke test completed.
```

典型性能：

```text
online getTime ~= 0.069s
P0 send ~= 0.08174MB
P1 send ~= 0.06202MB
P2 send ~= 0.06202MB
```

相比单层 Transformer smoke，通信量和时间上升是预期结果，因为当前已经是 4 层、2 头 encoder stack。

## 6. CUDA GEMM 路径测试

编译 CUDA GEMM 版本：

```sh
cd /home/user/hpmpc
make -j PARTY=all FUNCTION_IDENTIFIER=87 PROTOCOL=5 DATTYPE=64 BITLENGTH=64 FRACTIONAL=14 USE_CUDA_GEMM=2 NVCC=/usr/local/cuda/bin/nvcc
```

运行：

```sh
scripts/run.sh -p all -n 3
```

当前 TinyBERT smoke CUDA 结果：

```text
PPTI TinyBERT smoke test completed.
online getTime ~= 0.321s
```

CUDA 在当前 tiny 维度下比 CPU 慢是正常的，因为矩阵非常小，GPU kernel launch 和 host/device copy 开销远大于 GEMM 本身。CUDA 路径的意义在于验证：

- `prepare_GEMM` 可以切换到 CUDA backend。
- multi-head attention/FFN 中的 GEMM 都能走同一条接口。
- 真实 TinyBERT 大矩阵实验具备基础运行路径。

## 7. 编译宏切换真实 TinyBERT 形状

当前代码支持通过 `MACRO_FLAGS` 切换形状。例如真实 TinyBERT-like 形状可尝试：

```sh
make -j PARTY=all FUNCTION_IDENTIFIER=87 PROTOCOL=5 DATTYPE=64 BITLENGTH=64 FRACTIONAL=14 \
  USE_CUDA_GEMM=2 NVCC=/usr/local/cuda/bin/nvcc \
  MACRO_FLAGS="-DPPTI_SEQ_LEN=128 -DPPTI_HIDDEN=312 -DPPTI_NUM_HEADS=12 -DPPTI_NUM_LAYERS=4 -DPPTI_FFN_HIDDEN=1200"
```

注意：这一步目前只是形状编译入口。真正跑 TinyBERT 权重还需要完成权重导出和加载路径，否则仍然使用 dummy `weight_value()`。

## 8. 当前实现的限制

当前版本是协议骨架，不是最终精度版本：

1. 输入是 dummy embedding，还没有 tokenizer/embedding table。
2. 权重是 dummy deterministic values，还没有加载 HuggingFace TinyBERT 权重。
3. Softmax 使用二阶多项式 exp 近似，精度有限。
4. GELU 使用 cubic surrogate，尚未校准。
5. LayerNorm 使用 Newton reciprocal sqrt，但初值和迭代次数仍需针对真实数值范围调优。
6. 尚未实现 attention mask。
7. 尚未做 C++ fixed-point reveal trace 与 Python reference 的逐层误差对齐。

## 9. 下一步开发计划

推荐下一步顺序：

1. 增加 C++ trace reveal 模式，只在 debug 编译时 reveal 中间矩阵。
2. 用 `scripts/ppti_reference.py --dump` 对齐：
   - layer0 head0 scores
   - layer0 head0 probs
   - layer output
3. 实现 TinyBERT HuggingFace 权重导出脚本。
4. 定义 HPMPC 权重二进制布局：
   - per layer WQ/WK/WV/WO
   - FFN W1/W2
   - attention LayerNorm gamma/beta
   - FFN LayerNorm gamma/beta
5. 将 dummy `load_weights()` 替换成文件加载。
6. 加 attention mask。
7. 改进 Softmax 和 GELU 近似。
8. 用真实 TinyBERT 维度分别 benchmark CPU/CUDA。

## 10. 当前阶段结论

目前已经完成从“单层 Transformer 安全推理骨架”到“TinyBERT-style 4 层 multi-head encoder smoke test”的工程跨越。

这个阶段证明了：

- Transformer 的核心矩阵乘可以接到 HPMPC 现有 `prepare_GEMM` 路径。
- Attention Softmax 的插入位置和调用链已经明确。
- LayerNorm、GELU、residual、FFN 可以在 HPMPC 算术分享层组合出来。
- CPU 和 CUDA GEMM backend 均可运行。
- 后续工作可以集中在真实权重导出、定点精度校准和 Softmax/GELU 协议优化上。
