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
scripts/ppti_export_tinybert.py
docs/PPTI_WORKFLOW.md
StudyNote/PPTI-Demo.md
```

各文件作用：

| 文件 | 作用 |
| --- | --- |
| `protocol_executer.hpp` | 将 `FUNCTION_IDENTIFIER=87` 分发到 PPTI Transformer 程序 |
| `programs/transformer.hpp` | HPMPC 上的 TinyBERT-style 安全推理 smoke 实现 |
| `scripts/ppti_reference.py` | 无依赖明文参考，用同样的 dummy 输入、权重和近似函数复刻 C++ 逻辑 |
| `scripts/ppti_export_tinybert.py` | 导出 HuggingFace TinyBERT encoder 权重到 PPTI 顺序二进制 layout，也可生成 synthetic smoke 权重 |
| `scripts/ppti_export_tinybert_inputs.py` | 导出 TinyBERT embedding 权重和 tokenized input file |
| `docs/PPTI_WORKFLOW.md` | 开发流程、测试命令、后续路线 |
| `StudyNote/PPTI-Demo.md` | 本实验说明文档 |
| `StudyNote/PPTI-TaskFlow.md` | 后续任务流程表，按阶段记录目标、输入输出、命令和验收标准 |

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

当前 C++ 端会优先读取 `PPTI_MODEL_FILE` 指定的权重文件。文件格式为：

```text
int32 total_parameter_count
float32 parameters...
```

每层参数顺序：

```text
query.weight.T, query.bias
key.weight.T, key.bias
value.weight.T, value.bias
attention.output.dense.weight.T, attention.output.dense.bias
intermediate.dense.weight.T, intermediate.dense.bias
output.dense.weight.T, output.dense.bias
attention.output.LayerNorm.weight, attention.output.LayerNorm.bias
output.LayerNorm.weight, output.LayerNorm.bias
```

先用 synthetic 权重验证文件加载链路：

```sh
cd /home/user/hpmpc
python3 scripts/ppti_export_tinybert.py --synthetic --output models/ppti/tinybert_ppti_synthetic.bin
PPTI_MODEL_FILE=models/ppti/tinybert_ppti_synthetic.bin scripts/run.sh -p all -n 3
```

真实 TinyBERT 权重导出命令：

```sh
python3 scripts/ppti_export_tinybert.py \
  --model huawei-noah/TinyBERT_General_4L_312D \
  --output models/ppti/tinybert_4l_312d_ppti.bin
```

远端已经安装 CPU 版 PyTorch、`transformers` 和新版 `pillow`，并成功导出真实 TinyBERT encoder 权重：

```text
models/ppti/tinybert_4l_312d_ppti.bin
params=4568736
topology=layers:4 heads:12 hidden:312 ffn_hidden:1200
layout=ppti_tinybert_v1
```

该文件 header 与 C++ 真实 shape 参数公式一致：

```text
params=4568736
expected=4568736
```

同时已经导出真实 TinyBERT embedding 和一个 seq=16 的 tokenizer 输入样本：

```text
models/ppti/tinybert_embeddings_ppti.bin
models/ppti/sample_input_seq16.bin
embedding_params=9683856
embedding_shape=vocab:30522 max_position:512 type_vocab:2 hidden:312
```

Attention mask 已接入 attention Softmax。当前实现没有直接加大负数 mask，而是在多项式 exp 近似之后、row sum 之前用公开 mask 清零 masked token 的概率质量；这是为了避免二阶 exp 近似在大负数处因平方项失真。

阶段四已建立 C++ fixed-point trace 与 Python reference 的比较流程。当前 smoke 结果显示 `embedding_out` 完全对齐，第一处误差从 `layer0_head0_scores` 开始，最终输出 `max_abs_error` 约为 `0.68885958`。这说明下一步应先做 fixed-point Python reference 和真实 seq=16 小样本运行，再优化 Softmax/GELU。

阶段五已完成真实 `seq=16` 小样本端到端 CPU/CUDA 推理。使用真实 TinyBERT encoder 权重、真实 embedding 权重和 tokenizer input，CPU online `getTime` 约 `6.30s`，通信量约 P0 发送 `7.099MB`、P1/P2 发送 `5.201MB`。机器重启后 `nvidia-smi` 正常，按 RTX 2060 的 `sm_75` 重编 CUTLASS 对象后，CUDA 复测无 CUTLASS error，online `getTime` 约 `6.40s`。

当前代码支持通过 `MACRO_FLAGS` 切换形状。例如真实 TinyBERT-like 形状可尝试：

```sh
make -j PARTY=all FUNCTION_IDENTIFIER=87 PROTOCOL=5 DATTYPE=64 BITLENGTH=64 FRACTIONAL=14 \
  USE_CUDA_GEMM=2 NVCC=/usr/local/cuda/bin/nvcc \
  MACRO_FLAGS="-DPPTI_SEQ_LEN=128 -DPPTI_HIDDEN=312 -DPPTI_NUM_HEADS=12 -DPPTI_NUM_LAYERS=4 -DPPTI_FFN_HIDDEN=1200"
```

注意：真实权重运行需要同时满足 `MACRO_FLAGS` 的 shape 与导出的权重 shape 完全一致，否则 C++ 会因为参数数量不匹配而回退到 dummy 权重路径。

## 8. 当前实现的限制

当前版本是协议骨架，不是最终精度版本：

1. 真实 TinyBERT encoder 权重、embedding 权重和 tokenizer input 路径已经接入。
2. Attention mask 已接入，但 Softmax 使用二阶多项式 exp 近似，真实 score 范围下精度不足。
3. GELU 使用 cubic surrogate，尚未校准。
4. LayerNorm 使用 Newton reciprocal sqrt，但初值和迭代次数仍需针对真实数值范围调优。
5. C++ fixed-point reveal trace 与 Python reference 已建立第一轮对齐，当前第一处明显发散在 `layer0_head0_scores`。
6. 额外 reveal Q/K/V projection 全张量会扰动当前协议路径，后续需要改成轻量统计 trace。

## 9. 下一步开发计划

推荐下一步顺序：

1. 增加轻量统计 trace，围绕 Q/K projection、QK score matmul 和 scale 后输出 min/max/mean_abs。
2. 对 attention score 加 clamp/range reduction。
3. 改进 Softmax reciprocal 初值或归一化策略。
4. 校准 GELU 和 LayerNorm 近似。
5. 在 `seq=16` 误差收敛后再扩大到 `seq=32/64/128`。

## 10. 当前阶段结论

目前已经完成从“单层 Transformer 安全推理骨架”到“TinyBERT-style 4 层 multi-head encoder smoke test”的工程跨越。

这个阶段证明了：

- Transformer 的核心矩阵乘可以接到 HPMPC 现有 `prepare_GEMM` 路径。
- Attention Softmax 的插入位置和调用链已经明确。
- LayerNorm、GELU、residual、FFN 可以在 HPMPC 算术分享层组合出来。
- CPU 和 CUDA GEMM backend 均可运行。
- 后续工作可以集中在真实权重导出、定点精度校准和 Softmax/GELU 协议优化上。
