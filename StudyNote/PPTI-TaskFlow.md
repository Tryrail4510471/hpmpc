# PPTI Task Flow

本文档用于跟踪 PPTI 基于 HPMPC/PIGEON 实现 TinyBERT 隐私推理的后续工程流程。它不是成果总结，而是执行手册：每个阶段都给出目标、输入、输出、命令和验收标准。

## 当前基线

代码基线：

```text
commit: 65a7fe9 Add TinyBERT weight export path
remote: Tryrail4510471/hpmpc.git
branch: master
```

已具备能力：

- `FUNCTION_IDENTIFIER=87` 接入 PPTI Transformer 程序。
- TinyBERT-style encoder smoke test 已跑通。
- `prepare_GEMM` CPU 路径已跑通。
- CUDA GEMM 可编译，但 tiny GEMM 下 CUTLASS 会打印 internal error，暂不作为性能结论。
- `scripts/ppti_export_tinybert.py` 可导出 HuggingFace TinyBERT encoder 权重。
- `scripts/ppti_export_tinybert_inputs.py` 可导出 embedding 权重和 input id 文件。
- 真实 TinyBERT encoder 权重已导出：

```text
models/ppti/tinybert_4l_312d_ppti.bin
params=4568736
layers=4
heads=12
hidden=312
ffn_hidden=1200
```

当前缺口：

- token ids 到 input embedding 的路径已接入。
- attention mask 已进入 input file layout，并已接入 attention Softmax。
- 真实 TinyBERT 完整安全推理尚未运行。
- 尚未做 C++ fixed-point trace 与 PyTorch 明文逐层对齐。
- Softmax/GELU 仍是 smoke 级近似。

## 阶段 0：环境和基线确认

目标：确认远端仓库、依赖、权重文件和 smoke test 都处于可运行状态。

输入：

```text
/home/user/hpmpc
models/ppti/tinybert_4l_312d_ppti.bin
models/ppti/tinybert_ppti_synthetic.bin
```

命令：

```sh
cd /home/user/hpmpc
git status --short --branch
python3 scripts/ppti_reference.py
python3 scripts/ppti_export_tinybert.py --synthetic --output models/ppti/tinybert_ppti_synthetic.bin
```

验收标准：

```text
git status 只有 ignored build/model artifacts
ppti_reference_shape=4x8
synthetic params=2400
```

风险点：

- 不要提交 `models/ppti/*.bin`。
- 不要把 PAT/token 写入 remote URL 或文档。

## 阶段 1：真实权重文件加载验证

目标：确认 C++ 端能够接受真实 TinyBERT encoder 权重文件。

输入：

```text
PPTI_MODEL_FILE=models/ppti/tinybert_4l_312d_ppti.bin
MACRO_FLAGS="-DPPTI_SEQ_LEN=128 -DPPTI_HIDDEN=312 -DPPTI_NUM_HEADS=12 -DPPTI_NUM_LAYERS=4 -DPPTI_FFN_HIDDEN=1200"
```

命令：

```sh
cd /home/user/hpmpc
make -j PARTY=all FUNCTION_IDENTIFIER=87 PROTOCOL=5 DATTYPE=64 BITLENGTH=64 FRACTIONAL=14 USE_CUDA_GEMM=0 \
  MACRO_FLAGS="-DPPTI_SEQ_LEN=128 -DPPTI_HIDDEN=312 -DPPTI_NUM_HEADS=12 -DPPTI_NUM_LAYERS=4 -DPPTI_FFN_HIDDEN=1200"

PPTI_VALIDATE_MODEL_ONLY=1 \
PPTI_MODEL_FILE=models/ppti/tinybert_4l_312d_ppti.bin \
scripts/run.sh -p all -n 3
```

验收标准：

```text
真实 shape 编译通过
模型 header params=4568736
C++ expected params=4568736
验证模式快速结束
```

已完成状态：完成。

## 阶段 2：Embedding 路径

目标：把真实输入从 dummy embedding 替换为 TinyBERT 的 embedding 输出。

需要实现：

1. 导出 embedding 权重：
   - word embeddings
   - position embeddings
   - token type embeddings
   - embeddings LayerNorm gamma/beta
2. 导出输入样本：
   - token ids
   - segment ids
   - position ids
   - attention mask
3. C++ 端新增 input loader：
   - 优先读 `PPTI_INPUT_FILE`
   - 缺文件时回退 dummy input
4. C++ 端新增 embedding 计算：
   - gather word/position/token type embedding
   - 三者相加
   - embedding LayerNorm

建议文件：

```text
scripts/ppti_export_tinybert_inputs.py
programs/transformer.hpp
scripts/ppti_reference.py
```

建议二进制 layout：

```text
input file:
  int32 seq_len
  int32 token_ids[seq_len]
  int32 token_type_ids[seq_len]
  int32 position_ids[seq_len]
  int32 attention_mask[seq_len]

embedding file:
  int32 total_float_params
  float32 word_embeddings[vocab_size * hidden]
  float32 position_embeddings[max_position * hidden]
  float32 token_type_embeddings[type_vocab_size * hidden]
  float32 embedding_layernorm_gamma[hidden]
  float32 embedding_layernorm_beta[hidden]
```

验收标准：

```text
synthetic input embedding smoke test 跑通
真实 TinyBERT tokenizer 生成 input file
Python reference 能打印 embedding output
C++ validate mode 能确认 embedding 文件参数数量
```

已完成状态：完成。

实际新增文件：

```text
scripts/ppti_export_tinybert_inputs.py
programs/transformer.hpp
```

实际导出命令：

```sh
cd /home/user/hpmpc
python3 scripts/ppti_export_tinybert_inputs.py \
  --model huawei-noah/TinyBERT_General_4L_312D \
  --seq-len 16 \
  --text "privacy preserving transformer inference" \
  --embedding-output models/ppti/tinybert_embeddings_ppti.bin \
  --input-output models/ppti/sample_input_seq16.bin
```

实际导出结果：

```text
embedding_file=models/ppti/tinybert_embeddings_ppti.bin
embedding_params=9683856
input_file=models/ppti/sample_input_seq16.bin
input_seq_len=16
embedding_shape=vocab:30522 max_position:512 type_vocab:2 hidden:312
layout=ppti_tinybert_embedding_v1
```

真实 shape validation-only 命令：

```sh
make -j PARTY=all FUNCTION_IDENTIFIER=87 PROTOCOL=5 DATTYPE=64 BITLENGTH=64 FRACTIONAL=14 USE_CUDA_GEMM=0 \
  MACRO_FLAGS="-DPPTI_SEQ_LEN=16 -DPPTI_HIDDEN=312 -DPPTI_NUM_HEADS=12 -DPPTI_NUM_LAYERS=4 -DPPTI_FFN_HIDDEN=1200"

PPTI_VALIDATE_MODEL_ONLY=1 \
PPTI_MODEL_FILE=models/ppti/tinybert_4l_312d_ppti.bin \
PPTI_EMBEDDING_FILE=models/ppti/tinybert_embeddings_ppti.bin \
PPTI_INPUT_FILE=models/ppti/sample_input_seq16.bin \
scripts/run.sh -p all -n 3
```

说明：

- C++ 端当前会读取 `PPTI_EMBEDDING_FILE` 和 `PPTI_INPUT_FILE`，计算 `word + position + token_type` embedding，并执行 embedding LayerNorm。
- `attention_mask` 已经写入 input file，但尚未用于 attention Softmax，下一阶段实现。

## 阶段 3：Attention Mask

目标：在 Softmax 前加入 attention mask，使 padding token 不参与注意力。

需要实现：

```text
scores_h[row, col] += mask[col]
mask[col] = 0 for valid token
mask[col] = large negative value for padding token
```

实现方式：

```text
scores
  -> subtract row max
  -> exp polynomial approximation
  -> multiply exp_approx[col] by public attention_mask[col]
  -> row_sum over unmasked positions
  -> normalize
```

说明：

- mask 视为公开输入。
- 当前 Softmax 仍使用二阶多项式 `exp(x) ~= 1 + x + 0.5x^2`。
- 如果直接向 score 加很大的负数，平方项会让 masked 位置重新变大，因此当前采用 exp 后清零的实现。
- 后续更换 Softmax 近似后，可以再改回标准 additive mask。

验收标准：

```text
无 padding 时输出与旧路径一致
有 padding 时 masked position 的 attention probability 接近 0
Python reference 与 C++ trace 对齐
```

已完成状态：完成。

实际验证：

```sh
python3 scripts/ppti_export_tinybert_inputs.py --synthetic --seq-len 4 --synthetic-pad-from 2 \
  --embedding-output models/ppti/tinybert_embeddings_mask_synthetic.bin \
  --input-output models/ppti/sample_input_seq4_masked.bin

PPTI_EMBEDDING_FILE=models/ppti/tinybert_embeddings_mask_synthetic.bin \
PPTI_INPUT_FILE=models/ppti/sample_input_seq4_masked.bin \
scripts/run.sh -p all -n 3
```

测试 input 文件内容：

```text
seq=4
token_ids=[1, 2, 0, 0]
token_type_ids=[0, 1, 0, 0]
position_ids=[0, 1, 2, 3]
attention_mask=[1, 1, 0, 0]
```

结果：

```text
masked synthetic smoke test 通过
真实 seq=16 validation-only 通过
```

## 阶段 4：Trace 对齐

目标：建立 C++ fixed-point 与 Python/PyTorch 明文的逐层误差对齐流程。

建议新增宏：

```text
PPTI_TRACE=1
PPTI_TRACE_LAYER=0
PPTI_TRACE_HEAD=0
```

建议 reveal 点：

```text
embedding_out
layer0_head0_scores
layer0_head0_probs
layer0_attention_out
layer0_after_attention_layernorm
layer0_ffn_hidden
layer0_output
final_output
```

验收标准：

```text
smoke shape 下每个 reveal 点有 Python reference 对应值
记录 max_abs_error / mean_abs_error
明确误差主要来自 Softmax/GELU/rsqrt 哪一项
```

已完成状态：完成。

实际新增：

```text
PPTI_TRACE=1 编译开关
scripts/ppti_compare_trace.py
scripts/ppti_reference.py --trace
```

Makefile 已支持这些 PPTI 宏：

```text
PPTI_SEQ_LEN
PPTI_HIDDEN
PPTI_NUM_HEADS
PPTI_NUM_LAYERS
PPTI_FFN_HIDDEN
PPTI_TRACE
```

trace 命令：

```sh
cd /home/user/hpmpc
python3 scripts/ppti_reference.py --trace > /tmp/ppti_python_trace.log

make -j PARTY=all FUNCTION_IDENTIFIER=87 PROTOCOL=5 DATTYPE=64 BITLENGTH=64 FRACTIONAL=14 \
  USE_CUDA_GEMM=0 PPTI_TRACE=1

scripts/run.sh -p all -n 3 > /tmp/ppti_cpp_trace.log

python3 scripts/ppti_compare_trace.py \
  --reference /tmp/ppti_python_trace.log \
  --candidate /tmp/ppti_cpp_trace.log
```

trace 点：

```text
embedding_out
layer0_head0_scores
layer0_head0_probs
layer0_out
layer1_out
layer2_out
layer3_out
final_output
```

当前 smoke 误差：

| trace | max_abs_error | mean_abs_error |
| --- | ---: | ---: |
| embedding_out | 0 | 0 |
| layer0_head0_scores | 0.0242097552 | 0.0125943368 |
| layer0_head0_probs | 0.004290848 | 0.00309257975 |
| layer0_out | 0.81499594 | 0.220199583 |
| layer1_out | 1.04883559 | 0.235772405 |
| layer2_out | 0.814369096 | 0.197987481 |
| layer3_out | 0.68885958 | 0.154242283 |
| final_output | 0.68885958 | 0.154242283 |

结论：

- `embedding_out` 完全对齐，说明输入/embedding 路径正确。
- 第一处明显误差从 `layer0_head0_scores` 开始，主要来自 HPMPC fixed-point GEMM/截断与 Python float reference 的差异。
- `layer0_head0_probs` 误差较小，但经过 LayerNorm、GELU 和后续层后被放大。
- 下一步优先做 fixed-point Python reference 或 C++ trace 的定点仿真，而不是盲目改协议。

## 阶段 5：真实 TinyBERT 小样本端到端

目标：用真实 TinyBERT 权重、真实 tokenizer 输入、真实 shape 跑一次端到端安全推理。

推荐先跑：

```text
seq_len=16
hidden=312
heads=12
layers=4
ffn_hidden=1200
```

原因：

- 权重 shape 真实。
- 序列长度较小，先控制 attention 的 `seq_len^2` 成本。
- 更容易定位 Softmax 和 mask 问题。

命令模板：

```sh
make -j PARTY=all FUNCTION_IDENTIFIER=87 PROTOCOL=5 DATTYPE=64 BITLENGTH=64 FRACTIONAL=14 USE_CUDA_GEMM=0 \
  MACRO_FLAGS="-DPPTI_SEQ_LEN=16 -DPPTI_HIDDEN=312 -DPPTI_NUM_HEADS=12 -DPPTI_NUM_LAYERS=4 -DPPTI_FFN_HIDDEN=1200"

PPTI_MODEL_FILE=models/ppti/tinybert_4l_312d_ppti.bin \
PPTI_INPUT_FILE=models/ppti/sample_input_seq16.bin \
scripts/run.sh -p all -n 3
```

验收标准：

```text
协议完整结束
记录 online getTime
记录通信量
输出值可 reveal
```

## 阶段 6：近似函数升级

目标：把 smoke 级近似替换为可用于真实 TinyBERT 精度评估的近似。

优先顺序：

1. Softmax exp approximation
2. LayerNorm reciprocal sqrt
3. GELU approximation

Softmax 候选：

```text
range reduction + low-degree polynomial
piecewise polynomial
lookup-table with MPC-friendly selection
```

GELU 候选：

```text
tanh polynomial approximation
piecewise quadratic/cubic
hard-gelu style approximation
```

验收标准：

```text
smoke trace 误差下降
真实小样本输出不发散
通信和时间可接受
```

## 阶段 7：性能评估

目标：得到 CPU/CUDA、不同 seq_len、不同近似方案下的效率数据。

测试矩阵：

| seq_len | hidden | layers | backend | 目标 |
| --- | --- | --- | --- | --- |
| 4 | 8 | 4 | CPU | smoke regression |
| 16 | 312 | 4 | CPU | 真实权重小输入 |
| 32 | 312 | 4 | CPU | 中等输入 |
| 64 | 312 | 4 | CPU/CUDA | 性能拐点 |
| 128 | 312 | 4 | CPU/CUDA | TinyBERT 标准输入 |

记录项：

```text
online getTime
chrono
P0/P1/P2 send MB
P0/P1/P2 receive MB
是否出现 CUTLASS error
```

CUDA 注意：

- tiny GEMM 不适合评价 CUDA。
- 需要给 CUTLASS 设定支持的小矩阵 fallback 或只在大矩阵启用 CUDA。
- 如果 CUTLASS internal error 仍出现，先做 CPU baseline，不阻塞协议正确性。

## 提交流程

每完成一个阶段：

```sh
cd /home/user/hpmpc
git status --short
git add <changed files>
git commit -m "<concise message>"
git push origin master
```

注意：

- 不提交 `models/ppti/*.bin`。
- 不提交 `executables/`、`core/cuda/bin/`、`*.gch`。
- 如果用 PAT 推送，不要写入 `origin` URL。

## 下一步执行建议

下一步直接进入阶段 5，并并行准备阶段 6：

```text
用真实 TinyBERT seq=16 尝试端到端安全推理，同时建立 fixed-point Python reference。
```

最小可交付结果：

```text
真实 seq=16 协议完整结束
记录 online getTime / 通信量
fixed-point reference 可以复现 layer0_head0_scores 误差方向
StudyNote/PPTI-TaskFlow.md 更新阶段 5 状态
```
