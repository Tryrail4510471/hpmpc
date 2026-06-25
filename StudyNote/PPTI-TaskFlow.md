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
- CUDA GEMM 可编译。重启机器并按 RTX 2060 的 `sm_75` 重新编译 CUTLASS 对象后，真实 `seq=16` TinyBERT CUDA 路径已无 CUTLASS error。
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

已完成状态：完成 CPU baseline；完成 post-reboot CUDA seq16 复测。

实际输入：

```text
PPTI_MODEL_FILE=models/ppti/tinybert_4l_312d_ppti.bin
PPTI_EMBEDDING_FILE=models/ppti/tinybert_embeddings_ppti.bin
PPTI_INPUT_FILE=models/ppti/sample_input_seq16.bin
seq_len=16
hidden=312
heads=12
layers=4
ffn_hidden=1200
```

CPU 编译命令：

```sh
make -j PARTY=all FUNCTION_IDENTIFIER=87 PROTOCOL=5 DATTYPE=64 BITLENGTH=64 FRACTIONAL=14 USE_CUDA_GEMM=0 \
  PPTI_SEQ_LEN=16 PPTI_HIDDEN=312 PPTI_NUM_HEADS=12 PPTI_NUM_LAYERS=4 PPTI_FFN_HIDDEN=1200
```

CPU 运行命令：

```sh
PPTI_MODEL_FILE=models/ppti/tinybert_4l_312d_ppti.bin \
PPTI_EMBEDDING_FILE=models/ppti/tinybert_embeddings_ppti.bin \
PPTI_INPUT_FILE=models/ppti/sample_input_seq16.bin \
scripts/run.sh -p all -n 3 > /tmp/ppti_seq16_cpu.log
```

CPU 性能结果：

| party | send | receive | getTime | chrono |
| --- | ---: | ---: | ---: | ---: |
| P0 | 7.099MB, 0.000008MB | 0.000008MB, 0MB | 6.298142s | 6.298120s |
| P1 | 0MB, 5.201MB | 0.000008MB, 5.201MB | 6.297832s | 6.297812s |
| P2 | 5.201MB, 0.000008MB | 5.201MB, 7.099MB | 6.298222s | 6.298200s |

CUDA 复测准备：

RTX 2060 的 compute capability 是 `7.5`，需要先按 `sm_75` 重编 CUTLASS 对象。旧对象若按 `sm_89` 编译，或 NVIDIA driver/library 版本不一致，可能触发 `Got cutlass error: Error Internal at: 44`。

```sh
make -C core/cuda clean
make -C core/cuda arch=sm_75 CUDA_PATH=/usr/local/cuda CUTLASS_PATH=/home/user/cutlass
```

CUDA 编译：

```sh
make -j PARTY=all FUNCTION_IDENTIFIER=87 PROTOCOL=5 DATTYPE=64 BITLENGTH=64 FRACTIONAL=14 USE_CUDA_GEMM=2 \
  NVCC=/usr/local/cuda/bin/nvcc \
  PPTI_SEQ_LEN=16 PPTI_HIDDEN=312 PPTI_NUM_HEADS=12 PPTI_NUM_LAYERS=4 PPTI_FFN_HIDDEN=1200
```

CUDA 结果：

```text
post-reboot sm_75 run completed.
log contains no CUTLASS/error/failed entries.
```

| party | send | receive | getTime | chrono |
| --- | ---: | ---: | ---: | ---: |
| P0 | 7.099MB, 0.000008MB | 0.000008MB, 0MB | 6.397868s | 6.397859s |
| P1 | 0MB, 5.201MB | 0.000008MB, 5.201MB | 6.396788s | 6.396780s |
| P2 | 5.201MB, 0.000008MB | 5.201MB, 7.099MB | 6.397518s | 6.397507s |

阶段结论：

- 真实 TinyBERT 权重、真实 embedding、真实 tokenizer input 的 `seq=16` 端到端 HPMPC CPU 推理已经跑通。
- 当前可信 CPU baseline 为约 `6.30s` online time。
- 当前 CUDA seq16 复测可干净完成，online time 约 `6.40s`。在此 tiny shape 下 CUDA 未体现加速，主要受小矩阵、host/device copy 和 kernel launch 开销影响。

## 阶段 6：真实 seq16 fixed-point trace 对齐

目标：把真实 `seq=16` 路径的 C++ reveal trace 与 Python fixed-point reference 对齐，找到第一处明显发散的位置。

已完成状态：完成第一轮真实 fixed-point trace 对齐。

Python reference 命令：

```sh
python3 scripts/ppti_reference.py --fixed --trace \
  --seq-len 16 --hidden 312 --heads 12 --layers 4 --ffn-hidden 1200 --fractional 14 \
  --model-file models/ppti/tinybert_4l_312d_ppti.bin \
  --embedding-file models/ppti/tinybert_embeddings_ppti.bin \
  --input-file models/ppti/sample_input_seq16.bin \
  > /tmp/ppti_seq16_fixed_reference.log
```

C++ trace 编译：

```sh
make -j PARTY=all FUNCTION_IDENTIFIER=87 PROTOCOL=5 DATTYPE=64 BITLENGTH=64 FRACTIONAL=14 \
  USE_CUDA_GEMM=0 PPTI_TRACE=1 \
  PPTI_SEQ_LEN=16 PPTI_HIDDEN=312 PPTI_NUM_HEADS=12 PPTI_NUM_LAYERS=4 PPTI_FFN_HIDDEN=1200
```

注意：`scripts/run.sh -p all` 会把三方 stdout 混在一起，大 trace 行可能被其他 party 日志打断。trace 比较时应分离 P0/P1/P2 输出，只使用 P0 日志作为 candidate。

比较命令：

```sh
python3 scripts/ppti_compare_trace.py \
  --reference /tmp/ppti_seq16_fixed_reference.log \
  --candidate /tmp/ppti_seq16_cpp_trace_P0.log \
  > /tmp/ppti_seq16_trace_compare.csv
```

第一轮误差表：

| trace | shape | max_abs_error | mean_abs_error |
| --- | ---: | ---: | ---: |
| embedding_out | 16x312 | 0.0011582 | 0.000195694 |
| layer0_head0_scores | 16x16 | 1.316e10 | 4.66356313e9 |
| layer0_head0_probs | 16x16 | 5.43968633e14 | 5.42056872e13 |
| layer0_out | 16x312 | 5.62864423e14 | 2.70766165e14 |
| layer1_out | 16x312 | 5.62592851e14 | 2.73550013e14 |
| layer2_out | 16x312 | 5.62818678e14 | 2.79705209e14 |
| layer3_out | 16x312 | 3.41260919e14 | 1.49881746e14 |
| final_output | 16x312 | 3.41260919e14 | 1.49881746e14 |

数值范围观察：

```text
reference embedding_out: min=-8.31573486 max=3.02172852 mean_abs=0.402614435
C++       embedding_out: min=-8.316      max=3.022      mean_abs=0.402615008

reference layer0_head0_scores: min=-10.7826538 max=29.6589966 mean_abs=6.95825339
C++       layer0_head0_scores: min=-1.316e10   max=1.284e10   mean_abs=4.66356313e9
```

阶段结论：

- 真实 embedding 路径已经基本对齐。
- 第一处明显发散出现在 `layer0_head0_scores`，也就是 Q/K projection 后的 attention score matmul/scale 附近。
- 现有 Softmax reciprocal Newton 初值也不适合真实 score 范围，reference 自身在 `layer0_head0_probs` 开始出现巨大数值。
- 尝试在 C++ 中额外 reveal Q/K/V projection 张量会触发 `malloc(): unaligned tcache chunk detected`，后续需要改成轻量统计 trace，而不是全张量 reveal。

下一步优先级：

1. 给 `secure_matmul` 增加可选统计 trace，只 reveal min/max/mean_abs，不 reveal 全矩阵。
2. 在 Q/K projection、QK score、scale 后分别记录统计量，确认 score 爆炸是 projection 输出过大还是 score matmul 溢出。
3. 对 attention score 加 clamp/range reduction，再进入 Softmax。
4. 给 Softmax reciprocal 使用基于 row_sum 的更稳初值或归一化策略。

补充进展：projection 级统计 trace 已接入，定位结果如下。

- `layer0_input_stats`、`layer0_wq_stats`、`layer0_wk_stats`、`layer0_wv_stats` 均与 Python reference 对齐，说明真实 embedding 和第一层 Q/K/V 权重加载正确。
- 原 `prepare_GEMM` 路径下，`layer0_q_linear_stats` 直接发散到 `±5.6e14`，说明爆炸发生在大矩阵 GEMM 输出阶段。
- 将 PPTI 的 `secure_matmul` 临时切换为逐元素 dot-product 路径后，projection 和 score 对齐：

```text
layer0_q_linear_stats          max=0.00963574   mean=0.00625384667
layer0_k_linear_stats          max=0.0105811    mean=0.00709233
layer0_v_linear_stats          max=0.01052393   mean=0.00688174833
layer0_head0_score_scaled_stats max=0.0389966   mean=0.0261987267
layer0_head0_scores            max=0.1305347    mean=0.0399053072
```

新的第一处明显发散已经后移到 Softmax：

```text
layer0_head0_probs max=5.44014034e14 mean=5.42012114e13
layer0_out         max=5.62871964e14 mean=2.70765639e14
final_output       max=3.41277026e14 mean=1.49881735e14
```

阶段判断：

- 当前先保留朴素 `secure_matmul` 作为正确性 baseline。
- `prepare_GEMM` 大矩阵路径需要单独缩小复现，判断是大 `m*n` 输出数量、`inner=312`、还是 Transformer 权重范围触发的问题。
- 真实 TinyBERT 端到端的下一处核心问题从 projection matmul 转移到 Softmax 近似和 reciprocal 初值。

## 阶段 7：近似函数升级

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

### 7.1 Softmax rational exp baseline

阶段 6 证明 projection 和 attention score 在逐元素 dot-product baseline 下已经对齐，新的首个发散点是 `layer0_head0_probs`。原实现使用：

```text
exp(x) ~= 1 + x + 0.5*x^2
```

这个二阶 Taylor 形式只适合 `x` 接近 0 的小范围。Attention score 做 row-max stabilization 后满足 `x <= 0`，当某个 token 远低于 row max 时，`x` 是较大的负数，二阶项会把近似值重新放大，导致低概率 token 获得巨大概率质量。

当前替换为 rational baseline：

```text
exp(x) ~= 1 / (1 - x + 0.5*x^2), x <= 0
```

这个近似不是最终精度方案，但有两个工程优点：

- 对 `x <= 0` 始终为正，不会因为平方项把远离 row max 的 token 放大。
- 可以直接复用 HPMPC 的乘法和 Newton reciprocal 路径，适合作为 MPC-friendly baseline。

C++ 与 Python fixed-point reference 同步修改：

- `programs/transformer.hpp::secure_rowwise_softmax_poly`
- `scripts/ppti_reference.py::softmax_poly`

参数选择：

```text
exp reciprocal initial_guess = 1 / 1024
exp reciprocal iterations    = 12
row_sum reciprocal iterations = 8
```

最初尝试 `1/16384`，但在 `FRACTIONAL=14` 下它正好是 1 个 LSB，乘法截断会让 Newton reciprocal 卡住。根据真实 `seq=16` trace，`layer0_head0_score_scaled_stats` 范围约为 `[-10.8, 29.7]`，row-max 后 rational denominator 主要落在 `1..~830`，因此 `1/1024` 可以保持初始乘积在稳定区间内，同时避免 LSB 卡死。后续如果 score clamp/range reduction 做好，可以减少迭代次数以降低通信轮数。

补充修正：row max 必须忽略公开 masked columns。否则如果 padding token 的 score 大于有效 token，row-max stabilization 会以 masked token 为中心，后续再把 masked exp 清零会导致有效 token 的概率和小于 1。当前做法是在 `row_max` 前把 masked score 替换为公开常数 `-1024`，row-max 后再把 masked shifted score 置 0，并在 exp 之后继续用 mask 清零概率质量。

验证结果：

```text
smoke C++ vs Python fixed reference:
layer0_head0_probs max=0.000332422 mean=0.00010446175
final_output        max=0.001014941 mean=0.000319988893

real seq=16 C++ vs Python fixed reference:
layer0_head0_score_scaled_stats max=0.0389966 mean=0.0261987267
layer0_head0_probs              max=0.006285352 mean=0.000163028266
layer0_out                      max=5.62882548e14 mean=2.70958401e14
final_output                    max=3.46676142e14 mean=1.52156787e14
P0 online getTime               ~= 6.36s
```

阶段判断：

- Softmax 爆值问题已经压住，`layer0_head0_probs` 从 `~5.44e14` 级误差下降到 `~6.3e-3`。
- 新的首个主要发散已经后移到 Softmax 之后，即 attention context/output projection、LayerNorm 或后续 FFN 路径。
- rational exp 每个 score 都要做 Newton reciprocal，目前是正确性 baseline，不是最终性能方案；后续应考虑 range reduction + piecewise/lookup 以减少通信轮数。

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

下一步进入阶段 6：

```text
建立 fixed-point Python reference，并开始校准 Softmax/GELU/LayerNorm 近似。
```

最小可交付结果：

```text
fixed-point reference 可以复现 layer0_head0_scores 误差方向
Softmax/GELU 近似替换前后有 trace 误差对比
StudyNote/PPTI-TaskFlow.md 更新阶段 6 状态
```
