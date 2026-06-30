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

## 阶段 8：Softmax 后路径对齐

目标：阶段 7 已把 `layer0_head0_probs` 的误差从 `~5.44e14` 降到 `max ~= 0.0063`，但 `layer0_out/final_output` 仍然存在巨大差异。阶段 8 的任务是沿 Softmax 之后的链路定位新的首个主要发散点。

新增 trace 点：

```text
layer0_head0_context_stats
layer0_concat_context_stats
layer0_wo_stats
layer0_attn_out_linear_stats
layer0_attn_out_stats
layer0_attn_residual_pre_ln_stats
layer0_attn_residual_post_ln_stats
layer0_ffn_hidden_linear_stats
layer0_ffn_hidden_gelu_stats
layer0_ffn_out_stats
layer0_ffn_residual_pre_ln_stats
layer0_ffn_residual_post_ln_stats
```

排查顺序：

```text
probs * V
concat heads
attention output projection
attention residual
attention LayerNorm
FFN first projection
GELU
FFN second projection
FFN residual
final LayerNorm
```

验收标准：

```text
找出 layer0_out 巨大误差最早出现的 trace 点
确认该点是协议实现问题、fixed-point 近似问题，还是 Python/C++ reference 不一致
```

阶段 8 实测结果：

```text
real seq=16 C++ vs Python fixed reference:
layer0_head0_probs                  max=0.006285352 mean=0.000163028266
layer0_head0_context_stats          max=0.01053516  mean=0.00730345433
layer0_concat_context_stats         max=0.01160742  mean=0.004390771
layer0_attn_out_linear_stats        max=0.02069727  mean=0.01238228
layer0_attn_out_stats               max=0.02223926  mean=0.0131482977
layer0_attn_residual_pre_ln_stats   max=0.06151611  mean=0.0248404243
layer0_attn_residual_post_ln_stats  max=0.0294043   mean=0.0170682947
layer0_ffn_hidden_linear_stats      max=0.1601343   mean=0.0822365767
layer0_ffn_hidden_gelu_stats        max=252.882     mean=120.04788
layer0_ffn_out_stats                max=5231.235    mean=2029.3894
layer0_ffn_residual_post_ln_stats   max=5.62848948e14 mean=4.6544643e14
P0 online getTime                   ~= 6.48s
```

原始统计：

```text
reference layer0_ffn_hidden_linear_stats: min=-62.9501343 max=58.3748779 mean_abs=1.50869753
reference layer0_ffn_hidden_gelu_stats:   min=-34082.882 max=24906.9573 mean_abs=45.054341
reference layer0_ffn_out_stats:           min=-660431.235 max=920442.444 mean_abs=2128.48919

C++ layer0_ffn_hidden_linear_stats:       min=-62.79 max=58.29 mean_abs=1.507
C++ layer0_ffn_hidden_gelu_stats:         min=-3.383e4 max=2.48e4 mean_abs=44.75
C++ layer0_ffn_out_stats:                 min=-6.552e5 max=9.196e5 mean_abs=2114
```

阶段判断：

- Softmax 后到 attention residual/attention LayerNorm 的 C++ 与 Python fixed reference 仍保持小误差。
- 新的主要数值问题不是协议实现不一致，而是当前 cubic GELU surrogate 在真实 TinyBERT FFN pre-activation 范围 `[-63, 58]` 下严重放大数值。
- FFN output 被放大到 `~9e5` 后，后续 LayerNorm reciprocal sqrt 也失稳，最终表现为 `layer0_out/final_output` 的 `1e14` 级异常。
- 下一阶段应优先替换 GELU approximation，再校准 LayerNorm reciprocal sqrt。

## 阶段 9：GELU approximation 与 LayerNorm rsqrt 稳定化

目标：阶段 8 已定位到当前 cubic GELU surrogate 在真实 TinyBERT FFN pre-activation `[-63, 58]` 下严重放大数值。本阶段先建立不爆炸的 correctness baseline。

原实现：

```text
gelu_poly(x) = 0.5*x + 0.125*x^3
```

问题：

```text
FFN pre-activation min/max ~= [-63, 58]
cubic term 会产生 ~3e4 级输出
FFN output 进一步到 ~9e5
post-FFN LayerNorm reciprocal sqrt 失稳
```

当前替换为 ReLU-GELU baseline：

```text
gelu_baseline(x) = max(x, 0)
```

实现方式：

- C++：`secure_gelu_poly` 内用 `max_min_sint<0, BITLENGTH>` 对每个 secret value 和公开 0 求 max。
- Python fixed reference：`max(x, 0)`。
- C++/Python：attention LayerNorm 和 FFN LayerNorm 都切到同一套稳定参数：

```text
centered_scale=128
rsqrt_iterations=24
rsqrt_initial_guess=1/64
epsilon=0.001 / centered_scale^2
```

- Trace：新增每层关键统计，形如 `layerN_attn_residual_post_ln_stats`、`layerN_ffn_residual_post_ln_stats`、`layerN_out`。

阶段判断：

- 这不是最终 TinyBERT GELU 精度方案，而是第一步 correctness baseline。
- ReLU baseline 先压住 cubic GELU 的三次项爆炸；LayerNorm 的 centered scaling 解决多层 attention/FFN rsqrt 初值失稳。
- 后续再替换为更贴近 GELU 的 hard-GELU、piecewise polynomial 或 lookup/range-reduction 方案。

阶段 9 实测结果：

```text
real seq=16 C++ vs Python fixed reference:
layer0_attn_residual_post_ln_stats  max=28.7936377 mean=9.78902855
layer0_ffn_residual_post_ln_stats   max=2.6417783  mean=0.901285394
layer0_out                          max=2.6417783  mean=0.2086785
layer1_attn_residual_post_ln_stats  max=35.5218042 mean=17.5961049
layer1_ffn_residual_post_ln_stats   max=3.7191846  mean=1.44889311
layer1_out                          max=5.4381221  mean=0.450146234
layer2_attn_residual_post_ln_stats  max=5.81895508 mean=2.14596096
layer2_ffn_residual_post_ln_stats   max=5.56136475 mean=2.17044002
layer2_out                          max=7.63491211 mean=0.71829986
layer3_attn_residual_post_ln_stats  max=2.12800244 mean=1.15842895
layer3_ffn_residual_post_ln_stats   max=2.01873633 mean=0.9147061
layer3_out                          max=2.25173633 mean=0.245499895
final_output                        max=2.25173633 mean=0.245499895
```

原始输出范围：

```text
Python final_output: min=-3.07116699 max=4.35473633 mean_abs=0.3119489596
C++ final_output:    min=-3.703      max=2.336      mean_abs=0.2183957588
```

阶段结论：

- `layer0_ffn_hidden_gelu_stats` 不再出现 `~3e4` 放大。
- `layer0_ffn_out_stats` 不再出现 `~9e5` 放大。
- 4 层真实 TinyBERT seq=16 端到端不再出现 `1e14` 级爆值。
- 当前仍是 correctness baseline：final 输出误差还在 `max ~= 2.25`，需要继续做 GELU 精度、LayerNorm 近似和固定点参数校准。
- P0 trace 日志有时会先出现一组全零 trace，后面才是真实 reveal trace；对比时以有效 trace 为准。

验收标准：

```text
layer0_ffn_hidden_gelu_stats 不再出现 ~3e4 放大：完成
layer0_ffn_out_stats 不再出现 ~9e5 放大：完成
layer0_ffn_residual_post_ln_stats 不再出现 1e14 爆值：完成
真实 seq=16 C++ vs Python fixed reference 仍能逐层对齐：完成
```

## 阶段 10：性能评估

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

阶段 10 第一轮结果：

```text
seq=16 CPU no-trace:
P0 send=9.915MB / 0.000008MB  getTime=6.524174s
P1 send=0MB / 8.291MB         getTime=6.525956s
P2 send=8.291MB / 0.000008MB  getTime=6.524253s

seq=16 CUDA no-trace:
P0 send=9.915MB / 0.000008MB  getTime=7.345492s
P1 send=0MB / 8.291MB         getTime=7.346379s
P2 send=8.291MB / 0.000008MB  getTime=7.345073s

seq=32 CPU no-trace:
P0 send=25.83MB / 0.000008MB  getTime=9.160846s
P1 send=0MB / 21.96MB         getTime=9.160161s
P2 send=21.96MB / 0.000008MB  getTime=9.159894s
```

seq=32 输入导出命令：

```sh
python3 scripts/ppti_export_tinybert_inputs.py \
  --model huawei-noah/TinyBERT_General_4L_312D \
  --seq-len 32 \
  --text "privacy preserving transformer inference with secure multi party computation baseline benchmark" \
  --embedding-output models/ppti/tinybert_embeddings_ppti.bin \
  --input-output models/ppti/sample_input_seq32.bin
```

seq=32 C++ vs Python fixed reference：

```text
layer0_head0_probs max=0.005459473 mean=0.000100898513
layer0_out         max=3.10990518  mean=0.204591555
layer1_out         max=6.0576191   mean=0.6396979
layer2_out         max=7.6165815   mean=0.549741695
layer3_out         max=3.82263965  mean=0.308568136
final_output       max=3.82263965  mean=0.308568136
```

seq=32 原始输出范围：

```text
Python final_output: min=-5.28063965 max=4.45996094 mean_abs=0.4510626182
C++ final_output:    min=-2.044      max=4.337      mean_abs=0.2465803552
```

阶段判断：

- Stage 9 稳定化参数可以从 `seq=16` 扩展到 `seq=32`，未出现 `1e14` 级数值爆炸。
- `seq=16` CUDA 可跑通且无 CUTLASS error，但当前 no-trace 时间约 `7.35s`，慢于 CPU 的 `6.52s`。
- `seq=32` 通信量显著上升：P0 从 `9.915MB` 增至 `25.83MB`，P1/P2 从 `8.291MB` 增至 `21.96MB`。
- `seq=32` online time 从 `~6.52s` 增至 `~9.16s`。
- P0 trace 日志仍可能先输出一组全零 trace，后续才是真实 reveal trace；对比时需要使用有效 trace。

## 阶段 11：prepare_GEMM 大矩阵路径修复

目标：修复 TinyBERT projection 走 `prepare_GEMM` 时的 `1e14` 级爆值，并保留 manual dot baseline 作为 correctness oracle。

根因：

```text
TinyBERT 导出权重 layout: [inner, cols] row-major
manual dot 访问方式:      rhs[k * cols + c]
prepare_GEMM 访问方式:    B[c * inner + k]
```

因此，直接把 Transformer 的 RHS 传给 `prepare_GEMM` 时，GEMM 会按 `[cols, inner]` 读取 `[inner, cols]` 的权重，导致真实 Q/K/V projection 错位，进一步触发 `layer0_q_linear_stats ~= 5e14` 的错误。

实现：

```text
PPTI_MATMUL_BACKEND=0: manual dot baseline, 默认路径
PPTI_MATMUL_BACKEND=1: prepare_GEMM adapter
```

adapter 在调用 `prepare_GEMM` 前做一次 RHS layout 转换：

```text
rhs_for_gemm[c * inner + k] = rhs[k * cols + c]
prepare_GEMM(lhs, rhs_for_gemm, out, rows, cols, inner, false)
```

正确性验证：

```text
smoke GEMM adapter vs Python fixed reference:
embedding_out        max=0
layer0_head0_scores  max=0.0001222656 mean=0.0000727783299
layer0_head0_probs   max=0.000332422  mean=0.00010446175
final_output         max=0.76280518   mean=0.275500872

real seq=16 GEMM adapter vs Python fixed reference:
layer0_q_linear_stats           max=0.00963574  mean=0.00625384667
layer0_head0_score_scaled_stats max=0.0389966   mean=0.0261987267
layer0_head0_probs              max=0.006285352 mean=0.000163028266
layer0_out                      max=2.6417783   mean=0.2086785
layer1_out                      max=5.4381221   mean=0.450146234
layer2_out                      max=7.63491211  mean=0.71829986
layer3_out                      max=2.25173633  mean=0.245499895
final_output                    max=2.25173633  mean=0.245499895
```

性能验证：

```text
seq=16 manual dot CPU no-trace:
getTime ~= 6.52s

seq=16 prepare_GEMM adapter CPU no-trace:
P0 getTime=7.306864s
P1 getTime=7.308431s
P2 getTime=7.307701s

seq=16 prepare_GEMM adapter CUDA no-trace:
P0 getTime=7.701905s
P1 getTime=7.706207s
P2 getTime=7.705598s
```

阶段判断：

- `prepare_GEMM` 大矩阵爆值的直接原因已经定位并修复：不是权重文件错误，而是 RHS layout 不匹配。
- `PPTI_MATMUL_BACKEND=1` 已可作为 HPMPC GEMM 正确性路径使用。
- 当前 adapter 每次 matmul 都重排 RHS，且 seq16 形状下 CPU/CUDA GEMM adapter 都慢于 manual dot baseline；下一步性能优化应缓存或预导出 GEMM layout 权重，并减少小矩阵 CUDA 调用。
- 默认仍保持 `PPTI_MATMUL_BACKEND=0`，避免影响已有 correctness/performance baseline。

## 阶段 12：GELU approximation 升级

目标：把阶段 9 的 ReLU-GELU correctness baseline 替换为更接近 true GELU 的 MPC-friendly 近似，同时保持真实 `seq=16` 端到端稳定。

候选评估：

```text
true GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
ReLU baseline: max(x, 0)
standard hard-GELU: x * clamp(0.5 + x / 6, 0, 1)
tuned hard-GELU: x * clamp(0.5 + 0.3125 * x, 0, 1)
```

真实 layer0 FFN pre-activation 分布：

```text
min=-63.7238745 max=54.4401052 mean_abs=1.5958276
```

相对 true GELU 的 layer0 approximation error：

```text
ReLU baseline              max=0.1699712074 mean=0.0977432140
standard hard-GELU x/6     max=0.2953695466 mean=0.1349476164
tuned alpha search:
  alpha=0.30               max=0.0796308238 mean=0.0247579766
  alpha=0.325              max=0.0953306773 mean=0.0225662337
```

实现选择：

```text
gelu_tuned(x) = x * clamp(0.5 + 0.3125 * x, 0, 1)
```

说明：

- `0.3125 = 5/16`，固定点表达友好。
- 比 ReLU 更接近 true GELU。
- 比标准 `x/6` hard-GELU 更适合当前 TinyBERT FFN 激活分布。
- MPC 实现需要两次 `max_min_sint` clamp 和一次 secret multiplication。

真实 seq=16 C++ vs Python fixed reference：

```text
layer0_head0_probs             max=0.006285352 mean=0.000163028266
layer0_ffn_hidden_gelu_stats   max=67.288826  mean=22.5197127
layer0_out                     max=4.259229   mean=0.261641119
layer1_out                     max=5.8960229  mean=0.586889103
layer2_out                     max=5.5986333  mean=0.566933372
layer3_out                     max=2.25368652 mean=0.317721248
final_output                   max=2.25368652 mean=0.317721248
```

真实 seq=16 no-trace CPU 性能：

```text
P0 send=13.59MB / 0.000008MB  getTime=6.746801s
P1 send=0MB / 10.74MB         getTime=6.747408s
P2 send=10.74MB / 0.000008MB  getTime=6.746765s
```

阶段判断：

- tuned hard-GELU 在真实 layer0 分布上比 ReLU 更接近 true GELU，mean approximation error 从 `0.0977` 降到约 `0.02-0.03` 区间。
- 端到端仍稳定，未出现 `1e14` 级爆值。
- 相对 Python fixed reference 的 final mean error 从 ReLU baseline 的 `0.2455` 变为 `0.3177`，说明更接近 true GELU 的函数近似会放大当前 LayerNorm/fixed-point 对齐误差。
- 速度从 ReLU baseline 的 `~6.52s` 变为 `~6.75s`，通信量增加，原因是 hard-GELU 多了 clamp 比较和一次乘法。
- 后续需要配合 LayerNorm/fixed-point 校准，不能只替换 GELU。

## 阶段 13：LayerNorm / fixed-point 参数校准

目标：降低 tuned hard-GELU 路径下的 C++ fixed trace vs Python fixed reference 误差，并保持 attention/FFN 逐层稳定。

实现：

- C++ 新增可配置 LayerNorm 参数：

```text
PPTI_LN_CENTERED_SCALE
PPTI_LN_RSQRT_ITERATIONS
PPTI_LN_RSQRT_INITIAL_GUESS
```

- Python reference 同步新增：

```text
--ln-centered-scale
--ln-rsqrt-iterations
--ln-rsqrt-initial-guess
```

- Makefile `CONFIG_OPTIONS` 已加入 `PPTI_MATMUL_BACKEND` 和 `PPTI_LN_*`，后续扫参应使用正常 make 变量，不再用 `MACRO_FLAGS` 临时覆盖，否则容易绕过 HPMPC 配置更新机制。

候选结果：

```text
baseline tuned hard-GELU:
LN centered_scale=128
rsqrt_iterations=24
rsqrt_initial_guess=1/64
final_output max=2.25368652 mean=0.317721248
seq16 CPU getTime ~= 6.75s

candidate scale=64, init=1/32:
final_output max=4144999.53 mean=1084967.15
结论：淘汰，LayerNorm/rsqrt 失稳。

candidate scale=256, init=1/128:
layer0_head0_probs                 max=0.006285352 mean=0.000163028266
layer0_attn_residual_post_ln_stats max=0.01136182  mean=0.00764328367
layer0_ffn_hidden_linear_stats     max=0.0903931   mean=0.0551504367
layer0_ffn_hidden_gelu_stats       max=0.0587041   mean=0.0197157167
layer0_ffn_out_stats               max=5.320972    mean=1.79608625
layer0_ffn_residual_post_ln_stats  max=0.0702319   mean=0.0287691723
layer0_out                         max=0.184137012 mean=0.0117048779
layer1_out                         max=0.123886426 mean=0.00582938099
layer2_out                         max=0.056611133 mean=0.00545621854
layer3_out                         max=0.070651855 mean=0.00279345366
final_output                       max=0.070651855 mean=0.00279345366
```

正式默认值更新：

```text
PPTI_LN_CENTERED_SCALE=256.0f
PPTI_LN_RSQRT_ITERATIONS=24
PPTI_LN_RSQRT_INITIAL_GUESS=1/128
```

性能：

```text
seq16 CPU no-trace, tuned hard-GELU + LN scale=256:
P0 send=13.59MB / 0.000008MB  getTime=7.087780s
P1 send=0MB / 10.74MB         getTime=7.086622s
P2 send=10.74MB / 0.000008MB  getTime=7.085820s
```

阶段判断：

- `scale=256, init=1/128` 是当前最优 LayerNorm/fixed-point baseline。
- final fixed-reference mean error 从 `0.3177` 降到 `0.00279`，并且 attention Softmax 误差保持在 Stage 12 水平。
- 通信量不变，耗时约从 `6.75s` 增到 `7.09s`，属于可接受范围。
- 下一步应在 `seq=32` 上复验该默认配置，再进入 GEMM layout 缓存和 Softmax 低通信优化。

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

下一步进入阶段 14：

```text
复验 seq=32，并优化 GEMM adapter 的权重布局缓存。
```

最小可交付结果：

```text
seq32 fixed trace 使用 LN scale=256 复验
评估预转置/预导出 GEMM layout 对性能的影响
```

## 阶段 14：seq32 复验与 GEMM layout cache 评估

目标：

```text
确认 Stage 13 的 LayerNorm 默认参数能推广到 seq=32。
评估是否可以通过缓存/预加载 GEMM RHS layout 加速 PPTI_MATMUL_BACKEND=1。
```

seq32 trace 复验结果：

```text
layer0_head0_probs max=0.005459473 mean=0.000100898513
layer0_out         max=0.369475590 mean=0.018789278400
layer1_out         max=0.202377340 mean=0.007456045940
layer2_out         max=0.056631055 mean=0.005591327830
layer3_out         max=0.069527051 mean=0.002721626490
final_output       max=0.069527051 mean=0.002721626490
```

seq32 no-trace 性能：

```text
P0 send=33.18MB / 0.000008MB  getTime=9.483334s
P1 send=0MB / 26.86MB         getTime=9.478727s
P2 send=26.86MB / 0.000008MB  getTime=9.480891s
```

结论：

- `LN_CENTERED_SCALE=256, RSQRT_ITERATIONS=24, INIT=1/128` 在 seq32 上继续稳定。
- seq32 的 final fixed-reference mean error 为 `0.00272`，与 seq16 的 `0.00279` 同一量级。
- 通信量从 seq16 的 P0 `13.59MB` 增加到 seq32 的 P0 `33.18MB`，符合 attention/FFN 规模增长预期。

GEMM layout cache 实验：

```text
方案 A：加载后复制一份 GEMM RHS layout cache。
结果：P1/P2 double free or corruption，P0 残留等待。

方案 B：PPTI_MATMUL_BACKEND=1 时直接按 GEMM RHS layout 接收权重。
结果：仍触发 P1/P2 double free or corruption。
```

判断：

- 当前 `prepare_GEMM` 路径对 RHS share 对象的生命周期/释放有额外假设，不能简单把真实 TinyBERT 权重改成长生命周期 cache。
- 已恢复到稳定实现：`PPTI_MATMUL_BACKEND=1` 保留原始临时 RHS layout adapter，默认 correctness baseline 仍使用 manual dot。
- Stage 14 的优化结论是“缓存权重 layout 不是当前可安全落地的加速点”。下一步应优先做 Softmax 低通信近似、LayerNorm round/scale 简化，或从导出格式层面设计专门的 GEMM-only 权重文件并单独验证 share 对象生命周期。

恢复后稳定性验证：

```text
seq16 CPU manual dot no-trace:
P0 send=13.59MB / 0.000008MB  getTime=8.595128s
P1 send=0MB / 10.74MB         getTime=8.592344s
P2 send=10.74MB / 0.000008MB  getTime=8.592262s
```

## 阶段 15：Softmax Newton 轮数参数化与候选扫描

目标：

```text
降低 attention Softmax rational baseline 的 reciprocal Newton 轮数。
保留 12/8 作为默认精度 baseline，同时允许通过编译参数扫描速度/误差折中。
```

新增参数：

```text
PPTI_SOFTMAX_EXP_ITERATIONS
PPTI_SOFTMAX_ROWSUM_ITERATIONS
```

默认值保持：

```text
exp reciprocal iterations    = 12
row_sum reciprocal iterations = 8
```

对应实现：

```text
programs/transformer.hpp
scripts/ppti_reference.py
Makefile CONFIG_OPTIONS
```

seq16 no-trace 性能扫描：

```text
12/8 baseline:
P0 send=13.59MB  P1/P2 send=10.74MB  getTime ~= 8.59s

10/8 candidate:
P0 send=13.20MB  P1/P2 send=10.35MB  getTime ~= 6.82s

10/6 candidate:
P0 send=13.17MB  P1/P2 send=10.32MB  getTime ~= 7.28s

8/6 candidate:
P0 send=12.78MB  P1/P2 send=9.928MB  getTime ~= 6.23s
```

相对 12/8 C++ trace baseline 的漂移：

```text
10/8:
layer0_head0_probs max=0.00110 mean=0.000195586
final_output       max=0.01910 mean=0.00904820

10/6:
layer0_head0_probs max=0.00110 mean=0.000196445
final_output       max=0.10020 mean=0.03078600

8/6:
layer0_head0_probs max=0.00171 mean=0.000306172
final_output       max=0.13360 mean=0.04562570
```

阶段判断：

- `10/8` 是当前最稳妥的性能候选：速度明显改善，final drift 仍在 `~0.009` 量级。
- `8/6` 速度最快，但端到端漂移偏大，暂不建议作为默认。
- `10/6` 没有体现稳定收益，row_sum reciprocal 减轮对多层输出扰动更大。
- 代码默认值仍保持 `12/8`，避免无意降低当前精度 baseline。性能实验可显式传入 `PPTI_SOFTMAX_EXP_ITERATIONS=10 PPTI_SOFTMAX_ROWSUM_ITERATIONS=8`。

示例命令：

```sh
make -j PARTY=all FUNCTION_IDENTIFIER=87 PROTOCOL=5 DATTYPE=64 BITLENGTH=64 FRACTIONAL=14 \
  USE_CUDA_GEMM=0 PPTI_TRACE=0 \
  PPTI_SEQ_LEN=16 PPTI_HIDDEN=312 PPTI_NUM_HEADS=12 PPTI_NUM_LAYERS=4 PPTI_FFN_HIDDEN=1200 \
  PPTI_SOFTMAX_EXP_ITERATIONS=10 PPTI_SOFTMAX_ROWSUM_ITERATIONS=8
```

## 阶段 16：Softmax 10/8 候选在 seq32 上复验

目标：

```text
确认 Stage 15 的 10/8 Softmax 候选不是只在 seq16 上有效。
在 seq32 上同时比较性能、通信量和相对 12/8 baseline 的 trace drift。
```

同环境 seq32 no-trace 对比：

```text
12/8 baseline:
P0 send=33.18MB / 0.000008MB  getTime=10.782942s
P1 send=0MB / 26.86MB         getTime=10.777614s
P2 send=26.86MB / 0.000008MB  getTime=10.780986s

10/8 candidate:
P0 send=31.61MB / 0.000008MB  getTime=10.790858s
P1 send=0MB / 25.29MB         getTime=10.787334s
P2 send=25.29MB / 0.000008MB  getTime=10.789845s
```

seq32 上 `10/8` 相对 `12/8` C++ trace baseline 的漂移：

```text
layer0_head0_probs max=0.00073 mean=0.0000895605
layer0_out         max=0.24860 mean=0.0271454397
layer1_out         max=0.00790 mean=0.0028171985
layer2_out         max=0.01490 mean=0.0054503796
layer3_out         max=0.02520 mean=0.0064717668
final_output       max=0.02520 mean=0.0064717668
```

阶段判断：

- `10/8` 在 seq32 上没有明显降低 wall-clock time，当前 CPU 路径的瓶颈不完全在 Softmax reciprocal 轮数。
- `10/8` 能稳定减少通信量：P0 从 `33.18MB` 降到 `31.61MB`，P1/P2 从 `26.86MB` 降到 `25.29MB`。
- seq32 final drift mean 为 `0.00647`，比 seq16 的 `0.00905` 更小，说明该候选在 seq32 上没有放大误差。
- 结论：`10/8` 可作为“省通信、轻微扰动”的实验候选；默认仍保持 `12/8`。下一步优化应该转向 LayerNorm rsqrt iteration 扫描或 Softmax 结构性替换，而不是继续盲目减少 row_sum 轮数。

## 阶段 17：LayerNorm rsqrt iteration 扫描

目标：

```text
在 Stage 13 的 scale=256, init=1/128 稳定配置上减少 reciprocal sqrt Newton 轮数。
评估 24 -> 23/22/20/16/12 是否能降低耗时，同时保持逐层 trace 稳定。
```

seq16 no-trace 性能扫描：

```text
24 iterations:
P0 send=13.59MB  P1/P2 send=10.74MB  getTime ~= 7.08s

20 iterations:
P0 send=13.58MB  P1/P2 send=10.73MB  getTime ~= 7.02s

16 iterations:
P0 send=13.56MB  P1/P2 send=10.71MB  getTime ~= 7.78s

12 iterations:
P0 send=13.54MB  P1/P2 send=10.70MB  getTime ~= 8.04s

23 iterations:
P0 send=13.59MB  P1/P2 send=10.74MB  getTime ~= 6.65s

22 iterations:
P0 send=13.58MB  P1/P2 send=10.73MB  getTime ~= 7.40s
```

相对 24-iteration C++ trace baseline 的漂移：

```text
23 iterations:
layer0_out   max=0.4002 mean=0.104171466
final_output max=0.3674 mean=0.175488536

22 iterations:
layer0_out   max=0.8371 mean=0.224064377
final_output max=0.6500 mean=0.269629275

20 iterations:
layer0_out   max=1.7506 mean=0.464294601
final_output max=1.7332 mean=0.793736506

16 iterations:
layer0_out   max=1.6004 mean=0.665083359
final_output max=1.8554 mean=0.908531132

12 iterations:
layer0_out   max=1.5717 mean=0.771270204
final_output max=2.0093 mean=0.821134712
```

20-iteration 初值补偿实验：

```text
20 iterations, init=1/64:
layer0_out   max=0.9646 mean=0.261143782
final_output max=0.9200 mean=0.401008482

20 iterations, init=1/256:
layer0_out   max=1.7629 mean=0.648405579
final_output max=1.6084 mean=0.875118692
```

阶段判断：

- LayerNorm rsqrt 对迭代数极其敏感；即使只从 24 降到 23，第一层后输出已经明显漂移。
- 20/16/12 虽然通信量略降，但数值误差不可接受，且 wall-clock time 不稳定，不具备默认优化价值。
- 调整初值没有救回 20 轮，`1/64` 和 `1/256` 都比当前 `1/128` 更差。
- 当前结论：`PPTI_LN_RSQRT_ITERATIONS=24` 继续作为稳定默认值。后续 LayerNorm 优化应改为结构性方案，例如分段/查表 rsqrt、公开尺度估计后的更好初值、或减少 LayerNorm 调用频率，而不是简单减少 Newton 轮数。

## 阶段 18：LayerNorm rsqrt 初值边界扫描

目标：

```text
验证 Stage 17 的失败是否只是初值太小导致的。
在 20-iteration 下扫描更大的 public initial guess，寻找能否用更好初值替代 24-iteration baseline。
```

20-iteration 初值粗扫，相对 24-iteration baseline 的 final drift：

```text
init=1/128: final_output max=1.7332 mean=0.793736506
init=1/64:  final_output max=0.9200 mean=0.401008482
init=1/32:  final_output max=0.2715 mean=0.101596284
init=1/16:  final_output max=0.6158 mean=0.201383944
init=1/8:   final_output ~= 1e10 级，发散
init=1/4:   final_output ~= 1e10 级，发散
init=1/2:   final_output ~= 1e10 级，发散
init=1:     final_output ~= 1e10 级，发散
```

20-iteration 初值细扫：

```text
init=1/96: final_output max=1.7416 mean=0.680815753
init=1/80: final_output max=1.3100 mean=0.556016066
init=1/64: final_output max=0.9200 mean=0.401008482
init=1/48: final_output max=0.4609 mean=0.231605793
init=1/40: final_output max=0.3744 mean=0.195382660
init=1/36: final_output max=0.3760 mean=0.150754535
init=1/32: final_output max=0.2715 mean=0.101596284
init=1/28: final_output max=0.1304 mean=0.046488442
init=1/24: final_output max=0.0727 mean=0.024753001
init=1/23: final_output max=0.1334 mean=0.045036250
init=1/22: final_output max=0.2000 mean=0.067033558
init=1/21: final_output max=0.2690 mean=0.089748201
init=1/20: final_output max=0.3396 mean=0.112657035
init=1/18: final_output max=0.4836 mean=0.159056218
```

额外检查：

```text
21 iterations, init=1/24:
final_output max=0.61791 mean=0.201740773
seq16 CPU getTime ~= 6.60s

22 iterations, init=1/24:
final_output max=0.80910 mean=0.263890717
seq16 CPU getTime ~= 7.84s

20 iterations, init=1/24:
final_output max=0.07270 mean=0.024753001
seq16 CPU getTime ~= 8.00s
```

阶段判断：

- 更大的 public initial guess 确实能显著改善 20-iteration rsqrt；最佳点约为 `init=1/24`。
- 但最佳点仍有 `final_output mean drift ~= 0.0248`，比当前可接受的 Softmax `10/8` 候选漂移更大。
- `init=1/24` 附近的收敛区间很窄，继续增加迭代到 21/22 反而变差，说明它靠近 Newton 收敛边界，不适合作为默认稳定配置。
- wall-clock time 没有稳定收益：20-iteration + init=1/24 本次约 `8.00s`，不优于 24-iteration baseline 的常见 `~7.1s`。
- 结论：默认仍保持 `24 iterations, init=1/128`。可记录 `20 iterations, init=1/24` 作为研究候选，但不进入默认路径。真正可落地的 LayerNorm 优化仍应做分段/查表 rsqrt 或先估计 variance range 再选择保守初值。
