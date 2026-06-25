# PPTI Development Workflow

PPTI targets private/protected Transformer inference on top of HPMPC/PIGEON.
The current implementation is a TinyBERT-style smoke test wired into the
existing HPMPC runtime as `FUNCTION_IDENTIFIER=87`.

## Current Status

- `protocol_executer.hpp` dispatches `FUNCTION_IDENTIFIER=87` to
  `programs/transformer.hpp`.
- `programs/transformer.hpp` implements a TinyBERT-style encoder stack:
  - default smoke dimensions: `seq_len=4`, `hidden=8`, `heads=2`,
    `layers=4`, `ffn_hidden=16`
  - secret/public dummy input and model parameter loading
  - per-layer Q/K/V projections through `prepare_GEMM`
  - per-head attention score GEMM and public scale by `1 / sqrt(head_dim)`
  - row-wise stable Softmax slot using masked row max, rational exp, and
    secret row-sum reciprocal
  - attention value GEMM, head concatenation, and output projection
  - residual plus LayerNorm with learned gamma/beta
  - FFN with a cubic GELU surrogate
  - second residual plus LayerNorm
- `scripts/ppti_reference.py` mirrors the same TinyBERT-style stack in
  plaintext Python.

## Verified Commands

CPU smoke test:

```sh
make -j PARTY=all FUNCTION_IDENTIFIER=87 PROTOCOL=5 DATTYPE=64 BITLENGTH=64 FRACTIONAL=14 USE_CUDA_GEMM=0
scripts/run.sh -p all -n 3
```

CUDA GEMM smoke test:

```sh
make -j PARTY=all FUNCTION_IDENTIFIER=87 PROTOCOL=5 DATTYPE=64 BITLENGTH=64 FRACTIONAL=14 USE_CUDA_GEMM=2 NVCC=/usr/local/cuda/bin/nvcc
scripts/run.sh -p all -n 3
```

Plaintext reference:

```sh
python3 scripts/ppti_reference.py --dump
```

Trace comparison:

```sh
python3 scripts/ppti_reference.py --trace > /tmp/ppti_python_trace.log
make -j PARTY=all FUNCTION_IDENTIFIER=87 PROTOCOL=5 DATTYPE=64 BITLENGTH=64 FRACTIONAL=14 USE_CUDA_GEMM=0 PPTI_TRACE=1
scripts/run.sh -p all -n 3 > /tmp/ppti_cpp_trace.log
python3 scripts/ppti_compare_trace.py --reference /tmp/ppti_python_trace.log --candidate /tmp/ppti_cpp_trace.log
```

Current smoke trace result:

```text
embedding_out max_abs_error=0
layer0_head0_scores max_abs_error=0.0242097552
layer0_head0_probs max_abs_error=0.004290848
final_output max_abs_error=0.68885958
```

Real TinyBERT seq=16 CPU baseline:

```text
shape=seq16 hidden312 heads12 layers4 ffn1200
P0 send=7.099MB getTime=6.298142s
P1 send=5.201MB getTime=6.297832s
P2 send=5.201MB getTime=6.298222s
```

CUDA seq=16 post-reboot run:

```text
GPU=RTX 2060 compute_cap=7.5
CUTLASS objects rebuilt with arch=sm_75
completed with no CUTLASS/error/failed entries
P0 send=7.099MB getTime=6.397868s
P1 send=5.201MB getTime=6.396788s
P2 send=5.201MB getTime=6.397518s
```

Real seq=16 fixed-point trace comparison:

```text
python3 scripts/ppti_reference.py --fixed --trace --seq-len 16 --hidden 312 --heads 12 --layers 4 --ffn-hidden 1200 --fractional 14 \
  --model-file models/ppti/tinybert_4l_312d_ppti.bin \
  --embedding-file models/ppti/tinybert_embeddings_ppti.bin \
  --input-file models/ppti/sample_input_seq16.bin
```

First comparison result:

```text
embedding_out         max=0.0011582       mean=0.000195694
layer0_head0_scores   max=1.316e10        mean=4.66356313e9
layer0_head0_probs    max=5.43968633e14   mean=5.42056872e13
layer0_out            max=5.62864423e14   mean=2.70766165e14
final_output          max=3.41260919e14   mean=1.49881746e14
```

Projection-level statistical trace narrowed this down further:

```text
layer0_input_stats             max=0.00027148   mean=0.000183685
layer0_wq_stats                max=0.00011719   mean=0.0000740674
layer0_q_linear_stats original prepare_GEMM max=5.627e14 mean=4.667e14
layer0_q_linear_stats manual dot baseline   max=0.00963574 mean=0.00625385
layer0_head0_score_scaled_stats manual dot  max=0.0389966 mean=0.0261987
```

Interpretation: real embeddings and Q/K/V weights are aligned. The original
`prepare_GEMM` path currently explodes on the large TinyBERT projection shape,
while the manual dot-product baseline aligns projection and QK score statistics
with the Python fixed-point reference. The first large divergence has therefore
moved to attention Softmax:

```text
layer0_head0_probs max=5.44014034e14 mean=5.42012114e13
layer0_out         max=5.62871964e14 mean=2.70765639e14
final_output       max=3.41277026e14 mean=1.49881735e14
```

Stage 7 replaced the quadratic Taylor exp with a rational baseline:

```text
exp(x) ~= 1 / (1 - x + 0.5*x^2), x <= 0
```

and fixed masked Softmax row-max handling by replacing masked scores with a
public `-1024` before row-max stabilization. Real seq=16 trace comparison now
shows:

```text
layer0_head0_probs max=0.006285352 mean=0.000163028266
layer0_out         max=5.62882548e14 mean=2.70958401e14
final_output       max=3.46676142e14 mean=1.52156787e14
P0 online getTime  ~= 6.36s
```

Interpretation: the Softmax probability explosion is fixed for the current
correctness baseline. The next major divergence is after Softmax.

Synthetic model-file smoke test:

```sh
python3 scripts/ppti_export_tinybert.py --synthetic --output models/ppti/tinybert_ppti_synthetic.bin
PPTI_MODEL_FILE=models/ppti/tinybert_ppti_synthetic.bin scripts/run.sh -p all -n 3
```

Synthetic embedding/input smoke test:

```sh
python3 scripts/ppti_export_tinybert_inputs.py --synthetic --seq-len 4 \
  --embedding-output models/ppti/tinybert_embeddings_synthetic.bin \
  --input-output models/ppti/sample_input_seq4.bin

PPTI_EMBEDDING_FILE=models/ppti/tinybert_embeddings_synthetic.bin \
PPTI_INPUT_FILE=models/ppti/sample_input_seq4.bin \
scripts/run.sh -p all -n 3
```

HuggingFace TinyBERT export, after installing `torch` and `transformers`:

```sh
python3 scripts/ppti_export_tinybert.py \
  --model huawei-noah/TinyBERT_General_4L_312D \
  --output models/ppti/tinybert_4l_312d_ppti.bin
```

Verified real TinyBERT export:

```text
params=4568736
topology=layers:4 heads:12 hidden:312 ffn_hidden:1200
layout=ppti_tinybert_v1
```

Real model-file header check:

```text
params=4568736
expected=4568736
```

Verified real embedding/input export:

```text
embedding_params=9683856
input_seq_len=16
embedding_shape=vocab:30522 max_position:512 type_vocab:2 hidden:312
layout=ppti_tinybert_embedding_v1
```

Real TinyBERT shape experiments can be selected at compile time after the
weight-loader path is ready:

```sh
make -j PARTY=all FUNCTION_IDENTIFIER=87 PROTOCOL=5 DATTYPE=64 BITLENGTH=64 FRACTIONAL=14 \
  USE_CUDA_GEMM=2 NVCC=/usr/local/cuda/bin/nvcc \
  MACRO_FLAGS="-DPPTI_SEQ_LEN=128 -DPPTI_HIDDEN=312 -DPPTI_NUM_HEADS=12 -DPPTI_NUM_LAYERS=4 -DPPTI_FFN_HIDDEN=1200"
```

## Softmax Insertion Point

The attention Softmax is inserted immediately after:

```text
Q_h = X * WQ sliced to head h
K_h = X * WK sliced to head h
scores_h = Q_h * K_h^T / sqrt(head_dim)
```

and before:

```text
context_h = softmax(scores_h) * V_h
context = concat(context_0, ..., context_h) * WO
```

The current implementation is `secure_rowwise_softmax_poly` in
`programs/transformer.hpp`. It is separate from the PIGEON classification-head
Softmax path because attention requires a real row-wise probability
distribution, not only last-layer argmax behavior. Stage 7 currently uses a
rational exp baseline; it is intended for correctness debugging, not final
performance.

## Next Milestones

The detailed execution checklist lives in `StudyNote/PPTI-TaskFlow.md`.

1. Add a Python fixed-point reference so trace error can separate quantization
   from protocol/approximation error.
2. Trace the post-Softmax path: context matmul, output projection, residual and
   LayerNorm.
3. Replace the rational Softmax baseline with a lower-round lookup or
   range-reduction design.
4. Replace the cubic GELU surrogate with a calibrated approximation and measure
   task accuracy.
5. Calibrate fixed-point precision and truncation.
6. Scale constants from smoke dimensions to TinyBERT dimensions, then benchmark
   CPU and CUDA paths separately.
