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
  - residual plus LayerNorm with learned gamma/beta and stabilized rsqrt
    baseline
  - FFN with a tuned hard-GELU correctness/accuracy baseline
  - second residual plus stabilized LayerNorm
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
direct `prepare_GEMM` call exploded because Transformer exports weights in
`[inner, cols]` row-major layout, while HPMPC GEMM reads the RHS as
`[cols, inner]`. The manual dot-product baseline aligned projection and QK
score statistics with the Python fixed-point reference, moving the first large
divergence to attention Softmax:

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

Stage 8 added post-Softmax trace points and localized the next instability:

```text
layer0_attn_residual_post_ln_stats max=0.0294043 mean=0.0170682947
layer0_ffn_hidden_linear_stats     max=0.1601343 mean=0.0822365767
layer0_ffn_hidden_gelu_stats       max=252.882   mean=120.04788
layer0_ffn_out_stats               max=5231.235  mean=2029.3894
layer0_ffn_residual_post_ln_stats  max=5.62848948e14 mean=4.6544643e14
```

Interpretation: C++ and Python fixed-point traces stay close through attention
output and attention LayerNorm. The next numerical problem is the cubic GELU
surrogate: real TinyBERT FFN pre-activations reach roughly `[-63, 58]`, where
the cubic term grows to `~3e4`. That pushes FFN output to `~9e5`, after which
LayerNorm reciprocal sqrt becomes unstable.

Stage 9 replaces the cubic GELU surrogate with a ReLU-GELU correctness
baseline and runs both attention LayerNorm and FFN LayerNorm through a
stabilized reciprocal-sqrt path:

```text
gelu_baseline(x) = max(x, 0)
centered_scale=128
rsqrt_iterations=24
rsqrt_initial_guess=1/64
epsilon=0.001 / centered_scale^2
```

Real seq=16 fixed-point trace comparison after Stage 9:

```text
layer0_out   max=2.6417783 mean=0.2086785
layer1_out   max=5.4381221 mean=0.450146234
layer2_out   max=7.63491211 mean=0.71829986
layer3_out   max=2.25173633 mean=0.245499895
final_output max=2.25173633 mean=0.245499895
```

Interpretation: the 4-layer TinyBERT seq=16 path no longer produces `1e14`
explosions. This is still a correctness baseline, not a final accuracy design:
the final output has non-trivial approximation error and should be improved
with a calibrated GELU approximation, better LayerNorm/range parameters, and
fixed-point precision tuning.

Stage 10 no-trace performance baseline:

```text
seq=16 CPU  getTime ~= 6.52s  P0 send=9.915MB  P1/P2 send=8.291MB
seq=16 CUDA getTime ~= 7.35s  P0 send=9.915MB  P1/P2 send=8.291MB
seq=32 CPU  getTime ~= 9.16s  P0 send=25.83MB  P1/P2 send=21.96MB
```

Stage 10 seq=32 fixed-point trace comparison:

```text
layer0_head0_probs max=0.005459473 mean=0.000100898513
layer0_out         max=3.10990518  mean=0.204591555
layer1_out         max=6.0576191   mean=0.6396979
layer2_out         max=7.6165815   mean=0.549741695
layer3_out         max=3.82263965  mean=0.308568136
final_output       max=3.82263965  mean=0.308568136
```

Interpretation: the Stage 9 stabilized correctness baseline extends from
`seq=16` to `seq=32` without `1e14` explosions. CUDA currently runs cleanly but
does not accelerate this baseline at `seq=16`, because the path is still
dominated by small/medium matrix shapes, comparison-heavy approximations, and
the manual dot-product correctness oracle.

Stage 11 adds a selectable HPMPC GEMM adapter:

```text
PPTI_MATMUL_BACKEND=0  manual dot baseline, default
PPTI_MATMUL_BACKEND=1  prepare_GEMM adapter
```

The adapter converts RHS layout before calling GEMM:

```text
rhs_for_gemm[c * inner + k] = rhs[k * cols + c]
prepare_GEMM(lhs, rhs_for_gemm, out, rows, cols, inner, false)
```

Real seq=16 GEMM adapter trace comparison:

```text
layer0_q_linear_stats           max=0.00963574  mean=0.00625384667
layer0_head0_score_scaled_stats max=0.0389966   mean=0.0261987267
layer0_head0_probs              max=0.006285352 mean=0.000163028266
final_output                    max=2.25173633  mean=0.245499895
```

Stage 11 performance check:

```text
seq=16 manual dot CPU           getTime ~= 6.52s
seq=16 prepare_GEMM adapter CPU getTime ~= 7.31s
seq=16 prepare_GEMM adapter CUDA getTime ~= 7.70s
```

Interpretation: the large projection explosion is fixed by matching the RHS
layout expected by `prepare_GEMM`. The adapter is correct, but not yet faster:
it currently pays an RHS re-layout cost at every matmul and still uses many
small/medium GEMM calls. The next optimization is to cache or pre-export GEMM
layout weights and only use CUDA where the matrix shape is large enough.

Stage 12 replaces the ReLU-GELU baseline with tuned hard-GELU:

```text
gelu_tuned(x) = x * clamp(0.5 + 0.3125 * x, 0, 1)
```

Layer0 true-GELU approximation check on real seq=16 FFN pre-activations:

```text
ReLU baseline              max=0.1699712074 mean=0.0977432140
standard hard-GELU x/6     max=0.2953695466 mean=0.1349476164
tuned alpha ~= 0.30-0.325  mean ~= 0.0226-0.0248
```

Real seq=16 fixed trace after tuned hard-GELU:

```text
layer0_head0_probs max=0.006285352 mean=0.000163028266
final_output       max=2.25368652  mean=0.317721248
```

No-trace seq=16 CPU after tuned hard-GELU:

```text
getTime ~= 6.75s
P0 send=13.59MB
P1/P2 send=10.74MB
```

Interpretation: tuned hard-GELU is much closer to true GELU than ReLU on the
observed FFN activation distribution and remains numerically stable. It costs
extra communication because it adds clamp comparisons and one multiplication.
The final fixed-reference error is higher than the ReLU baseline, so the next
step is LayerNorm/fixed-point calibration rather than more GELU changes alone.

Stage 13 calibrates LayerNorm/fixed-point parameters and makes them configurable:

```text
PPTI_LN_CENTERED_SCALE
PPTI_LN_RSQRT_ITERATIONS
PPTI_LN_RSQRT_INITIAL_GUESS
```

Current default:

```text
PPTI_LN_CENTERED_SCALE=256
PPTI_LN_RSQRT_ITERATIONS=24
PPTI_LN_RSQRT_INITIAL_GUESS=1/128
```

Real seq=16 fixed trace after calibration:

```text
layer0_head0_probs max=0.006285352 mean=0.000163028266
layer0_out         max=0.184137012 mean=0.0117048779
layer1_out         max=0.123886426 mean=0.00582938099
layer2_out         max=0.056611133 mean=0.00545621854
layer3_out         max=0.070651855 mean=0.00279345366
final_output       max=0.070651855 mean=0.00279345366
```

No-trace seq=16 CPU after calibration:

```text
getTime ~= 7.09s
P0 send=13.59MB
P1/P2 send=10.74MB
```

Interpretation: LayerNorm scaling was the main source of tuned hard-GELU fixed
trace mismatch. Increasing the centered scaling from `128` to `256` and using
`1/128` as the rsqrt initial guess reduces final mean error from `0.3177` to
`0.00279` without changing the protocol structure.

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

1. Replace the rational Softmax baseline with a lower-round lookup or
   range-reduction design.
2. Reduce LayerNorm cost by testing smaller Newton iteration counts after the
   `scale=256` calibration.
3. Revisit GEMM acceleration with a dedicated pre-exported layout format rather
   than long-lived copied share objects.
4. Scale to `seq=64/128` after a stable accelerated matmul or Softmax path lands.
5. Scale constants from smoke dimensions to TinyBERT dimensions, then benchmark
   CPU and CUDA paths separately.

## Stage 14 Optimization Note

The calibrated LayerNorm defaults were revalidated at `seq=32`:

```text
final_output max=0.069527051 mean=0.00272162649
seq32 CPU getTime ~= 9.48s
P0 send=33.18MB, P1/P2 send=26.86MB
```

Two GEMM RHS layout cache variants were tested and rejected for now:

```text
copied long-lived RHS layout cache -> P1/P2 double free or corruption
direct GEMM-layout weight loading  -> P1/P2 double free or corruption
```

The stable code path was restored. The current working assumption is that
`prepare_GEMM` has lifecycle assumptions for RHS share objects that make a naive
long-lived TinyBERT weight cache unsafe. Future GEMM work should use a dedicated
export/load path and validate share ownership independently before measuring
performance.

## Stage 15 Softmax Iteration Tuning

The rational attention Softmax now exposes its Newton reciprocal iteration
counts as compile-time/reference parameters:

```text
PPTI_SOFTMAX_EXP_ITERATIONS
PPTI_SOFTMAX_ROWSUM_ITERATIONS
```

Defaults remain `12/8` to preserve the current accuracy baseline. The best
measured performance candidate so far is `10/8`:

```text
12/8 baseline:  seq16 CPU ~= 8.59s, P0 send=13.59MB, P1/P2 send=10.74MB
10/8 candidate: seq16 CPU ~= 6.82s, P0 send=13.20MB, P1/P2 send=10.35MB
```

Trace drift against the C++ `12/8` baseline:

```text
10/8 final_output max=0.01910 mean=0.00904820
8/6  final_output max=0.13360 mean=0.04562570
```

Recommendation: keep `12/8` as default and use `10/8` as the current explicit
speed candidate. More aggressive settings such as `8/6` are faster but move the
multi-layer output too far for the current TinyBERT accuracy target.
