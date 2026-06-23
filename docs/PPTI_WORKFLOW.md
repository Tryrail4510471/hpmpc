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
  - row-wise stable Softmax slot using row max, polynomial exp, and secret
    row-sum reciprocal
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
distribution, not only last-layer argmax behavior.

## Next Milestones

1. Add fixed-point trace comparison between `programs/transformer.hpp` and
   `scripts/ppti_reference.py`.
2. Add HuggingFace/Pygeon export for TinyBERT weights in the layout expected by
   the HPMPC Transformer program.
3. Replace the quadratic exp approximation with a better MPC-friendly
   approximation or lookup/range-reduction design.
4. Replace the cubic GELU surrogate with a calibrated approximation and measure
   task accuracy.
5. Add attention mask support before Softmax.
6. Scale constants from smoke dimensions to TinyBERT dimensions, then benchmark
   CPU and CUDA paths separately.
