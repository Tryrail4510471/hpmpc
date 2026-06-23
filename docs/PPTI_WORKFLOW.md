# PPTI Development Workflow

PPTI targets private/protected Transformer inference on top of HPMPC/PIGEON.
The current implementation is a tiny Transformer block smoke test wired into
the existing HPMPC runtime as `FUNCTION_IDENTIFIER=87`.

## Current Status

- `protocol_executer.hpp` dispatches `FUNCTION_IDENTIFIER=87` to
  `programs/transformer.hpp`.
- `programs/transformer.hpp` implements a 4-token, 8-hidden-dim Transformer
  block:
  - secret/public dummy input and model parameter loading
  - Q/K/V projections through `prepare_GEMM`
  - attention score GEMM and public scale by `1 / sqrt(hidden)`
  - row-wise stable Softmax slot using row max, polynomial exp, and secret
    row-sum reciprocal
  - attention value GEMM and output projection
  - residual plus LayerNorm with learned gamma/beta
  - FFN with a cubic GELU surrogate
  - second residual plus LayerNorm
- `scripts/ppti_reference.py` mirrors the same tiny block in plaintext Python.

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

## Softmax Insertion Point

The attention Softmax is inserted immediately after:

```text
Q = X * WQ
K = X * WK
scores = Q * K^T / sqrt(hidden)
```

and before:

```text
context = softmax(scores) * V
```

The current implementation is `secure_rowwise_softmax_poly` in
`programs/transformer.hpp`. It is separate from the PIGEON classification-head
Softmax path because attention requires a real row-wise probability
distribution, not only last-layer argmax behavior.

## Next Milestones

1. Add fixed-point trace comparison between `programs/transformer.hpp` and
   `scripts/ppti_reference.py`.
2. Replace the quadratic exp approximation with a better MPC-friendly
   approximation or lookup/range-reduction design.
3. Replace the cubic GELU surrogate with a calibrated approximation and measure
   task accuracy.
4. Add attention mask support before Softmax.
5. Add HuggingFace/Pygeon export for TinyBERT/BERT weights in the layout expected
   by the HPMPC Transformer program.
6. Scale constants from the tiny block to TinyBERT dimensions, then benchmark CPU
   and CUDA paths separately.
