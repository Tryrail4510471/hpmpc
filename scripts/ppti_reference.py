#!/usr/bin/env python3
"""Plaintext reference for the PPTI TinyBERT smoke test.

This mirrors programs/transformer.hpp numerically, using the same smoke-test
dimensions, dummy inputs, dummy weights, polynomial Softmax, polynomial GELU,
and LayerNorm steps. The default topology is TinyBERT-like: 4 encoder layers
with multi-head self-attention.
"""

from __future__ import annotations

import argparse
import math
from typing import Iterable, List


SEQ_LEN = 4
HIDDEN = 8
NUM_HEADS = 2
NUM_LAYERS = 4
FFN_HIDDEN = 16
HEAD_DIM = HIDDEN // NUM_HEADS
LAYER_WEIGHT_STRIDE = 10000


Matrix = List[List[float]]


def input_value(i: int) -> float:
    return ((i % 7) - 3) * 0.0625


def weight_value(i: int) -> float:
    return ((i % 11) - 5) * 0.03125


def make_matrix(rows: int, cols: int, values: Iterable[float]) -> Matrix:
    flat = list(values)
    return [flat[r * cols : (r + 1) * cols] for r in range(rows)]


def load_inputs() -> Matrix:
    return make_matrix(SEQ_LEN, HIDDEN, (input_value(i) for i in range(SEQ_LEN * HIDDEN)))


def load_weights(rows: int, cols: int, offset: int) -> Matrix:
    return make_matrix(rows, cols, (weight_value(i + offset) for i in range(rows * cols)))


def load_bias(cols: int, offset: int) -> list[float]:
    return [weight_value(i + offset) for i in range(cols)]


def load_layer_norm_params(offset: int) -> tuple[list[float], list[float]]:
    gamma = [1.0 + weight_value(offset + i) for i in range(HIDDEN)]
    beta = [weight_value(offset + 1000 + i) for i in range(HIDDEN)]
    return gamma, beta


def matmul(a: Matrix, b: Matrix) -> Matrix:
    rows, inner, cols = len(a), len(a[0]), len(b[0])
    return [[sum(a[r][k] * b[k][c] for k in range(inner)) for c in range(cols)] for r in range(rows)]


def transpose(a: Matrix) -> Matrix:
    return [list(col) for col in zip(*a)]


def add(a: Matrix, b: Matrix) -> Matrix:
    return [[x + y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def add_bias(a: Matrix, bias: list[float]) -> Matrix:
    return [[x + bias[i] for i, x in enumerate(row)] for row in a]


def scale(a: Matrix, factor: float) -> Matrix:
    return [[x * factor for x in row] for row in a]


def extract_head(values: Matrix, head: int) -> Matrix:
    start = head * HEAD_DIM
    end = start + HEAD_DIM
    return [row[start:end] for row in values]


def concat_heads(heads: list[Matrix]) -> Matrix:
    out: Matrix = []
    for r in range(SEQ_LEN):
        row: list[float] = []
        for head in heads:
            row.extend(head[r])
        out.append(row)
    return out


def softmax_poly(scores: Matrix, attention_mask: list[int] | None = None) -> Matrix:
    out: Matrix = []
    for row in scores:
        row_max = max(row)
        shifted = [x - row_max for x in row]
        exp_approx = [1.0 + x + 0.5 * x * x for x in shifted]
        if attention_mask is not None:
            exp_approx = [x if attention_mask[i] != 0 else 0.0 for i, x in enumerate(exp_approx)]
        denom = sum(exp_approx)
        out.append([x / denom for x in exp_approx])
    return out


def gelu_poly(values: Matrix) -> Matrix:
    return [[0.5 * x + 0.125 * x * x * x for x in row] for row in values]


def layer_norm(values: Matrix, gamma: list[float], beta: list[float], epsilon: float = 0.001) -> Matrix:
    out: Matrix = []
    for row in values:
        mean = sum(row) / len(row)
        centered = [x - mean for x in row]
        variance = sum(x * x for x in centered) / len(centered) + epsilon
        inv_std = 1.0 / math.sqrt(variance)
        out.append([centered[i] * inv_std * gamma[i] + beta[i] for i in range(len(row))])
    return out


def multi_head_attention(x: Matrix, base: int, traces: dict[str, Matrix], layer: int, attention_mask: list[int]) -> Matrix:
    wq = load_weights(HIDDEN, HIDDEN, base + 0)
    bq = load_bias(HIDDEN, base + 800)
    wk = load_weights(HIDDEN, HIDDEN, base + 1000)
    bk = load_bias(HIDDEN, base + 1800)
    wv = load_weights(HIDDEN, HIDDEN, base + 2000)
    bv = load_bias(HIDDEN, base + 2800)
    wo = load_weights(HIDDEN, HIDDEN, base + 3000)
    bo = load_bias(HIDDEN, base + 3800)

    q = add_bias(matmul(x, wq), bq)
    k = add_bias(matmul(x, wk), bk)
    v = add_bias(matmul(x, wv), bv)

    contexts: list[Matrix] = []
    for head in range(NUM_HEADS):
        qh = extract_head(q, head)
        kh = extract_head(k, head)
        vh = extract_head(v, head)
        scores = scale(matmul(qh, transpose(kh)), 1.0 / math.sqrt(HEAD_DIM))
        probs = softmax_poly(scores, attention_mask)
        contexts.append(matmul(probs, vh))
        if layer == 0 and head == 0:
            traces["layer0_head0_scores"] = scores
            traces["layer0_head0_probs"] = probs

    return add_bias(matmul(concat_heads(contexts), wo), bo)


def encoder_layer(hidden: Matrix, layer: int, traces: dict[str, Matrix]) -> Matrix:
    base = layer * LAYER_WEIGHT_STRIDE
    w1 = load_weights(HIDDEN, FFN_HIDDEN, base + 4000)
    b1 = load_bias(FFN_HIDDEN, base + 4800)
    w2 = load_weights(FFN_HIDDEN, HIDDEN, base + 5000)
    b2 = load_bias(HIDDEN, base + 5800)
    attn_gamma, attn_beta = load_layer_norm_params(base + 6000)
    ffn_gamma, ffn_beta = load_layer_norm_params(base + 7000)

    attention_mask = traces.get("attention_mask_vector", [[1 for _ in range(SEQ_LEN)]])[0]
    attn_out = multi_head_attention(hidden, base, traces, layer, [int(x) for x in attention_mask])
    attn_residual = layer_norm(add(hidden, attn_out), attn_gamma, attn_beta)
    ffn_hidden = gelu_poly(add_bias(matmul(attn_residual, w1), b1))
    ffn_out = add_bias(matmul(ffn_hidden, w2), b2)
    out = layer_norm(add(attn_residual, ffn_out), ffn_gamma, ffn_beta)
    traces[f"layer{layer}_out"] = out
    return out


def forward() -> tuple[Matrix, dict[str, Matrix]]:
    hidden = load_inputs()
    traces: dict[str, Matrix] = {"input": hidden, "attention_mask_vector": [[1 for _ in range(SEQ_LEN)]]}
    for layer in range(NUM_LAYERS):
        hidden = encoder_layer(hidden, layer, traces)
    return hidden, traces


def print_matrix(name: str, values: Matrix) -> None:
    print(f"{name}:")
    for row in values:
        print("  " + " ".join(f"{x:+.8f}" for x in row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PPTI TinyBERT plaintext reference.")
    parser.add_argument("--dump", action="store_true", help="Print intermediate matrices.")
    args = parser.parse_args()

    output, traces = forward()
    print(f"ppti_reference_out0={output[0][0]:+.8f}")
    print(f"ppti_reference_shape={len(output)}x{len(output[0])}")
    print(f"ppti_reference_topology=layers:{NUM_LAYERS} heads:{NUM_HEADS} hidden:{HIDDEN} seq:{SEQ_LEN}")
    if args.dump:
        for name, values in traces.items():
            print_matrix(name, values)
        print_matrix("output", output)


if __name__ == "__main__":
    main()
