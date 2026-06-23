#!/usr/bin/env python3
"""Plaintext reference for the PPTI tiny Transformer smoke test.

This mirrors programs/transformer.hpp numerically, using the same dimensions,
dummy inputs, dummy weights, polynomial Softmax, polynomial GELU, and LayerNorm
steps. It is intentionally dependency-free so it can run on a fresh Ubuntu host.
"""

from __future__ import annotations

import argparse
import math
from typing import Iterable, List


SEQ_LEN = 4
HIDDEN = 8
FFN_HIDDEN = 16


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


def scale(a: Matrix, factor: float) -> Matrix:
    return [[x * factor for x in row] for row in a]


def softmax_poly(scores: Matrix) -> Matrix:
    out: Matrix = []
    for row in scores:
        row_max = max(row)
        shifted = [x - row_max for x in row]
        exp_approx = [1.0 + x + 0.5 * x * x for x in shifted]
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


def forward() -> tuple[Matrix, dict[str, Matrix]]:
    x = load_inputs()
    wq = load_weights(HIDDEN, HIDDEN, 0)
    wk = load_weights(HIDDEN, HIDDEN, 1000)
    wv = load_weights(HIDDEN, HIDDEN, 2000)
    wo = load_weights(HIDDEN, HIDDEN, 3000)
    w1 = load_weights(HIDDEN, FFN_HIDDEN, 4000)
    w2 = load_weights(FFN_HIDDEN, HIDDEN, 5000)
    ln1_gamma, ln1_beta = load_layer_norm_params(6000)
    ln2_gamma, ln2_beta = load_layer_norm_params(7000)

    q = matmul(x, wq)
    k = matmul(x, wk)
    v = matmul(x, wv)
    scores = scale(matmul(q, transpose(k)), 1.0 / math.sqrt(HIDDEN))
    probs = softmax_poly(scores)
    context = matmul(probs, v)
    attn_out = matmul(context, wo)
    residual = layer_norm(add(x, attn_out), ln1_gamma, ln1_beta)
    ffn_hidden = gelu_poly(matmul(residual, w1))
    ffn_out = layer_norm(add(matmul(ffn_hidden, w2), residual), ln2_gamma, ln2_beta)
    return ffn_out, {"scores": scores, "probs": probs, "residual": residual}


def print_matrix(name: str, values: Matrix) -> None:
    print(f"{name}:")
    for row in values:
        print("  " + " ".join(f"{x:+.8f}" for x in row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PPTI tiny Transformer plaintext reference.")
    parser.add_argument("--dump", action="store_true", help="Print intermediate matrices.")
    args = parser.parse_args()

    output, intermediates = forward()
    print(f"ppti_reference_out0={output[0][0]:+.8f}")
    print(f"ppti_reference_shape={len(output)}x{len(output[0])}")
    if args.dump:
        for name, values in intermediates.items():
            print_matrix(name, values)
        print_matrix("output", output)


if __name__ == "__main__":
    main()
