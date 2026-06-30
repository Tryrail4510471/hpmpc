#!/usr/bin/env python3
"""Plaintext/fixed-point reference for the PPTI TinyBERT path.

The reference mirrors programs/transformer.hpp.  It can run the original
deterministic smoke path, or load the exported PPTI model, embedding and input
files used by the C++ program.  With --fixed, arithmetic is quantized after the
same operations that trigger fixed-point truncation in the MPC path.
"""

from __future__ import annotations

import argparse
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


LAYER_WEIGHT_STRIDE = 10000
UINT_BITS = 64
UINT_MOD = 1 << UINT_BITS
SIGN_BIT = 1 << (UINT_BITS - 1)


Matrix = list[list[float]]


@dataclass
class Config:
    seq_len: int
    hidden: int
    heads: int
    layers: int
    ffn_hidden: int
    fractional: int
    fixed: bool
    ln_centered_scale: float = 256.0
    ln_rsqrt_iterations: int = 24
    ln_rsqrt_initial_guess: float = 1.0 / 128.0
    softmax_exp_iterations: int = 12
    softmax_rowsum_iterations: int = 8

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads


@dataclass
class ModelReader:
    params: list[float] | None = None
    cursor: int = 0

    def take(self, count: int, fallback: Iterable[float]) -> list[float]:
        if self.params is None:
            return list(fallback)
        if self.cursor + count > len(self.params):
            raise SystemExit(f"Model file ended at {self.cursor}, need {count} more values.")
        out = self.params[self.cursor : self.cursor + count]
        self.cursor += count
        return out


@dataclass
class InputBundle:
    hidden: Matrix | None
    attention_mask: list[int]


def wrap_u64(value: int) -> int:
    return value % UINT_MOD


def to_signed(value: int) -> int:
    value = wrap_u64(value)
    return value - UINT_MOD if value & SIGN_BIT else value


def float_to_fixed(value: float, fractional: int) -> int:
    return wrap_u64(int(round(value * (1 << fractional))))


def fixed_to_float(value: int, fractional: int) -> float:
    return to_signed(value) / float(1 << fractional)


def q(value: float, cfg: Config) -> float:
    if not cfg.fixed:
        return float(value)
    return fixed_to_float(float_to_fixed(float(value), cfg.fractional), cfg.fractional)


def q_add(a: float, b: float, cfg: Config) -> float:
    if not cfg.fixed:
        return a + b
    return fixed_to_float(float_to_fixed(a, cfg.fractional) + float_to_fixed(b, cfg.fractional), cfg.fractional)


def q_sub(a: float, b: float, cfg: Config) -> float:
    if not cfg.fixed:
        return a - b
    return fixed_to_float(float_to_fixed(a, cfg.fractional) - float_to_fixed(b, cfg.fractional), cfg.fractional)


def q_mul(a: float, b: float, cfg: Config) -> float:
    if not cfg.fixed:
        return a * b
    product = to_signed(float_to_fixed(a, cfg.fractional)) * to_signed(float_to_fixed(b, cfg.fractional))
    # The MPC path truncates after fixed-point multiplications.
    return fixed_to_float(product >> cfg.fractional, cfg.fractional)


def q_sum(values: Iterable[float], cfg: Config) -> float:
    total = 0.0
    for value in values:
        total = q_add(total, value, cfg)
    return total


def input_value(i: int) -> float:
    return ((i % 7) - 3) * 0.0625


def weight_value(i: int) -> float:
    return ((i % 11) - 5) * 0.03125


def make_matrix(rows: int, cols: int, values: Iterable[float], cfg: Config) -> Matrix:
    flat = [q(v, cfg) for v in values]
    return [flat[r * cols : (r + 1) * cols] for r in range(rows)]


def read_i32(f) -> int:
    data = f.read(4)
    if len(data) != 4:
        raise SystemExit("Unexpected EOF while reading int32.")
    return struct.unpack("<i", data)[0]


def read_f32_values(f, count: int) -> list[float]:
    data = f.read(4 * count)
    if len(data) != 4 * count:
        raise SystemExit(f"Unexpected EOF while reading {count} float32 values.")
    return list(struct.unpack(f"<{count}f", data))


def read_i32_values(f, count: int) -> list[int]:
    data = f.read(4 * count)
    if len(data) != 4 * count:
        raise SystemExit(f"Unexpected EOF while reading {count} int32 values.")
    return list(struct.unpack(f"<{count}i", data))


def expected_model_params(cfg: Config) -> int:
    linear = lambda i, o: i * o + o
    attention = 4 * linear(cfg.hidden, cfg.hidden)
    ffn = linear(cfg.hidden, cfg.ffn_hidden) + linear(cfg.ffn_hidden, cfg.hidden)
    layer_norm = 4 * cfg.hidden
    return cfg.layers * (attention + ffn + layer_norm)


def load_model(path: str | None, cfg: Config) -> ModelReader:
    if not path:
        return ModelReader()
    with Path(path).open("rb") as f:
        total = read_i32(f)
        expected = expected_model_params(cfg)
        if total != expected:
            raise SystemExit(f"{path}: model params={total}, expected={expected}.")
        return ModelReader(read_f32_values(f, total))


def build_embedding_inputs(
    embedding_file: str | None, input_file: str | None, cfg: Config
) -> InputBundle:
    if not embedding_file or not input_file:
        hidden = make_matrix(
            cfg.seq_len, cfg.hidden, (input_value(i) for i in range(cfg.seq_len * cfg.hidden)), cfg
        )
        return InputBundle(hidden, [1 for _ in range(cfg.seq_len)])

    with Path(input_file).open("rb") as f:
        seq_len = read_i32(f)
        if seq_len != cfg.seq_len:
            raise SystemExit(f"{input_file}: seq_len={seq_len}, expected={cfg.seq_len}.")
        token_ids = read_i32_values(f, cfg.seq_len)
        token_type_ids = read_i32_values(f, cfg.seq_len)
        position_ids = read_i32_values(f, cfg.seq_len)
        attention_mask = read_i32_values(f, cfg.seq_len)

    with Path(embedding_file).open("rb") as f:
        total = read_i32(f)
        vocab_size = read_i32(f)
        max_position = read_i32(f)
        type_vocab_size = read_i32(f)
        hidden = read_i32(f)
        if hidden != cfg.hidden:
            raise SystemExit(f"{embedding_file}: hidden={hidden}, expected={cfg.hidden}.")
        expected = vocab_size * cfg.hidden + max_position * cfg.hidden + type_vocab_size * cfg.hidden + 2 * cfg.hidden
        if total != expected:
            raise SystemExit(f"{embedding_file}: params={total}, expected={expected}.")
        word = read_f32_values(f, vocab_size * cfg.hidden)
        position = read_f32_values(f, max_position * cfg.hidden)
        token_type = read_f32_values(f, type_vocab_size * cfg.hidden)
        gamma = read_f32_values(f, cfg.hidden)
        beta = read_f32_values(f, cfg.hidden)

    values: Matrix = []
    for r in range(cfg.seq_len):
        token_id = token_ids[r]
        token_type_id = token_type_ids[r]
        position_id = position_ids[r]
        if not (0 <= token_id < vocab_size):
            raise SystemExit(f"token id {token_id} out of range.")
        if not (0 <= token_type_id < type_vocab_size):
            raise SystemExit(f"token type id {token_type_id} out of range.")
        if not (0 <= position_id < max_position):
            raise SystemExit(f"position id {position_id} out of range.")

        row = []
        for c in range(cfg.hidden):
            value = (
                word[token_id * cfg.hidden + c]
                + position[position_id * cfg.hidden + c]
                + token_type[token_type_id * cfg.hidden + c]
            )
            row.append(q(value, cfg))

        # C++ embedding LayerNorm is plaintext float before fixed sharing.
        # Under --fixed, quantize the final shared embedding only.
        mean = sum(row) / cfg.hidden
        centered = [x - mean for x in row]
        variance = sum(x * x for x in centered) / cfg.hidden + 0.001
        inv_std = 1.0 / math.sqrt(variance)
        values.append([q(centered[c] * inv_std * gamma[c] + beta[c], cfg) for c in range(cfg.hidden)])

    return InputBundle(values, attention_mask)


def load_weights(reader: ModelReader, rows: int, cols: int, offset: int, cfg: Config) -> Matrix:
    values = reader.take(rows * cols, (weight_value(i + offset) for i in range(rows * cols)))
    return make_matrix(rows, cols, values, cfg)


def load_bias(reader: ModelReader, cols: int, offset: int, cfg: Config) -> list[float]:
    return [q(x, cfg) for x in reader.take(cols, (weight_value(i + offset) for i in range(cols)))]


def load_layer_norm_params(reader: ModelReader, offset: int, cfg: Config) -> tuple[list[float], list[float]]:
    gamma = reader.take(cfg.hidden, (1.0 + weight_value(offset + i) for i in range(cfg.hidden)))
    beta = reader.take(cfg.hidden, (weight_value(offset + 1000 + i) for i in range(cfg.hidden)))
    return [q(x, cfg) for x in gamma], [q(x, cfg) for x in beta]


def matmul(a: Matrix, b: Matrix, cfg: Config) -> Matrix:
    rows, inner, cols = len(a), len(a[0]), len(b[0])
    out: Matrix = []
    for r in range(rows):
        row = []
        for c in range(cols):
            total = 0.0
            for k in range(inner):
                total = q_add(total, q_mul(a[r][k], b[k][c], cfg), cfg)
            row.append(total)
        out.append(row)
    return out


def transpose(a: Matrix) -> Matrix:
    return [list(col) for col in zip(*a)]


def add(a: Matrix, b: Matrix, cfg: Config) -> Matrix:
    return [[q_add(x, y, cfg) for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def add_bias(a: Matrix, bias: list[float], cfg: Config) -> Matrix:
    return [[q_add(x, bias[i], cfg) for i, x in enumerate(row)] for row in a]


def scale(a: Matrix, factor: float, cfg: Config) -> Matrix:
    factor = q(factor, cfg)
    return [[q_mul(x, factor, cfg) for x in row] for row in a]


def extract_head(values: Matrix, head: int, cfg: Config) -> Matrix:
    start = head * cfg.head_dim
    end = start + cfg.head_dim
    return [row[start:end] for row in values]


def concat_heads(heads: list[Matrix], cfg: Config) -> Matrix:
    out: Matrix = []
    for r in range(cfg.seq_len):
        row: list[float] = []
        for head in heads:
            row.extend(head[r])
        out.append(row)
    return out


def reciprocal_newton(values: list[float], iterations: int, initial_guess: float, cfg: Config) -> list[float]:
    reciprocal = [q(initial_guess, cfg) for _ in values]
    for _ in range(iterations):
        product = [q_mul(x, y, cfg) for x, y in zip(values, reciprocal)]
        product = [q_sub(q(2.0, cfg), x, cfg) for x in product]
        reciprocal = [q_mul(y, p, cfg) for y, p in zip(reciprocal, product)]
    return reciprocal


def reciprocal_sqrt_newton(values: list[float], iterations: int, initial_guess: float, cfg: Config) -> list[float]:
    y = [q(initial_guess, cfg) for _ in values]
    for _ in range(iterations):
        y_squared = [q_mul(v, v, cfg) for v in y]
        xy_squared = [q_mul(x, ys, cfg) for x, ys in zip(values, y_squared)]
        xy_half = [q_mul(x, 0.5, cfg) for x in xy_squared]
        correction = [q_sub(q(1.5, cfg), x, cfg) for x in xy_half]
        y = [q_mul(v, corr, cfg) for v, corr in zip(y, correction)]
    return y


def softmax_poly(scores: Matrix, attention_mask: list[int], cfg: Config) -> Matrix:
    out: Matrix = []
    for row in scores:
        masked_row = [x if attention_mask[i] != 0 else q(-1024.0, cfg) for i, x in enumerate(row)]
        row_max = max(masked_row)
        shifted = [q_sub(x, row_max, cfg) for x in masked_row]
        shifted = [x if attention_mask[i] != 0 else q(0.0, cfg) for i, x in enumerate(shifted)]
        squared = [q_mul(x, x, cfg) for x in shifted]
        squared = [q_mul(x, 0.5, cfg) for x in squared]
        exp_denom = [q_add(q_sub(q(1.0, cfg), x, cfg), sq, cfg) for x, sq in zip(shifted, squared)]
        exp_approx = reciprocal_newton(exp_denom, cfg.softmax_exp_iterations, 1.0 / 1024.0, cfg)
        exp_approx = [q_mul(x, 1.0 if attention_mask[i] != 0 else 0.0, cfg) for i, x in enumerate(exp_approx)]
        denom = q_sum(exp_approx, cfg)
        inv_sum = reciprocal_newton([denom], cfg.softmax_rowsum_iterations, 1.0 / len(row), cfg)[0]
        out.append([q_mul(x, inv_sum, cfg) for x in exp_approx])
    return out


def gelu_poly(values: Matrix, cfg: Config) -> Matrix:
    out: Matrix = []
    for row in values:
        next_row = []
        for x in row:
            gate = q_add(q_mul(x, 0.3125, cfg), q(0.5, cfg), cfg)
            gate = min(max(gate, q(0.0, cfg)), q(1.0, cfg))
            next_row.append(q_mul(x, gate, cfg))
        out.append(next_row)
    return out


def layer_norm(
    values: Matrix,
    gamma: list[float],
    beta: list[float],
    cfg: Config,
    centered_scale: float = 1.0,
    rsqrt_iterations: int = 3,
    rsqrt_initial_guess: float = 1.0,
) -> Matrix:
    out: Matrix = []
    inv_cols = q(1.0 / len(values[0]), cfg)
    for row in values:
        mean = q_mul(q_sum(row, cfg), inv_cols, cfg)
        centered = [q_sub(x, mean, cfg) for x in row]
        if centered_scale != 1.0:
            centered = [q_mul(x, 1.0 / centered_scale, cfg) for x in centered]
        squared = [q_mul(x, x, cfg) for x in centered]
        variance = q_add(q_mul(q_sum(squared, cfg), inv_cols, cfg), q(0.001 / (centered_scale * centered_scale), cfg), cfg)
        inv_std = reciprocal_sqrt_newton([variance], rsqrt_iterations, rsqrt_initial_guess, cfg)[0]
        norm = [q_mul(x, inv_std, cfg) for x in centered]
        norm = [q_mul(x, gamma[i], cfg) for i, x in enumerate(norm)]
        out.append([q_add(x, beta[i], cfg) for i, x in enumerate(norm)])
    return out


def matrix_stats(values: Matrix) -> Matrix:
    flat = [x for row in values for x in row]
    if not flat:
        return [[0.0, 0.0, 0.0]]
    return [[min(flat), max(flat), sum(abs(x) for x in flat) / len(flat)]]


def multi_head_attention(
    x: Matrix,
    reader: ModelReader,
    base: int,
    traces: dict[str, Matrix],
    layer: int,
    attention_mask: list[int],
    cfg: Config,
) -> Matrix:
    wq = load_weights(reader, cfg.hidden, cfg.hidden, base + 0, cfg)
    bq = load_bias(reader, cfg.hidden, base + 800, cfg)
    wk = load_weights(reader, cfg.hidden, cfg.hidden, base + 1000, cfg)
    bk = load_bias(reader, cfg.hidden, base + 1800, cfg)
    wv = load_weights(reader, cfg.hidden, cfg.hidden, base + 2000, cfg)
    bv = load_bias(reader, cfg.hidden, base + 2800, cfg)
    wo = load_weights(reader, cfg.hidden, cfg.hidden, base + 3000, cfg)
    bo = load_bias(reader, cfg.hidden, base + 3800, cfg)

    if layer == 0:
        traces["layer0_input_stats"] = matrix_stats(x)
        traces["layer0_wq_stats"] = matrix_stats(wq)
        traces["layer0_wk_stats"] = matrix_stats(wk)
        traces["layer0_wv_stats"] = matrix_stats(wv)
    q_linear = matmul(x, wq, cfg)
    k_linear = matmul(x, wk, cfg)
    v_linear = matmul(x, wv, cfg)
    if layer == 0:
        traces["layer0_q_linear_stats"] = matrix_stats(q_linear)
        traces["layer0_k_linear_stats"] = matrix_stats(k_linear)
        traces["layer0_v_linear_stats"] = matrix_stats(v_linear)
    q_mat = add_bias(q_linear, bq, cfg)
    k_mat = add_bias(k_linear, bk, cfg)
    v_mat = add_bias(v_linear, bv, cfg)
    if layer == 0:
        traces["layer0_q_stats"] = matrix_stats(q_mat)
        traces["layer0_k_stats"] = matrix_stats(k_mat)
        traces["layer0_v_stats"] = matrix_stats(v_mat)

    contexts: list[Matrix] = []
    for head in range(cfg.heads):
        qh = extract_head(q_mat, head, cfg)
        kh = extract_head(k_mat, head, cfg)
        vh = extract_head(v_mat, head, cfg)
        raw_scores = matmul(qh, transpose(kh), cfg)
        scores = scale(raw_scores, 1.0 / math.sqrt(cfg.head_dim), cfg)
        probs = softmax_poly(scores, attention_mask, cfg)
        context = matmul(probs, vh, cfg)
        contexts.append(context)
        if layer == 0 and head == 0:
            traces["layer0_head0_q_stats"] = matrix_stats(qh)
            traces["layer0_head0_k_stats"] = matrix_stats(kh)
            traces["layer0_head0_v_stats"] = matrix_stats(vh)
            traces["layer0_head0_score_raw_stats"] = matrix_stats(raw_scores)
            traces["layer0_head0_score_scaled_stats"] = matrix_stats(scores)
            traces["layer0_head0_scores"] = scores
            traces["layer0_head0_probs"] = probs
            traces["layer0_head0_context_stats"] = matrix_stats(context)

    concat_context = concat_heads(contexts, cfg)
    if layer == 0:
        traces["layer0_concat_context_stats"] = matrix_stats(concat_context)
        traces["layer0_wo_stats"] = matrix_stats(wo)
    attn_out_linear = matmul(concat_context, wo, cfg)
    if layer == 0:
        traces["layer0_attn_out_linear_stats"] = matrix_stats(attn_out_linear)
    attn_out = add_bias(attn_out_linear, bo, cfg)
    if layer == 0:
        traces["layer0_attn_out_stats"] = matrix_stats(attn_out)
    return attn_out


def encoder_layer(
    hidden: Matrix,
    reader: ModelReader,
    layer: int,
    traces: dict[str, Matrix],
    attention_mask: list[int],
    cfg: Config,
) -> Matrix:
    base = layer * LAYER_WEIGHT_STRIDE
    attn_out = multi_head_attention(hidden, reader, base, traces, layer, attention_mask, cfg)

    w1 = load_weights(reader, cfg.hidden, cfg.ffn_hidden, base + 4000, cfg)
    b1 = load_bias(reader, cfg.ffn_hidden, base + 4800, cfg)
    w2 = load_weights(reader, cfg.ffn_hidden, cfg.hidden, base + 5000, cfg)
    b2 = load_bias(reader, cfg.hidden, base + 5800, cfg)
    attn_gamma, attn_beta = load_layer_norm_params(reader, base + 6000, cfg)
    ffn_gamma, ffn_beta = load_layer_norm_params(reader, base + 7000, cfg)

    attn_residual_pre_ln = add(hidden, attn_out, cfg)
    traces[f"layer{layer}_attn_residual_pre_ln_stats"] = matrix_stats(attn_residual_pre_ln)
    attn_residual = layer_norm(
        attn_residual_pre_ln,
        attn_gamma,
        attn_beta,
        cfg,
        cfg.ln_centered_scale,
        cfg.ln_rsqrt_iterations,
        cfg.ln_rsqrt_initial_guess,
    )
    traces[f"layer{layer}_attn_residual_post_ln_stats"] = matrix_stats(attn_residual)
    ffn_hidden_linear = matmul(attn_residual, w1, cfg)
    traces[f"layer{layer}_ffn_hidden_linear_stats"] = matrix_stats(ffn_hidden_linear)
    ffn_hidden = gelu_poly(add_bias(ffn_hidden_linear, b1, cfg), cfg)
    traces[f"layer{layer}_ffn_hidden_gelu_stats"] = matrix_stats(ffn_hidden)
    ffn_out = add_bias(matmul(ffn_hidden, w2, cfg), b2, cfg)
    traces[f"layer{layer}_ffn_out_stats"] = matrix_stats(ffn_out)
    ffn_residual_pre_ln = add(attn_residual, ffn_out, cfg)
    traces[f"layer{layer}_ffn_residual_pre_ln_stats"] = matrix_stats(ffn_residual_pre_ln)
    out = layer_norm(
        ffn_residual_pre_ln,
        ffn_gamma,
        ffn_beta,
        cfg,
        cfg.ln_centered_scale,
        cfg.ln_rsqrt_iterations,
        cfg.ln_rsqrt_initial_guess,
    )
    traces[f"layer{layer}_ffn_residual_post_ln_stats"] = matrix_stats(out)
    traces[f"layer{layer}_out"] = out
    return out


def forward(
    cfg: Config, model_file: str | None, embedding_file: str | None, input_file: str | None
) -> tuple[Matrix, dict[str, Matrix]]:
    reader = load_model(model_file, cfg)
    input_bundle = build_embedding_inputs(embedding_file, input_file, cfg)
    if input_bundle.hidden is None:
        raise SystemExit("No input hidden states were built.")

    hidden = input_bundle.hidden
    traces: dict[str, Matrix] = {"embedding_out": hidden}
    for layer in range(cfg.layers):
        hidden = encoder_layer(hidden, reader, layer, traces, input_bundle.attention_mask, cfg)
    traces["final_output"] = hidden
    return hidden, traces


def print_matrix(name: str, values: Matrix) -> None:
    print(f"{name}:")
    for row in values:
        print("  " + " ".join(f"{x:+.8f}" for x in row))


def print_trace(name: str, values: Matrix) -> None:
    rows = len(values)
    cols = len(values[0]) if rows else 0
    flat = [x for row in values for x in row]
    print("PPTI_TRACE " + name + f" {rows} {cols} " + " ".join(f"{x:.9g}" for x in flat))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PPTI TinyBERT reference.")
    parser.add_argument("--model-file", default="")
    parser.add_argument("--embedding-file", default="")
    parser.add_argument("--input-file", default="")
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--ffn-hidden", type=int, default=16)
    parser.add_argument("--fractional", type=int, default=14)
    parser.add_argument("--ln-centered-scale", type=float, default=256.0)
    parser.add_argument("--ln-rsqrt-iterations", type=int, default=24)
    parser.add_argument("--ln-rsqrt-initial-guess", type=float, default=1.0 / 128.0)
    parser.add_argument("--softmax-exp-iterations", type=int, default=12)
    parser.add_argument("--softmax-rowsum-iterations", type=int, default=8)
    parser.add_argument("--fixed", action="store_true", help="Emulate fixed-point truncation after MPC multiply steps.")
    parser.add_argument("--dump", action="store_true", help="Print intermediate matrices.")
    parser.add_argument("--trace", action="store_true", help="Print machine-readable PPTI_TRACE lines.")
    args = parser.parse_args()

    cfg = Config(
        args.seq_len,
        args.hidden,
        args.heads,
        args.layers,
        args.ffn_hidden,
        args.fractional,
        args.fixed,
        args.ln_centered_scale,
        args.ln_rsqrt_iterations,
        args.ln_rsqrt_initial_guess,
        args.softmax_exp_iterations,
        args.softmax_rowsum_iterations,
    )
    if cfg.hidden % cfg.heads != 0:
        raise SystemExit("--hidden must be divisible by --heads.")

    output, traces = forward(
        cfg,
        args.model_file or None,
        args.embedding_file or None,
        args.input_file or None,
    )
    mode = "fixed" if args.fixed else "float"
    print(f"ppti_reference_out0={output[0][0]:+.8f}")
    print(f"ppti_reference_shape={len(output)}x{len(output[0])}")
    print(
        f"ppti_reference_topology=layers:{cfg.layers} heads:{cfg.heads} "
        f"hidden:{cfg.hidden} seq:{cfg.seq_len} ffn:{cfg.ffn_hidden} mode:{mode}"
    )
    if args.trace:
        trace_names = [
            "embedding_out",
            "layer0_input_stats",
            "layer0_wq_stats",
            "layer0_wk_stats",
            "layer0_wv_stats",
            "layer0_q_linear_stats",
            "layer0_k_linear_stats",
            "layer0_v_linear_stats",
            "layer0_q_stats",
            "layer0_k_stats",
            "layer0_v_stats",
            "layer0_head0_q_stats",
            "layer0_head0_k_stats",
            "layer0_head0_v_stats",
            "layer0_head0_score_raw_stats",
            "layer0_head0_score_scaled_stats",
            "layer0_head0_scores",
            "layer0_head0_probs",
            "layer0_head0_context_stats",
            "layer0_concat_context_stats",
            "layer0_wo_stats",
            "layer0_attn_out_linear_stats",
            "layer0_attn_out_stats",
            "final_output",
        ]
        layer_suffixes = [
            "attn_residual_pre_ln_stats",
            "attn_residual_post_ln_stats",
            "ffn_hidden_linear_stats",
            "ffn_hidden_gelu_stats",
            "ffn_out_stats",
            "ffn_residual_pre_ln_stats",
            "ffn_residual_post_ln_stats",
            "out",
        ]
        for layer in range(cfg.layers):
            for suffix in layer_suffixes:
                trace_names.insert(-1, f"layer{layer}_{suffix}")

        for name in trace_names:
            if name in traces:
                print_trace(name, traces[name])
    if args.dump:
        for name, values in traces.items():
            print_matrix(name, values)
        print_matrix("output", output)


if __name__ == "__main__":
    main()
