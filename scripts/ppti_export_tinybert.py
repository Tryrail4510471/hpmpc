#!/usr/bin/env python3
"""Export TinyBERT encoder weights for the PPTI HPMPC demo.

The output format matches programs/transformer.hpp:

    int32 total_parameter_count
    float32 parameters in fixed layer order

Per encoder layer order:
    query.weight.T, query.bias
    key.weight.T, key.bias
    value.weight.T, value.bias
    attention.output.dense.weight.T, attention.output.dense.bias
    intermediate.dense.weight.T, intermediate.dense.bias
    output.dense.weight.T, output.dense.bias
    attention.output.LayerNorm.weight, attention.output.LayerNorm.bias
    output.LayerNorm.weight, output.LayerNorm.bias

Use --synthetic to create a deterministic smoke-test file without installing
transformers or downloading a model.
"""

from __future__ import annotations

import argparse
import array
import struct
from pathlib import Path
from typing import Iterable


def weight_value(i: int) -> float:
    return ((i % 11) - 5) * 0.03125


def append_synthetic(values: list[list[float]], size: int, offset: int) -> None:
    values.append([weight_value(offset + i) for i in range(size)])


def expected_params(layers: int, hidden: int, ffn_hidden: int) -> int:
    linear = lambda in_features, out_features: in_features * out_features + out_features
    attention = 4 * linear(hidden, hidden)
    ffn = linear(hidden, ffn_hidden) + linear(ffn_hidden, hidden)
    layer_norm = 4 * hidden
    return layers * (attention + ffn + layer_norm)


def synthetic_values(layers: int, hidden: int, ffn_hidden: int) -> list[list[float]]:
    values: list[list[float]] = []
    stride = 10000
    for layer in range(layers):
        base = layer * stride
        append_synthetic(values, hidden * hidden, base + 0)
        append_synthetic(values, hidden, base + 800)
        append_synthetic(values, hidden * hidden, base + 1000)
        append_synthetic(values, hidden, base + 1800)
        append_synthetic(values, hidden * hidden, base + 2000)
        append_synthetic(values, hidden, base + 2800)
        append_synthetic(values, hidden * hidden, base + 3000)
        append_synthetic(values, hidden, base + 3800)
        append_synthetic(values, hidden * ffn_hidden, base + 4000)
        append_synthetic(values, ffn_hidden, base + 4800)
        append_synthetic(values, ffn_hidden * hidden, base + 5000)
        append_synthetic(values, hidden, base + 5800)
        values.append([1.0 + weight_value(base + 6000 + i) for i in range(hidden)])
        append_synthetic(values, hidden, base + 7000)
        values.append([1.0 + weight_value(base + 7000 + i) for i in range(hidden)])
        append_synthetic(values, hidden, base + 8000)
    return values


def tensor_to_list(tensor) -> list[float]:
    return [float(x) for x in tensor.detach().cpu().contiguous().view(-1).tolist()]


def append_linear(values: list[list[float]], linear) -> None:
    values.append(tensor_to_list(linear.weight.transpose(0, 1)))
    values.append(tensor_to_list(linear.bias))


def append_layer_norm(values: list[list[float]], layer_norm) -> None:
    values.append(tensor_to_list(layer_norm.weight))
    values.append(tensor_to_list(layer_norm.bias))


def load_huggingface_values(model_name: str, layers: int) -> tuple[list[list[float]], int, int, int, int]:
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise SystemExit("transformers is required for HuggingFace export. Use --synthetic for a local smoke file.") from exc

    model = AutoModel.from_pretrained(model_name)
    config = model.config
    encoder_layers = model.encoder.layer
    if len(encoder_layers) < layers:
        raise SystemExit(f"Model has {len(encoder_layers)} layers, but --layers={layers} was requested.")

    values: list[list[float]] = []
    for layer in encoder_layers[:layers]:
        append_linear(values, layer.attention.self.query)
        append_linear(values, layer.attention.self.key)
        append_linear(values, layer.attention.self.value)
        append_linear(values, layer.attention.output.dense)
        append_linear(values, layer.intermediate.dense)
        append_linear(values, layer.output.dense)
        append_layer_norm(values, layer.attention.output.LayerNorm)
        append_layer_norm(values, layer.output.LayerNorm)

    hidden = int(config.hidden_size)
    heads = int(config.num_attention_heads)
    ffn_hidden = int(config.intermediate_size)
    return values, hidden, heads, layers, ffn_hidden


def write_bin(path: Path, chunks: Iterable[Iterable[float]]) -> int:
    flat = array.array("f")
    for chunk in chunks:
        flat.extend(float(x) for x in chunk)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<i", len(flat)))
        flat.tofile(f)
    return len(flat)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TinyBERT weights for PPTI/HPMPC.")
    parser.add_argument("--output", default="models/ppti/tinybert_ppti.bin")
    parser.add_argument("--model", default="huawei-noah/TinyBERT_General_4L_312D")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=8, help="Synthetic mode hidden size.")
    parser.add_argument("--heads", type=int, default=2, help="Synthetic mode attention heads.")
    parser.add_argument("--ffn-hidden", type=int, default=16, help="Synthetic mode FFN size.")
    parser.add_argument("--synthetic", action="store_true", help="Write deterministic smoke weights instead of loading HF.")
    args = parser.parse_args()

    if args.synthetic:
        values = synthetic_values(args.layers, args.hidden, args.ffn_hidden)
        hidden, heads, layers, ffn_hidden = args.hidden, args.heads, args.layers, args.ffn_hidden
    else:
        values, hidden, heads, layers, ffn_hidden = load_huggingface_values(args.model, args.layers)

    total = write_bin(Path(args.output), values)
    expected = expected_params(layers, hidden, ffn_hidden)
    if total != expected:
        raise SystemExit(f"Exported {total} parameters, expected {expected}.")

    print(f"wrote={args.output}")
    print(f"params={total}")
    print(f"topology=layers:{layers} heads:{heads} hidden:{hidden} ffn_hidden:{ffn_hidden}")
    print("layout=ppti_tinybert_v1")


if __name__ == "__main__":
    main()
