#!/usr/bin/env python3
"""Export PPTI TinyBERT embedding weights and input-id files.

Embedding file layout:

    int32 total_float_params
    int32 vocab_size
    int32 max_position
    int32 type_vocab_size
    int32 hidden
    float32 word_embeddings[vocab_size * hidden]
    float32 position_embeddings[max_position * hidden]
    float32 token_type_embeddings[type_vocab_size * hidden]
    float32 embedding_layernorm_gamma[hidden]
    float32 embedding_layernorm_beta[hidden]

Input file layout:

    int32 seq_len
    int32 token_ids[seq_len]
    int32 token_type_ids[seq_len]
    int32 position_ids[seq_len]
    int32 attention_mask[seq_len]

Use --synthetic for a dependency-free smoke file pair.
"""

from __future__ import annotations

import argparse
import array
import struct
from pathlib import Path
from typing import Iterable


def weight_value(i: int) -> float:
    return ((i % 11) - 5) * 0.03125


def write_floats(path: Path, header: Iterable[int], chunks: Iterable[Iterable[float]]) -> int:
    flat = array.array("f")
    for chunk in chunks:
        flat.extend(float(x) for x in chunk)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<i", len(flat)))
        for item in header:
            f.write(struct.pack("<i", int(item)))
        flat.tofile(f)
    return len(flat)


def write_int_vectors(path: Path, seq_len: int, vectors: Iterable[Iterable[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<i", int(seq_len)))
        for vector in vectors:
            data = array.array("i", (int(x) for x in vector))
            if len(data) != seq_len:
                raise SystemExit(f"Input vector length {len(data)} does not match seq_len={seq_len}.")
            data.tofile(f)


def synthetic_embedding_chunks(vocab_size: int, max_position: int, type_vocab_size: int, hidden: int) -> list[list[float]]:
    return [
        [weight_value(i) for i in range(vocab_size * hidden)],
        [weight_value(100000 + i) for i in range(max_position * hidden)],
        [weight_value(200000 + i) for i in range(type_vocab_size * hidden)],
        [1.0 + weight_value(300000 + i) for i in range(hidden)],
        [weight_value(400000 + i) for i in range(hidden)],
    ]


def synthetic_input(seq_len: int, vocab_size: int, type_vocab_size: int, pad_from: int) -> tuple[list[int], list[int], list[int], list[int]]:
    token_ids = [(i + 1) % vocab_size for i in range(seq_len)]
    token_type_ids = [i % type_vocab_size for i in range(seq_len)]
    position_ids = list(range(seq_len))
    attention_mask = [1 for _ in range(seq_len)]
    if 0 <= pad_from < seq_len:
        for i in range(pad_from, seq_len):
            token_ids[i] = 0
            token_type_ids[i] = 0
            attention_mask[i] = 0
    return token_ids, token_type_ids, position_ids, attention_mask


def tensor_to_list(tensor) -> list[float]:
    return [float(x) for x in tensor.detach().cpu().contiguous().view(-1).tolist()]


def load_huggingface_embeddings(model_name: str) -> tuple[list[list[float]], int, int, int, int]:
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise SystemExit("transformers is required for HuggingFace embedding export. Use --synthetic for smoke files.") from exc

    model = AutoModel.from_pretrained(model_name)
    embeddings = model.embeddings
    word = embeddings.word_embeddings
    position = embeddings.position_embeddings
    token_type = embeddings.token_type_embeddings
    layer_norm = embeddings.LayerNorm
    hidden = int(model.config.hidden_size)
    vocab_size = int(word.num_embeddings)
    max_position = int(position.num_embeddings)
    type_vocab_size = int(token_type.num_embeddings)

    chunks = [
        tensor_to_list(word.weight),
        tensor_to_list(position.weight),
        tensor_to_list(token_type.weight),
        tensor_to_list(layer_norm.weight),
        tensor_to_list(layer_norm.bias),
    ]
    return chunks, vocab_size, max_position, type_vocab_size, hidden


def load_huggingface_input(model_name: str, text: str, seq_len: int) -> tuple[list[int], list[int], list[int], list[int]]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("transformers is required for HuggingFace input export. Use --synthetic for smoke files.") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoded = tokenizer(
        text,
        max_length=seq_len,
        padding="max_length",
        truncation=True,
        return_token_type_ids=True,
    )
    token_ids = [int(x) for x in encoded["input_ids"]]
    token_type_ids = [int(x) for x in encoded.get("token_type_ids", [0] * seq_len)]
    attention_mask = [int(x) for x in encoded["attention_mask"]]
    position_ids = list(range(seq_len))
    return token_ids, token_type_ids, position_ids, attention_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PPTI TinyBERT embedding and input files.")
    parser.add_argument("--model", default="huawei-noah/TinyBERT_General_4L_312D")
    parser.add_argument("--embedding-output", default="models/ppti/tinybert_embeddings_ppti.bin")
    parser.add_argument("--input-output", default="models/ppti/sample_input_seq16.bin")
    parser.add_argument("--text", default="privacy preserving transformer inference")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-pad-from", type=int, default=-1, help="Set synthetic attention_mask to 0 from this index.")
    parser.add_argument("--hidden", type=int, default=8, help="Synthetic hidden size.")
    parser.add_argument("--vocab-size", type=int, default=32, help="Synthetic vocab size.")
    parser.add_argument("--max-position", type=int, default=32, help="Synthetic max position embeddings.")
    parser.add_argument("--type-vocab-size", type=int, default=2, help="Synthetic token type vocab size.")
    args = parser.parse_args()

    if args.synthetic:
        chunks = synthetic_embedding_chunks(args.vocab_size, args.max_position, args.type_vocab_size, args.hidden)
        token_ids, token_type_ids, position_ids, attention_mask = synthetic_input(
            args.seq_len, args.vocab_size, args.type_vocab_size, args.synthetic_pad_from
        )
        vocab_size, max_position, type_vocab_size, hidden = (
            args.vocab_size,
            args.max_position,
            args.type_vocab_size,
            args.hidden,
        )
    else:
        chunks, vocab_size, max_position, type_vocab_size, hidden = load_huggingface_embeddings(args.model)
        token_ids, token_type_ids, position_ids, attention_mask = load_huggingface_input(
            args.model, args.text, args.seq_len
        )

    total = write_floats(Path(args.embedding_output), [vocab_size, max_position, type_vocab_size, hidden], chunks)
    expected = vocab_size * hidden + max_position * hidden + type_vocab_size * hidden + 2 * hidden
    if total != expected:
        raise SystemExit(f"Exported {total} embedding params, expected {expected}.")
    write_int_vectors(Path(args.input_output), args.seq_len, [token_ids, token_type_ids, position_ids, attention_mask])

    print(f"embedding_file={args.embedding_output}")
    print(f"embedding_params={total}")
    print(f"input_file={args.input_output}")
    print(f"input_seq_len={args.seq_len}")
    print(f"embedding_shape=vocab:{vocab_size} max_position:{max_position} type_vocab:{type_vocab_size} hidden:{hidden}")
    print("layout=ppti_tinybert_embedding_v1")


if __name__ == "__main__":
    main()
