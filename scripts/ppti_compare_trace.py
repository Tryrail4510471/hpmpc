#!/usr/bin/env python3
"""Compare PPTI C++ trace output against the Python reference trace."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TraceTensor:
    rows: int
    cols: int
    values: list[float]


def load_trace(path: Path) -> dict[str, TraceTensor]:
    traces: dict[str, TraceTensor] = {}
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("PPTI_TRACE "):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        name = parts[1]
        rows = int(parts[2])
        cols = int(parts[3])
        values = [float(x) for x in parts[4:]]
        if len(values) != rows * cols:
            raise SystemExit(f"{path}: trace {name} has {len(values)} values, expected {rows * cols}.")
        traces[name] = TraceTensor(rows, cols, values)
    return traces


def compare(reference: TraceTensor, candidate: TraceTensor) -> tuple[float, float]:
    if reference.rows != candidate.rows or reference.cols != candidate.cols:
        raise SystemExit(
            f"Shape mismatch: reference {reference.rows}x{reference.cols}, "
            f"candidate {candidate.rows}x{candidate.cols}."
        )
    diffs = [abs(a - b) for a, b in zip(reference.values, candidate.values)]
    return max(diffs), sum(diffs) / len(diffs) if diffs else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PPTI_TRACE outputs.")
    parser.add_argument("--reference", required=True, help="Python reference trace file.")
    parser.add_argument("--candidate", required=True, help="C++ HPMPC trace log.")
    parser.add_argument("--names", default="", help="Comma-separated trace names to compare. Default: all reference traces.")
    parser.add_argument("--fail-at", type=float, default=None, help="Fail if any max_abs_error exceeds this value.")
    args = parser.parse_args()

    reference = load_trace(Path(args.reference))
    candidate = load_trace(Path(args.candidate))
    if not reference:
        raise SystemExit(f"No PPTI_TRACE lines found in {args.reference}.")
    if not candidate:
        raise SystemExit(f"No PPTI_TRACE lines found in {args.candidate}.")

    failed = False
    print("trace_name,rows,cols,max_abs_error,mean_abs_error")
    names = [name for name in args.names.split(",") if name] if args.names else list(reference)
    for name in names:
        if name not in reference:
            print(f"{name},missing_reference,missing_reference,missing_reference,missing_reference")
            failed = True
            continue
        if name not in candidate:
            print(f"{name},missing,missing,missing,missing")
            failed = True
            continue
        max_abs, mean_abs = compare(reference[name], candidate[name])
        print(f"{name},{reference[name].rows},{reference[name].cols},{max_abs:.9g},{mean_abs:.9g}")
        if args.fail_at is not None and max_abs > args.fail_at:
            failed = True

    extra = sorted(set(candidate) - set(reference))
    for name in extra:
        print(f"{name},extra,extra,extra,extra")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
