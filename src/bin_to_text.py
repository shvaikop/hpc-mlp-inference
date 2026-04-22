#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np


def parse_shape(shape_str: str | None):
    if shape_str is None:
        return None
    parts = shape_str.lower().replace("x", ",").split(",")
    shape = tuple(int(p.strip()) for p in parts if p.strip())
    if any(d <= 0 for d in shape):
        raise ValueError("All shape dimensions must be positive.")
    return shape


def main():
    parser = argparse.ArgumentParser(
        description="Convert binary float32/int64 files to readable text."
    )
    parser.add_argument("input", type=Path, help="Path to binary input file")
    parser.add_argument(
        "--dtype",
        required=True,
        choices=["float32", "int64"],
        help="Element type stored in the binary file",
    )
    parser.add_argument(
        "--shape",
        type=str,
        default=None,
        help='Optional output shape, e.g. "16,512" or "4096x512"',
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file (.txt or .csv). If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default=",",
        help="Delimiter for text output, default is ','",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=10,
        help="Maximum rows to print when output is stdout",
    )
    parser.add_argument(
        "--max-cols",
        type=int,
        default=10,
        help="Maximum columns to print when output is stdout",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Floating-point precision for text output",
    )

    args = parser.parse_args()

    dtype_map = {
        "float32": np.float32,
        "int64": np.int64,
    }

    arr = np.fromfile(args.input, dtype=dtype_map[args.dtype])

    shape = parse_shape(args.shape)
    if shape is not None:
        expected = np.prod(shape)
        if arr.size != expected:
            raise ValueError(
                f"Cannot reshape array of size {arr.size} into shape {shape} "
                f"(expected {expected} elements)."
            )
        arr = arr.reshape(shape)

    print(f"Loaded: {args.input}")
    print(f"dtype: {arr.dtype}, shape: {arr.shape}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if np.issubdtype(arr.dtype, np.floating):
            fmt = f"%.{args.precision}f"
        else:
            fmt = "%d"

        if arr.ndim == 1:
            np.savetxt(args.output, arr.reshape(-1, 1), fmt=fmt, delimiter=args.delimiter)
        elif arr.ndim == 2:
            np.savetxt(args.output, arr, fmt=fmt, delimiter=args.delimiter)
        else:
            flat = arr.reshape(arr.shape[0], -1)
            np.savetxt(args.output, flat, fmt=fmt, delimiter=args.delimiter)

        print(f"Wrote readable output to: {args.output}")
    else:
        if arr.ndim == 1:
            preview = arr[: args.max_rows]
        elif arr.ndim == 2:
            preview = arr[: args.max_rows, : args.max_cols]
        else:
            preview = arr.reshape(arr.shape[0], -1)[: args.max_rows, : args.max_cols]

        np.set_printoptions(
            precision=args.precision,
            suppress=True,
            linewidth=200,
        )
        print("\nPreview:")
        print(preview)


if __name__ == "__main__":
    main()
