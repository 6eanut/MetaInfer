#!/usr/bin/env python3
"""Read metadata of a safetensors file (or all *.safetensors in a directory)
without loading tensors into RAM. Output a compact JSON description.

Usage:
  python parse_safetensors.py <path> [--out FILE]
  <path> is a .safetensors file or a directory containing them.

Output schema (list of per-file objects):
  [{
    "file": "...",
    "format": "safetensors",
    "metadata": {...},                # the __metadata__ if present
    "tensors": [
      {"name": "...", "dtype": "BF16", "shape": [int, ...], "nbytes": int}
    ]
  }]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import sys
from typing import Any


def _read_safetensors_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            raise ValueError(f"{path}: too short to be a safetensors file")
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        if header_len > (1 << 40):
            raise ValueError(f"{path}: absurd header length {header_len}")
        header_bytes = f.read(header_len)
    header = json.loads(header_bytes.decode("utf-8"))
    return header, 8 + header_len


_DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1,
    "U8": 1, "U16": 2, "U32": 4, "U64": 8,
    "BOOL": 1,
    "F8_E4M3": 1, "F8_E5M2": 1,
}


def _nbytes(dtype: str, shape: list[int]) -> int:
    unit = _DTYPE_BYTES.get(dtype.upper())
    if unit is None:
        return 0
    n = 1
    for d in shape:
        n *= int(d)
    return n * unit


def describe_file(path: str) -> dict:
    header, _data_off = _read_safetensors_header(path)
    metadata = header.get("__metadata__", {}) or {}
    tensors = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(info, dict):
            continue
        dtype = info.get("dtype", "?")
        shape = [int(x) for x in info.get("shape", [])]
        tensors.append({
            "name": name,
            "dtype": dtype,
            "shape": shape,
            "nbytes": _nbytes(dtype, shape),
        })
    tensors.sort(key=lambda t: t["name"])
    return {
        "file": os.path.abspath(path),
        "format": "safetensors",
        "metadata": metadata,
        "tensor_count": len(tensors),
        "total_nbytes": sum(t["nbytes"] for t in tensors),
        "tensors": tensors,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-tensors", action="store_true",
                    help="omit per-tensor rows (just summary counts)")
    args = ap.parse_args()

    if os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, "*.safetensors")))
        files += sorted(glob.glob(os.path.join(args.path, "**", "*.safetensors"), recursive=True))
        files = list(dict.fromkeys(files))
    elif os.path.isfile(args.path):
        files = [args.path]
    else:
        print(f"no such file or directory: {args.path}", file=sys.stderr)
        return 2

    if not files:
        print(f"no safetensors files under {args.path}", file=sys.stderr)
        return 2

    out = []
    for f in files:
        try:
            d = describe_file(f)
        except Exception as e:
            d = {"file": os.path.abspath(f), "format": "safetensors", "error": str(e)}
        if args.no_tensors and "tensors" in d:
            d = {k: v for k, v in d.items() if k != "tensors"}
        out.append(d)

    text = json.dumps(out, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
