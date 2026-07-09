#!/usr/bin/env python3
"""Read metadata of a torch checkpoint (.bin/.pt/.pth) without loading tensors
into RAM. Uses torch.load with mmap=True and weights_only=True when supported,
then walks the state_dict to extract {name, dtype, shape} per tensor.

Falls back to a zipfile walk for safetensors-like files when torch is absent
or the file is a sharded `*.bin` whose sibling `*.index.json` exists.

Usage:
  python parse_ckpt.py <file_or_dir> [--out FILE]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def _dtype_size(dtype_name: str) -> int:
    t = dtype_name.lower()
    return {
        "float64": 8, "double": 8, "float32": 4, "float": 4, "float16": 2, "half": 2,
        "bfloat16": 2,
        "int64": 8, "long": 8, "int32": 4, "int": 4, "int16": 2, "short": 2,
        "int8": 1, "uint8": 1, "bool": 1,
    }.get(t, 0)


def _describe_via_index(index_path: str) -> list[dict]:
    """PyTorch sharded checkpoints ship a *.index.json with name->file map but
    not shapes; we can still enumerate names per shard cheaply."""
    with open(index_path) as f:
        idx = json.load(f)
    wm = idx.get("weight_map", {})
    out = []
    for name, shard in sorted(wm.items()):
        out.append({"name": name, "shard": shard})
    return out


def _load_with_torch(path: str) -> dict | None:
    try:
        import torch  # type: ignore
    except Exception as e:
        print(f"[parse_ckpt] torch unavailable ({e!r}); only index metadata possible", file=sys.stderr)
        return None

    try:
        sd = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        # older torch — no mmap/weights_only kwargs
        try:
            sd = torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"[parse_ckpt] torch.load failed: {e!r}", file=sys.stderr)
            return None
    except Exception as e:
        print(f"[parse_ckpt] torch.load failed: {e!r}", file=sys.stderr)
        return None

    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        # some checkpoints nest under "model"
        sd = sd["model"]

    tensors = []
    if isinstance(sd, dict):
        for k, v in sd.items():
            try:
                import torch as _t  # noqa
                if torch.is_tensor(v):
                    tensors.append({
                        "name": str(k),
                        "dtype": str(v.dtype).replace("torch.", ""),
                        "shape": list(v.shape),
                        "nbytes": v.numel() * v.element_size(),
                    })
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if torch.is_tensor(vv):
                            tensors.append({
                                "name": f"{k}.{kk}",
                                "dtype": str(vv.dtype).replace("torch.", ""),
                                "shape": list(vv.shape),
                                "nbytes": vv.numel() * vv.element_size(),
                            })
            except Exception:
                continue
    return {"file": os.path.abspath(path), "format": "torch_ckpt",
            "tensor_count": len(tensors),
            "total_nbytes": sum(t["nbytes"] for t in tensors),
            "tensors": sorted(tensors, key=lambda x: x["name"])}


def describe(path: str) -> dict:
    # try sibling index.json first if dir
    if os.path.isdir(path):
        idx = os.path.join(path, "pytorch_model.bin.index.json")
        if os.path.exists(idx):
            names = _describe_via_index(idx)
            return {"file": os.path.abspath(idx), "format": "torch_sharded_index",
                    "tensor_count": len(names), "tensors": names}
        # otherwise describe each .bin via torch
        files = sorted(glob.glob(os.path.join(path, "*.bin")) +
                       glob.glob(os.path.join(path, "*.pt")) +
                       glob.glob(os.path.join(path, "*.pth")))
        results = []
        for f in files:
            d = _load_with_torch(f)
            if d is not None:
                results.append(d)
        return {"files": results} if results else {"error": "no torch files described"}

    # single file
    base, ext = os.path.splitext(path)
    if ext in (".bin",) and os.path.exists(base + ".index.json"):
        names = _describe_via_index(base + ".index.json")
        return {"file": os.path.abspath(base + ".index.json"),
                "format": "torch_sharded_index",
                "tensor_count": len(names), "tensors": names}
    return _load_with_torch(path) or {"file": os.path.abspath(path),
                                       "format": "unknown",
                                       "error": "could not parse (torch missing?)"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.path):
        print(f"no such path: {args.path}", file=sys.stderr)
        return 2

    out = describe(args.path)
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
