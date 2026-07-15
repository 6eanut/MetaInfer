"""
Evaluator harness for fused MoE Triton kernel optimization.

Loads evolved kernel, runs correctness against reference, measures GPU performance,
returns combined score for OpenEvolve selection.

Cascade evaluation:
- Stage 1: Fast filter on 7 critical test cases
- Stage 2: Full correctness + performance on all 27 test cases
"""

import importlib.util
import traceback
import sys
import math
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import triton
import triton.language as tl

from openevolve.evaluation_result import EvaluationResult

_TASK_DIR = Path(__file__).parent
_REFERENCE_DIR = _TASK_DIR / "reference"

# ============================================================
# Test Case Definitions (from IMPLEMENTATION_PLAN.md Section 3.3)
# ============================================================

TEST_CASES = [
    # Group G1: Normal, even K
    {"name": "g1_fp16_even_small",  "M": 1,   "N": 1024, "K": 256,  "E": 8,  "topk": 2, "dtype": "float16", "seed": 1001},
    {"name": "g1_fp16_even_medium", "M": 64,  "N": 1024, "K": 256,  "E": 8,  "topk": 2, "dtype": "float16", "seed": 1002},
    {"name": "g1_fp16_even_large",  "M": 512, "N": 4096, "K": 2048, "E": 64, "topk": 6, "dtype": "float16", "seed": 1003},
    {"name": "g1_bf16_even",        "M": 64,  "N": 1024, "K": 512,  "E": 8,  "topk": 2, "dtype": "bfloat16","seed": 1004},
    {"name": "g1_topk1",            "M": 64,  "N": 1024, "K": 256,  "E": 8,  "topk": 1, "dtype": "float16", "seed": 1005},
    {"name": "g1_topk8",            "M": 64,  "N": 1024, "K": 256,  "E": 64, "topk": 8, "dtype": "float16", "seed": 1006},
    {"name": "g1_single_expert",    "M": 64,  "N": 1024, "K": 256,  "E": 1,  "topk": 1, "dtype": "float16", "seed": 1007},
    # Group G2: Uneven K
    {"name": "g2_fp16_uneven_k",    "M": 64,  "N": 1024, "K": 511,  "E": 8,  "topk": 2, "dtype": "float16", "seed": 2001},
    {"name": "g2_bf16_uneven_k",    "M": 64,  "N": 128,  "K": 511,  "E": 8,  "topk": 2, "dtype": "bfloat16","seed": 2002},
    {"name": "g2_large_uneven",     "M": 222,"N": 4096, "K": 1536, "E": 64, "topk": 6, "dtype": "float16", "seed": 2003},
    # Group G3: fp8_w8a8 block-wise
    {"name": "g3_fp8_block",        "M": 64,  "N": 1024, "K": 256,  "E": 8,  "topk": 2, "dtype": "float16",
     "use_fp8_w8a8": True, "block_shape": [128, 128], "seed": 3001},
    {"name": "g3_fp8_block_large",  "M": 222,"N": 4096, "K": 2048, "E": 64, "topk": 6, "dtype": "float16",
     "use_fp8_w8a8": True, "block_shape": [128, 128], "seed": 3002},
    # Group G4: fp8_w8a8 per-channel
    {"name": "g4_fp8_per_channel",  "M": 64,  "N": 1024, "K": 256,  "E": 8,  "topk": 2, "dtype": "float16",
     "use_fp8_w8a8": True, "per_channel_quant": True, "seed": 4001},
    # Group G5: fp8_w8a8 tensor-wise
    {"name": "g5_fp8_tensor",       "M": 64,  "N": 1024, "K": 256,  "E": 8,  "topk": 2, "dtype": "float16",
     "use_fp8_w8a8": True, "seed": 5001},
    # Group G6: int8_w8a8
    {"name": "g6_int8_w8a8",        "M": 64,  "N": 1024, "K": 256,  "E": 8,  "topk": 2, "dtype": "float16",
     "use_int8_w8a8": True, "per_channel_quant": True, "seed": 6001},
    # Group G7: int8_w8a16
    {"name": "g7_int8_w8a16",       "M": 64,  "N": 1024, "K": 256,  "E": 8,  "topk": 2, "dtype": "float16",
     "use_int8_w8a16": True, "seed": 7001},
    # Group G8: Bias
    {"name": "g8_bias_fp16",        "M": 64,  "N": 1024, "K": 256,  "E": 8,  "topk": 2, "dtype": "float16",
     "has_bias": True, "seed": 8001},
    {"name": "g8_bias_fp8",         "M": 64,  "N": 1024, "K": 256,  "E": 8,  "topk": 2, "dtype": "float16",
     "use_fp8_w8a8": True, "block_shape": [128, 128], "has_bias": True, "seed": 8002},
    # Group G9/G10: MUL_ROUTED_WEIGHT
    {"name": "g9_weighted",         "M": 64,  "N": 1024, "K": 256,  "E": 8,  "topk": 2, "dtype": "float16",
     "mul_routed_weight": True, "seed": 9001},
    {"name": "g10_no_weight",       "M": 64,  "N": 1024, "K": 256,  "E": 8,  "topk": 2, "dtype": "float16",
     "mul_routed_weight": False, "seed": 10001},
    # Group G11: filter_expert — NOT tested standalone (requires EP mapping in moe_align_block_size)
    # {"name": "g11_filter",          "M": 64,  "N": 1024, "K": 256,  "E": 8,  "topk": 2, "dtype": "float16",
    #  "filter_expert": True, "seed": 11001},
    # Group G12/G13: c_sorted
    {"name": "g12_sorted",          "M": 64,  "N": 1024, "K": 256,  "E": 8,  "topk": 2, "dtype": "float16",
     "c_sorted": True, "seed": 12001},
    {"name": "g13_unsorted",        "M": 64,  "N": 1024, "K": 256,  "E": 8,  "topk": 2, "dtype": "float16",
     "c_sorted": False, "seed": 13001},
    # Realistic workload sizes (DeepSeek-V4 like)
    {"name": "real_dsv4_up",        "M": 222, "N": 7168, "K": 2048, "E": 64, "topk": 6, "dtype": "bfloat16","seed": 14001},
    # NOTE: real_dsv4_down crashes on DCU due to VM fault (K=7168 too large for BLOCK_SIZE_K=32)
    # {"name": "real_dsv4_down",      "M": 222, "N": 2048, "K": 7168, "E": 64, "topk": 6, "dtype": "bfloat16","seed": 14002},
    {"name": "real_dsv4_large_batch","M":1024, "N": 7168, "K": 2048, "E": 256,"topk": 8, "dtype": "float16", "seed": 14003},
]

FAST_TEST_CASES = [tc for tc in TEST_CASES if tc["name"] in {
    "g1_fp16_even_medium", "g2_fp16_uneven_k", "g3_fp8_block",
    "g4_fp8_per_channel", "g6_int8_w8a8", "g8_bias_fp16",
    "real_dsv4_up",
}]

DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4,
}

# ============================================================
# moe_align_block_size (pure PyTorch)
# ============================================================

def moe_align_block_size(topk_ids, block_size, num_experts):
    """Align token-expert assignments to block_size multiples."""
    M, topk = topk_ids.shape
    flat_ids = topk_ids.reshape(-1).to(torch.int32)
    total_tokens = flat_ids.numel()
    device = topk_ids.device

    counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    for e in range(num_experts):
        counts[e] = (flat_ids == e).sum().item()

    padded_counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    for e in range(num_experts):
        cnt = counts[e].item()
        if cnt > 0:
            padded_counts[e] = ((cnt + block_size - 1) // block_size) * block_size

    num_tokens_post_padded = padded_counts.sum().item()
    sorted_ids = torch.full((num_tokens_post_padded,), total_tokens,
                            dtype=torch.int32, device=device)
    expert_ids_list = []
    offset = 0
    for e in range(num_experts):
        cnt = counts[e].item()
        if cnt == 0:
            continue
        mask = flat_ids == e
        e_tokens = torch.where(mask)[0].to(torch.int32)
        num_e = e_tokens.numel()
        sorted_ids[offset:offset + num_e] = e_tokens
        num_blocks = padded_counts[e].item() // block_size
        for _ in range(num_blocks):
            expert_ids_list.append(e)
        offset += padded_counts[e].item()

    expert_ids = torch.tensor(expert_ids_list, dtype=torch.int32, device=device)
    return sorted_ids, expert_ids, num_tokens_post_padded

# ============================================================
# launch_kernel
# ============================================================

def launch_kernel(kernel_fn, a, b, topk_weights, topk_ids,
                  bias=None, a_scale=None, b_scale=None,
                  block_shape=None, per_channel_quant=False,
                  mul_routed_weight=True, filter_expert=False,
                  c_sorted=False, swap_ab=False,
                  use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False,
                  config=None):
    """Set up all arguments and launch the Triton kernel."""
    M, K = a.shape
    E, N, K2 = b.shape
    topk = topk_ids.shape[1]
    assert K == K2, f"K mismatch: {K} vs {K2}"

    if config is None:
        config = DEFAULT_CONFIG

    BLOCK_SIZE_M = config["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = config["BLOCK_SIZE_N"]
    BLOCK_SIZE_K = config["BLOCK_SIZE_K"]
    GROUP_SIZE_M = config["GROUP_SIZE_M"]

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, BLOCK_SIZE_M, E)

    # When c_sorted=True, kernel writes to offsets 0..num_tokens_post_padded-1
    # which may exceed M*topk. Allocate a 2D C of (num_tokens_post_padded, N).
    # When c_sorted=False, kernel writes to flat token indices within [0, M*topk).
    if c_sorted:
        c = torch.zeros((num_tokens_post_padded, N), device=a.device, dtype=a.dtype)
    else:
        c = torch.zeros((M, topk, N), device=a.device, dtype=a.dtype)

    even_Ks = K % BLOCK_SIZE_K == 0
    compute_type = tl.bfloat16 if a.dtype == torch.bfloat16 else tl.float16

    group_n = block_shape[0] if block_shape else 0
    group_k = block_shape[1] if block_shape else 0

    stride_am = a.stride(0)
    stride_ak = a.stride(1)
    stride_be = b.stride(0)
    stride_bk = b.stride(1)
    stride_bn = b.stride(2)
    if c_sorted:
        stride_cm = c.stride(0)  # = N for (num_tokens, N)
        stride_cn = c.stride(1)  # = 1
    else:
        stride_cm = c.stride(1)  # = N for (M, topk, N)
        stride_cn = c.stride(2)  # = 1
    stride_bias_e = bias.stride(0) if bias is not None else 0
    stride_bias_n = bias.stride(1) if bias is not None else 0
    stride_asm = a_scale.stride(0) if a_scale is not None and a_scale.ndim >= 2 else 0
    stride_ask = a_scale.stride(1) if a_scale is not None and a_scale.ndim >= 2 else 0
    stride_bse = b_scale.stride(0) if b_scale is not None and b_scale.ndim >= 1 else 0
    stride_bsk = b_scale.stride(2) if b_scale is not None and b_scale.ndim == 3 else 0
    stride_bsn = b_scale.stride(1) if b_scale is not None and b_scale.ndim >= 2 else 0

    a_desc = None
    b_desc = None
    num_tokens_post_padded_t = torch.tensor(num_tokens_post_padded, dtype=torch.int32, device=a.device)

    grid = lambda meta: (
        triton.cdiv(num_tokens_post_padded, meta["BLOCK_SIZE_M"])
        * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )

    kernel_fn[grid](
        a, a_desc, b, b_desc, bias, c,
        a_scale, b_scale, topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_padded_t,
        N, K, num_tokens_post_padded, M * topk,
        stride_am, stride_ak,
        stride_be, stride_bk, stride_bn,
        stride_bias_e, stride_bias_n,
        stride_cm, stride_cn,
        stride_asm, stride_ask,
        stride_bse, stride_bsk, stride_bsn,
        group_n, group_k,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M,
        mul_routed_weight, topk, compute_type,
        use_fp8_w8a8, use_int8_w8a8, use_int8_w8a16,
        per_channel_quant, even_Ks, c_sorted, filter_expert, swap_ab,
        num_warps=config.get("num_warps", 4),
        num_stages=config.get("num_stages", 4),
    )
    return c

# ============================================================
# Utility
# ============================================================

def _compute_perf_score(ref_time_ms, evolved_time_ms):
    speedup = ref_time_ms / max(evolved_time_ms, 1e-6)
    if speedup < 0.5:
        return 0.0
    elif speedup <= 2.0:
        return 0.5 + (speedup - 1.0) * 0.5
    else:
        return 1.0

# ============================================================
# FusedMoEEvaluator
# ============================================================

class FusedMoEEvaluator:
    def __init__(self):
        self.ref_kernel = None

    def _load_ref_kernel(self):
        if self.ref_kernel is not None:
            return
        sys.path.insert(0, str(_REFERENCE_DIR))
        import original_kernel
        self.ref_kernel = original_kernel.get_reference_kernel()
        sys.path.pop(0)

    def _load_program_kernel(self, program_path):
        spec = importlib.util.spec_from_file_location("_evolved_prog", program_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "get_kernel"):
            raise ValueError("Program missing get_kernel() function")
        return module.get_kernel()

    def _generate_inputs(self, tc):
        torch.manual_seed(tc["seed"])
        M, N_val, K_val, E, topk = tc["M"], tc["N"], tc["K"], tc["E"], tc["topk"]
        dtype = getattr(torch, tc["dtype"])
        device = "cuda"

        a = torch.randn(M, K_val, device=device, dtype=dtype) * 0.01
        b = torch.randn(E, N_val, K_val, device=device, dtype=dtype) * 0.01
        topk_weights = torch.rand(M, topk, device=device, dtype=dtype)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_ids = torch.randint(0, E, (M, topk), device=device, dtype=torch.int64)

        a_scale = None
        b_scale = None
        bias = None

        use_fp8 = tc.get("use_fp8_w8a8", False)
        use_int8_a8 = tc.get("use_int8_w8a8", False)
        use_int8_a16 = tc.get("use_int8_w8a16", False)

        if use_fp8 or use_int8_a8:
            a_scale = torch.rand(M, (K_val + 127) // 128, device=device, dtype=torch.float32) + 0.5
            block_shape = tc.get("block_shape")
            if block_shape:
                gn, gk = block_shape
                b_scale = torch.rand(E, (N_val + gn - 1) // gn, (K_val + gk - 1) // gk,
                                     device=device, dtype=torch.float32) + 0.5
            elif tc.get("per_channel_quant"):
                b_scale = torch.rand(E, N_val, device=device, dtype=torch.float32) + 0.5
            else:
                b_scale = torch.rand(E, device=device, dtype=torch.float32) + 0.5
        elif use_int8_a16:
            b_scale = torch.rand(E, N_val, device=device, dtype=torch.float32) + 0.5

        if tc.get("has_bias"):
            bias = torch.randn(E, N_val, device=device, dtype=dtype) * 0.01

        return a, b, topk_weights, topk_ids, bias, a_scale, b_scale

    def _get_check_flags(self, tc):
        return dict(
            use_fp8_w8a8=tc.get("use_fp8_w8a8", False),
            use_int8_w8a8=tc.get("use_int8_w8a8", False),
            use_int8_w8a16=tc.get("use_int8_w8a16", False),
            block_shape=tc.get("block_shape"),
            per_channel_quant=tc.get("per_channel_quant", False),
            mul_routed_weight=tc.get("mul_routed_weight", True),
            filter_expert=tc.get("filter_expert", False),
            c_sorted=tc.get("c_sorted", False),
            swap_ab=tc.get("swap_ab", False),
        )

    def check_correctness(self, kernel_fn, test_cases):
        self._load_ref_kernel()
        results = []
        for tc in test_cases:
            try:
                a, b, tw, ti, bias, a_sc, b_sc = self._generate_inputs(tc)
                flags = self._get_check_flags(tc)
                with torch.no_grad():
                    ref = launch_kernel(self.ref_kernel, a, b, tw, ti, bias=bias,
                                        a_scale=a_sc, b_scale=b_sc, **flags)
                    evo = launch_kernel(kernel_fn, a, b, tw, ti, bias=bias,
                                        a_scale=a_sc, b_scale=b_sc, **flags)
                max_err = (ref - evo).abs().max().item()
                has_nan = torch.isnan(evo).any().item() or torch.isinf(evo).any().item()
                passed = max_err < 1e-3 and not has_nan
                results.append({"name": tc["name"], "max_error": max_err,
                                "passed": passed, "has_nan": has_nan})
            except Exception as e:
                results.append({"name": tc["name"], "error": str(e)[:200], "passed": False})
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        all_passed = all(r["passed"] for r in results)
        return all_passed, results

    def measure_performance(self, kernel_fn, test_cases, warmup=5, repeat=20):
        self._load_ref_kernel()
        results = {}
        for tc in test_cases:
            try:
                a, b, tw, ti, bias, a_sc, b_sc = self._generate_inputs(tc)
                flags = self._get_check_flags(tc)
                for _ in range(warmup):
                    launch_kernel(kernel_fn, a, b, tw, ti, bias=bias,
                                  a_scale=a_sc, b_scale=b_sc, **flags)
                torch.cuda.synchronize()
                times = []
                for _ in range(repeat):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    launch_kernel(kernel_fn, a, b, tw, ti, bias=bias,
                                  a_scale=a_sc, b_scale=b_sc, **flags)
                    end.record()
                    torch.cuda.synchronize()
                    times.append(start.elapsed_time(end))
                results[tc["name"]] = {
                    "median_ms": float(np.median(times)),
                    "min_ms": float(np.min(times)),
                    "mean_ms": float(np.mean(times)),
                }
            except Exception as e:
                results[tc["name"]] = {"error": str(e)[:200]}
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return results

    def evaluate(self, program_path):
        try:
            kernel = self._load_program_kernel(program_path)

            all_passed, correct_results = self.check_correctness(kernel, TEST_CASES)
            if not all_passed:
                failed = [r for r in correct_results if not r["passed"]]
                return EvaluationResult(
                    metrics={"combined_score": 0.0, "correctness": 0.0, "perf_score": 0.0},
                    artifacts={"failed_tests": str(failed)[:2000], "status": "correctness_failed"}
                )

            ref_perf = self.measure_performance(self.ref_kernel, TEST_CASES)
            evo_perf = self.measure_performance(kernel, TEST_CASES)

            perf_scores = []
            weights = []
            for tc in TEST_CASES:
                name = tc["name"]
                if "error" in ref_perf.get(name, {}) or "error" in evo_perf.get(name, {}):
                    continue
                ref_t = ref_perf[name]["median_ms"]
                evo_t = evo_perf[name]["median_ms"]
                perf_scores.append(_compute_perf_score(ref_t, evo_t))
                weights.append(tc["M"] * tc["N"] * tc["K"] / 1e9)

            if not perf_scores:
                return EvaluationResult(
                    metrics={"combined_score": 0.0, "correctness": 1.0, "perf_score": 0.0},
                    artifacts={"status": "perf_measurement_failed"}
                )

            total_w = sum(weights)
            perf_score = float(np.exp(
                sum(w * np.log(max(s, 1e-6)) for w, s in zip(weights, perf_scores)) / total_w
            ))
            combined = 0.3 + 0.7 * perf_score

            speedups = {}
            for tc in TEST_CASES:
                name = tc["name"]
                if name in ref_perf and name in evo_perf and "median_ms" in ref_perf[name]:
                    speedups[name] = ref_perf[name]["median_ms"] / max(evo_perf[name].get("median_ms", 1e-6), 1e-6)

            avg_speedup = float(np.mean(list(speedups.values()))) if speedups else 0.0

            return EvaluationResult(
                metrics={
                    "combined_score": float(combined),
                    "correctness": 1.0,
                    "perf_score": float(perf_score),
                    "avg_speedup": avg_speedup,
                },
                artifacts={
                    "speedups": str(speedups)[:4000],
                    "ref_perf_summary": str({k: v.get("median_ms", 0) for k, v in ref_perf.items()})[:2000],
                    "evo_perf_summary": str({k: v.get("median_ms", 0) for k, v in evo_perf.items()})[:2000],
                }
            )
        except Exception as e:
            return EvaluationResult(
                metrics={"combined_score": 0.0, "correctness": 0.0, "perf_score": 0.0},
                artifacts={"error_type": type(e).__name__, "error": str(e)[:2000],
                           "traceback": traceback.format_exc()[:3000]}
            )

    def evaluate_stage1(self, program_path):
        try:
            kernel = self._load_program_kernel(program_path)
            all_passed, results = self.check_correctness(kernel, FAST_TEST_CASES)
            score = 0.3 if all_passed else 0.0
            return EvaluationResult(
                metrics={"combined_score": score, "correctness": float(all_passed)},
                artifacts={"stage": 1, "results": str(results)[:2000]}
            )
        except Exception as e:
            return EvaluationResult(
                metrics={"combined_score": 0.0},
                artifacts={"error": str(e)[:2000]}
            )

    def evaluate_stage2(self, program_path):
        return self.evaluate(program_path)


_evaluator = FusedMoEEvaluator()

def evaluate(program_path):
    return _evaluator.evaluate(program_path)

def evaluate_stage1(program_path):
    return _evaluator.evaluate_stage1(program_path)

def evaluate_stage2(program_path):
    return _evaluator.evaluate_stage2(program_path)

if __name__ == "__main__":
    import sys
    print("FusedMoE Evaluator smoke test")
    print(f"  Test cases: {len(TEST_CASES)} total, {len(FAST_TEST_CASES)} fast")
    print(f"  Reference dir: {_REFERENCE_DIR}")
    if len(sys.argv) > 1:
        path = sys.argv[1]
        result = evaluate(path)
        print(f"  Result: {result.metrics}")
