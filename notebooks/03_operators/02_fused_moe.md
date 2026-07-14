# FusedMoE Kernel Optimization

## Overview

FusedMoE (Fused Mixture of Experts) is a Triton kernel in SGLang that fuses
the gating + expert computation for MoE layers. It's the primary optimization
target for the `fusedmoe-evolve` task type.

## Target Kernel

- **File:** `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py`
- **Function:** `fused_moe_kernel` (line 339)
- **Lines:** 339-626 (287 lines of Triton kernel code)

## Kernel Architecture

The kernel processes MoE layers in a single fused Triton kernel:

1. **Input:** Token activations `A [M, K]`, expert weights `B [E, K, N]`
2. **Routing:** `topk_weights [M, top_k]`, `sorted_token_ids`, `expert_ids`
3. **Output:** Combined expert output `C [M, top_k, N]`

### Key Optimization Axes

| Axis | Description | Impact |
|------|-------------|--------|
| Block size tuning | BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K | Memory coalescing, occupancy |
| SplitK | Splits K dimension across CTAs | Better SM utilization for large K |
| even_Ks path | Optimized path for K % BLOCK_SIZE_K == 0 | Faster, fewer bounds checks |
| TMA (Hopper) | Tensor Memory Accelerator for H100/H20 | Reduced register pressure |
| wgmma (Hopper) | Warp-group matrix multiply-accumulate | Higher throughput on SM90 |
| fp8 compute | W8A8 with block-wise quantization | 2x throughput vs bf16 |

### Common Optimization Patterns

1. **Memory Coalescing:** Align block sizes to cache line boundaries (128 bytes)
2. **Register Pressure:** Tune block sizes to stay under 255 registers/SM
3. **Pipeline Stalls:** Insert `tl.dot` operations to hide memory latency
4. **Shared Memory:** Use `tl.static_range` for compile-time unrolling
5. **Grid Tuning:** Adjust grid size based on M to balance SM utilization

## Evaluator Design

The evaluator for OpenEvolve (`evaluator.py`) must:

1. **Score function:** `evaluate(program_text: str) -> dict`
2. **Return format:** `{"combined_score": float, "metrics": {...}, "artifacts": {...}}`
3. **Scoring rubric:**
   - -1000: Crash, NaN, Inf
   - 0-50: Correctness failure (proportional)
   - 50-100: Correctness pass (50 base + 50 × speedup_ratio)

### Reference Implementation

A PyTorch eager reference MoE computes:

```python
# Reference: torch.einsum + topk routing
output = torch.zeros(M, top_k, N, dtype=dtype, device=device)
for i in range(M):
    for t in range(top_k):
        e = expert_ids[i, t]
        w = topk_weights[i, t]
        output[i, t] = w * (A[i] @ B[e])
```

## Hardware-Specific Notes

### H100 / H20 (SM 90, Hopper)
- TMA loads reduce register pressure
- wgmma MMA operations are faster than mma.sync
- fp8 tensor cores: 2x bf16 throughput
- Shared memory: 228 KB per SM

### A100 (SM 80, Ampere)
- Async copy (cp.async) for overlapping loads
- mma.sync for matrix operations
- Shared memory: 164 KB per SM
- No fp8 support

### RTX 4090 (SM 89, Ada)
- Similar to Hopper but no TMA
- Higher clock, fewer SMs
- Consumer card constraints: 24 GB VRAM

### MI300 (CDNA3, AMD)
- ROCm HIP backend
- Matrix cores via rocBLAS/rocWMMA
- Different shared memory model

## Known Pitfalls

1. **Non-aligned K:** The even_Ks path assumes K % BLOCK_SIZE_K == 0. Fails silently with wrong results for odd K.
2. **Dtype mismatch:** Mixing fp16/bf16 tensors without explicit casts produces wrong results.
3. **Triton version sensitivity:** Kernel behavior changes between Triton versions (2.1.x vs 3.x).
4. **Grid launch:** Wrong grid dimensions cause hangs or partial computation.
5. **NaN propagation:** MoE routing with zero expert scores can produce NaN gradients.

## Testing

The validation oracle runs 7 test cases covering:
- Small/medium/large token counts (M=128, 1024, 4096)
- bf16 and fp8 dtypes
- Non-aligned K and M (edge cases)
- DeepSeek V2/V3 typical shapes (E=256, N=2048, K=5120, top_k=8)

Tolerances:
- bf16: rtol=0.02, atol=0.01
- fp8: rtol=0.05, atol=0.05
