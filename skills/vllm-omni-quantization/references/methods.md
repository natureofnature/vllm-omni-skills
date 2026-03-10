# AR Quantization Methods Reference

This file covers autoregressive and general `vllm` quantization paths used by `vllm-omni`. For diffusion `fp8` and `gguf`, see `diffusion.md`.

## AWQ (Activation-Aware Weight Quantization)

- **Precision**: 4-bit weights, 16-bit activations
- **Memory savings**: ~4x vs BF16
- **Speed**: 1.2-1.5x faster decode
- **Calibration**: Requires a small calibration dataset (~128 samples); lightweight
- **Tool**: [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
- **HuggingFace naming**: `-AWQ` suffix (e.g., `Qwen2.5-Omni-7B-AWQ`)

AWQ identifies which weights are "salient" (high activation magnitude) and protects them during quantization, preserving quality better than naive round-to-nearest.

### Key Config Options

| Option | Values | Effect |
|--------|--------|--------|
| `w_bit` | 4, 8 | Weight precision |
| `q_group_size` | 64, 128 | Quantization group size (smaller = better quality, more overhead) |
| `zero_point` | True/False | Zero-point quantization (True recommended) |
| `version` | GEMM, GEMV | Kernel variant (GEMM for throughput, GEMV for small batch) |

## GPTQ (Generative Pre-Trained Transformer Quantization)

- **Precision**: 4-bit or 8-bit weights
- **Memory savings**: ~4x (4-bit), ~2x (8-bit)
- **Speed**: 1.2-1.4x faster decode
- **Calibration**: Requires calibration dataset; slower than AWQ (~hours for large models)
- **Tool**: [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ)
- **HuggingFace naming**: `-GPTQ-Int4` or `-GPTQ-Int8` suffix

GPTQ uses second-order information (Hessian) to minimize quantization error layer by layer.

### Key Config Options

| Option | Values | Effect |
|--------|--------|--------|
| `bits` | 4, 8 | Quantization precision |
| `group_size` | 32, 64, 128 | Group size (32 = best quality, highest overhead) |
| `desc_act` | True/False | Descending activation order (True = better quality, slower) |

## FP8 (8-bit Floating Point)

- **Precision**: 8-bit float (E4M3 format)
- **Memory savings**: ~2x vs BF16
- **Speed**: 1.1-1.3x faster
- **Hardware requirement**: NVIDIA H100/H200/B100 (compute capability 9.0+) for weight FP8; A100+ for KV cache FP8
- **No calibration needed** for weight FP8 (dynamic quantization)
- **HuggingFace naming**: `-FP8` or `-fp8` suffix

```bash
# Weight FP8 (H100+ only)
vllm serve <model> --omni --quantization fp8

# KV cache FP8 (A100+)
vllm serve <model> --omni --kv-cache-dtype fp8
```

## SqueezeLLM

- **Precision**: Mixed (sparse + quantized)
- **Memory savings**: ~2-3x
- **Speed**: 1.1-1.3x
- **Use case**: When quality is critical and memory savings of 2x are sufficient

Less widely used than AWQ/GPTQ. Fewer pre-quantized models available on HuggingFace.

## BF16 / FP16 (Dtype, Not Quantization)

Changing dtype reduces memory without a quantization step:

| Dtype | VRAM | Hardware |
|-------|------|----------|
| BF16 | Baseline | Ampere+ (A100, RTX 30/40, H100) -- recommended |
| FP16 | Same as BF16 | All CUDA GPUs -- use for older architectures |
| FP32 | 2x BF16 | For debugging only |

```bash
vllm serve <model> --omni --dtype bfloat16   # default on Ampere+
vllm serve <model> --omni --dtype float16    # for Volta/Turing (V100, RTX 20 series)
```

## Method Comparison Summary

| Method | VRAM Savings | Calibration | Hardware | Quality Loss |
|--------|-------------|-------------|----------|-------------|
| AWQ 4-bit | ~4x | Light (~minutes) | Any CUDA | Minimal |
| GPTQ 4-bit | ~4x | Heavy (~hours) | Any CUDA | Minimal |
| FP8 weights | ~2x | None | H100+ | Minimal |
| FP8 KV cache | KV only | None | A100+ | Minimal |
| SqueezeLLM | ~2-3x | Medium | Any CUDA | Low |
