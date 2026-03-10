# Quantization Compatibility by Modality

## Summary

Quantization in `vllm-omni` is split:

- AR and omni-model quantization follow upstream `vllm`
- diffusion quantization exists for selected DiT models through `fp8` and `gguf`

## Compatibility Matrix

| Model Type | AR Weight Quantization | Diffusion Quantization | KV Cache FP8 | Notes |
|------------|------------------------|------------------------|--------------|-------|
| **Omni models** (Qwen-Omni) | Yes | N/A | Yes | Quantize the AR backbone |
| **AR-only text models** | Yes | N/A | Yes | Full upstream support |
| **Multi-stage AR+DiT** (Qwen-Image, BAGEL) | Partial | Qwen-Image FP8 only | AR stage only | Qwen-Image DiT supports diffusion FP8; other hybrids vary by model |
| **DiT-only image models** (Z-Image, FLUX.2-klein, SD3) | No | Z-Image FP8 and GGUF; FLUX.2-klein GGUF; SD3 none | N/A | Check model-specific diffusion support before assuming coverage |
| **Video models** (Wan2.2) | No | No | N/A | Use cache and parallelism instead |
| **TTS models** (Qwen3-TTS) | Yes | N/A | Yes | Quality may degrade for voice cloning |
| **Audio models** (Stable-Audio) | No | No | N/A | Diffusion architecture, no quantization path documented here |

## Omni Models (Qwen2.5-Omni, Qwen3-Omni)

The AR backbone (thinker/talker stages) can be quantized. This reduces memory for the language model portion:

```bash
vllm serve Qwen/Qwen2.5-Omni-7B-AWQ --omni --quantization awq
```

**Expected impact**:
- Text understanding/generation quality: minimal degradation (~95%+ of BF16)
- Audio output quality: slight degradation, especially for voice characteristics
- Multi-modal understanding: minimal degradation

## Image Generation Models (FLUX, SD3, Qwen-Image, BAGEL, Z-Image)

Diffusion quantization is now model-specific rather than globally unsupported.

- `Qwen-Image`, `Qwen-Image-2512`: diffusion `fp8` supported; start with `ignored_layers=["img_mlp"]`
- `Tongyi-MAI/Z-Image-Turbo`: diffusion `fp8` supported; all-layer quantization is the default starting point
- `FLUX.2-klein`: diffusion `gguf` supported through model-specific adapter logic
- `SD3`, `Bagel`, most video diffusion models: no diffusion quantization path documented here

For hybrids with an AR stage, AR quantization can still save memory even if the DiT stage is not quantized:

```bash
vllm serve Qwen/Qwen-Image-AWQ --omni --quantization awq
```

For unsupported diffusion models, use:

- **TeaCache** or **Cache-DiT** for speed
- **CPU offloading** for memory
- **BF16/FP16** dtype choices

## Video Models (Wan2.2)

No documented diffusion weight quantization support. Use:
- **TeaCache** or **Cache-DiT** for speed
- **CPU offloading** for memory
- **Tensor parallelism** for both

## TTS Models (Qwen3-TTS)

The AR decoder can be quantized. Trade-offs:
- Intelligibility: maintained
- Voice cloning fidelity (CustomVoice): slight degradation at 4-bit
- Recommendation: use AWQ 8-bit instead of 4-bit for TTS to preserve voice quality

```bash
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-AWQ --omni --quantization awq
```

## Choosing Based on Modality

| Primary Use Case | Recommended Approach |
|-----------------|---------------------|
| Text or omni AR backbone | AWQ 4-bit or GPTQ 4-bit |
| Qwen-Image diffusion | diffusion `fp8`, often with `ignored_layers=["img_mlp"]` |
| Z-Image diffusion | diffusion `fp8` |
| FLUX.2-klein diffusion | diffusion `gguf` |
| Unsupported diffusion models | CPU offloading + cache acceleration |
| TTS (Qwen3-TTS) | AWQ 8-bit for quality |
| Audio understanding (MiMo-Audio) | AWQ 4-bit |
