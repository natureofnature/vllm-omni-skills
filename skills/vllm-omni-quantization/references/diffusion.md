# Diffusion Quantization Reference

The local `vllm-omni` diffusion quantization methods are:

- `fp8`
- `gguf`

Use this file when working on DiT quantization, adding a new method such as `int8` or `nvfp4`, or debugging loader, mapping, quality, or performance issues.

## Ownership Boundary

- Upstream `vllm`: `QuantizationConfig`, quant methods, kernels, generic loader behavior, generic post-load processing, non-diffusion hardware rules
- `vllm-omni`: `OmniDiffusionConfig`, wrappers under `vllm_omni/diffusion/quantization/`, threading `quant_config` into diffusion linear layers, GGUF adapters, diffusion docs and tests

Rule: if a method is missing generic kernels, loader semantics, or config classes, fix upstream `vllm` first. `vllm-omni` should add a thin diffusion wrapper, not a private quantization stack.

## Existing Methods

### FP8

- Online quantization from BF16 or FP16 weights
- Reuses upstream `vllm` FP8 infrastructure
- Flow: user config -> `OmniDiffusionConfig` -> `DiffusionFp8Config` -> `get_vllm_quant_config_for_layers()` -> transformer linear layers
- `DiffusersPipelineLoader` must still run `process_weights_after_loading()` for modules with quant methods

Starting points:

- `Qwen-Image`, `Qwen-Image-2512`: begin with `ignored_layers=["img_mlp"]`
- `Tongyi-MAI/Z-Image-Turbo`: start with all layers quantized

Prefer dynamic activation scaling unless static calibration is explicitly required.

### GGUF

- Native quantized transformer-weight loading
- Requires `quantization_config.gguf_model`
- Resolve GGUF as a local file, `repo/file.gguf`, or `repo:quant_type`
- If the GGUF repo lacks `model_index.json`, use the base HF repo for config and use GGUF only for transformer weights

Implementation rules:

- Do not use `state_dict()` to discover GGUF loadable names; use `named_parameters()` and `named_buffers()`
- Tensor-name mapping must be explicit per architecture
- Do not rely on a fake generic fallback adapter
- Guard fused QKV and KV rewrites so `to_qkv` or `add_kv_proj` are not rewritten twice
- GGUF linear methods expect 2D input; flatten and restore shape around matmul
- Prefer eager mode and `fp16` unless measurement says otherwise

## Adding a New Method

Follow this order:

1. Record the exact date context and current repo behavior.
2. Confirm the method type: online quantization, native quantized checkpoint, or GGUF-like external transformer weights.
3. Check upstream `vllm` for config class, kernel path, loader path, and hardware rules.
4. If upstream support exists, add a wrapper in `vllm_omni/diffusion/quantization/` and register it in `_QUANT_CONFIG_REGISTRY`.
5. Normalize user config in `vllm_omni/diffusion/data.py` via `quantization` or `quantization_config` without mutating input mappings.
6. Update `vllm_omni/diffusion/model_loader/diffusers_loader.py` for the correct load path.
7. Thread `quant_config` through every relevant diffusion linear layer.
8. Add config, loader, and smoke tests, then update docs and examples.

For `int8`, `nvfp4`, or future methods: if the generic method is not already present in upstream `vllm`, do not implement it first as a diffusion-only stack.

## Common Failures

| Symptom | Likely Cause | Fix |
|--------|--------------|-----|
| `Unknown quantization method` | wrapper not registered | add config wrapper and registry entry |
| config behaves strangely | input mapping was mutated or precedence is unclear | copy the mapping and define conflict behavior |
| only some layers quantize | `quant_config` not threaded through all linear layers | audit every transformer linear constructor |
| FP8 quality collapses | sensitive layers should stay BF16 | start with `img_mlp` in `ignored_layers` |
| GGUF adapter mismatch | model is unsupported or mapping is missing | add a model-specific adapter |
| GGUF shape mismatch | remap or qkv or ffn split logic is wrong | fix mapping per architecture and re-test with a fixed seed |
| quantized path is slower than BF16 | dtype mismatch, kernel behavior, or compile issue | verify dtype, test smaller sizes, compare one-step runs, try eager |

## Verification

- Record the exact date context in notes and docs
- Confirm whether the mode is `fp8`, `gguf`, or a new method
- Use a fixed seed and identical prompt, image size, and inference steps
- Compare against a BF16 baseline for both quality and speed
- For GGUF, verify architecture-specific mapping before full end-to-end runs
- For new methods, pass config and loader-path tests before claiming the model path works
- Fail clearly on unsupported models; do not leave half-working fallback behavior
