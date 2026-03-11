# Fish Speech S2 Pro

## Architecture

Fish Speech S2 Pro (`fishaudio/s2-pro`) is a 4B dual-AR TTS system by FishAudio:

- **Stage 0 (Slow AR + Fast AR)**: Qwen3-based Slow AR generates semantic tokens autoregressively. Fast AR predicts residual codebook tokens (9 codebooks) from each semantic token in a single forward pass.
- **Stage 1 (DAC Decoder)**: Converts codec codes [10 codebooks x T frames] into 44.1 kHz audio waveform via the DAC codec decoder.

Key differences from Qwen3-TTS:
- **Dual-AR in one stage**: Slow AR and Fast AR run together in Stage 0 (vs Qwen3-TTS's single code predictor)
- **DAC codec**: 10 codebooks (1 semantic + 9 residual), 44.1 kHz output (vs Qwen3-TTS's SpeechTokenizer at 24 kHz)
- **No task_type**: Uses `ref_audio` + `ref_text` for voice cloning directly (no CustomVoice/VoiceDesign/Base variants)
- **Token mapping**: Semantic tokens use IDs 151678-155773 (4096 codes), special tokens like `<|voice|>` = 151673

## Model

- **HF ID**: `fishaudio/s2-pro`
- **Parameters**: ~4B
- **Min VRAM**: 16 GB
- **Output sample rate**: 44.1 kHz
- **model_type**: `fish_qwen3_omni`

## Key Files

```
vllm_omni/model_executor/models/fish_speech/
├── configuration_fish_speech.py   # Config classes (FishSpeechConfig, SlowAR, FastAR)
├── fish_speech_slow_ar.py         # Stage 0: Slow AR (semantic token generation)
├── fish_speech_fast_ar.py         # Stage 0: Fast AR (residual codebook prediction)
├── fish_speech_dac_decoder.py     # Stage 1: DAC codec decoder
├── dac_encoder.py                 # CPU-side DAC encoder for voice cloning
└── __init__.py

vllm_omni/model_executor/stage_configs/fish_speech_s2_pro.yaml
vllm_omni/model_executor/stage_input_processors/fish_speech.py
vllm_omni/transformers_utils/configs/fish_speech.py
vllm_omni/entrypoints/openai/serving_speech.py  # Online serving (shared with Qwen3-TTS)
```

## Prerequisites

```bash
pip install fish-speech  # Required for DAC codec model architecture
```

## Serving

```bash
vllm-omni serve fishaudio/s2-pro \
  --stage-configs-path vllm_omni/model_executor/stage_configs/fish_speech_s2_pro.yaml \
  --omni --port 8091 \
  --trust-remote-code --enforce-eager \
  --gpu-memory-utilization 0.9
```

## Offline Inference

```python
from vllm_omni import Omni

omni = Omni(
    model="fishaudio/s2-pro",
    stage_configs_path="vllm_omni/model_executor/stage_configs/fish_speech_s2_pro.yaml",
)

# Build prompt with Fish Speech chat template
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("fishaudio/s2-pro", trust_remote_code=True)
messages = [{"role": "user", "content": "<|speaker:0|>Hello, this is a test."}]
prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
voice_id = tokenizer.encode("<|voice|>", add_special_tokens=False)
prompt_ids = prompt_ids + voice_id

inputs = [{"prompt_token_ids": prompt_ids, "additional_information": {"text": ["Hello"], "max_new_tokens": [4096]}}]
for stage_outputs in omni.generate(inputs):
    for output in stage_outputs.request_output:
        audio = output.outputs[0].multimodal_output
        # audio["audio"] is the waveform, audio["sr"] is 44100
```

See `examples/offline_inference/fish_speech/end2end.py` for the full script.

## Voice Cloning

Provide `ref_audio` (path or URL) and `ref_text` to clone a voice:

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, this is a cloned voice.",
    "voice": "default",
    "ref_audio": "https://example.com/reference.wav",
    "ref_text": "Transcript of the reference audio."
  }' --output cloned.wav
```

Internally, the server DAC-encodes the reference audio on CPU, extracts semantic codes (codebook 0), converts them to token IDs (151678 + code_value), and prepends as a system message with `<|audio_start|>...<|audio_end|>` markers.

## Debugging Notes

Common issues encountered during integration (in priority order):

1. **Hop length**: DAC hop = 2048 (512 decoder upsample * 4 quantizer upsample). Wrong value causes noise in streaming.
2. **Token mapping**: Semantic codes must be clamped to [0, 4095]. `im_end` token (151645) minus `semantic_begin_id` (151678) = negative -- must clamp.
3. **Autocast**: Use `torch.cuda.amp.autocast(enabled=False)` not `autocast(dtype=torch.float32)` for the DAC decoder.
4. **Repetition penalty**: Must be 1.0 (vLLM's global penalty != reference model's windowed penalty).
5. **max_new_tokens**: Must be explicitly passed to Stage-0's sampling_params (not just stored in additional_information).
