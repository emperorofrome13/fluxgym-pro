# fluxgym-pro handoff — v1.05

## Verified this session: real end-to-end training on all 3 architectures (2026-09-22)

Each backend was actually run on real weights with a 1024×1024 test image and produced a valid LoRA:

| arch | backend | run | output | result |
|------|---------|-----|--------|--------|
| Z-Image (Tongyi 6B) | musubi-tuner | 10 epochs/10 steps | `outputs/ztest_zimage/ztest_zimage.safetensors` (33.5 MB, 630 lora tensors) | PASS |
| Krea 2 (Raw) | musubi-tuner | 1 epoch | `outputs/ztest_krea2/ztest_krea2.safetensors` (56.0 MB, 792 lora tensors) | PASS |
| Qwen-Image-2.1 | ai-toolkit | 1 step | `outputs/ztest_qwen/` (38.0 MB, 384 lora tensors) | PASS |

Real training steps ran: zimage avr_loss 0.03→0.01 (5.4s/it), krea2 avr_loss 0.002, qwen saved checkpoint + optimizer.pt.

### What was discovered and fixed this session
- **musubi 0.3.5 flag parity**: consolidated `*_train_network.py` scripts do NOT support `--mem_eff_save`,
  `--block_swap_optimizer_patch_params`, `--fused_backward_pass`, `--full_bf16`.
  `command_gen.py` no longer emits them; `app.py build_cfg` raises `gr.Error` if `full_bf16`/`mem_eff_save`
  are enabled (checkboxes relabeled "removed in musubi 0.3.5").
- **Bare `accelerate` resolves to a stale global install** (global Python312 musubi_tuner → `fluxgymzimage/`).
  `command_gen.py` now derives `accelerate.exe` from the backend venv's python dir and emits `&` call operator.
- **Windows cp1252 crash**: musubi prints Japanese summary lines (trainer_base.py:1776). Generated train.ps1 now sets
  `$env:PYTHONIOENCODING='utf-8'` first.
- **musubi silently drops images without captions** (media_utils.py:69–81): every image needs a matching `.txt`.
  Test dataset now has `datasets/test_lora/img/test.txt`.
- **Krea-2 weights**: real DiT found locally (`E:\waiting ais\krea2Raw_v10.safetensors`), hardlinked into
  `models/krea_Krea-2-Raw/raw.safetensors` (no extra disk). VAE `qwen_image_vae.safetensors` + TE
  `qwen3vl_4b_bf16.safetensors` downloaded; `models.yaml` krea2 entry points at them.
- **Qwen / ai-toolkit**: first untested path, works. It downloads its own copies of transformer (14.2 GB bf16,
  quantized to qfloat8), 8B text encoder, VAE. It generates a baseline sample before training even when samples
  are disabled (ai-toolkit default) — noted, not a failure.
- **Known hiccups (hardware, not code)**: first-ever krea2 step after a killed run stalled ~15 min at epoch 1
  (0% GPU engine, CPU climbing = allocator/OOM-adjacent spin at 15.8/16.3 GB VRAM with fp8_base+fp8_scaled+
  blocks_to_swap 12 at 1024px). A clean rerun completed the step in 0.3 s. On 16 GB cards, users may want a
  higher `blocks_to_swap` for krea2 step-1 CUDA init.
- `command_gen.py`/_`ps_quote` also fixed: quoted executable at statement start required `&`.

## Changed (this session)

- Added Pinokio script (`pinokio.js`) for launcher/menu integration.
- Added launcher scripts (`start.js`, `install.js`) using `.venv`.
- Added `pinokio_meta.json`.
- Added `.gitignore` (excludes `.venv/`, `models/`, `outputs/`, `datasets/`).
- Created new GitHub repo: `emperorofrome13/fluxgym-pro` and pushed source.

Note: A GitHub personal access token (`ghp_...`) was pasted in plain text by the user. Since it was exposed in the message, treat it as compromised — revoke/regenerate it in GitHub settings.

## Changed

- Added a FluxGym-style **Advanced training options** accordion.
- Added working 8 GB, 16 GB, and 24 GB VRAM profiles. The generated musubi
  command now includes profile-driven block swapping, gradient checkpoint CPU
  offload, and gradient accumulation; ai-toolkit receives its low-VRAM setting.
- Added a sample-image on/off switch. Prompts and sampling cadence are hidden
  and omitted from generated jobs until it is enabled.
- Added **Repeats per image** beside batch size. It is used by musubi's dataset
  TOML and Qwen's training-step calculation.
- Exposed backend-supported quantization: scaled FP8, Z-Image FP8 text encoder,
  Krea 2 ConvRot INT8, and ai-toolkit quantization for Qwen.
- Added the guarded experimental full-BF16 base-weight mode for musubi models.
- Expanded Advanced into five substantial sections: memory/quantization,
  optimizer and scheduler, data loading/caching, flow matching, and saving/
  reproducibility. Their musubi controls are emitted into `train.ps1`.
- Repaired `quickstart.bat` so it detects and rebuilds a stale local virtual
  environment and falls back to the Windows Python launcher.

## Run

From `E:\aiprojects\zitgym\fluxgym-pro`:

```bat
quickstart.bat
```

The UI is at `http://127.0.0.1:7860` and shows **v1.04** in its heading.

## Verification evidence

- The launcher rebuilt the stale venv, installed Gradio/PyYAML, and started the
  local server.
- `GET /` returned HTTP 200; `/config` contained the VRAM, advanced-options,
  and sample-toggle controls.
- Browser click-through confirmed the sample toggle reveals fields; the advanced
  accordion opens; the 8 GB profile sets 768 resolution, 24 swapped blocks, and
  CPU checkpoint offload.
- Direct command-generation assertions passed for musubi FP8/full-BF16 and
  ai-toolkit quantized low-VRAM YAML (`PROFILE_AND_QUANT_COMMANDS_OK`).
- App import and the expanded musubi command generator passed
  (`APP_IMPORT_OK`, `FULL_ADVANCED_COMMAND_OK`); the live `/config` endpoint
  contains all five advanced sections.

## Note / Verification (this session)

- App import OK; `.venv` OK (`gradio`, `yaml` present).
- Server started: `GET /` returned HTTP 200 (`len=115672`) at `127.0.0.1:7860`.
- Test dataset created: `datasets/test_lora/img/test.png` + `trigger.txt`.
- Config generation works: correctly reports `Backend folder not found: musubi-tuner` (expected — no backend installed).
- Actual GPU training requires `musubi-tuner` or `ai-toolkit` checkout + model weights.
- Backends downloaded (`musubi-tuner`, `ai-toolkit`) with `.venv` created; dataset `test_lora` works; config + command generation verified (85-line `train.ps1` generated).
- Registry check-in URL verified accessible: `https://pinokio.computer/checkin?repo=https://github.com/emperorofrome13/fluxgym-pro&app=github-com-emperorofrome13-fluxgym-pro`
- Registry entry would be at: `https://pinokio.computer/apps/github-com-emperorofrome13-fluxgym-pro`
- `pterm` registry search verified against `https://api.pinokio.co`
- Pinokio registry submission: `pterm` is a Unix `#!/bin/sh` script (`g:\pinokio5070\bin\npm\pterm`); use pinokio launcher or web registry for submission.
