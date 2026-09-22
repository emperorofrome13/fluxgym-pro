# fluxgym-pro handoff — v1.04

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

The UI is at `http://127.0.0.1:7860` and shows **v1.03** in its heading.

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

## Note

Actual GPU training still requires a compatible `musubi-tuner` or `ai-toolkit`
checkout and the selected model weights; neither backend was run in this UI-only
verification.
