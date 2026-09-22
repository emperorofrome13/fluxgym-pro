# fluxgym-pro

Dead simple LoRA training UI for **Z-Image**, **Krea 2** and **Qwen-Image-2.1**.
Fork of [cocktailpeanut/fluxgym](https://github.com/cocktailpeanut/fluxgym) with
the sd-scripts backend swapped for the trainers that actually support these models.

## Backends

| Model | Backend | Why |
|---|---|---|
| Z-Image (Tongyi-MAI, 6B) | [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner) | native `zimage_*` scripts |
| Krea 2 (Raw / Turbo) | musubi-tuner | native `krea2_*` scripts |
| Qwen-Image-2.1 | [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) | day-0 `qwen_image_2` arch; musubi's `qwen_image` scripts only cover the older 20B model |

You never see this seam — the UI is identical for all three; only the generated
training script differs (PowerShell on Windows, with `$LASTEXITCODE` checks).

## Quickstart

```bat
quickstart.bat
```

That creates a local venv, installs UI deps, and opens http://127.0.0.1:7860.
(The UI itself needs no torch — training runs in the backend's own venv.)

Manual equivalent:

```powershell
pip install -r requirements.txt
python app.py
```

## Backends: one-time setup

Clone the backends next to this folder (or anywhere — the path is a UI field):

```powershell
# musubi-tuner, for Z-Image and Krea 2
git clone https://github.com/kohya-ss/musubi-tuner
cd musubi-tuner
python -m venv .venv
.venv\Scripts\pip install -e .          # + CUDA torch per musubi README

# ai-toolkit, for Qwen-Image-2.1
cd ..
git clone --recurse-submodules https://github.com/ostris/ai-toolkit
cd ai-toolkit
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

fluxgym-pro auto-detects `backend\.venv\Scripts\python.exe` and falls back to
the system Python. Download the model weights listed in `models.yaml`
(HuggingFace repos) — the UI tells you exactly which file paths it expects if
one is missing.

## Using it

1. **Dataset** — name it, upload images, click *Create dataset*.
2. **AI captions** *(optional)* — click *AI captions* to auto-tag every image
   with a WD14 tagger (CPU, downloads the model once). Add a trigger word and
   it gets prepended to each caption.
3. **Train** — pick base model, LoRA name, keep defaults, *Preview commands*
   to inspect the exact script, then *Start training*.
   **Repeats per image** controls how often each image appears in an epoch.
4. **Sample prompts** *(optional)* — one prompt per line; training saves sample
   images every N epochs so you can watch the LoRA learn.
5. **Status** — live `train.log` tail; your LoRA lands in `outputs/<name>/`.

## VRAM and precision

Choose the **GPU VRAM profile** before previewing a run. The app applies real
backend settings, not just a label: 8 GB uses 768 px, CPU block swapping, and
checkpoint CPU offload; 16 GB uses 1024 px with moderate block swapping; 24 GB
uses the model's normal resolution with no block swapping. All profiles keep
batch size at 1; CPU swapping trades a great deal of speed for lower VRAM.

Open **Advanced training options** to change the profile defaults. Z-Image and
Krea 2 support scaled FP8 DiT weights; Z-Image can also run its text encoder in
FP8. Krea 2 additionally supports ConvRot INT8 as an FP8 alternative (Triton
is needed for its speedup). Qwen-Image-2.1 uses ai-toolkit's quantized-model
and quantized-text-encoder settings instead. `bf16`/`fp16` remains the compute
precision; it is not base-weight quantization.

The advanced panel also exposes musubi's optimizer and scheduler, warmup,
gradient clipping, optimizer arguments, cache and data-loader sizing, flow
matching shift/guidance, checkpoint precision, optimizer-state saving,
memory-efficient saves, and seed. They are written directly into the generated
PowerShell command; options labelled **musubi only** are intentionally not sent
to Qwen's different ai-toolkit backend.

**Full BF16 base weights** is available only for the musubi models. It lowers
VRAM versus FP32 base weights, but is experimental and automatically pairs
Adafactor with fused backward pass. For most GPUs, use the default scaled FP8
profile first.

Everything (dataset TOML, YAML, script, log, checkpoints) is written to
`outputs/<your lora name>/`, so a run is fully reproducible.

## Adding models

Edit `models.yaml`. One entry = one dropdown option. `arch` must be one of
`zimage | krea2 | qwen_image_2` (see `command_gen.py` for the script mapping).

## GPU note

The UI never touches your GPU. Training runs in the backend venv you point at
via the *backend folder* field. Point it at a different box and copy the
generated `train.ps1` there — it's self-contained apart from the dataset dir.
