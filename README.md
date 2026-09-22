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
install.bat    # one-time: backends + model checkpoints (asks yes/no + HF token for Krea 2)
quickstart.bat # every time: starts the UI at http://127.0.0.1:7860
```

`install.bat` creates the UI venv, installs UI deps, then walks you through the
big one-time downloads. It asks yes/no for each component, and asks for your
Hugging Face token only when you choose **Krea 2** (its DiT repo is gated):

| Prompt | What it installs | Size |
|---|---|---|
| Backends | `musubi-tuner` + `ai-toolkit`, cloned next to this folder, with their own venvs + CUDA torch | hours of pip, once |
| Z-Image | transformer + VAE + text encoder into `models/Tongyi-MAI_Z-Image` | ~19.6 GB |
| Krea 2 | gated DiT + Qwen VAE + Qwen3-VL text encoder into `models/krea_Krea-2-Raw` | ~34 GB |
| Qwen pre-seed *(optional)* | warms the Hugging Face cache; ai-toolkit self-downloads on first training anyway | ~14 GB |

Every step is skippable and re-runnable (downloads resume). The token is used in
memory only — nothing is saved to disk. Krea 2 also requires accepting the repo
terms at https://huggingface.co/krea/Krea-2-Raw before the download works.

Manual equivalent:

```powershell
pip install -r requirements.txt
python app.py
```

## Backends: one-time setup

The easy path is `install.bat` above. Doing it manually — clone the backends next
to this folder (or anywhere — the path is a UI field):

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
2. **Captions** — every image needs a `<image>.txt` caption or the trainers
   silently skip it. Click *AI captions* to auto-tag the whole folder with a
   WD14 tagger (CPU, downloads the model once). Add a trigger word and it gets
   prepended to each caption.
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

**Full BF16 base weights** and **memory-efficient checkpoint saves** were removed
with musubi-tuner 0.3.5 (its consolidated `*_train_network.py` scripts no longer
accept those flags) — the app refuses them. Use the default scaled FP8 base
weights, which is the recommended low-VRAM path for Z-Image and Krea 2.

Everything (dataset TOML, YAML, script, log, checkpoints) is written to
`outputs/<your lora name>/`, so a run is fully reproducible.

## Adding models

Edit `models.yaml`. One entry = one dropdown option. `arch` must be one of
`zimage | krea2 | qwen_image_2` (see `command_gen.py` for the script mapping).

## GPU note

The UI never touches your GPU. Training runs in the backend venv you point at
via the *backend folder* field. Point it at a different box and copy the
generated `train.ps1` there — it's self-contained apart from the dataset dir.
