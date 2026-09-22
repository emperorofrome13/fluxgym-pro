"""
fluxgym-pro — dead simple LoRA training UI for Z-Image, Krea 2 and Qwen-Image-2.1.

Fork of cocktailpeanut/fluxgym with the sd-scripts backend swapped for:
  * musubi-tuner  -> Z-Image and Krea 2 LoRA training
  * ai-toolkit    -> Qwen-Image-2.1 LoRA training (day-0 upstream support)

Pure UI + process launcher: no torch import, so it starts on any machine.
The heavy lifting happens in the generated train.ps1 scripts.
"""

import os
import sys
import shutil
import subprocess

import gradio as gr
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(ROOT, "datasets")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
MODELS_DIR = os.path.join(ROOT, "models")
MODELS_YAML = os.path.join(ROOT, "models.yaml")
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

with open(MODELS_YAML, "r", encoding="utf-8") as f:
    MODELS = yaml.safe_load(f)
MODEL_NAMES = list(MODELS.keys())

RUNNING = {}       # output_name -> training Popen
CAPTION_JOBS = {}  # dataset name  -> captioning Popen

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")

VRAM_PROFILES = {
    "8 GB (slowest, CPU offload)": {"resolution": 768, "blocks_to_swap": 24,
                                  "cpu_offload": True, "low_vram": True},
    "16 GB (balanced)": {"resolution": 1024, "blocks_to_swap": 12,
                           "cpu_offload": False, "low_vram": True},
    "24 GB (fastest)": {"resolution": None, "blocks_to_swap": 0,
                          "cpu_offload": False, "low_vram": False},
}


def _profile(profile, arch, model_resolution):
    """Return conservative, backend-supported memory settings for a VRAM preset."""
    settings = dict(VRAM_PROFILES.get(profile, VRAM_PROFILES["16 GB (balanced)"]))
    settings["resolution"] = settings["resolution"] or model_resolution
    # Krea 2 reserves two of its 28 blocks; musubi rejects a larger value.
    if arch == "krea2":
        settings["blocks_to_swap"] = min(26, settings["blocks_to_swap"])
    return settings


def _safe_name(name, what="name"):
    name = (name or "").strip().strip('"')
    if not name:
        raise gr.Error(f"Give the {what}.")
    if '"' in name or "\\" in name or any(c in name for c in '<>|:*?'):
        raise gr.Error(f"{what} cannot contain quotes, backslashes or <>|:*? characters.")
    if name in (".", "..") or name.startswith("."):
        raise gr.Error(f"{what} cannot start with a dot.")
    return name


# ---------------------------------------------------------------- dataset ---
def dataset_names():
    if not os.path.isdir(DATASETS_DIR):
        return []
    return sorted(
        d for d in os.listdir(DATASETS_DIR)
        if os.path.isdir(os.path.join(DATASETS_DIR, d, "img"))
    )


def create_dataset(name, files, trigger):
    name = _safe_name(name, "dataset name")
    if not files:
        raise gr.Error("Add at least one image.")
    img_dir = os.path.join(DATASETS_DIR, name, "img")
    os.makedirs(img_dir, exist_ok=True)
    trigger = (trigger or "").strip()
    n = 0
    for src in files:
        path = src.name if hasattr(src, "name") else src  # gradio 4/5 compat
        base, _ = os.path.splitext(os.path.basename(path))
        dst = os.path.join(img_dir, os.path.basename(path))
        shutil.copy2(path, dst)
        cap = os.path.join(img_dir, base + ".txt")
        content = ""
        if os.path.exists(cap):
            with open(cap, "r", encoding="utf-8") as f:
                content = f.read().strip()
        if trigger and not content.startswith(trigger):
            content = f"{trigger} {content}".strip()
        with open(cap, "w", encoding="utf-8") as f:
            f.write(content)
        n += 1
    with open(os.path.join(DATASETS_DIR, name, "trigger.txt"), "w",
              encoding="utf-8") as f:
        f.write(trigger)
    return (f"Dataset '{name}' created with {n} images.",
            gr.update(choices=dataset_names(), value=name))


def dataset_trigger(name):
    path = os.path.join(DATASETS_DIR, name or "", "trigger.txt")
    if not os.path.abspath(path).startswith(os.path.abspath(DATASETS_DIR)
                                            + os.sep):
        return ""
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def auto_caption(dataset_ui, dataset_dd):
    """Run AI captioning (WD14 tagger) over a dataset in the background, CPU-only."""
    name = _safe_name((dataset_ui or "").strip() or (dataset_dd or "").strip(),
                      "dataset name")
    if not name:
        raise gr.Error("Create a dataset first, then run AI captions.")
    img_dir = os.path.join(DATASETS_DIR, name, "img")
    if not os.path.isdir(img_dir):
        raise gr.Error(f"Dataset '{name}' not found. Create it first.")
    if not any(f.lower().endswith(IMG_EXTS) for f in os.listdir(img_dir)):
        raise gr.Error(f"Dataset '{name}' has no images.")

    cap_py = os.path.join(ROOT, "caption.py")
    ps1 = os.path.join(DATASETS_DIR, name, "caption.ps1")
    log_path = os.path.join(DATASETS_DIR, name, "caption.log")
    py = sys.executable
    with open(ps1, "w", encoding="utf-8-sig") as f:
        f.write(
            "$ErrorActionPreference = 'Stop'\n"
            f"$py = \"{py}\"\n"
            "& $py -c \"import onnxruntime, huggingface_hub, numpy, PIL\" 2>$null\n"
            "if ($LASTEXITCODE -ne 0) { & $py -m pip install --quiet onnxruntime huggingface_hub numpy pillow }\n"
            f"& $py \"{cap_py}\" \"{img_dir}\" \"{dataset_trigger(name)}\"\n"
            "if ($LASTEXITCODE -ne 0) { throw \"AI captioning failed - see caption.log\" }\n"
        )
    log = open(log_path, "w", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps1],
        cwd=DATASETS_DIR, stdout=log, stderr=subprocess.STDOUT)
    CAPTION_JOBS[name] = proc
    return (f"AI captioning started for '{name}' in the background (CPU-only, "
            f"first run downloads the WD14 tagger model). "
            f"Existing captions are kept. Log: {log_path}")


def check_caption(dataset_ui, dataset_dd):
    name = (dataset_ui or "").strip() or (dataset_dd or "").strip()
    if not name:
        return "(no dataset selected)"
    p = CAPTION_JOBS.get(name)
    if p is not None and p.poll() is None:
        return f"AI captioning for '{name}' is RUNNING (pid {p.pid})."
    if p is not None:
        return (f"AI captioning for '{name}' FINISHED with exit code {p.returncode}."
                if p.returncode == 0 else
                f"AI captioning for '{name}' FAILED (exit code {p.returncode}) - see caption.log.")
    return f"'{name}' was not captioned in this session. Click 'AI captions' to start."


# ------------------------------------------------------------------ config ---
def _n_images(train_dir):
    img_dir = os.path.join(train_dir, "img")
    if not os.path.isdir(img_dir):
        return 0
    return len([f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS)])


def _resolve_weights(m, backend):
    """Weight paths resolve to models/<repo>/..., then the backend folder; absolute wins."""
    repo_dir = os.path.join(MODELS_DIR, (m.get("repo") or "").replace("/", "_"))
    out = {}
    for key in ("dit_path", "vae_path", "text_encoder_path"):
        val = (m.get(key) or "").strip()
        if not val:
            continue
        if os.path.isabs(val):
            out[key] = os.path.abspath(val)
            continue
        for base in (repo_dir, backend):
            cand = os.path.abspath(os.path.join(base, val))
            if os.path.exists(cand):
                out[key] = cand
                break
        else:
            out[key] = os.path.abspath(os.path.join(repo_dir, val))
    return out


def _pick_python(backend):
    for cand in (".venv", "venv"):
        for sub in ("Scripts", "bin"):
            exe = os.path.join(backend, cand, sub,
                               "python.exe" if os.name == "nt" else "python")
            if os.path.isfile(exe):
                return exe
    return sys.executable


def build_cfg(base_model, dataset_name, output_name, backend_dir,
              learning_rate, network_dim, network_alpha, resolution,
              epochs, batch_size, repeats, save_every_n_epochs, mixed_precision,
              vram_profile, quantization, blocks_to_swap, cpu_offload,
              gradient_accumulation, optimizer, full_bf16,
              lr_scheduler, lr_warmup_steps, max_grad_norm, optimizer_args,
              loader_workers, persistent_workers, save_precision, save_state,
              mem_eff_save, timestep_sampling, flow_shift, guidance_scale,
              cache_batch_size, seed,
              samples_enabled, sample_prompts, sample_every):
    if base_model not in MODELS:
        raise gr.Error("Pick a base model.")
    output_name = _safe_name(output_name, "LoRA name")
    train_dir = os.path.join(DATASETS_DIR, dataset_name or "")
    if _n_images(train_dir) == 0:
        raise gr.Error(f"Dataset '{dataset_name}' has no images. Create one first.")

    m = MODELS[base_model]
    arch = m["arch"]
    backend = os.path.abspath((backend_dir or "").strip() or m["backend_dir"])
    if not os.path.isdir(backend):
        raise gr.Error(f"Backend folder not found: {backend}\n"
                       f"Clone it there (see README) or fix the path.")

    out_dir = os.path.join(OUTPUTS_DIR, output_name)
    os.makedirs(out_dir, exist_ok=True)
    profile = _profile(vram_profile, arch, m.get("default_resolution", 1024))
    prompts = ([line.strip() for line in (sample_prompts or "").splitlines()
                if line.strip()] if samples_enabled else [])
    quantization = quantization or "Automatic (recommended)"
    if quantization == "ConvRot INT8 (Krea 2 only)" and arch != "krea2":
        raise gr.Error("ConvRot INT8 is supported only by Krea 2 in musubi-tuner.")
    if quantization == "FP8 DiT + text encoder (Z-Image)" and arch != "zimage":
        raise gr.Error("FP8 text-encoder mode is supported only by Z-Image in this app.")
    if arch == "qwen_image_2" and quantization not in (
            "Automatic (recommended)", "None (largest VRAM use)"):
        raise gr.Error("Qwen-Image-2.1 uses ai-toolkit quantization; choose Automatic or None.")
    if full_bf16 and arch == "qwen_image_2":
        raise gr.Error("Full BF16 base weights are a musubi-tuner option, not an ai-toolkit option.")
    if full_bf16 and mixed_precision != "bf16":
        raise gr.Error("Full BF16 base weights require BF16 mixed precision.")
    if (full_bf16 or (optimizer == "adafactor" and int(blocks_to_swap or 0) > 0)) \
            and int(gradient_accumulation or 1) > 1:
        raise gr.Error("Fused backward pass does not support gradient accumulation. Set it to 1.")

    fp8_base = quantization in ("Automatic (recommended)", "FP8 DiT (scaled)",
                                 "FP8 DiT + text encoder (Z-Image)") and bool(m.get("fp8_base", False))
    if quantization == "FP8 DiT (scaled)":
        fp8_base = arch in ("zimage", "krea2")
    fp8_llm = quantization == "FP8 DiT + text encoder (Z-Image)"
    convrot_int8 = quantization == "ConvRot INT8 (Krea 2 only)"
    if convrot_int8:
        fp8_base = False

    cfg = {
        "arch": arch,
        "backend_dir": backend,
        "python_exe": _pick_python(backend),
        "train_data_dir": train_dir,
        "toml_path": os.path.join(out_dir, "dataset.toml"),
        "output_dir": out_dir,
        "output_name": output_name,
        "repo": m.get("repo", ""),
        "trigger": dataset_trigger(dataset_name),
        "resolution": int(resolution or profile["resolution"]),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "save_every_n_epochs": int(save_every_n_epochs),
        "learning_rate": float(learning_rate),
        "network_dim": int(network_dim),
        "network_alpha": int(network_alpha),
        "mixed_precision": mixed_precision or "bf16",
        "fp8_base": fp8_base,
        "fp8_llm": fp8_llm,
        "convrot_int8": convrot_int8,
        "blocks_to_swap": max(0, int(blocks_to_swap if blocks_to_swap is not None else profile["blocks_to_swap"])),
        "gradient_checkpointing_cpu_offload": bool(cpu_offload or profile["cpu_offload"]),
        "gradient_accumulation_steps": max(1, int(gradient_accumulation or 1)),
        "optimizer": "adafactor" if full_bf16 else (optimizer or "adamw8bit"),
        "full_bf16": bool(full_bf16),
        "lr_scheduler": lr_scheduler or "constant",
        "lr_warmup_steps": max(0, int(lr_warmup_steps or 0)),
        "max_grad_norm": max(0, float(max_grad_norm or 0)),
        "optimizer_args": [part.strip() for part in (optimizer_args or "").split("|")
                           if part.strip()],
        "loader_workers": max(0, int(loader_workers or 0)),
        "persistent_workers": bool(persistent_workers),
        "save_precision": save_precision or "bf16",
        "save_state": bool(save_state),
        "mem_eff_save": bool(mem_eff_save),
        "timestep_sampling": timestep_sampling or "Model default (recommended)",
        "flow_shift": float(flow_shift or 0),
        "guidance_scale": float(guidance_scale or 0),
        "cache_batch_size": max(1, int(cache_batch_size or 1)),
        "low_vram": profile["low_vram"],
        "quantize": quantization != "None (largest VRAM use)",
        "quantize_te": quantization != "None (largest VRAM use)",
        "repeats": max(1, int(repeats or m.get("num_repeats", 1))),
        "seed": int(seed if seed is not None else m.get("seed", 0)),
        "sample_prompts": prompts,
        "sample_every": int(sample_every or 0),
    }
    cfg.update(_resolve_weights(m, backend))

    if arch == "qwen_image_2":
        n = _n_images(train_dir)
        bs = max(1, cfg["batch_size"])
        cfg["steps"] = max(1, (n * cfg["repeats"] * cfg["epochs"]) // bs)
    return cfg


# ---------------------------------------------------------------- training ---
def preview_commands(base_model, dataset_name, output_name, backend_dir,
                     learning_rate, network_dim, network_alpha, resolution,
                     epochs, batch_size, repeats, save_every_n_epochs, mixed_precision,
                     vram_profile, quantization, blocks_to_swap, cpu_offload,
                     gradient_accumulation, optimizer, full_bf16,
                     lr_scheduler, lr_warmup_steps, max_grad_norm, optimizer_args,
                     loader_workers, persistent_workers, save_precision, save_state,
                     mem_eff_save, timestep_sampling, flow_shift, guidance_scale,
                     cache_batch_size, seed,
                     samples_enabled, sample_prompts, sample_every):
    args = [base_model, dataset_name, output_name, backend_dir,
            learning_rate, network_dim, network_alpha, resolution,
            epochs, batch_size, repeats, save_every_n_epochs, mixed_precision, vram_profile,
            quantization, blocks_to_swap, cpu_offload, gradient_accumulation,
            optimizer, full_bf16, lr_scheduler, lr_warmup_steps, max_grad_norm,
            optimizer_args, loader_workers, persistent_workers, save_precision,
            save_state, mem_eff_save, timestep_sampling, flow_shift, guidance_scale,
            cache_batch_size, seed, samples_enabled, sample_prompts, sample_every]
    try:
        cfg = build_cfg(*args)
    except gr.Error as e:
        return f"Cannot preview yet: {e}"
    from command_gen import full_script
    try:
        return full_script(cfg)
    except (RuntimeError, OSError) as e:
        return f"Cannot preview yet: {e}"


def start_training(base_model, dataset_name, output_name, backend_dir,
                   learning_rate, network_dim, network_alpha, resolution,
                   epochs, batch_size, repeats, save_every_n_epochs, mixed_precision,
                   vram_profile, quantization, blocks_to_swap, cpu_offload,
                   gradient_accumulation, optimizer, full_bf16,
                   lr_scheduler, lr_warmup_steps, max_grad_norm, optimizer_args,
                   loader_workers, persistent_workers, save_precision, save_state,
                   mem_eff_save, timestep_sampling, flow_shift, guidance_scale,
                   cache_batch_size, seed,
                   samples_enabled, sample_prompts, sample_every):
    args = [base_model, dataset_name, output_name, backend_dir,
            learning_rate, network_dim, network_alpha, resolution,
            epochs, batch_size, repeats, save_every_n_epochs, mixed_precision, vram_profile,
            quantization, blocks_to_swap, cpu_offload, gradient_accumulation,
            optimizer, full_bf16, lr_scheduler, lr_warmup_steps, max_grad_norm,
            optimizer_args, loader_workers, persistent_workers, save_precision,
            save_state, mem_eff_save, timestep_sampling, flow_shift, guidance_scale,
            cache_batch_size, seed, samples_enabled, sample_prompts, sample_every]
    cfg = build_cfg(*args)

    from command_gen import full_script, dataset_toml
    if cfg["arch"] != "qwen_image_2":
        with open(cfg["toml_path"], "w", encoding="utf-8") as f:
            f.write(dataset_toml(cfg["train_data_dir"], cfg["resolution"],
                                 cfg["batch_size"], cfg["repeats"]))

    try:
        script = full_script(cfg)
    except (RuntimeError, OSError) as e:
        raise gr.Error(str(e))

    ps1 = os.path.join(cfg["output_dir"], "train.ps1")
    with open(ps1, "w", encoding="utf-8-sig") as f:
        f.write(script)

    log = open(os.path.join(cfg["output_dir"], "train.log"),
               "w", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps1],
        cwd=cfg["output_dir"], stdout=log, stderr=subprocess.STDOUT)
    RUNNING[cfg["output_name"]] = proc

    return (gr.update(choices=run_names(), value=cfg["output_name"]),
            f"Training started. Log: {os.path.join(cfg['output_dir'], 'train.log')}\n\n"
            f"Generated commands:\n\n{script}")


# ------------------------------------------------------------------ status ---
def run_names():
    if not os.path.isdir(OUTPUTS_DIR):
        return []
    return sorted(d for d in os.listdir(OUTPUTS_DIR)
                  if os.path.isdir(os.path.join(OUTPUTS_DIR, d)))


def check_status(output_name):
    p = RUNNING.get(output_name)
    if p is None:
        return f"'{output_name}' was not started in this session (or already cleared)."
    if p.poll() is None:
        return f"RUNNING (pid {p.pid})"
    return f"FINISHED with exit code {p.returncode}"


def read_log(output_name):
    try:
        name = _safe_name(output_name, "run name")
    except gr.Error:
        return "(no log yet)"
    path = os.path.join(OUTPUTS_DIR, name, "train.log")
    if not os.path.isfile(path):
        return "(no log yet)"
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()[-8000:]


# --------------------------------------------------------------------- UI ---
def on_model_change(name):
    m = MODELS.get(name, {})
    return (m.get("default_lr", 1e-4), m.get("default_dim", 16),
            m.get("default_alpha", 16), m.get("default_resolution", 1024),
            m.get("backend_dir", ""), m.get("num_repeats", 1))


def on_vram_profile_change(profile, base_model):
    """Make the selected preset visible and editable before training starts."""
    m = MODELS.get(base_model, {})
    settings = _profile(profile, m.get("arch", "zimage"),
                        m.get("default_resolution", 1024))
    return (gr.update(value=settings["resolution"]),
            gr.update(value=settings["blocks_to_swap"]),
            gr.update(value=settings["cpu_offload"]))


def sample_controls(enabled):
    return (gr.update(visible=enabled), gr.update(visible=enabled))


with gr.Blocks(title="fluxgym-pro") as app:
    gr.Markdown(
        "# fluxgym-pro v1.04\n"
        "Dead simple LoRA training for **Z-Image**, **Krea 2** and "
        "**Qwen-Image-2.1** — a fork of fluxgym.\n\n"
        "1. Create a dataset  2. (optional) AI captions  3. Pick a base model  "
        "4. Preview  5. Start"
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("## 1. Dataset")
            ds_name = gr.Textbox(label="Dataset name", placeholder="my_lora")
            ds_upload = gr.File(label="Images", file_count="multiple",
                                file_types=["image"])
            ds_trigger = gr.Textbox(label="Trigger word (prepended to captions)",
                                    placeholder="ohwx woman")
            with gr.Row():
                ds_btn = gr.Button("Create dataset", variant="primary")
                ds_caption_btn = gr.Button("AI captions (WD14 tagger, CPU)")
                ds_caption_check = gr.Button("Check captioning")
            ds_status = gr.Markdown()
            ds_dropdown = gr.Dropdown(label="Dataset", choices=dataset_names())

            gr.Markdown("## Sample images")
            samples_enabled = gr.Checkbox(
                label="Generate sample images during training", value=False,
                info="Turn this on only when you want progress images; it slows training.")
            sample_prompts = gr.Textbox(
                label="Sample prompts (one per line)",
                placeholder="a photo of my subject at the beach\n"
                            "my subject holding a coffee cup",
                lines=3, visible=False)
            sample_every = gr.Number(label="Generate samples every N epochs",
                                     value=1, precision=0, minimum=1, visible=False)

        with gr.Column():
            gr.Markdown("## 2. Training")
            base_model = gr.Dropdown(label="Base model", choices=MODEL_NAMES,
                                     value=MODEL_NAMES[0] if MODEL_NAMES else None)
            backend_dir = gr.Textbox(label="Backend folder (musubi-tuner / ai-toolkit)",
                                     value=MODELS[MODEL_NAMES[0]]["backend_dir"] if MODEL_NAMES else "")
            output_name = gr.Textbox(label="LoRA name", placeholder="my_lora_v1")
            with gr.Row():
                learning_rate = gr.Number(
                    label="Learning rate",
                    value=MODELS[MODEL_NAMES[0]]["default_lr"] if MODEL_NAMES else 1e-4)
                network_dim = gr.Number(
                    label="LoRA dim (rank)",
                    value=MODELS[MODEL_NAMES[0]]["default_dim"] if MODEL_NAMES else 16,
                    precision=0)
                network_alpha = gr.Number(
                    label="LoRA alpha",
                    value=MODELS[MODEL_NAMES[0]]["default_alpha"] if MODEL_NAMES else 16,
                    precision=0)
            with gr.Row():
                resolution = gr.Number(
                    label="Resolution",
                    value=MODELS[MODEL_NAMES[0]]["default_resolution"] if MODEL_NAMES else 1024,
                    precision=0)
                epochs = gr.Number(label="Epochs", value=10, precision=0)
                batch_size = gr.Number(label="Batch size", value=1, precision=0)
                repeats = gr.Number(label="Repeats per image", value=1, precision=0,
                                    minimum=1,
                                    info="How many times each image appears per epoch.")
                save_every_n_epochs = gr.Number(label="Save every N epochs",
                                                value=2, precision=0)
            mixed_precision = gr.Dropdown(label="Mixed precision",
                                          choices=["bf16", "fp16"], value="bf16")
            vram_profile = gr.Radio(
                label="GPU VRAM profile",
                choices=list(VRAM_PROFILES), value="16 GB (balanced)",
                info="8 GB uses CPU block offload and is much slower; batch size stays 1.")
            with gr.Accordion("Advanced training options", open=False):
                gr.Markdown(
                    "These options are forwarded to the trainer. Sections labelled **musubi only** "
                    "apply to Z-Image and Krea 2; Qwen-Image-2.1 has the options ai-toolkit supports.")
                with gr.Accordion("Memory and quantization", open=True):
                    quantization = gr.Dropdown(
                        label="Frozen base-weight quantization",
                        choices=["Automatic (recommended)", "FP8 DiT (scaled)",
                                 "FP8 DiT + text encoder (Z-Image)",
                                 "ConvRot INT8 (Krea 2 only)", "None (largest VRAM use)"],
                        value="Automatic (recommended)")
                    with gr.Row():
                        blocks_to_swap = gr.Number(
                            label="DiT blocks swapped to CPU (profile default shown)",
                            value=12, precision=0, minimum=0,
                            info="Higher saves VRAM but slows every training step.")
                        gradient_accumulation = gr.Number(
                            label="Gradient accumulation steps", value=1, precision=0, minimum=1)
                    cpu_offload = gr.Checkbox(
                        label="CPU offload during gradient checkpointing", value=False)
                    full_bf16 = gr.Checkbox(
                        label="Full BF16 base weights (musubi only; lower VRAM, experimental)",
                        value=False,
                        info="This uses Adafactor plus fused backward pass automatically. It is not FP8.")
                with gr.Accordion("Optimizer and learning-rate schedule (musubi only)", open=False):
                    with gr.Row():
                        optimizer = gr.Dropdown(label="Optimizer", choices=["adamw8bit", "adafactor"],
                                                value="adamw8bit")
                        lr_scheduler = gr.Dropdown(
                            label="Learning-rate scheduler",
                            choices=["constant", "constant_with_warmup", "linear", "cosine"],
                            value="constant")
                    with gr.Row():
                        lr_warmup_steps = gr.Number(label="Warmup steps", value=0, precision=0, minimum=0)
                        max_grad_norm = gr.Number(
                            label="Maximum gradient norm (0 disables clipping)", value=1.0, minimum=0)
                    optimizer_args = gr.Textbox(
                        label="Optimizer arguments (pipe-separated)",
                        placeholder="relative_step=False | scale_parameter=False",
                        info="Advanced example for Adafactor. Leave blank for the optimizer default.")
                with gr.Accordion("Data loading and caching", open=False):
                    with gr.Row():
                        cache_batch_size = gr.Number(
                            label="Text-encoder cache batch size (musubi)", value=16,
                            precision=0, minimum=1)
                        loader_workers = gr.Number(
                            label="Data-loader worker processes (musubi)", value=2,
                            precision=0, minimum=0)
                    persistent_workers = gr.Checkbox(
                        label="Keep data-loader workers alive (musubi)", value=True,
                        info="Disable this if a Windows worker process hangs between epochs.")
                with gr.Accordion("Flow matching and sampling (musubi only)", open=False):
                    with gr.Row():
                        timestep_sampling = gr.Dropdown(
                            label="Timestep sampling",
                            choices=["Model default (recommended)", "shift", "krea2_shift"],
                            value="Model default (recommended)")
                        flow_shift = gr.Number(
                            label="Discrete flow shift (0 = model default)", value=0, minimum=0)
                    guidance_scale = gr.Number(
                        label="Training guidance scale (0 = model default)", value=0, minimum=0)
                with gr.Accordion("Saving and reproducibility", open=False):
                    with gr.Row():
                        save_precision = gr.Dropdown(
                            label="LoRA checkpoint precision (musubi)",
                            choices=["bf16", "fp16", "float"], value="bf16")
                        save_state = gr.Checkbox(
                            label="Save optimizer state for resume (musubi)", value=True)
                    with gr.Row():
                        mem_eff_save = gr.Checkbox(
                            label="Use memory-efficient checkpoint saving (musubi)", value=False)
                        flow_seed = gr.Number(label="Training seed", value=0, precision=0)
                    # Kept as a named component so the UI can expose the same seed users
                    # expect from FluxGym; build_cfg reads this value through seed below.
                    seed = flow_seed
            with gr.Row():
                btn_preview = gr.Button("Preview commands")
                btn_start = gr.Button("Start training", variant="primary")

    gr.Markdown("## Commands")
    cmd_out = gr.Code(label="train.ps1", lines=24)

    gr.Markdown("## Status")
    with gr.Row():
        st_name = gr.Dropdown(label="Run", choices=run_names())
        st_refresh = gr.Button("Refresh runs")
        st_check = gr.Button("Check status")
    st_status = gr.Textbox(label="Status", interactive=False)
    st_log = gr.Textbox(label="Log (tail)", lines=14, interactive=False)

    # ---- wiring
    UI_INPUTS = [base_model, ds_dropdown, output_name, backend_dir,
                 learning_rate, network_dim, network_alpha, resolution,
                 epochs, batch_size, repeats, save_every_n_epochs, mixed_precision, vram_profile,
                 quantization, blocks_to_swap, cpu_offload, gradient_accumulation,
                 optimizer, full_bf16, lr_scheduler, lr_warmup_steps, max_grad_norm,
                 optimizer_args, loader_workers, persistent_workers, save_precision,
                 save_state, mem_eff_save, timestep_sampling, flow_shift, guidance_scale,
                 cache_batch_size, seed, samples_enabled, sample_prompts, sample_every]

    ds_btn.click(create_dataset, [ds_name, ds_upload, ds_trigger],
                 [ds_status, ds_dropdown])
    ds_caption_btn.click(auto_caption, [ds_name, ds_dropdown], [ds_status])
    ds_caption_check.click(check_caption, [ds_name, ds_dropdown], [ds_status])

    base_model.change(on_model_change, [base_model],
                      [learning_rate, network_dim, network_alpha,
                       resolution, backend_dir, repeats])
    vram_profile.change(on_vram_profile_change, [vram_profile, base_model],
                        [resolution, blocks_to_swap, cpu_offload])
    samples_enabled.change(sample_controls, [samples_enabled],
                           [sample_prompts, sample_every])

    btn_preview.click(preview_commands, UI_INPUTS, [cmd_out])
    btn_start.click(start_training, UI_INPUTS, [st_name, cmd_out])

    st_refresh.click(lambda: gr.update(choices=run_names()), None, [st_name])
    st_check.click(check_status, [st_name], [st_status])
    st_name.change(read_log, [st_name], [st_log])


if __name__ == "__main__":
    app.queue().launch(server_name="127.0.0.1",
                       server_port=int(os.environ.get("FGP_PORT", "7860")),
                       inbrowser=os.environ.get("FGP_NO_BROWSER") != "1")
