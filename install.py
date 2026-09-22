# fluxgym-pro one-time setup: backends + model checkpoints.
# Run via install.bat (creates the UI venv, then runs this).
# Answers:
#   1) Backends       - clones musubi-tuner + ai-toolkit next to this repo (+ their venvs)
#   2) Z-Image        - ~19.6 GB public weights into models/Tongyi-MAI_Z-Image
#   3) Krea 2         - ~25 GB gated DiT (needs your HF token) + VAE + text encoder
#   4) Qwen pre-seed  - optional; ai-toolkit downloads its own copy on first training
#
# Environment: HF_TOKEN may be set instead of typing the token. Nothing is saved to disk.

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(ROOT)            # backend folders live next to this repo
MODELS = os.path.join(ROOT, "models")
PYTHON = sys.executable                    # we always run inside the repo's venv

# Default PyTorch CUDA wheel index (matches the proven RTX 40/50-series setup).
# On older cards, swap to cu124/cu121 from https://pytorch.org/get-started/locally/.
CUDA_INDEX = "https://download.pytorch.org/whl/cu128"

BACKENDS = {
    "musubi-tuner": {
        "url": "https://github.com/kohya-ss/musubi-tuner.git",
        "recurse": False,
        "install": [["-m", "pip", "install", "-e", "."]],
    },
    "ai-toolkit": {
        "url": "https://github.com/ostris/ai-toolkit.git",
        "recurse": True,
        "install": [["-m", "pip", "install", "-r", "requirements.txt"]],
    },
}

KREA_DIT = ("krea/Krea-2-Raw", "raw.safetensors", "raw.safetensors", "Krea 2 RAW DiT (gated, ~25 GB)")
KREA_VAE = ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors",
            "qwen_image_vae.safetensors", "Qwen-Image VAE (~242 MB)")
KREA_TE = ("Comfy-Org/Qwen3-VL", "text_encoders/qwen3vl_4b_bf16.safetensors",
           "qwen3vl_4b_bf16.safetensors", "Qwen3-VL text encoder (~8.9 GB)")


def say(msg):
    print(msg, flush=True)


def ask(question, default=True):
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        ans = input(f"{question} {suffix}: ").strip().lower()
        if ans == "":
            return default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("  please answer y or n")


def run(cmd, cwd=None):
    say("")
    say(">> " + " ".join(str(c) for c in cmd))
    code = subprocess.run(cmd, cwd=cwd).returncode
    if code != 0:
        raise SystemExit(f"ERROR: command failed with exit code {code}\n    {' '.join(str(c) for c in cmd)}")
    return True


def venv_py(name):
    return os.path.join(PARENT, name, ".venv", "Scripts", "python.exe")


def have_file(path, min_mb=1):
    return os.path.isfile(path) and os.path.getsize(path) > min_mb * 1024 * 1024


# ------------------------------------------------------------------ backends ---
def install_backends():
    say("")
    say("=" * 68)
    say(f"Backends will be cloned into: {PARENT}")
    say("(Same place the UI's default 'Backend folder' field looks: ../musubi-tuner)")
    say("=" * 68)
    if not ask("Install the two training backends (musubi-tuner, ai-toolkit)?"):
        say("  skipped backends. You can install them later (README, or run install.bat again).")
        return
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        raise SystemExit("ERROR: git not found on PATH. Install git from https://git-scm.com and rerun.")
    for name, meta in BACKENDS.items():
        dst = os.path.join(PARENT, name)
        py = venv_py(name)
        if os.path.isdir(dst) and (os.path.isdir(os.path.join(dst, ".git")) or os.path.isfile(py)):
            say(f"  [{name}] already present at {dst} - skipping.")
            continue
        say(f"  [{name}] cloning + creating venv (large install, several minutes)...")
        cmd = ["git", "clone"] + (["--recurse-submodules"] if meta["recurse"] else []) + [meta["url"], dst]
        run(cmd)
        run([PYTHON, "-m", "venv", os.path.join(dst, ".venv")])
        run([py, "-m", "pip", "install", "--upgrade", "pip"])
        run([py, "-m", "pip", "install", "torch", "torchvision", "--index-url", CUDA_INDEX])
        for inst in meta["install"]:
            run([py] + inst, cwd=dst)
        say(f"  [{name}] installed. torch index: {CUDA_INDEX}")
    say("")
    say("Backends ok. If your GPU needs a different CUDA wheel, reinstall torch inside the")
    say("backend venv with the index shown at https://pytorch.org/get-started/locally/.")


# ----------------------------------------------------------------- checkpoints ---
def install_zimage():
    say("")
    say("=" * 68)
    say("Z-Image (Tongyi-MAI, 6B): transformer + VAE + text encoder, ~19.6 GB, public.")
    say("Destination: models/Tongyi-MAI_Z-Image/")
    say("=" * 68)
    if not ask("Download Z-Image checkpoints?"):
        say("  skipped Z-Image.")
        return
    target = os.path.join(MODELS, "Tongyi-MAI_Z-Image")
    if os.path.isfile(os.path.join(target, "vae", "diffusion_pytorch_model.safetensors")) and \
            os.path.isfile(os.path.join(target, "transformer",
                                        "diffusion_pytorch_model-00001-of-00002.safetensors")):
        say("  Z-Image weights already present - skipping.")
        return
    from huggingface_hub import snapshot_download
    say("  downloading Tongyi-MAI/Z-Image ...")
    snapshot_download("Tongyi-MAI/Z-Image", local_dir=target)
    say(f"  done -> {target}")


def install_krea(token):
    say("")
    say("=" * 68)
    say("Krea 2: gated DiT (krea/Krea-2-Raw, ~25 GB, REQUIRES your HF token + repo access)")
    say("plus public VAE + text encoder. Destination: models/krea_Krea-2-Raw/")
    say("=" * 68)
    if not ask("Download Krea 2 checkpoints?"):
        say("  skipped Krea 2.")
        return
    if not token:
        token = ask_token()
    target = os.path.join(MODELS, "krea_Krea-2-Raw")
    os.makedirs(target, exist_ok=True)
    from huggingface_hub import hf_hub_download
    files = [KREA_DIT, KREA_VAE, KREA_TE]
    got = 0
    for repo, src, dest, label in files:
        if have_file(os.path.join(target, dest)):
            say(f"  [{dest}] already present - skipping.")
            got += 1
            continue
        say(f"  downloading {label} from {repo} ...")
        try:
            hf_hub_download(repo, src, local_dir=target, token=token or None)
        except Exception as exc:                      # gated / bad token -> hint
            raise SystemExit(
                f"ERROR downloading {label}: {exc}\n\n"
                f"If this is an auth/gating error for 'krea/Krea-2-Raw':\n"
                f"  1. Open https://huggingface.co/krea/Krea-2-Raw and click 'Agree and access repository'\n"
                f"  2. Create an access token at https://huggingface.co/settings/tokens (read permission)\n"
                f"  3. Paste it when asked (or set the HF_TOKEN environment variable)\n"
                f"  4. You may re-run this installer - it resumes finished downloads.")
        got += 1
    say(f"  Krea 2 weight downloads finished. Already/present: {got}/{len(files)}")


def ask_token():
    if os.environ.get("HF_TOKEN"):
        say("  using HF_TOKEN from environment.")
        return os.environ["HF_TOKEN"].strip()
    while True:
        tok = input("Paste your Hugging Face access token (read): ").strip()
        if tok:
            return tok
        print("  (empty is not accepted - re-run and answer 'n' to skip Krea 2 if you lack access)")


# -------------------------------------------------------------------------- qwen ---
def install_qwen():
    say("")
    say("=" * 68)
    say("Qwen-Image-2.1: ai-toolkit downloads its OWN copy (transformer + TE + VAE) on")
    say("the very first training run. Nothing is needed in models/ - this step only")
    say("pre-seeds the Hugging Face cache so the first run starts offline/faster (~14 GB).")
    say("=" * 68)
    if not ask("Pre-seed Qwen-Image-2.1 into the HF cache?", default=False):
        say("  skipped. First training will download it automatically.")
        return
    from huggingface_hub import snapshot_download
    say("  pre-seeding Qwen/Qwen-Image-2.1 ...")
    snapshot_download("Qwen/Qwen-Image-2.1")
    say("  done. ai-toolkit will reuse this cached copy.")


# ---------------------------------------------------------------------------- main ---
def main():
    say("")
    say("fluxgym-pro installer")
    say("=====================")
    say(f"repo (this folder): {ROOT}")
    say(f"models dir        : {MODELS}")
    say("")
    say("Everything is optional; answer n to skip anything. Re-running continues/resumes.")
    say("For Krea 2 you need a Hugging Face token and access to the gated repo.")

    dry = "--dry-run" in sys.argv
    if dry:
        say("  *** DRY RUN: no files are downloaded or cloned. ***")

    # machine preconditions
    if not dry:
        import shutil
        for line in ("git", "curl"):
            if shutil.which(line) is None:
                say(f"  note: '{line}' not found on PATH")
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        if not dry:
            raise SystemExit("ERROR: huggingface_hub missing. Run install.bat (it installs requirements.txt).")

    if dry:
        say("  (dry-run) would install backends + run all yes-branches")
        return

    install_backends()
    install_zimage()
    install_krea(os.environ.get("HF_TOKEN", "").strip())
    install_qwen()

    say("")
    say("=" * 68)
    say("SETUP COMPLETE")
    say("=" * 68)
    say("Next:")
    say(f"  1. Run quickstart.bat  (starts the UI at http://127.0.0.1:7860)") 
    say("  2. Backend folder field keeps its default ../musubi-tuner (or type an absolute path)")
    say("  3. Create a dataset with at least one image AND a .txt caption per image,")
    say("     then Train. Qwen-Image-2.1 self-downloads its weights on the first run.")
    say("")
    say("Sizes are large: Z-Image ~19.6 GB, Krea 2 DiT ~25 GB + VAE/TE, Qwen pre-seed ~14 GB.")
    say("Spare disk before installing. HF downloads resume after a disconnect.")
    say("")


if __name__ == "__main__":
    main()