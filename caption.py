"""
fluxgym-pro AI captioning (WD14 tagger, same as upstream fluxgym's WD14Captioner).

Runs on CPU via onnxruntime, so captions can be made while the GPU is busy.
CLI mode:  python caption.py <img_dir> [trigger_word]
"""

import os

MODEL_REPO = "SmilingWolf/wd-swinv2-tagger-v3"
INPUT_SIZE = 448
DEFAULT_THRESHOLD = 0.35
# tags that are useless in a LoRA caption (quality boilerplate)
BLACKLIST = {"general", "sensitive", "questionable", "explicit", "commentary", "masterpiece", "high score", "absurdres", "highres"}

# per-process caches: building an InferenceSession is expensive, so reuse one
_SESSIONS = {}
_LABELS = {}


def _get_session(model_path):
    sess = _SESSIONS.get(model_path)
    if sess is None:
        import onnxruntime as ort
        sess = ort.InferenceSession(model_path,
                                    providers=["CPUExecutionProvider"])
        _SESSIONS[model_path] = sess
    return sess


def _get_labels(csv_path):
    labels = _LABELS.get(csv_path)
    if labels is None:
        labels = _load_labels(csv_path)
        _LABELS[csv_path] = labels
    return labels


def _download(model_repo):
    from huggingface_hub import hf_hub_download
    model = hf_hub_download(model_repo, "model.onnx")
    csv = hf_hub_download(model_repo, "selected_tags.csv")
    return model, csv


def _load_labels(csv_path):
    import csv as _csv
    labels = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            labels.append((row["name"], int(row["category"])))
    return labels


def _predict(model_path, image_path):
    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    img = Image.open(image_path).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(bg, img).convert("RGB")
    # pad to square then resize (same as most WD14 front-ends)
    w, h = img.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), (255, 255, 255))
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    canvas = canvas.resize((INPUT_SIZE, INPUT_SIZE), Image.LANCZOS)

    x = np.asarray(canvas, dtype=np.float32)  # HWC
    x = x[:, :, ::-1]  # RGB -> BGR (WD14 convention)
    x = x[None, :, :, :]

    sess = _get_session(model_path)
    inp = sess.get_inputs()[0].name
    preds = sess.run(None, {inp: x})[0][0]
    return preds


def tag_image(image_path, threshold=DEFAULT_THRESHOLD, model_repo=MODEL_REPO,
              sep="comma"):
    """Return the caption string for one image."""
    model_path, csv_path = _download(model_repo)
    labels = _load_labels(csv_path)
    preds = _predict(model_path, image_path)

    tags = []
    for (name, cat), score in zip(_get_labels(csv_path), preds):
        if cat != 0:  # category 0 = general tags
            continue
        if score < threshold:
            continue
        name = name.replace("_", " ").strip()
        if not name or name.lower() in BLACKLIST:
            continue
        tags.append(name)

    if sep == "space":
        return " ".join(tags).strip()
    return ", ".join(tags)


def caption_dataset(img_dir, trigger="", threshold=DEFAULT_THRESHOLD, sep="comma",
                    progress_cb=None):
    """Caption every uncaptioned image in img_dir. Returns (done, skipped, errors)."""
    exts = (".png", ".jpg", ".jpeg", ".webp")
    images = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(exts))
    done = skipped = errors = 0
    for i, name in enumerate(images):
        base, _ = os.path.splitext(name)
        cap = os.path.join(img_dir, base + ".txt")
        if os.path.exists(cap) and os.path.getsize(cap) > 0:
            skipped += 1
            if progress_cb:
                progress_cb(i + 1, len(images))
            continue
        try:
            tags = tag_image(os.path.join(img_dir, name), threshold=threshold,
                             sep=sep)
        except Exception:
            errors += 1
            if progress_cb:
                progress_cb(i + 1, len(images))
            continue
        if trigger and trigger not in tags:
            tags = [trigger] + tags
        with open(cap, "w", encoding="utf-8") as f:
            f.write(", ".join(tags))
        done += 1
        if progress_cb:
            progress_cb(i + 1, len(images))
    return done, skipped, errors


def _cli():
    import sys
    if len(sys.argv) < 2:
        print("usage: python caption.py <img_dir> [trigger_word]")
        return 2
    img_dir = sys.argv[1]
    trigger = sys.argv[2].strip() if len(sys.argv) > 2 else ""

    def cb(i, n):
        print(f"captioning {i}/{n}", flush=True)

    done, skipped, errors = caption_dataset(img_dir, trigger=trigger,
                                            progress_cb=cb)
    print(f"captioned {done}, kept existing {skipped}, errors {errors}",
          flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(_cli())
