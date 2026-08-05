"""
src/config.py — Modul Konfigurasi Terpusat
Brain Disease Classification Pipeline (ONNX Runtime, FastAPI Cloud)
"""

import os
from pathlib import Path

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  KELAS PENYAKIT OTAK (5 KELAS)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMG_SIZE    = 224
NUM_CLASSES = 5

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

CLASSES = [
    "Alzheimer",
    "Intracranial_Hemorrhage",
    "Normal",
    "Stroke_Iskemik",
    "Tumor",
]

CLASS_DISPLAY = {
    "Alzheimer":               "Alzheimer",
    "Intracranial_Hemorrhage": "ICH",
    "Normal":                  "Normal",
    "Stroke_Iskemik":          "Ischemic Stroke",
    "Tumor":                   "Brain Tumor",
}

CLASS_COLORS = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1"]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STRUKTUR DIREKTORI PROYEK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR     = BASE_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURES_DIR    = OUTPUT_DIR / "figures"

# ── Hugging Face Model Repo ────────────────────────────────────────────
HF_REPO_ID = "delfidev/brain-hybrid-efficientnet-vit"


def download_model_from_hf(filename: str):
    """
    Download file .onnx dari Hugging Face Hub kalau belum ada lokal.
    Return None kalau gagal (file tidak ada, dsb) supaya api.py bisa
    fallback dengan aman (tanpa membuat server crash).
    """
    from huggingface_hub import hf_hub_download
    local_path = CHECKPOINT_DIR / filename
    if local_path.exists():
        return str(local_path)
    try:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        return hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            local_dir=str(CHECKPOINT_DIR),
        )
    except Exception as e:
        print(f"⚠️ Gagal download '{filename}' dari Hugging Face: {e}")
        return None


def init_folders() -> None:
    """Membuat seluruh struktur folder proyek yang diperlukan jika belum ada."""
    for d in [CHECKPOINT_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)


if __name__ == "__main__":
    init_folders()
    print("Modul Konfigurasi Terpusat (config.py) -- OK")
    print(f"Root proyek : {BASE_DIR}")
    print(f"HF_REPO_ID  : {HF_REPO_ID}")
