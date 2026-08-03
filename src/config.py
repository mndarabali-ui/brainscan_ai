"""
src/config.py — Modul Konfigurasi Terpusat
Brain Disease Classification Pipeline
Vision Transformer (google/vit-base-patch16-224) Pretrained
"""

import os
from pathlib import Path

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HYPERPARAMETER UTAMA & SEED
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEED        = 42
IMG_SIZE    = 224
BATCH_SIZE  = 16
NUM_CLASSES = 5
LR          = 5e-5
EPOCHS      = 30

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GEMINI API KEY — WAJIB di-set lewat environment variable, JANGAN ditulis di sini
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Jangan pernah taruh API key asli langsung di source code (apalagi yang ikut
# ter-commit ke git / ter-share ke orang lain) — siapa pun yang membaca file
# ini bisa memakai kuota/API key milik Anda. Set lewat environment variable:
#   export GEMINI_API_KEY="isi-key-anda"        (Linux/Mac)
#   setx GEMINI_API_KEY "isi-key-anda"           (Windows)
# Jika tidak di-set, sistem otomatis memakai generator laporan lokal (fallback)
# di gemini_client.py — jadi aplikasi tetap berjalan tanpa Gemini API.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  KELAS PENYAKIT OTAK (5 KELAS)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
# Posisi src/config.py adalah di proyek/src/, maka parent-nya adalah root proyek.
BASE_DIR = Path(__file__).resolve().parent.parent

# ── Data Directories ──────────────────────────────────────────────────────
DATA_DIR           = BASE_DIR / "data"
RAW_DIR            = DATA_DIR / "raw"
INTERIM_DIR        = DATA_DIR / "interim"
PROCESSED_DIR      = DATA_DIR / "processed"
SPLITS_DIR         = DATA_DIR / "splits"

# ── Output Directories ────────────────────────────────────────────────────
OUTPUT_DIR         = BASE_DIR / "outputs"
CHECKPOINT_DIR     = OUTPUT_DIR / "checkpoints"

# ── Hugging Face Model Repo ────────────────────────────────────────────
HF_REPO_ID = os.environ.get("HF_REPO_ID", "marksnb/brain-hybrid-efficientnet-vit")

def download_model_from_hf(filename: str):
    """
    Download file .onnx dari Hugging Face Hub kalau belum ada lokal.
    Kalau gagal (file nggak ada di repo, dll) return None supaya
    main.py bisa fallback ke perilaku lama (pakai bobot pretrained).
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

LOGS_DIR           = OUTPUT_DIR / "logs"
FIGURES_DIR        = OUTPUT_DIR / "figures"
TABLES_DIR         = OUTPUT_DIR / "tables"
REPORTS_DIR        = OUTPUT_DIR / "reports"

# ── File Paths Penting ────────────────────────────────────────────────────
TRAINING_LOG_FILE    = LOGS_DIR        / "training.log"
SPLITS_CSV_FILE      = SPLITS_DIR      / "train_val_test.csv"
BEST_MODEL_PATH      = CHECKPOINT_DIR  / "hybrid_vit_efficientnet_brain_fp32.onnx"

# ─── Output Figures (nama file sesuai spesifikasi) ────────────────────────
FIG_TRAINING_PERF      = FIGURES_DIR / "training_performance.png"
FIG_CONFUSION_MATRIX   = FIGURES_DIR / "confusion_matrix.png"
FIG_TRAINING_SUMMARY   = FIGURES_DIR / "training_summary.png"
FIG_CLASS_DIST         = FIGURES_DIR / "class_distribution_before.png"
FIG_AUGMENT_COMP       = FIGURES_DIR / "augmentation_comparison.png"

# ─── Output Tables & Reports ──────────────────────────────────────────────
TABLE_CLASSIF_REPORT   = TABLES_DIR  / "classification_report.csv"
TABLE_DATASET_DIST     = TABLES_DIR  / "dataset_distribution.csv"
TABLE_TRAINING_HISTORY = TABLES_DIR  / "training_history.csv"
TABLE_TEST_EVAL        = TABLES_DIR  / "test_evaluation_results.csv"
REPORT_AUDIT_SUMMARY = REPORTS_DIR / "audit_summary.json"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MODE TRAINING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# QUICK_TEST=True  -> cuma proses 2 batch/epoch, buat tes cepat pipeline jalan/tidak
# QUICK_TEST=False -> training penuh pakai seluruh data asli (WAJIB False untuk hasil final)
QUICK_TEST = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TARGET AUGMENTASI (class balancing)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Target jumlah sampel TRAIN per kelas setelah augmentasi offline.
# Total akhir = AUGMENT_TARGET_PER_CLASS x NUM_CLASSES -- TAPI HANYA kalau
# semua kelas raw < target ini. Kelas yang raw-nya sudah >= target TIDAK
# dikurangi/dipotong (augment.py cuma menambah, tidak pernah membuang data).
AUGMENT_TARGET_PER_CLASS = 19097

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  INIT_FOLDERS — Buat Semua Direktori Otomatis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def init_folders() -> None:
    """
    Membuat seluruh struktur folder proyek yang diperlukan jika belum ada.
    """
    all_dirs = [
        RAW_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        SPLITS_DIR,
        CHECKPOINT_DIR,
        LOGS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        REPORTS_DIR,
    ]
    for d in all_dirs:
        os.makedirs(d, exist_ok=True)


if __name__ == "__main__":
    init_folders()
    print("=" * 60)
    print("  Modul Konfigurasi Terpusat (config.py)")
    print("=" * 60)
    print(f"  Root proyek : {BASE_DIR}")
    print(f"  SEED        : {SEED}")
    print(f"  IMG_SIZE    : {IMG_SIZE}")
    print(f"  BATCH_SIZE  : {BATCH_SIZE}")
    print(f"  NUM_CLASSES : {NUM_CLASSES}")
    print(f"  EPOCHS      : {EPOCHS}")
    print("  Struktur folder berhasil diperiksa/dibuat.")
    print("=" * 60)
