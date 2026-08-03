"""
preprocess.py
-------------
Berisi:
  - train_transforms / val_transforms : objek T.Compose siap pakai
  - get_transforms()                  : mengembalikan (train_tf, val_tf)
  - print_preprocessing_info()        : cetak ringkasan ke log
  - plot_preprocessing_distribution() : grafik before/after balancing
"""

import os
import cv2
import logging
import warnings
import pandas as pd
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchvision.transforms as T

from src.config import (
    IMG_SIZE, CLASSES, CLASS_DISPLAY, CLASS_COLORS, FIGURES_DIR,
    DATA_DIR, SPLITS_DIR,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger("brain_pipeline")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TRANSFORMS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Statistik normalisasi HARUS SAMA dengan yang dipakai saat training
# (lihat notebook: IMAGENET_MEAN / IMAGENET_STD), karena backbone
# EfficientNet-B3 di-pretrain pakai statistik ImageNet ini.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Transformasi untuk data latihan (dengan augmentasi spesifik citra medis)
train_transforms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=8),  # Rotasi kecil spesifik medis
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.0, hue=0.0), # Citra medis grayscale tidak memerlukan warna/hue/sat
    T.RandomAffine(degrees=0, translate=(0.03, 0.03)), # Pergeseran kecil saja
    T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5)), # Gaussian Blur halus
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    T.RandomErasing(p=0.1, scale=(0.01, 0.05)), # Erasing diperkecil agar tidak menutupi patologi penting
])

# Transformasi untuk data validasi & pengujian (tanpa augmentasi)
val_transforms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Transform khusus untuk model Precheck -- model ini dilatih terpisah
# (train_precheck.py) memakai normalisasi [0.5,0.5,0.5], BUKAN statistik
# ImageNet, jadi harus dipakai transform yang beda dari val_transforms
# di atas supaya konsisten dengan cara precheck model dilatih.
precheck_transforms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def get_transforms():
    """Mengembalikan (train_transforms, val_transforms)."""
    return train_transforms, val_transforms


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PREPROCESSING INFO & PLOTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_preprocessing_info():
    """Tampilkan ringkasan preprocessing & augmentasi ke log/terminal."""
    from src.config import IMG_SIZE, BATCH_SIZE, LR, EPOCHS

    sep70 = "=" * 70
    logger.info("\n" + sep70)
    logger.info("  PREPROCESSING ANALYSIS")
    logger.info(sep70)

    info = {
        "Resize"              : f"{IMG_SIZE} x {IMG_SIZE} px",
        "Normalisasi"         : "mean=[0.5,0.5,0.5]  std=[0.5,0.5,0.5]  (range -> [-1,1])",
        "Augmentasi (train)"  : (
            "RandomHorizontalFlip(p=0.5) | RandomRotation(+-15) | "
            "ColorJitter(brightness=0.2, contrast=0.2) | "
            "RandomAffine(translate=5%) | RandomErasing(p=0.2)"
        ),
        "Augmentasi (val/test)": "Resize + ToTensor + Normalize  (tanpa augmentasi)",
        "Balancing"           : "WeightedRandomSampler  (per-class inverse-frequency weights)",
    }
    col = 25
    for k, v in info.items():
        logger.info(f"  {k:<{col}}: {v}")


def plot_preprocessing_distribution(train_df_before, train_df_after):
    """
    Grafik distribusi kelas SEBELUM augmentasi (train_df_before, hasil scan
    data/raw/ asli) vs SESUDAH augmentasi offline sungguhan (train_df_after,
    hasil nyata dari augment.run_augmentation() -- bukan proyeksi/estimasi).
    """
    COLOR_CT = "#E15759"
    COLOR_MRI = "#4E79A7"

    before_mods = {
        "Alzheimer": "MRI", "Intracranial_Hemorrhage": "CT",
        "Normal": "MRI", "Stroke_Iskemik": "CT", "Tumor": "CT"
    }

    before_counts_s = train_df_before["label"].value_counts().reindex(CLASSES, fill_value=0)
    after_counts_s  = train_df_after["label"].value_counts().reindex(CLASSES, fill_value=0)
    before = before_counts_s.tolist()
    after  = after_counts_s.tolist()
    total_before = int(sum(before))
    total_after  = int(sum(after))

    labels = [f"{CLASS_DISPLAY.get(c, c)}\n({before_mods.get(c, 'MRI')})" for c in CLASSES]
    colors = [COLOR_CT if before_mods.get(c, 'MRI') == 'CT' else COLOR_MRI for c in CLASSES]

    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Kiri: Sebelum Augmentasi (Distribusi Asli Train)
    bars_b = axes[0].bar(x, before, width=0.55, color=colors,
                         edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars_b, before):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(before, default=1) * 0.01,
                     f"{val:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[0].set_title("Sebelum Augmentasi  (Distribusi Asli Data Train)", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_ylabel("Jumlah Sampel")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0, max(before + after, default=1) * 1.15)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_CT, edgecolor='white', label='CT Scan'),
        Patch(facecolor=COLOR_MRI, edgecolor='white', label='MRI Otak')
    ]
    axes[0].legend(handles=legend_elements, loc="upper right")

    # Kanan: Sesudah Augmentasi Offline (Hasil NYATA, bukan estimasi)
    bars_a = axes[1].bar(x, after, width=0.55, color=colors,
                         edgecolor="white", linewidth=1.2, alpha=0.85)
    for bar, val in zip(bars_a, after):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(before + after, default=1) * 0.01,
                     f"{val:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[1].set_title("Sesudah Augmentasi Offline  (Hasil Nyata)", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_ylabel("Jumlah Sampel")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_ylim(0, max(before + after, default=1) * 1.15)
    axes[1].legend(handles=legend_elements, loc="upper right")

    axes[1].text(0.5, 0.90, f"Total Data Train Setelah Augmentasi = {total_after:,} data",
                 transform=axes[1].transAxes, ha="center", va="center",
                 color="#D62728", fontsize=11, fontweight="bold",
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='#D62728', boxstyle='round,pad=0.5'))

    fig.suptitle(f"Distribusi Data CT Scan & MRI Sebelum ({total_before:,}) dan Sesudah ({total_after:,}) Augmentasi",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = Path(FIGURES_DIR) / "preprocessing_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Grafik preprocessing disimpan -> {out}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  BRAIN SCAN CROP CONTOUR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def crop_brain_contour(img):
    """
    Mendeteksi area scan otak sirkular di dalam gambar dan memotong (crop)
    area luar seperti taskbar desktop, bingkai window, atau ruang hitam berlebih.
    """
    if img is None:
        return None
        
    h, w, c = img.shape
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold to binary (threshold 20, max 255)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
        
    # Sort contours by area descending and find the largest
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Get bounding box of the largest contour (the brain scan area)
    x, y, cw, ch = cv2.boundingRect(contours[0])
    
    # Check if the bounding box is valid and reasonably large
    # (e.g. at least 30% of original width/height to avoid tiny text contours)
    if cw > w * 0.3 and ch > h * 0.3:
        # Add a padding of 5 pixels to avoid clipping the edges
        pad = 5
        x = max(0, x - pad)
        y = max(0, y - pad)
        cw = min(w - x, cw + 2 * pad)
        ch = min(h - y, ch + 2 * pad)
        
        # Crop
        cropped = img[y:y+ch, x:x+cw]
        return cropped
    return img


def run_cropping_pipeline():
    processed_dir = DATA_DIR / "processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    logger.info("Memulai proses pemotongan (cropping) citra otomatis untuk membersihkan desktop/window borders...")
    
    # 1. Proses dataset splits
    for csv_name in ["train.csv", "val.csv", "test.csv"]:
        csv_path = os.path.join(SPLITS_DIR, csv_name)
        if not os.path.exists(csv_path):
            continue
            
        logger.info(f"Memproses {csv_name}...")
        df = pd.read_csv(csv_path)
        new_paths = []
        
        for idx, row in df.iterrows():
            orig_path = row['image_path']
            label = row['label']
            
            # Tentukan output path di folder processed
            filename = os.path.basename(orig_path)
            out_label_dir = os.path.join(processed_dir, label)
            os.makedirs(out_label_dir, exist_ok=True)
            out_path = os.path.join(out_label_dir, filename)
            
            # Jika file sudah di-crop sebelumnya, lewati
            if os.path.exists(out_path):
                new_paths.append(out_path)
                continue
                
            # Load, crop, dan simpan
            img = cv2.imread(orig_path)
            if img is not None:
                cropped_img = crop_brain_contour(img)
                cv2.imwrite(out_path, cropped_img)
                new_paths.append(out_path)
            else:
                # Fallback ke path asli jika gagal load
                new_paths.append(orig_path)
                
        df['image_path'] = new_paths
        df.to_csv(csv_path, index=False)
        logger.info(f"Selesai memproses {csv_name}. Berkas disimpan kembali dengan path gambar terpotong.")
 
    # 2. Proses folder samples agar berkas tes juga terpotong bersih
    samples_dir = "samples"
    if os.path.exists(samples_dir):
        logger.info("Memproses folder samples...")
        for filename in os.listdir(samples_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(samples_dir, filename)
                img = cv2.imread(filepath)
                if img is not None:
                    cropped_img = crop_brain_contour(img)
                    cv2.imwrite(filepath, cropped_img)
        logger.info("Selesai memproses folder samples.")


if __name__ == "__main__":
    # Setup basic logging to stdout when run directly
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_cropping_pipeline()