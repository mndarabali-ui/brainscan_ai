"""
explainability.py  (versi ONNX — tanpa torch)
------------------------------------------------
Menghasilkan heatmap "Peta Atensi Model" dari attention weight ViT layer
terakhir. BUKAN Grad-CAM klasik (yang butuh backward pass/gradient,
tidak didukung ONNX Runtime standar) — ini murni forward pass, membaca
output kedua ('attention') dari model ONNX yang sudah di-re-export
supaya menyertakan attention weight (lihat reexport_onnx_with_attention.py).

Logika pengolahan attention (rata-rata head, ambil attention CLS->patch,
reshape ke grid, resize, overlay) SAMA PERSIS dengan versi PyTorch
sebelumnya — cuma sumber datanya dari ONNX Runtime, bukan torch tensor.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from src.preprocess import val_transforms
from src.config import OUTPUT_DIR


def generate_attention_heatmap(image_path, save_name, onnx_session):
    """Menghasilkan peta panas (heatmap) fokus perhatian model AI pada gambar otak.

    onnx_session: ort.InferenceSession dari model hybrid (2 output: logits, attention),
                  di-load sekali di api.py lalu dipakai ulang di sini (tidak reload tiap panggilan).
    """
    # 1. Muat dan preprocess gambar
    orig_image = Image.open(image_path).convert("RGB")
    input_array = val_transforms(orig_image)  # [1, 3, H, W] float32

    # 2. Forward pass ONNX -> logits + attention weight layer terakhir
    input_name = onnx_session.get_inputs()[0].name
    logits, attention = onnx_session.run(None, {input_name: input_array})
    # attention shape: [1, num_heads, seq_len, seq_len]

    # 3. Rata-ratakan semua attention heads
    avg_attn = attention[0].mean(axis=0)  # [seq_len, seq_len]

    # 4. Ambil attention dari CLS token (index 0) ke semua patch token
    cls_attn = avg_attn[0, 1:]  # buang CLS-to-CLS, sisa [num_patches]

    # 5. Reshape ke grid persegi (feature map EfficientNet-B3 @224px -> grid 7x7)
    num_patches = int(round(cls_attn.shape[0] ** 0.5))
    heatmap = cls_attn.reshape(num_patches, num_patches)

    # 6. Normalisasi 0..1
    heatmap = np.maximum(heatmap, 0)
    max_val = np.max(heatmap)
    heatmap = heatmap / max_val if max_val != 0 else heatmap

    # 7. Gambar & gabungkan citra asli dengan peta panas
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(orig_image)
    axes[0].set_title("Gambar Medis Asli")
    axes[0].axis("off")

    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_resized = np.array(heatmap_img.resize(orig_image.size, Image.BILINEAR)) / 255.0

    axes[1].imshow(orig_image)
    axes[1].imshow(heatmap_resized, cmap="jet", alpha=0.4)
    axes[1].set_title("Peta Fokus Atensi AI (ViT Attention)")
    axes[1].axis("off")

    figure_dir = os.path.join(OUTPUT_DIR, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    save_path = os.path.join(figure_dir, save_name)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Sukses menghasilkan peta eksplanabilitas AI! Tersimpan di: {save_path}")
    return save_path
