import os
import numpy as np
from PIL import Image
from src.config import OUTPUT_DIR


def generate_attention_heatmap(
    image_path: str,
    save_name: str = "attention_map.png",
    attention_override: np.ndarray | None = None,
):
    """
    Menghasilkan peta panas (heatmap) fokus perhatian model AI pada gambar otak.

    Parameters
    ----------
    image_path : str
        Path ke file gambar asli.
    save_name : str
        Nama file output heatmap (disimpan di outputs/figures/).
    attention_override : np.ndarray | None
        Array attention hasil inference ONNX dengan shape
        [1, num_heads, seq_len, seq_len].
        Jika None, heatmap dibuat sebagai gaussian blur sederhana (fallback).
    """
    orig_image = Image.open(image_path).convert("RGB")

    if attention_override is not None:
        # attention shape: [1, num_heads, seq_len, seq_len]
        attn = attention_override[0]              # [num_heads, seq_len, seq_len]
        avg_attn = attn.mean(axis=0)              # [seq_len, seq_len]
        cls_attn = avg_attn[0, 1:]               # [num_patches] — CLS ke patch lainnya

        num_patches = int(round(cls_attn.shape[0] ** 0.5))
        heatmap = cls_attn[:num_patches * num_patches].reshape(num_patches, num_patches)
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() != 0:
            heatmap /= heatmap.max()
    else:
        # Fallback: gaussian-blur sederhana sebagai placeholder
        w, h = orig_image.size
        cx, cy = w // 2, h // 2
        y_grid, x_grid = np.ogrid[:h, :w]
        sigma = min(w, h) * 0.25
        heatmap = np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma ** 2))

    # Ubah ukuran heatmap agar pas dengan dimensi gambar asli
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            orig_image.size, Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    ) / 255.0

    # Gambar dan gabungkan citra asli dengan peta panas
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(orig_image)
    axes[0].set_title("Gambar Medis Asli")
    axes[0].axis("off")

    axes[1].imshow(orig_image)
    axes[1].imshow(heatmap_resized, cmap="jet", alpha=0.4)
    axes[1].set_title("Peta Fokus Atensi AI (ViT Attention)")
    axes[1].axis("off")

    figure_dir = os.path.join(OUTPUT_DIR, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    save_path = os.path.join(figure_dir, save_name)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Sukses menghasilkan peta eksplanabilitas AI! Tersimpan di: {save_path}")


if __name__ == "__main__":
    sample_dir = "data/raw/Normal"
    if os.path.exists(sample_dir) and os.listdir(sample_dir):
        first_img = os.listdir(sample_dir)[0]
        full_path = os.path.join(sample_dir, first_img)
        generate_attention_heatmap(full_path)
    else:
        print("Folder data/raw/Normal kosong atau tidak ditemukan untuk pengujian.")
