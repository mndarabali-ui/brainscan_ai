# BrainScan AI — Final Project

## Cara jalanin (dari root folder ini, sejajar dengan folder `src/`)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# (opsional) kalau mau pakai Gemini API buat laporan radiologi:
export GEMINI_API_KEY="isi-key-kamu"     # Windows: setx GEMINI_API_KEY "isi-key-kamu"

uvicorn src.app:app --reload
```

Buka `http://127.0.0.1:8000` di browser.

## Yang perlu diperhatikan
- Model `.pth` **otomatis di-download** dari Hugging Face
  (`delfidev/brain-hybrid-efficientnet-vit`) pas server pertama kali start,
  disimpan di `outputs/checkpoints/`. Kalau belum ada model precheck yang
  ke-upload, sistem fallback pakai bobot pretrained (ada warning di log,
  bukan error).
- `data/brainscan.db` otomatis dibuat sendiri pas ada pasien pertama
  didaftarkan — nggak perlu dibuat manual.
- File training (`train_classifier.py`, `data_loader.py`, dll) sengaja
  tidak diikutkan di sini karena tidak dipakai untuk menjalankan website;
  itu tetap ada di notebook/project training terpisah.
