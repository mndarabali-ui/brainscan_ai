# BrainScan AI — Catatan Deploy SnapDeploy + Hugging Face + Turso

## 1. Environment variables di SnapDeploy
Set minimal:

```env
HF_REPO_ID=delfidev/brain-hybrid-efficientnet-vit
PORT=8000
```

Kalau pakai Turso, set juga:

```env
TURSO_DATABASE_URL=libsql://nama-db-anda.turso.io
TURSO_AUTH_TOKEN=token-dari-turso
```

Opsional untuk laporan Gemini:

```env
GEMINI_API_KEY=isi_api_key_gemini
```

## 2. File model di Hugging Face
Pastikan repo `delfidev/brain-hybrid-efficientnet-vit` berisi file berikut persis namanya:

- `best_precheck_model.onnx`
- `hybrid_vit_efficientnet_brain_fp32.onnx`

Aplikasi akan download otomatis saat server start dan cache ke `outputs/checkpoints/`.

## 3. Start command SnapDeploy
Gunakan salah satu:

```bash
bash start.sh
```

atau langsung:

```bash
uvicorn src.app:app --host 0.0.0.0 --port $PORT
```

Jangan pakai `--reload` di production.

## 4. Catatan Turso
Kode `src/database.py` sekarang otomatis:

- memakai Turso jika `TURSO_DATABASE_URL` dan `TURSO_AUTH_TOKEN` tersedia;
- fallback ke SQLite lokal `data/brainscan.db` jika env Turso belum diset.

Jadi development lokal tetap bisa jalan tanpa Turso.

## 5. Health check
Buka:

```text
/api/status
```

Jika deploy benar, respons harus menunjukkan:

```json
{
  "status": "Online",
  "precheck_model_loaded": true,
  "classifier_model_loaded": true
}
```

Jika model belum loaded, cek nama file di Hugging Face dan log SnapDeploy.
