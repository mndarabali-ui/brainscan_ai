import os
import io
import base64
import shutil
import datetime

import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fpdf import FPDF

from src.config import CLASSES, OUTPUT_DIR, CHECKPOINT_DIR, download_model_from_hf
from src.preprocess import val_transforms, precheck_transforms
from src.gemini_client import generate_radiology_report
from src.explainability import generate_attention_heatmap

router = APIRouter()

# ─────────────────────────────────────────────────────────────
# Util tanggal Indonesia (dipakai di laporan PDF)
# ─────────────────────────────────────────────────────────────
_HARI_ID = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
_BULAN_ID = ["", "Januari", "Februari", "Maret", "April", "Mei", "Juni",
             "Juli", "Agustus", "September", "Oktober", "November", "Desember"]


def tanggal_indonesia_sekarang() -> str:
    """Format tanggal sekarang, misal: 'Selasa, 7 Juli 2026' — bukan tanggal tetap."""
    now = datetime.datetime.now()
    return f"{_HARI_ID[now.weekday()]}, {now.day} {_BULAN_ID[now.month]} {now.year}"


# ─────────────────────────────────────────────────────────────
# Helper: buat ONNX Runtime InferenceSession
# ─────────────────────────────────────────────────────────────
def _make_session(onnx_path: str) -> ort.InferenceSession:
    """Buat ORT session dengan CPU EP."""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )


# ─────────────────────────────────────────────────────────────
# Muat ONNX model via ONNX Runtime saat modul ini diimpor
# ─────────────────────────────────────────────────────────────
precheck_session: ort.InferenceSession | None = None
hybrid_session: ort.InferenceSession | None = None

try:
    print("Memuat model Precheck (ONNX Runtime)...")
    precheck_onnx = (
        download_model_from_hf("best_precheck_model.onnx")
        or os.path.join(CHECKPOINT_DIR, "best_precheck_model.onnx")
    )
    if os.path.exists(precheck_onnx):
        precheck_session = _make_session(precheck_onnx)
        print(f"Sukses memuat Precheck ONNX dari: {precheck_onnx}")
    else:
        print("Warning: best_precheck_model.onnx tidak ditemukan.")

    print("Memuat model Hybrid (ONNX Runtime)...")
    hybrid_onnx = (
        download_model_from_hf("hybrid_vit_efficientnet_brain_fp32.onnx")
        or os.path.join(CHECKPOINT_DIR, "hybrid_vit_efficientnet_brain_fp32.onnx")
    )
    if os.path.exists(hybrid_onnx):
        hybrid_session = _make_session(hybrid_onnx)
        print(f"Sukses memuat Hybrid ONNX dari: {hybrid_onnx}")
    else:
        print("Warning: hybrid_vit_efficientnet_brain_fp32.onnx tidak ditemukan.")

    print("Seluruh model AI berhasil dimuat.")
except Exception as e:
    print(f"Gagal memuat model AI: {str(e)}")


# ─────────────────────────────────────────────────────────────
# Helper: jalankan inference ONNX Runtime
# ─────────────────────────────────────────────────────────────
def _run_precheck(tensor_image_np: np.ndarray) -> tuple[int, float]:
    """
    Jalankan inference model Precheck.
    Return: (is_valid_idx, probability_score)
      is_valid_idx == 1  => gambar Valid (brain scan)
      is_valid_idx == 0  => gambar Invalid
    """
    outputs = precheck_session.run(
        None,
        {"input": tensor_image_np},
    )
    logits = outputs[0]  # shape [1, 2]
    # Softmax manual supaya tidak perlu torch
    exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_l / exp_l.sum(axis=1, keepdims=True)
    pred_idx = int(np.argmax(probs[0]))
    prob_score = float(probs[0][pred_idx])
    return pred_idx, prob_score


def _run_hybrid(tensor_image_np: np.ndarray) -> tuple[int, float, np.ndarray]:
    """
    Jalankan inference model Hybrid.
    Return: (predicted_class_idx, confidence_score, attention_map_array)
    """
    outputs = hybrid_session.run(
        None,
        {"input": tensor_image_np},
    )
    logits = outputs[0]       # shape [1, 5]
    attention = outputs[1]    # shape [1, num_heads, seq_len, seq_len]

    # Softmax manual
    exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_l / exp_l.sum(axis=1, keepdims=True)
    pred_idx = int(np.argmax(probs[0]))
    confidence = float(probs[0][pred_idx]) * 100
    return pred_idx, confidence, attention


# ─────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────
class PatientCreate(BaseModel):
    nik: str
    name: str
    age: int = None
    birth_date: str = None
    gender: str = None
    address: str = None
    phone: str = None


class PDFDownloadRequest(BaseModel):
    patient_name: str
    patient_age: str
    patient_gender: str
    patient_nik: str
    patient_birth_date: str = ""
    patient_address: str = ""
    patient_phone: str = ""
    report_text: str


# ─────────────────────────────────────────────────────────────
# Endpoint: status server
# ─────────────────────────────────────────────────────────────
@router.get("/api/status")
def get_status():
    """Mengecek status online server dan ketersediaan model AI"""
    return {
        "status": "Online",
        "precheck_model_loaded": precheck_session is not None,
        "classifier_model_loaded": hybrid_session is not None,
        "device": "onnxruntime",
    }


# ─────────────────────────────────────────────────────────────
# Endpoint: data pasien
# ─────────────────────────────────────────────────────────────
from src.database import upsert_patient, get_patient, add_scan_record, get_patient_history

@router.post("/api/patients/")
def register_patient(patient: PatientCreate):
    """Menyimpan atau memperbarui data profil pasien"""
    try:
        upsert_patient(
            nik=patient.nik,
            name=patient.name,
            age=patient.age,
            birth_date=patient.birth_date,
            gender=patient.gender,
            address=patient.address,
            phone=patient.phone,
        )
        return {"status": "Success", "message": "Data pasien berhasil disimpan."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menyimpan data pasien: {str(e)}")


@router.get("/api/patients/{nik}")
def get_patient_info(nik: str):
    """Mengambil data pasien berdasarkan NIK"""
    patient = get_patient(nik)
    if not patient:
        raise HTTPException(status_code=404, detail="Pasien tidak ditemukan.")
    return {"status": "Success", "patient": patient}


@router.get("/api/patients/{nik}/history")
def get_patient_scans_history(nik: str):
    """Mengambil riwayat scan pasien berdasarkan NIK"""
    try:
        history = get_patient_history(nik)
        return {"status": "Success", "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal mengambil riwayat scan: {str(e)}")


# ─────────────────────────────────────────────────────────────
# Endpoint utama: analisis gambar scan otak
# ─────────────────────────────────────────────────────────────
@router.post("/api/analyze/")
async def analyze_brain_image(file: UploadFile = File(...), patient_nik: str = Form(None)):
    """
    Endpoint utama untuk mengunggah gambar scan otak, menjalankan pre-check,
    menjalankan klasifikasi penyakit, memvisualisasikan atensi model (XAI),
    dan menghasilkan laporan radiologi AI.
    """
    # 1. Validasi Ekstensi File
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Format file harus berupa gambar (PNG, JPG, JPEG).")

    try:
        # 2. Simpan file unggahan sementara untuk visualisasi heatmap
        os.makedirs("temp_uploads", exist_ok=True)
        temp_file_path = os.path.join("temp_uploads", file.filename)
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 3. Baca gambar dan konversi ke numpy array float32
        image = Image.open(temp_file_path).convert("RGB")

        # Precheck: normalisasi [0.5, 0.5, 0.5]
        precheck_np = precheck_transforms(image).unsqueeze(0).numpy().astype(np.float32)
        # Hybrid: normalisasi ImageNet
        hybrid_np = val_transforms(image).unsqueeze(0).numpy().astype(np.float32)

        # 4. TAHAP 1: Precheck
        is_valid = True
        precheck_prob_val = 0.99
        if precheck_session is not None:
            is_valid_idx, precheck_prob_val = _run_precheck(precheck_np)
            is_valid = (is_valid_idx == 1)

        if not is_valid:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            return {
                "status": "Invalid",
                "filename": file.filename,
                "message": "Gambar tidak dikenali sebagai scan otak yang valid (CT-Scan/MRI). Hubungi Administrator.",
                "precheck_confidence": f"{precheck_prob_val * 100:.2f}%",
            }

        # 5. TAHAP 2: Klasifikasi Utama
        if hybrid_session is None:
            raise HTTPException(status_code=500, detail="Model utama klasifikasi tidak termuat di server.")

        predicted_idx, confidence_score, attention = _run_hybrid(hybrid_np)
        predicted_class = CLASSES[predicted_idx]

        # 6. TAHAP 3: Eksplanabilitas AI (XAI) — Peta Atensi Heatmap (pakai attention dari ONNX)
        heatmap_filename = f"heatmap_{os.path.splitext(file.filename)[0]}.png"
        generate_attention_heatmap(
            image_path=temp_file_path,
            save_name=heatmap_filename,
            attention_override=attention,
        )

        # 7. TAHAP 4: Laporan radiologi AI
        modality = "CT" if "ct" in file.filename.lower() else "MRI"
        report_text = generate_radiology_report(predicted_idx, confidence_score, modality)

        # 8. Encode heatmap dan gambar asli ke base64
        heatmap_path = os.path.join(OUTPUT_DIR, "figures", heatmap_filename)
        with open(heatmap_path, "rb") as img_file:
            heatmap_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        with open(temp_file_path, "rb") as img_file:
            original_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        # 9. Simpan ke database (opsional)
        if patient_nik:
            try:
                add_scan_record(
                    patient_nik=patient_nik,
                    filename=file.filename,
                    modality=modality,
                    predicted_class=predicted_class,
                    confidence=confidence_score,
                    report_text=report_text,
                    original_b64=f"data:image/png;base64,{original_base64}",
                    heatmap_b64=f"data:image/png;base64,{heatmap_base64}",
                )
            except Exception as db_err:
                print(f"Gagal menyimpan riwayat scan ke database: {str(db_err)}")

        # 10. Kembalikan respons JSON
        return {
            "status": "Valid",
            "filename": file.filename,
            "modality_detected": modality,
            "prediction": {
                "class_name": predicted_class,
                "class_index": predicted_idx,
                "confidence": f"{confidence_score:.2f}%",
            },
            "precheck_confidence": f"{precheck_prob_val * 100:.2f}%",
            "radiology_report": report_text,
            "original_image_b64": f"data:image/png;base64,{original_base64}",
            "heatmap_image_b64": f"data:image/png;base64,{heatmap_base64}",
        }

    except Exception as e:
        if "temp_file_path" in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan internal analisis: {str(e)}")


# ─────────────────────────────────────────────────────────────
# Endpoint: download laporan PDF
# ─────────────────────────────────────────────────────────────
@router.post("/api/download-pdf/")
def download_pdf(data: PDFDownloadRequest):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("helvetica", size=10)

        # 1. Header (Kop Surat)
        pdf.set_font("helvetica", "B", 14)
        pdf.cell(0, 8, "PUSAT RADIOLOGI DIGITAL & DIAGNOSTIK AI", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.set_font("helvetica", size=9)
        pdf.cell(0, 5, "Jl. Semilasari Barat No. 88, Sektor Kecerdasan Buatan, Denpasar", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.cell(0, 5, "Email: support@brainscan.ai | Telp: (021) 555-2026", new_x="LMARGIN", new_y="NEXT", align="C")

        pdf.ln(3)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        # 2. Document Title
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 7, "DOKUMEN LAPORAN HASIL PEMERIKSAAN RADIOLOGI (OPINI AI)", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(4)

        # 3. Patient Details
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(0, 6, "I. IDENTITAS PASIEN & PEMERIKSAAN", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", size=9)

        details = [
            ("Nama Pasien", data.patient_name, "Jenis Kelamin", data.patient_gender),
            ("Umur", f"{data.patient_age} Tahun", "Tanggal Lahir", data.patient_birth_date),
            ("NIK Pasien", data.patient_nik, "No. Telepon", data.patient_phone),
            ("Alamat", data.patient_address, "Tanggal Analisis", tanggal_indonesia_sekarang()),
        ]

        col_width = 40
        val_width = 55
        for row in details:
            pdf.set_font("helvetica", "B", 9)
            pdf.cell(col_width, 6, f"{row[0]}:", border=0)
            pdf.set_font("helvetica", "", 9)
            pdf.cell(val_width, 6, str(row[1]), border=0)

            pdf.set_font("helvetica", "B", 9)
            pdf.cell(col_width, 6, f"{row[2]}:", border=0)
            pdf.set_font("helvetica", "", 9)
            pdf.cell(val_width, 6, str(row[3]), border=0, new_x="LMARGIN", new_y="NEXT")

        pdf.ln(3)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        # 4. Report Text Content
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(0, 6, "II. LAPORAN PEMERIKSAAN (RADIOLOGY REPORT)", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        pdf.set_font("helvetica", "", 9.5)
        lines = data.report_text.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("1. ", "2. ", "3. ", "4. ")):
                pdf.ln(2)
                pdf.set_font("helvetica", "B", 10)
                pdf.multi_cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("helvetica", "", 9.5)
            elif stripped.startswith(("* ", "- ")):
                pdf.set_font("helvetica", "", 9.5)
                pdf.set_x(15)
                pdf.multi_cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
            elif stripped.startswith(("*Catatan:", "Catatan:")):
                pdf.ln(4)
                pdf.set_font("helvetica", "I", 8.5)
                pdf.multi_cell(0, 4.5, line, new_x="LMARGIN", new_y="NEXT")
            else:
                pdf.multi_cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")

        # 5. Signatures
        pdf.ln(15)
        current_y = pdf.get_y()

        if current_y > 240:
            pdf.add_page()
            current_y = pdf.get_y()

        pdf.set_font("helvetica", "", 9.5)
        pdf.set_xy(130, current_y)
        pdf.cell(60, 5, f"Denpasar, {tanggal_indonesia_sekarang()}", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.set_x(130)
        pdf.cell(60, 5, "Pusat Radiologi Digital & Diagnostik AI", new_x="LMARGIN", new_y="NEXT", align="C")

        pdf.ln(10)
        pdf.set_x(130)
        pdf.set_font("helvetica", "B", 9.5)
        pdf.cell(60, 5, "dr. _________________________, Sp.Rad", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.set_x(130)
        pdf.set_font("helvetica", "", 8.5)
        pdf.cell(60, 5, "NIP. ___________________________", new_x="LMARGIN", new_y="NEXT", align="C")

        pdf_bytes = bytes(pdf.output())

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=Laporan_Radiologi_BrainScan.pdf"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses PDF: {str(e)}")
