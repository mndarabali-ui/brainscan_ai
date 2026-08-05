
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

from src.config import CLASSES, OUTPUT_DIR, download_model_from_hf
from src.preprocess import val_transforms, precheck_transforms
from src.gemini_client import generate_radiology_report
from src.explainability import generate_attention_heatmap
from src.database import upsert_patient, get_patient, add_scan_record, get_patient_history

router = APIRouter()

# ─────────────────────────────────────────────────────────────
# Util tanggal Indonesia (dipakai di laporan PDF)
# ─────────────────────────────────────────────────────────────
_HARI_ID = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
_BULAN_ID = ["", "Januari", "Februari", "Maret", "April", "Mei", "Juni",
             "Juli", "Agustus", "September", "Oktober", "November", "Desember"]


def tanggal_indonesia_sekarang() -> str:
    now = datetime.datetime.now()
    return f"{_HARI_ID[now.weekday()]}, {now.day} {_BULAN_ID[now.month]} {now.year}"


def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax numpy biasa, menggantikan torch.nn.functional.softmax."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


# ─────────────────────────────────────────────────────────────
# Muat model-model AI (ONNX Runtime session) secara global, SEKALI,
# saat modul ini diimpor (yaitu saat app.py memanggil `from src.api import router`)
#
# PENTING: pakai FP32 untuk KEDUA model (bukan FP16/INT8) --
# FP16 hybrid terbukti gagal di-load (mixed dtype bug di graph ONNX-nya),
# dan INT8 (baik precheck maupun hybrid) terbukti akurasinya jatuh drastis
# saat divalidasi. FP32 adalah satu-satunya versi yang sudah terverifikasi
# 100% cocok dengan model PyTorch aslinya.
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# Opsi sesi ONNX Runtime yang menekan pemakaian RAM -- penting untuk
# hosting dengan RAM terbatas (mis. 512MB free tier). Trade-off:
# eksekusi sedikit lebih lambat, tapi puncak pemakaian RAM turun jauh.
# ─────────────────────────────────────────────────────────────
def _low_memory_session_options():
    opts = ort.SessionOptions()
    opts.enable_mem_pattern = False       # matikan pre-alokasi buffer besar
    opts.enable_cpu_mem_arena = False     # matikan arena allocator (lebih hemat, tapi lebih lambat)
    opts.intra_op_num_threads = 1         # 1 thread -- kurangi overhead memori per-thread
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    return opts


precheck_session = None
hybrid_session = None

try:
    print("⏳ Memuat model Precheck (ONNX)...")
    precheck_path = download_model_from_hf("best_precheck_model.onnx")
    if precheck_path:
        precheck_session = ort.InferenceSession(
            precheck_path, sess_options=_low_memory_session_options(),
            providers=["CPUExecutionProvider"],
        )
        print(f"💾 Sukses memuat model Precheck dari {precheck_path}")
    else:
        print("⚠️ Warning: best_precheck_model.onnx tidak ditemukan di HF repo. Precheck akan dilewati (semua gambar dianggap valid).")

    print("⏳ Memuat model Utama Hybrid (ONNX)...")
    hybrid_path = download_model_from_hf("hybrid_vit_efficientnet_brain_fp32.onnx")
    if hybrid_path:
        hybrid_session = ort.InferenceSession(
            hybrid_path, sess_options=_low_memory_session_options(),
            providers=["CPUExecutionProvider"],
        )
        print(f"💾 Sukses memuat model Classifier utama dari {hybrid_path}")
    else:
        print("⚠️ Warning: hybrid_vit_efficientnet_brain_fp32.onnx tidak ditemukan di HF repo.")

    print("✨ Proses pemuatan model AI selesai.")
except Exception as e:
    print(f"❌ Gagal memuat model AI: {str(e)}")


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
        "runtime": "onnxruntime (CPU)",
    }


# ─────────────────────────────────────────────────────────────
# Endpoint: data pasien
# ─────────────────────────────────────────────────────────────
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
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Format file harus berupa gambar (PNG, JPG, JPEG).")

    try:
        temp_file_path = os.path.join("temp_uploads", file.filename)
        os.makedirs("temp_uploads", exist_ok=True)
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        image = Image.open(temp_file_path).convert("RGB")

        # 1. TAHAP 1: Precheck (Menyaring Gambar Valid Brain Scan vs Gambar Noise/Invalid)
        is_valid = True
        precheck_prob_val = 0.99
        if precheck_session is not None:
            precheck_input = precheck_transforms(image)
            input_name = precheck_session.get_inputs()[0].name
            precheck_logits = precheck_session.run(None, {input_name: precheck_input})[0]
            precheck_prob = softmax(precheck_logits)[0]
            is_valid_idx = int(np.argmax(precheck_prob))
            precheck_prob_val = float(precheck_prob[is_valid_idx])
            # Indeks 1: Valid, Indeks 0: Invalid (Sesuai dengan dataset latihan precheck)
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

        # 2. TAHAP 2: Klasifikasi Utama (5 Kelas Penyakit Otak)
        if hybrid_session is None:
            raise HTTPException(status_code=500, detail="Model utama klasifikasi tidak termuat di server.")

        hybrid_input = val_transforms(image)
        input_name = hybrid_session.get_inputs()[0].name
        # Model hybrid punya 2 output: [logits, attention] (lihat reexport_onnx_with_attention.py)
        hybrid_logits = hybrid_session.run(None, {input_name: hybrid_input})[0]
        hybrid_prob = softmax(hybrid_logits)[0]

        predicted_idx = int(np.argmax(hybrid_prob))
        confidence_score = float(hybrid_prob[predicted_idx]) * 100
        predicted_class = CLASSES[predicted_idx]

        # 3. TAHAP 3: Eksplanabilitas AI (XAI) - Hasilkan Peta Atensi Heatmap
        heatmap_filename = f"heatmap_{os.path.splitext(file.filename)[0]}.png"
        generate_attention_heatmap(temp_file_path, save_name=heatmap_filename, onnx_session=hybrid_session)

        # 4. TAHAP 4: Kirim Hasil Ke Gemini / Laporan Lokal
        modality = "CT" if "ct" in file.filename.lower() else "MRI"
        report_text = generate_radiology_report(predicted_idx, confidence_score, modality)

        # 5. Encode gambar visualisasi heatmap dan gambar asli menjadi base64
        heatmap_path = os.path.join(OUTPUT_DIR, "figures", heatmap_filename)

        with open(heatmap_path, "rb") as img_file:
            heatmap_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        with open(temp_file_path, "rb") as img_file:
            original_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

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
                print(f"⚠️ Gagal menyimpan riwayat scan ke database: {str(db_err)}")

        return {
            "status": "Valid",
            "filename": file.filename,
            "modality_detected": modality,
            "prediction": {
                "class_name": predicted_class,
                "class_index": predicted_idx,
                "confidence": f"{confidence_score:.2f}%",
            },
            "radiology_report": report_text,
            "original_image_b64": f"data:image/png;base64,{original_base64}",
            "heatmap_image_b64": f"data:image/png;base64,{heatmap_base64}",
        }

    except Exception as e:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
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

        pdf.set_font("helvetica", "B", 14)
        pdf.cell(0, 8, "PUSAT RADIOLOGI DIGITAL & DIAGNOSTIK AI", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.set_font("helvetica", size=9)
        pdf.cell(0, 5, "Jl. Semilasari Barat No. 88, Sektor Kecerdasan Buatan, Denpasar", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.cell(0, 5, "Email: support@brainscan.ai | Telp: (021) 555-2026", new_x="LMARGIN", new_y="NEXT", align="C")

        pdf.ln(3)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 7, "DOKUMEN LAPORAN HASIL PEMERIKSAAN RADIOLOGI (OPINI AI)", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(4)

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
