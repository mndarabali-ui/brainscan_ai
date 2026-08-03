import os
import torch
import datetime
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import shutil
import base64
from pydantic import BaseModel

from src.config import CLASSES, OUTPUT_DIR, CHECKPOINT_DIR, download_model_from_hf
from src.preprocess import val_transforms, precheck_transforms
from src.models.precheck_model import BrainPreCheckModel
from src.models.classifier_model import BrainHybridModel
from src.gemini_client import generate_radiology_report
from src.explainability import generate_attention_heatmap
from src.database import upsert_patient, get_patient, add_scan_record, get_patient_history

# Inisialisasi Aplikasi FastAPI
app = FastAPI(
    title="BrainScan AI Framework API",
    description="API untuk analisis otomatis CT-Scan & MRI menggunakan arsitektur Hybrid CNN-Transformer",
    version="1.0"
)

# Model Data Pydantic untuk Input Pasien
class PatientCreate(BaseModel):
    nik: str
    name: str
    age: int = None
    birth_date: str = None
    gender: str = None
    address: str = None
    phone: str = None


# Aktifkan CORS agar frontend dapat berkomunikasi dengan lancar
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Atur perangkat keras (GPU jika ada, jika tidak gunakan CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_HARI_ID  = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
_BULAN_ID = ["", "Januari", "Februari", "Maret", "April", "Mei", "Juni",
             "Juli", "Agustus", "September", "Oktober", "November", "Desember"]


def tanggal_indonesia_sekarang() -> str:
    """Format tanggal sekarang, misal: 'Selasa, 7 Juli 2026' — bukan tanggal tetap."""
    now = datetime.datetime.now()
    return f"{_HARI_ID[now.weekday()]}, {now.day} {_BULAN_ID[now.month]} {now.year}"

# Buat folder output yang diperlukan
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)
os.makedirs("temp_uploads", exist_ok=True)

# Muat model-model AI secara global pada startup server
precheck_model = None
hybrid_model = None

try:
    print("⏳ Memuat model Precheck...")
    precheck_model = BrainPreCheckModel().to(device)
    precheck_checkpoint = download_model_from_hf("best_precheck_model.onnx") or os.path.join(CHECKPOINT_DIR, "best_precheck_model.onnx")
    if os.path.exists(precheck_checkpoint):
        try:
            precheck_model.load_state_dict(torch.load(precheck_checkpoint, map_location=device))
            print(f"Sukses memuat bobot model Precheck dari {precheck_checkpoint}")
        except RuntimeError as e:
            print(f"Warning: Checkpoint precheck tidak kompatibel dengan arsitektur EfficientNet-B0 baru, menggunakan bobot pretrained bawaan.")
    else:
        print("Warning: best_precheck_model.onnx tidak ditemukan, menggunakan bobot pretrained bawaan.")
    precheck_model.eval()

    print("Memuat model Utama Hybrid...")
    hybrid_model = BrainHybridModel().to(device)
    hybrid_checkpoint = download_model_from_hf("hybrid_vit_efficientnet_brain_fp32.onnx") or os.path.join(CHECKPOINT_DIR, "hybrid_vit_efficientnet_brain_fp32.onnx")
    if os.path.exists(hybrid_checkpoint):
        try:
            ckpt = torch.load(hybrid_checkpoint, map_location=device)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                hybrid_model.load_state_dict(ckpt["model_state_dict"])
            else:
                hybrid_model.load_state_dict(ckpt)
            print(f"Sukses memuat bobot model Classifier utama dari {hybrid_checkpoint}")
        except RuntimeError as e:
            print(f"Warning: Checkpoint classifier tidak kompatibel dengan arsitektur baru: {str(e)}. Menggunakan bobot pretrained bawaan.")
    else:
        print("Warning: hybrid_vit_efficientnet_brain_fp32.onnx tidak ditemukan, menggunakan bobot pretrained bawaan.")
    hybrid_model.eval()
    print("Seluruh model AI berhasil dimuat.")
except Exception as e:
    print(f"Gagal memuat model AI: {str(e)}")

@app.get("/api/status")
def get_status():
    """Mengecek status online server dan ketersediaan model AI"""
    return {
        "status": "Online",
        "precheck_model_loaded": precheck_model is not None,
        "classifier_model_loaded": hybrid_model is not None,
        "device": str(device)
    }

@app.post("/api/patients/")
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
            phone=patient.phone
        )
        return {"status": "Success", "message": "Data pasien berhasil disimpan."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menyimpan data pasien: {str(e)}")

@app.get("/api/patients/{nik}")
def get_patient_info(nik: str):
    """Mengambil data pasien berdasarkan NIK"""
    patient = get_patient(nik)
    if not patient:
        raise HTTPException(status_code=404, detail="Pasien tidak ditemukan.")
    return {"status": "Success", "patient": patient}

@app.get("/api/patients/{nik}/history")
def get_patient_scans_history(nik: str):
    """Mengambil riwayat scan pasien berdasarkan NIK"""
    try:
        history = get_patient_history(nik)
        return {"status": "Success", "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal mengambil riwayat scan: {str(e)}")

@app.post("/api/analyze/")
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
        temp_file_path = os.path.join("temp_uploads", file.filename)
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 3. Baca gambar untuk pemrosesan tensor PyTorch
        image = Image.open(temp_file_path).convert("RGB")
        # Dua tensor terpisah: precheck pakai normalisasi [0.5,0.5,0.5]
        # (sesuai cara dia dilatih), hybrid pakai normalisasi ImageNet
        # (sesuai cara model utama dilatih di notebook)
        precheck_tensor = precheck_transforms(image).unsqueeze(0).to(device)
        tensor_image = val_transforms(image).unsqueeze(0).to(device)
        
        # 4. TAHAP 1: Precheck (Menyaring Gambar Valid Brain Scan vs Gambar Noise/Invalid)
        is_valid = True
        precheck_prob_val = 0.99
        if precheck_model is not None:
            with torch.no_grad():
                precheck_outputs = precheck_model(precheck_tensor)
                precheck_prob = F.softmax(precheck_outputs, dim=1)
                is_valid_idx = torch.argmax(precheck_prob, dim=1).item()
                precheck_prob_val = precheck_prob[0][is_valid_idx].item()
                # Indeks 1: Valid, Indeks 0: Invalid (Sesuai dengan dataset latihan precheck)
                is_valid = (is_valid_idx == 1)

        # Jika gambar dinyatakan invalid, hentikan proses analisis awal
        if not is_valid:
            # Hapus file sementara
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            return {
                "status": "Invalid",
                "filename": file.filename,
                "message": "Gambar tidak dikenali sebagai scan otak yang valid (CT-Scan/MRI). Hubungi Administrator.",
                "precheck_confidence": f"{precheck_prob_val * 100:.2f}%"
            }

        # 5. TAHAP 2: Klasifikasi Utama (5 Kelas Penyakit Otak)
        if hybrid_model is None:
            raise HTTPException(status_code=500, detail="Model utama klasifikasi tidak termuat di server.")
            
        with torch.no_grad():
            hybrid_outputs = hybrid_model(tensor_image)
            hybrid_prob = F.softmax(hybrid_outputs, dim=1)
            confidence, predicted_idx = torch.max(hybrid_prob, dim=1)
            
            confidence_score = confidence.item() * 100
            predicted_class = CLASSES[predicted_idx.item()]
            
        # 6. TAHAP 3: Eksplanabilitas AI (XAI) - Hasilkan Peta Atensi Heatmap
        heatmap_filename = f"heatmap_{os.path.splitext(file.filename)[0]}.png"
        generate_attention_heatmap(temp_file_path, save_name=heatmap_filename)
        
        # 7. TAHAP 4: Kirim Hasil Ke Gemini / Laporan Lokal
        modality = "CT" if "ct" in file.filename.lower() else "MRI"
        report_text = generate_radiology_report(predicted_idx.item(), confidence_score, modality)
        
        # 8. Encode gambar visualisasi heatmap dan gambar asli menjadi base64 untuk dikirim langsung ke frontend
        # Ini mencegah isu caching browser pada pemuatan statis
        heatmap_path = os.path.join(OUTPUT_DIR, "figures", heatmap_filename)
        
        # Baca visualisasi heatmap
        with open(heatmap_path, "rb") as img_file:
            heatmap_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
        # Baca gambar asli
        with open(temp_file_path, "rb") as img_file:
            original_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        # Hapus file sementara setelah diproses
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        # Simpan ke database jika patient_nik tersedia
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
                    heatmap_b64=f"data:image/png;base64,{heatmap_base64}"
                )
            except Exception as db_err:
                print(f"Gagal menyimpan riwayat scan ke database: {str(db_err)}")

        # 9. Kembalikan respons akhir dalam format JSON
        return {
            "status": "Valid",
            "filename": file.filename,
            "modality_detected": modality,
            "prediction": {
                "class_name": predicted_class,
                "class_index": predicted_idx.item(),
                "confidence": f"{confidence_score:.2f}%"
            },
            "radiology_report": report_text,
            "original_image_b64": f"data:image/png;base64,{original_base64}",
            "heatmap_image_b64": f"data:image/png;base64,{heatmap_base64}"
        }
        
    except Exception as e:
        # Bersihkan jika ada file sementara tersisa
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan internal analisis: {str(e)}")


from fastapi.responses import StreamingResponse
from fpdf import FPDF

class PDFDownloadRequest(BaseModel):
    patient_name: str
    patient_age: str
    patient_gender: str
    patient_nik: str
    patient_birth_date: str = ""
    patient_address: str = ""
    patient_phone: str = ""
    report_text: str

@app.post("/api/download-pdf/")
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
        
        # Line divider
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
        
        # Create key-value table
        details = [
            ("Nama Pasien", data.patient_name, "Jenis Kelamin", data.patient_gender),
            ("Umur", f"{data.patient_age} Tahun", "Tanggal Lahir", data.patient_birth_date),
            ("NIK Pasien", data.patient_nik, "No. Telepon", data.patient_phone),
            ("Alamat", data.patient_address, "Tanggal Analisis", tanggal_indonesia_sekarang())
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
            if stripped.startswith("1. ") or stripped.startswith("2. ") or stripped.startswith("3. ") or stripped.startswith("4. "):
                pdf.ln(2)
                pdf.set_font("helvetica", "B", 10)
                pdf.multi_cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("helvetica", "", 9.5)
            elif stripped.startswith("* ") or stripped.startswith("- "):
                pdf.set_font("helvetica", "", 9.5)
                pdf.set_x(15)
                pdf.multi_cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
            elif stripped.startswith("*Catatan:") or stripped.startswith("Catatan:"):
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
            headers={"Content-Disposition": "attachment; filename=Laporan_Radiologi_BrainScan.pdf"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses PDF: {str(e)}")

# Mount folder figures sebagai static files agar bisa diakses (opsional fallback)
app.mount("/outputs/figures", StaticFiles(directory=os.path.join(OUTPUT_DIR, "figures")), name="figures")

# Serve file static frontend secara langsung
# html=True akan menyajikan index.html secara default jika rute / dipanggil
app.mount("/", StaticFiles(directory="src/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # Jalankan server (jalankan dari root proyek: `python -m src.main`
    # atau `uvicorn src.main:app --reload` dari folder root, BUKAN dari dalam folder src/)
    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=True)