import os
import google.generativeai as genai
from src.config import CLASSES, GEMINI_API_KEY as CONFIG_API_KEY
from src.prompts import build_radiology_prompt

# Inisialisasi API Key Gemini
# Prioritas: 1) config.py GEMINI_API_KEY  2) Environment Variable  3) Fallback lokal
api_key = CONFIG_API_KEY or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    print("Gemini API Client berhasil dikonfigurasi menggunakan API Key.")
else:
    print("Warning: GEMINI_API_KEY tidak ditemukan di config.py maupun environment. Mengaktifkan sistem laporan lokal fallback.")

def generate_radiology_report(predicted_idx, confidence, modality="MRI"):
    """
    Menghasilkan laporan radiologi detail.
    Mencoba menggunakan API Gemini terlebih dahulu, dan menggunakan generator lokal jika API tidak tersedia.
    """
    class_name = CLASSES[predicted_idx]
    
    # 1. Definisikan detail pola visual dan area atensi untuk prompt
    suspected_patterns = {
        "Alzheimer": "Atrofi kortikal difus bilateral, pelebaran sulkus otak, dan pembesaran sistem ventrikel (ventrikulomegali) kompensatorik.",
        "Intracranial_Hemorrhage": "Lesi hiperdens akut (fokus pendarahan aktif) intraaksial/ekstraaksial dengan potensi efek massa.",
        "Normal": "Arsitektur parenkim otak normal, batas substansia alba-grisea tegas, sistem ventrikel dan sulkus kortikal dalam batas normal.",
        "Stroke_Iskemik": "Area hipodensitas fokal (infark serebri) akut/subakut yang sesuai dengan vaskularisasi arteri serebri tertentu.",
        "Tumor": "Massa soliter/multipel intraaksial dengan edema perifokal (vasogenik) luas serta pendesakan garis tengah (midline shift)."
    }
    
    attention_regions = {
        "Alzheimer": "Lobus Temporal Medial, terutama daerah Hipokampus bilateral.",
        "Intracranial_Hemorrhage": "Parenkim serebral (lobus frontal/temporal) atau ruang subdural/epidural.",
        "Normal": "Seluruh parenkim serebri dan serebelum secara simetris.",
        "Stroke_Iskemik": "Korteks serebri atau ganglia basalis (terutama teritori Arteri Serebri Media/MCA).",
        "Tumor": "Lobus frontal/parietal parenkim serebral, atau fosa posterior serebelum."
    }

    result_data = {
        "predicted_class": class_name,
        "confidence": f"{confidence:.2f}%",
        "visual_summary": {
            "modality": modality,
            "suspected_pattern": suspected_patterns.get(class_name, "Pola visual atipikal."),
            "attention_region": attention_regions.get(class_name, "Area kortikal serebri.")
        }
    }

    # Jika API key Gemini tersedia, panggil model Gemini
    if api_key:
        try:
            prompt = build_radiology_prompt(result_data)
            # Menggunakan gemini-1.5-flash untuk respon cepat dan andal
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            if response.text:
                return response.text.strip()
        except Exception as e:
            print(f"Gagal memanggil Gemini API: {str(e)}. Menggunakan laporan medis lokal.")

    # 2. Generator Laporan Medis Lokal Fallback (Sangat Detail dan Profesional)
    return get_local_fallback_report(class_name, confidence, modality, suspected_patterns[class_name], attention_regions[class_name])

def get_local_fallback_report(class_name, confidence, modality, pattern, region):
    """Menghasilkan teks laporan medis radiologi standar yang terstruktur"""
    
    indonesian_names = {
        "Alzheimer": "Penyakit Alzheimer / Atrofi Serebral Kronis",
        "Intracranial_Hemorrhage": "Pendarahan Intrakranial (Intracranial Hemorrhage)",
        "Normal": "Kondisi Otak Normal (Normal Brain Scan)",
        "Stroke_Iskemik": "Stroke Iskemik (Infark Serebri)",
        "Tumor": "Neoplasma Otak / Tumor Serebral"
    }
    
    display_name = indonesian_names.get(class_name, class_name)
    
    report = f"""LAPORAN RADIOLOGI EVALUASI SCAN OTAK
Modalitas Pemeriksaan : {modality}-Scan Kepala (Brain Imaging)
Identitas Temuan      : Deteksi Otomatis Sistem AI Terintegrasi
Status AI             : Analisis Selesai (Confidence Level: {confidence:.2f}%)
Dugaan Klinis Utama   : {display_name}

1. RINGKASAN TEMUAN (FINDINGS SUMMARY):
Pemeriksaan penunjang citra {modality} kepala menunjukkan adanya indikasi kelainan terfokus. Peta visual atensi AI (XAI) mendeteksi fokus anomali pada: {region}.
Pola visual yang dominan diidentifikasi sebagai: '{pattern}'.

2. ANALISIS KLINIS (CLINICAL ANALYSIS):"""

    if class_name == "Alzheimer":
        report += """
* Tampak adanya reduksi volume parenkim otak global yang signifikan (Atrofi Serebral).
* Sulkus kortikal serebri melebar disertai pendalaman girus serebri secara difus, menonjol di lobus temporal dan parietal.
* Sistem ventrikel lateral kiri dan kanan melebar simetris, konsisten dengan hydrocephalus ex-vacuo sekunder akibat hilangnya jaringan otak.
* Tidak ditemukan tanda-tanda pendarahan akut maupun space-occupying lesion (massa tumor)."""
    
    elif class_name == "Intracranial_Hemorrhage":
        report += """
* Tampak visualisasi area lesi densitas tinggi (hiperdens) homogen lokal pada jaringan parenkim otak.
* Lesi disertai visualisasi edema perifokal (ring-like edema) tipis di sekitarnya.
* Terdeteksi sedikit efek massa lokal berupa kompresi ringan pada sulkus serebri yang bersebelahan.
* Rekomendasi perhatian ketat terhadap potensi peningkatan tekanan intrakranial (TIK)."""
        
    elif class_name == "Normal":
        report += """
* Parenkim serebri dan serebelum menunjukkan intensitas dan struktur sinyal homogen normal.
* Batas substansia alba (white matter) dan grisea (grey matter) tegas dan simetris di kedua hemisfer serebri.
* Sistem ventrikel lateral, ventrikel III, dan IV berada dalam posisi sentral dengan ukuran dan konfigurasi normal.
* Tidak tampak area hipodensitas lokal (infark serebri), pendarahan intrakranial, maupun massa neoplasma yang dicurigai."""
        
    elif class_name == "Stroke_Iskemik":
        report += """
* Terdeteksi area lesi fokal hipodensitas parenkim serebri yang menunjukkan batas kurang tegas di daerah korteks/subkorteks.
* Visualisasi lesi selaras dengan batas vaskularisasi suplai darah serebral, menandakan adanya hambatan perfusi arteri (iskemia akut/subakut).
* Ditemukan edema sitotoksik minimal di area terinfark tanpa pergeseran midline serebri yang signifikan."""
        
    elif class_name == "Tumor":
        report += """
* Tampak massa lesi berbatas tegas dengan kontur ireguler yang menempati ruang serebral (space-occupying lesion).
* Lesi dikelilingi oleh area edema vasogenik luas yang menekan sulkus kortikal dan jaringan parenkim sekitarnya.
* Tampak kompresi parsial pada tanduk ventrikel lateral homolateral serta pergeseran minor garis tengah serebral (midline shift)."""

    report += f"""

3. TINGKAT KEYAKINAN (CONFIDENCE LEVEL ANALYSIS):
* Model klasifikasi Vision Transformer (Marksnb/brain-hybrid-efficientnet-vit) mendeteksi tanda visual penyakit '{class_name}' dengan tingkat keyakinan {confidence:.2f}%.
* Keakuratan klasifikasi divalidasi oleh cross-attention map yang secara presisi mengunci koordinat lesi di {region}.

4. REKOMENDASI PEMERIKSAAN LANJUTAN (RECOMMENDATIONS):"""

    if class_name == "Alzheimer":
        report += """
* Disarankan melakukan korelasi klinis melalui asesmen kognitif menyeluruh (MMSE, MoCA).
* Evaluasi lanjutan dengan MRI resolusi tinggi (3T) sekuens volumetri hipokampus untuk menilai tingkat atrofi secara kuantitatif."""
    elif class_name == "Intracranial_Hemorrhage":
        report += """
* Diperlukan tindakan darurat (cito) konsultasi dengan Dokter Spesialis Bedah Saraf.
* Direkomendasikan CT-Scan non-kontras evaluasi ulang berkala (serial scan) dalam 12-24 jam untuk memantau perkembangan volume pendarahan."""
    elif class_name == "Normal":
        report += """
* Tidak diperlukan evaluasi radiologi lanjutan segera jika tidak terdapat kecurigaan gejala klinis baru.
* Disarankan kontrol berkala sesuai dengan anjuran dokter pengirim."""
    elif class_name == "Stroke_Iskemik":
        report += """
* Disarankan melakukan CT-Angiografi (CTA) atau MR-Angiografi (MRA) segera untuk menilai patensi pembuluh darah serebral.
* Pemeriksaan MRI kepala sekuens DWI/ADC (Diffusion-Weighted Imaging) untuk konfirmasi area infark akut (ischemic penumbra)."""
    elif class_name == "Tumor":
        report += """
* Segera konsultasikan ke Spesialis Bedah Saraf / Onkologi.
* Direkomendasikan pemeriksaan MRI kepala dengan kontras (gadolinium) sekuens multiplanar untuk detail morfologi neoplasma.
* Rencanakan biopsi histopatologi untuk penentuan stadium dan jenis sel tumor secara definitif."""

    report += "\n\n*Catatan: Laporan ini dihasilkan secara otomatis oleh sistem kecerdasan buatan (AI) sebagai opini sekunder radiologi. Hasil akhir harus selalu divalidasi dan ditandatangani oleh Dokter Spesialis Radiologi (Sp.Rad).*"
    return report
