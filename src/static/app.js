document.addEventListener('DOMContentLoaded', () => {
    const btnStart = document.getElementById("btn-start");
    const patientForm = document.getElementById("patient-form");
    const dashboardGrid = document.getElementById("dashboard-grid");
    const pipelineSection = document.getElementById("pipeline-section");
    const ablationSection= document.getElementById("ablation-section");

    // Buka Tutup/Buka Tombol Riwayat
    const openHistory =
        document.getElementById("open-history");

        const historyModal =
        document.getElementById("history-modal");

        const closeHistory =
        document.getElementById("close-history");

        openHistory?.addEventListener("click", () => {
            historyModal.style.display = "flex";
        });

        closeHistory?.addEventListener("click", () => {
            historyModal.style.display = "none";
        });

btnStart?.addEventListener("click", () => {
    const nama = document.getElementById("patient-name").value.trim();
    const umur = document.getElementById("patient-age").value.trim();
    const gender = document.getElementById("patient-gender").value;
    const nik = document.getElementById("patient-nik").value.trim();
    const birthDate = document.getElementById("tanggal_lahir").value;
    const address = document.getElementById("patient-alamat").value.trim();
    const phone = document.getElementById("patient-whatsaap").value.trim();

    if (!nama || !umur || !gender || !nik) {
        alert("Lengkapi data pasien (Nama, Umur, Jenis Kelamin, dan NIK) terlebih dahulu!");
        return;
    }

    // Simpan data pasien secara lokal
    localStorage.setItem("patient_name", nama);
    localStorage.setItem("patient_age", umur);
    localStorage.setItem("patient_gender", gender);
    localStorage.setItem("patient_nik", nik);
    localStorage.setItem("patient_birth_date", birthDate || "");
    localStorage.setItem("patient_address", address || "");
    localStorage.setItem("patient_phone", phone || "");

    // Kirim data pasien ke backend untuk disimpan di SQLite
    const patientData = {
        nik: nik,
        name: nama,
        age: parseInt(umur) || null,
        birth_date: birthDate || null,
        gender: gender || null,
        address: address || null,
        phone: phone || null
    };

    fetch('/api/patients/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(patientData)
    })
    .then(res => {
        if (!res.ok) throw new Error("Gagal mendaftarkan pasien di database.");
        return res.json();
    })
    .then(data => {
        console.log("Pasien berhasil disimpan ke database SQLite:", data);
    })
    .catch(err => {
        console.error("Gagal menyimpan pasien:", err);
    });

    // Sembunyikan form
    document.getElementById("patient-modal").style.display = "none";

    // Tampilkan dashboard scan
    dashboardGrid.style.display = "grid";
});

    // DOM Elements
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const btnUpload = document.getElementById('btn-upload');
    const filePreviewContainer = document.getElementById('file-preview-container');
    const fileNamePreview = document.getElementById('file-name-preview');
    const btnCancelFile = document.getElementById('btn-cancel-file');
    
    const placeholderVisual = document.getElementById('placeholder-visual');
    const visualContainer = document.getElementById('visual-container');
    const imgOriginal = document.getElementById('img-original');
    const imgHeatmap = document.getElementById('img-heatmap');
    
    const metricStatus = document.getElementById('metric-status');
    const metricDiagnosis = document.getElementById('metric-diagnosis');
    const metricConfidence = document.getElementById('metric-confidence');
    
    const chartConfVal = document.getElementById('chart-conf-val');
    const chartConfTooltip = document.getElementById('chart-conf-tooltip');
    
    const stepPrecheck = document.getElementById('step-precheck');
    const stepClassifier = document.getElementById('step-classifier');
    const stepReport = document.getElementById('step-report');
    
    const reportSection = document.getElementById('report-section');
    const reportTextContent = document.getElementById('report-text-content');
    const btnDownloadPdf = document.getElementById('btn-download-pdf');
    
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingTitle = document.getElementById('loading-title');
    const loadingDesc = document.getElementById('loading-desc');

    let selectedFile = null;

    // 1. Drag & Drop Handlers
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileSelection(files[0]);
        }
    });

    // 2. Click to Select File Handlers
    dropzone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });

    btnCancelFile.addEventListener('click', (e) => {
        e.stopPropagation(); // Stop click propagating to dropzone
        resetFileSelection();
    });

    function handleFileSelection(file) {
        // Validasi ekstensi
        const allowedExtensions = /(\.png|\.jpg|\.jpeg)$/i;
        if (!allowedExtensions.exec(file.name)) {
            alert('Format file salah! Harap pilih gambar dengan format PNG, JPG, atau JPEG.');
            return;
        }

        selectedFile = file;
        fileNamePreview.textContent = file.name;
        filePreviewContainer.style.display = 'flex';
        
        // Ganti button dropzone menjadi pemicu analisis
        btnUpload.textContent = 'MULAI ANALISIS DIAGNOSTIK';
        btnUpload.style.background = 'linear-gradient(135deg, #00dfd8, #0070f3)';
        btnUpload.style.color = '#0a0e1a';
    }

    function resetFileSelection() {
        selectedFile = null;
        fileInput.value = '';
        filePreviewContainer.style.display = 'none';
        btnUpload.textContent = 'UPLOAD & ANALISIS SEKARANG';
        btnUpload.removeAttribute('style');
    }

    // 3. Trigger Analysis Click
    btnUpload.addEventListener('click', (e) => {
        e.stopPropagation(); // Stop click propagating to dropzone
        
        if (!selectedFile) {
            fileInput.click();
            return;
        }

        runBrainAnalysis();
    });

    // 4. API Request and Pipeline Updates
    async function runBrainAnalysis() {
        showLoading('Tahap 1: Precheck...', 'Menyaring dan mengecek validitas orientasi gambar.');
        resetDashboardResults();

        const formData = new FormData();
        formData.append('file', selectedFile);
        
        const patientNik = localStorage.getItem("patient_nik");
        if (patientNik) {
            formData.append('patient_nik', patientNik);
        }

        try {
            // Jalankan pre-check step visual
            stepPrecheck.classList.add('active');
            
            // Simulasi sedikit waktu pemrosesan agar user merasakan alur pipeline berjalan
            await new Promise(r => setTimeout(r, 1000));

            const response = await fetch('/api/analyze/', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Terjadi kesalahan sistem analisis.');
            }

            const data = await response.json();
            
            if (data.status === 'Invalid') {
                // Skenario 1: Citra Bukan Scan Otak
                hideLoading();
                
                metricStatus.textContent = 'INVALID';
                metricStatus.className = 'value badge invalid';
                metricDiagnosis.textContent = 'Gambar Ditolak';
                metricConfidence.textContent = data.precheck_confidence || '99.00%';
                
                alert(`Analisis dihentikan: ${data.message}`);
                return;
            }

            // Skenario 2: Gambar Valid Scan Otak, Lanjut Klasifikasi
            showLoading('Tahap 2: Klasifikasi Utama...', 'Mengekstraksi fitur CNN-Transformer dan memetakan pola patologis.');
            stepClassifier.classList.add('active');
            await new Promise(r => setTimeout(r, 1200));

            // Perbarui Metrik Utama
            metricStatus.textContent = 'VALID';
            metricStatus.className = 'value badge valid';
            
            // Format nama diagnosis agar lebih rapi
            const diagnosisMap = {
                "Alzheimer": "Alzheimer / Atrofi Serebral",
                "Intracranial_Hemorrhage": "Pendarahan Intrakranial (ICH)",
                "Normal": "Otak Normal (No Anomalies)",
                "Stroke_Iskemik": "Stroke Iskemik (Infark)",
                "Tumor": "Tumor Serebral (Neoplasma)"
            };
            const displayDiagnosis = diagnosisMap[data.prediction.class_name] || data.prediction.class_name;
            metricDiagnosis.textContent = displayDiagnosis;
            metricConfidence.textContent = data.prediction.confidence;

            const nik = localStorage.getItem("patient_nik");
            const historyData =
                JSON.parse(localStorage.getItem("patient_history")) || [];

            historyData.push({
                nik: nik,
                nama: localStorage.getItem("patient_name"),
                diagnosis: displayDiagnosis,
                confidence: data.prediction.confidence,
                tanggal: new Date().toLocaleString("id-ID")
            });

            localStorage.setItem(
                "patient_history",
                JSON.stringify(historyData)
            );

            // Perbarui Chart Ketinggian Confidence (Hingga max 100%)
            const confFloat = parseFloat(data.prediction.confidence);
            chartConfVal.style.height = `${confFloat}%`;
            chartConfTooltip.textContent = `${confFloat.toFixed(1)}%`;
            chartConfTooltip.style.opacity = '1';

            // Tampilkan Gambar Asli vs Heatmap
            imgOriginal.src = data.original_image_b64;
            imgHeatmap.src = data.heatmap_image_b64;
            placeholderVisual.style.display = 'none';
            visualContainer.style.display = 'flex';

            // Skenario 3: Membuka Laporan Radiologi AI
            showLoading('Tahap 3: Laporan Gemini AI...', 'Menyusun laporan diagnostik deskriptif radiologi.');
            stepReport.classList.add('active');
            await new Promise(r => setTimeout(r, 1000));

            // Tampilkan laporan teks
            reportTextContent.textContent = data.radiology_report;
            reportSection.style.display = 'block';

            // Scroll perlahan ke hasil analisis
            reportSection.scrollIntoView({ behavior: 'smooth' });

        } catch (error) {
            console.error(error);
            alert(`Gagal menganalisis gambar: ${error.message}`);
            resetPipelineSteps();
        } finally {
            hideLoading();
        }
    }

    // 5. PDF Generation via Backend (Direct Data Passing)
    btnDownloadPdf.addEventListener('click', async () => {
        if (!selectedFile) return;

        showLoading('Mengunduh PDF...', 'Menyiapkan berkas laporan radiologi resmi.');
        
        // Wait 100ms to ensure the visual loading overlay is fully rendered by the browser
        await new Promise(resolve => setTimeout(resolve, 100));

        const textContainer = document.getElementById('report-text-content');
        if (!textContainer || !textContainer.textContent.trim()) {
            hideLoading();
            alert('Teks laporan radiologi belum terisi. Harap jalankan analisis kembali.');
            return;
        }

        const payload = {
            patient_name: localStorage.getItem("patient_name") || "Pasien Anonim",
            patient_age: localStorage.getItem("patient_age") || "-",
            patient_gender: localStorage.getItem("patient_gender") || "-",
            patient_nik: localStorage.getItem("patient_nik") || "-",
            patient_birth_date: localStorage.getItem("patient_birth_date") || "-",
            patient_address: localStorage.getItem("patient_address") || "-",
            patient_phone: localStorage.getItem("patient_phone") || "-",
            report_text: textContainer.textContent
        };

        try {
            const response = await fetch('/api/download-pdf/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error('Gagal mengunduh file PDF dari server.');
            }

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = 'Laporan_Radiologi_BrainScan.pdf';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } catch (err) {
            console.error('Gagal mengunduh PDF:', err);
            alert('Gagal menghasilkan PDF dari server: ' + err.message);
        } finally {
            hideLoading();
        }
    });

    // Helper functions
    function showLoading(title, desc) {
        loadingTitle.textContent = title;
        loadingDesc.textContent = desc;
        loadingOverlay.style.display = 'flex';
    }

    function hideLoading() {
        loadingOverlay.style.display = 'none';
    }

    function resetPipelineSteps() {
        stepPrecheck.classList.remove('active');
        stepClassifier.classList.remove('active');
        stepReport.classList.remove('active');
    }

    function resetDashboardResults() {
        resetPipelineSteps();
        metricStatus.textContent = 'MENUNGGU';
        metricStatus.className = 'value badge';
        metricDiagnosis.textContent = '-';
        metricConfidence.textContent = '-';
        chartConfVal.style.height = '0%';
        chartConfTooltip.style.opacity = '0';
        
        placeholderVisual.style.display = 'flex';
        visualContainer.style.display = 'none';
        reportSection.style.display = 'none';
        imgOriginal.src = '';
        imgHeatmap.src = '';
        reportTextContent.textContent = '';
    }

    // 6. Search Patient History from DB
    const btnSearchHistory = document.getElementById("btn-search-history");
    const searchNikInput = document.getElementById("search-nik");
    const historyResultDiv = document.getElementById("history-result");

    btnSearchHistory?.addEventListener("click", async () => {
        const queryNik = searchNikInput.value.trim();
        if (!queryNik) {
            alert("Masukkan NIK terlebih dahulu!");
            return;
        }

        historyResultDiv.innerHTML = "<div class='loading-small'>Mencari data...</div>";

        try {
            const res = await fetch(`/api/patients/${queryNik}/history`);
            if (!res.ok) {
                throw new Error("Gagal mencari riwayat pasien.");
            }
            const data = await res.json();
            
            if (!data.history || data.history.length === 0) {
                historyResultDiv.innerHTML = "<p class='no-history'>Tidak ada riwayat pemeriksaan untuk NIK ini.</p>";
                return;
            }

            let htmlContent = `
                <div class="patient-info-header" style="margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid rgba(255,255,255,0.1);">
                    <h4 style="color: #00dfd8; margin: 0;">Profil Pasien: ${data.history[0].name}</h4>
                    <p style="margin: 5px 0 0 0; font-size: 13px; color: #a0aec0;">NIK: ${data.history[0].patient_nik} | ${data.history[0].gender} | ${data.history[0].age} Thn</p>
                </div>
                <div class="history-list" style="max-height: 300px; overflow-y: auto;">
            `;

            data.history.forEach((scan, index) => {
                const diagMap = {
                    "Alzheimer": "Alzheimer / Atrofi Serebral",
                    "Intracranial_Hemorrhage": "Pendarahan Intrakranial (ICH)",
                    "Normal": "Otak Normal (No Anomalies)",
                    "Stroke_Iskemik": "Stroke Iskemik (Infark)",
                    "Tumor": "Tumor Serebral (Neoplasma)"
                };
                const displayDiag = diagMap[scan.predicted_class] || scan.predicted_class;
                const formattedDate = new Date(scan.created_at).toLocaleString("id-ID");

                htmlContent += `
                    <div class="history-item" style="background: rgba(255,255,255,0.03); padding: 12px; margin-bottom: 10px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.05); display: flex; flex-direction: column; gap: 8px;">
                        <div class="history-meta" style="display: flex; justify-content: space-between; font-size: 11px; color: #a0aec0;">
                            <span class="history-date">${formattedDate}</span>
                            <span class="history-modality" style="background: rgba(0,223,216,0.1); color: #00dfd8; padding: 2px 6px; border-radius: 4px; font-weight: bold;">${scan.modality}</span>
                        </div>
                        <div class="history-diagnosis" style="display: flex; justify-content: space-between; align-items: center;">
                            <strong style="color: #fff; font-size: 14px;">${displayDiag}</strong>
                            <span class="history-conf" style="color: #0070f3; font-size: 12px; font-weight: 500;">Akurasi: ${scan.confidence.toFixed(2)}%</span>
                        </div>
                        <button class="btn-load-past-report" onclick="window.loadPastScanReport(${index})" style="background: #0070f3; color: white; border: none; padding: 6px 12px; border-radius: 4px; font-size: 12px; cursor: pointer; align-self: flex-end; transition: background 0.2s;">Lihat Laporan</button>
                    </div>
                `;
            });

            htmlContent += `</div>`;
            historyResultDiv.innerHTML = htmlContent;

            // Simpan data history pencarian secara global untuk diload jika diklik
            window.lastSearchResults = data.history;

        } catch (err) {
            historyResultDiv.innerHTML = `<p class='error-msg' style='color: #ff3366;'>Error: ${err.message}</p>`;
        }
    });

    // Handler global untuk me-load laporan masa lalu
    window.loadPastScanReport = (index) => {
        const scan = window.lastSearchResults?.[index];
        if (!scan) return;

        // Simpan data pasien secara lokal saat me-load riwayat masa lalu
        localStorage.setItem("patient_name", scan.name || "Pasien Anonim");
        localStorage.setItem("patient_age", scan.age || "-");
        localStorage.setItem("patient_gender", scan.gender || "-");
        localStorage.setItem("patient_nik", scan.patient_nik || "-");
        localStorage.setItem("patient_birth_date", scan.birth_date || "");
        localStorage.setItem("patient_address", scan.address || "");
        localStorage.setItem("patient_phone", scan.phone || "");

        // Tutup modal riwayat
        if (historyModal) historyModal.style.display = "none";

        // Tampilkan dashboard scan
        dashboardGrid.style.display = "grid";

        // Set metrik utama
        metricStatus.textContent = 'VALID (RIWAYAT)';
        metricStatus.className = 'value badge valid';
        
        const diagnosisMap = {
            "Alzheimer": "Alzheimer / Atrofi Serebral",
            "Intracranial_Hemorrhage": "Pendarahan Intrakranial (ICH)",
            "Normal": "Otak Normal (No Anomalies)",
            "Stroke_Iskemik": "Stroke Iskemik (Infark)",
            "Tumor": "Tumor Serebral (Neoplasma)"
        };
        const displayDiagnosis = diagnosisMap[scan.predicted_class] || scan.predicted_class;
        metricDiagnosis.textContent = displayDiagnosis;
        metricConfidence.textContent = `${scan.confidence.toFixed(2)}%`;

        // Perbarui Chart
        chartConfVal.style.height = `${scan.confidence}%`;
        chartConfTooltip.textContent = `${scan.confidence.toFixed(1)}%`;
        chartConfTooltip.style.opacity = '1';

        // Tampilkan Gambar Asli vs Heatmap
        imgOriginal.src = scan.original_image_b64;
        imgHeatmap.src = scan.heatmap_image_b64;
        placeholderVisual.style.display = 'none';
        visualContainer.style.display = 'flex';

        // Tampilkan laporan teks
        reportTextContent.textContent = scan.radiology_report;
        reportSection.style.display = 'block';

        // Scroll
        reportSection.scrollIntoView({ behavior: 'smooth' });
    };
});


