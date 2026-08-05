"""
database.py
------------
Database dipisah dari backend (Render) — pakai Turso, bukan file SQLite
lokal. Alasan: Render (seperti kebanyakan platform hosting compute)
tidak menyimpan data secara permanen di disk lokalnya — hilang tiap
kali service restart/redeploy.

Turso dipilih karena protokolnya SQLite-compatible, jadi SEMUA query SQL
di bawah ini (CREATE TABLE, INSERT, SELECT, ON CONFLICT) sama persis
dengan versi sqlite3 biasa — cuma cara membuka koneksinya yang beda.

Environment variable yang WAJIB di-set di Render (Settings > Environment):
    TURSO_DATABASE_URL   -> contoh: libsql://nama-db-kamu.turso.io
    TURSO_AUTH_TOKEN     -> token dari `turso db tokens create nama-db-kamu`
"""

import os
import libsql_client

TURSO_DATABASE_URL = os.environ.get("TURSO_DATABASE_URL", "")
TURSO_AUTH_TOKEN = os.environ.get("TURSO_AUTH_TOKEN", "")

_client = None


def get_client():
    """Membuat (atau reuse) satu koneksi client ke Turso. Dibuat sekali
    dan dipakai berulang (bukan buka-tutup tiap query), lebih efisien
    untuk koneksi jaringan (HTTP) dibanding sqlite3.connect() lokal."""
    global _client
    if _client is None:
        if not TURSO_DATABASE_URL or not TURSO_AUTH_TOKEN:
            raise RuntimeError(
                "TURSO_DATABASE_URL / TURSO_AUTH_TOKEN belum di-set. "
                "Set sebagai environment variable di Render Settings > Environment."
            )
        _client = libsql_client.create_client_sync(
            url=TURSO_DATABASE_URL,
            auth_token=TURSO_AUTH_TOKEN,
        )
    return _client


def _rows_to_dicts(result_set):
    """Ubah hasil query jadi list of dict, biar cara pakainya di api.py
    tetap sama seperti waktu masih sqlite3.Row."""
    cols = result_set.columns
    return [dict(zip(cols, row)) for row in result_set.rows]


def init_db():
    """Menginisialisasi tabel-tabel di Turso jika belum ada (idempotent,
    aman dipanggil berkali-kali setiap kali server start)."""
    client = get_client()

    client.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        nik TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        age INTEGER,
        birth_date TEXT,
        gender TEXT,
        address TEXT,
        phone TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    client.execute("""
    CREATE TABLE IF NOT EXISTS scans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_nik TEXT NOT NULL,
        filename TEXT,
        modality TEXT,
        predicted_class TEXT NOT NULL,
        confidence REAL NOT NULL,
        radiology_report TEXT,
        original_image_b64 TEXT,
        heatmap_image_b64 TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (patient_nik) REFERENCES patients(nik) ON DELETE CASCADE
    )
    """)

    client.execute("""
    CREATE TABLE IF NOT EXISTS dataset_distribution (
        kelas TEXT PRIMARY KEY,
        sebelum_balancing INTEGER,
        setelah_balancing INTEGER
    )
    """)

    client.execute("""
    CREATE TABLE IF NOT EXISTS training_history (
        epoch INTEGER PRIMARY KEY,
        train_loss REAL,
        val_loss REAL,
        train_acc REAL,
        val_acc REAL,
        train_f1 REAL,
        val_f1 REAL,
        epoch_time_seconds REAL
    )
    """)

    client.execute("""
    CREATE TABLE IF NOT EXISTS test_evaluation_results (
        actual_class TEXT,
        predicted_class TEXT,
        count INTEGER,
        PRIMARY KEY (actual_class, predicted_class)
    )
    """)

    print(f"Database Turso berhasil diinisialisasi: {TURSO_DATABASE_URL}")


def save_dataset_distribution_to_db(df):
    client = get_client()
    client.execute("DELETE FROM dataset_distribution")
    for _, row in df.iterrows():
        client.execute(
            "INSERT INTO dataset_distribution (kelas, sebelum_balancing, setelah_balancing) VALUES (?, ?, ?)",
            [row["Kelas"], int(row["Sebelum_Balancing"]), int(row["Setelah_Balancing"])],
        )


def save_training_history_to_db(df):
    client = get_client()
    client.execute("DELETE FROM training_history")
    for _, row in df.iterrows():
        client.execute(
            """INSERT INTO training_history
               (epoch, train_loss, val_loss, train_acc, val_acc, train_f1, val_f1, epoch_time_seconds)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [int(row["Epoch"]), float(row["Train_Loss"]), float(row["Val_Loss"]),
             float(row["Train_Acc"]), float(row["Val_Acc"]), float(row["Train_F1"]),
             float(row["Val_F1"]), float(row["Epoch_Time_Seconds"])],
        )


def save_test_evaluation_to_db(cm, classes):
    client = get_client()
    client.execute("DELETE FROM test_evaluation_results")
    for i, act_cls in enumerate(classes):
        for j, pred_cls in enumerate(classes):
            client.execute(
                "INSERT INTO test_evaluation_results (actual_class, predicted_class, count) VALUES (?, ?, ?)",
                [act_cls, pred_cls, int(cm[i, j])],
            )


def upsert_patient(nik, name, age=None, birth_date=None, gender=None, address=None, phone=None):
    client = get_client()
    client.execute(
        """INSERT INTO patients (nik, name, age, birth_date, gender, address, phone)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(nik) DO UPDATE SET
               name = excluded.name,
               age = excluded.age,
               birth_date = excluded.birth_date,
               gender = excluded.gender,
               address = excluded.address,
               phone = excluded.phone""",
        [nik, name, age, birth_date, gender, address, phone],
    )


def get_patient(nik):
    client = get_client()
    rs = client.execute("SELECT * FROM patients WHERE nik = ?", [nik])
    rows = _rows_to_dicts(rs)
    return rows[0] if rows else None


def add_scan_record(patient_nik, filename, modality, predicted_class, confidence, report_text, original_b64=None, heatmap_b64=None):
    client = get_client()
    client.execute(
        """INSERT INTO scans
           (patient_nik, filename, modality, predicted_class, confidence, radiology_report, original_image_b64, heatmap_image_b64)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [patient_nik, filename, modality, predicted_class, confidence, report_text, original_b64, heatmap_b64],
    )


def get_patient_history(nik):
    client = get_client()
    rs = client.execute(
        """SELECT s.*, p.name, p.age, p.gender, p.birth_date, p.address, p.phone
           FROM scans s
           JOIN patients p ON s.patient_nik = p.nik
           WHERE s.patient_nik = ?
           ORDER BY s.created_at DESC""",
        [nik],
    )
    return _rows_to_dicts(rs)


try:
    init_db()
except Exception as e:
    print(f"⚠️ Peringatan: gagal konek ke Turso saat startup ({e}). "
          f"Fitur riwayat pasien tidak akan berfungsi sampai TURSO_DATABASE_URL/TURSO_AUTH_TOKEN di-set.")
