import os
import sqlite3
from contextlib import contextmanager

from src.config import DATA_DIR

# Local SQLite fallback. Di production, jika env Turso tersedia, aplikasi akan pakai Turso.
DB_PATH = os.environ.get("SQLITE_DB_PATH", os.path.join(DATA_DIR, "brainscandb.db"))
TURSO_DATABASE_URL = os.environ.get("TURSO_DATABASE_URL", "").strip()
TURSO_AUTH_TOKEN = os.environ.get("TURSO_AUTH_TOKEN", "").strip()
USE_TURSO = bool(TURSO_DATABASE_URL and TURSO_AUTH_TOKEN)


def _normalize_sql(sql: str) -> str:
    """Konversi SQL ringan agar kompatibel dengan SQLite lokal dan Turso/libSQL."""
    return sql.replace("TIMESTAMP DEFAULT CURRENT_TIMESTAMP", "TEXT DEFAULT CURRENT_TIMESTAMP")


@contextmanager
def get_db_connection():
    """Membuka koneksi database.

    - Production: Turso/libSQL jika TURSO_DATABASE_URL + TURSO_AUTH_TOKEN diset.
    - Local/dev: SQLite file lokal data/brainscan.db.
    """
    if USE_TURSO:
        try:
            import libsql_experimental as libsql
        except ImportError as exc:
            raise RuntimeError(
                "Turso env sudah diset, tetapi package libsql-experimental belum terinstall. "
                "Pastikan requirements.txt berisi libsql-experimental."
            ) from exc

        conn = libsql.connect("brainscan-turso", sync_url=TURSO_DATABASE_URL, auth_token=TURSO_AUTH_TOKEN)
        try:
            conn.sync()
            yield conn
            conn.commit()
            conn.sync()
        finally:
            conn.close()
    else:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def _rows_to_dicts(rows):
    result = []
    for row in rows:
        if isinstance(row, sqlite3.Row):
            result.append(dict(row))
        elif hasattr(row, "keys"):
            result.append({k: row[k] for k in row.keys()})
        else:
            # libSQL biasanya mengembalikan tuple; caller SELECT harus urut sesuai kolom.
            result.append(row)
    return result


def init_db():
    """Menginisialisasi tabel-tabel di database jika belum ada."""
    statements = [
        """
        CREATE TABLE IF NOT EXISTS patients (
            nik TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            age INTEGER,
            birth_date TEXT,
            gender TEXT,
            address TEXT,
            phone TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
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
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_nik) REFERENCES patients(nik) ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS dataset_distribution (
            kelas TEXT PRIMARY KEY,
            sebelum_balancing INTEGER,
            setelah_balancing INTEGER
        )
        """,
        """
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
        """,
        """
        CREATE TABLE IF NOT EXISTS test_evaluation_results (
            actual_class TEXT,
            predicted_class TEXT,
            count INTEGER,
            PRIMARY KEY (actual_class, predicted_class)
        )
        """,
    ]
    with get_db_connection() as conn:
        cur = conn.cursor()
        for statement in statements:
            cur.execute(_normalize_sql(statement))
    target = "Turso" if USE_TURSO else DB_PATH
    print(f"Database berhasil diinisialisasi di: {target}")


def save_dataset_distribution_to_db(df):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM dataset_distribution")
        for _, row in df.iterrows():
            cursor.execute(
                """
                INSERT INTO dataset_distribution (kelas, sebelum_balancing, setelah_balancing)
                VALUES (?, ?, ?)
                """,
                (row["Kelas"], int(row["Sebelum_Balancing"]), int(row["Setelah_Balancing"])),
            )


def save_training_history_to_db(df):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM training_history")
        for _, row in df.iterrows():
            cursor.execute(
                """
                INSERT INTO training_history (epoch, train_loss, val_loss, train_acc, val_acc, train_f1, val_f1, epoch_time_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (int(row["Epoch"]), float(row["Train_Loss"]), float(row["Val_Loss"]), float(row["Train_Acc"]), float(row["Val_Acc"]), float(row["Train_F1"]), float(row["Val_F1"]), float(row["Epoch_Time_Seconds"])),
            )


def save_test_evaluation_to_db(cm, classes):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM test_evaluation_results")
        for i, act_cls in enumerate(classes):
            for j, pred_cls in enumerate(classes):
                cursor.execute(
                    """
                    INSERT INTO test_evaluation_results (actual_class, predicted_class, count)
                    VALUES (?, ?, ?)
                    """,
                    (act_cls, pred_cls, int(cm[i, j])),
                )


def upsert_patient(nik, name, age=None, birth_date=None, gender=None, address=None, phone=None):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO patients (nik, name, age, birth_date, gender, address, phone)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(nik) DO UPDATE SET
                name = excluded.name,
                age = excluded.age,
                birth_date = excluded.birth_date,
                gender = excluded.gender,
                address = excluded.address,
                phone = excluded.phone
            """,
            (nik, name, age, birth_date, gender, address, phone),
        )


def get_patient(nik):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT nik, name, age, birth_date, gender, address, phone, created_at FROM patients WHERE nik = ?", (nik,))
        row = cursor.fetchone()
    if not row:
        return None
    if isinstance(row, sqlite3.Row):
        return dict(row)
    return dict(zip(["nik", "name", "age", "birth_date", "gender", "address", "phone", "created_at"], row))


def add_scan_record(patient_nik, filename, modality, predicted_class, confidence, report_text, original_b64=None, heatmap_b64=None):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO scans (patient_nik, filename, modality, predicted_class, confidence, radiology_report, original_image_b64, heatmap_image_b64)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (patient_nik, filename, modality, predicted_class, confidence, report_text, original_b64, heatmap_b64),
        )


def get_patient_history(nik):
    columns = [
        "id", "patient_nik", "filename", "modality", "predicted_class", "confidence",
        "radiology_report", "original_image_b64", "heatmap_image_b64", "created_at",
        "name", "age", "gender", "birth_date", "address", "phone",
    ]
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT s.id, s.patient_nik, s.filename, s.modality, s.predicted_class, s.confidence,
                   s.radiology_report, s.original_image_b64, s.heatmap_image_b64, s.created_at,
                   p.name, p.age, p.gender, p.birth_date, p.address, p.phone
            FROM scans s
            JOIN patients p ON s.patient_nik = p.nik
            WHERE s.patient_nik = ?
            ORDER BY s.created_at DESC
            """,
            (nik,),
        )
        rows = cursor.fetchall()
    result = []
    for row in rows:
        if isinstance(row, sqlite3.Row):
            result.append(dict(row))
        else:
            result.append(dict(zip(columns, row)))
    return result


# Inisialisasi DB saat modul di-import pertama kali
init_db()
