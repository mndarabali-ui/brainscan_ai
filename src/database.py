import os
import sqlite3
import requests

from src.config import DATA_DIR

# ============================================================
# CONFIG
# ============================================================

DB_PATH = os.environ.get("SQLITE_DB_PATH", os.path.join(DATA_DIR, "brainscandb.db"))

TURSO_DATABASE_URL = os.environ.get("TURSO_DATABASE_URL", "").strip()
TURSO_AUTH_TOKEN = os.environ.get("TURSO_AUTH_TOKEN", "").strip()

USE_TURSO = bool(TURSO_DATABASE_URL and TURSO_AUTH_TOKEN)
​
============================================================
TURSO HTTP API HELPERS
============================================================
def _turso_http_url():
"""
Convert URL Turso dari:
libsql://xxx.turso.io
menjadi:
https://xxx.turso.io
"""
if TURSO_DATABASE_URL.startswith("libsql://"):
return TURSO_DATABASE_URL.replace("libsql://", "https://", 1)
return TURSO_DATABASE_URL
def _turso_value(value):
"""
Convert Python value ke format arg Turso HTTP API.
"""
if value is None:
return {"type": "null"}
if isinstance(value, bool):
return {"type": "integer", "value": "1" if value else "0"}
if isinstance(value, int):
return {"type": "integer", "value": str(value)}
if isinstance(value, float):
return {"type": "float", "value": value}
return {"type": "text", "value": str(value)}
def _turso_args(args=None):
"""
Convert tuple/list Python ke args Turso.
"""
return [_turso_value(arg) for arg in (args or [])]
def turso_execute(sql, args=None):
"""
Execute SQL ke Turso lewat HTTP API.
"""
url = f"{_turso_http_url()}/v2/pipeline"
headers = {
"Authorization": f"Bearer {TURSO_AUTH_TOKEN}",
"Content-Type": "application/json",
}
payload = {
"requests": [
{
"type": "execute",
"stmt": {
"sql": sql,
"args": _turso_args(args),
},
},
{
"type": "close"
},
]
}
response = requests.post(url, headers=headers, json=payload, timeout=30)
if not response.ok:
raise RuntimeError(
f"Turso HTTP error {response.status_code}: {response.text}"
)
return response.json()
def _parse_turso_rows(result_json, columns):
"""
Parse hasil SELECT dari Turso HTTP API menjadi list[dict].
"""
try:
responses = result_json.get("results", [])
execute_result = responses[0].get("response", {}).get("result", {})
rows = execute_result.get("rows", [])
except Exception:
return []
result = []
for row in rows:
item = {}
for col_name, cell in zip(columns, row):
if cell is None:
item[col_name] = None
continue
value_type = cell.get("type")
value = cell.get("value")
if value_type == "null":
item[col_name] = None
elif value_type == "integer":
try:
item[col_name] = int(value)
except Exception:
item[col_name] = value
elif value_type == "float":
try:
item[col_name] = float(value)
except Exception:
item[col_name] = value
else:
item[col_name] = value
result.append(item)
return result
============================================================
SQLITE HELPERS
============================================================
def sqlite_execute(sql, args=None, fetch=False):
"""
Execute SQL ke SQLite lokal.
"""
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
try:
cursor = conn.cursor()
cursor.execute(sql, args or [])
if fetch:
rows = cursor.fetchall()
return [dict(row) for row in rows]
conn.commit()
return None
finally:
conn.close()
============================================================
UNIVERSAL DB HELPERS
============================================================
def execute(sql, args=None, fetch=False, columns=None):
"""
Wrapper universal:
jika USE_TURSO=True -> pakai Turso HTTP API
jika False -> pakai SQLite lokal
"""
if USE_TURSO:
result = turso_execute(sql, args)
if fetch:
return _parse_turso_rows(result, columns or [])
return None
return sqlite_execute(sql, args, fetch=fetch)
def _normalize_sql(sql: str) -> str:
"""
SQL ringan agar aman untuk SQLite lokal dan Turso.
"""
return sql.replace(
"TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
"TEXT DEFAULT CURRENT_TIMESTAMP"
)
============================================================
INIT DATABASE
============================================================
def init_db():
"""
Membuat tabel jika belum ada.
"""
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
for statement in statements:
execute(_normalize_sql(statement))
target = "Turso HTTP API" if USE_TURSO else DB_PATH
print(f"Database berhasil diinisialisasi di: {target}")
============================================================
PATIENT FUNCTIONS
============================================================
def upsert_patient(nik, name, age=None, birth_date=None, gender=None, address=None, phone=None):
"""
Menyimpan atau memperbarui data pasien.
"""
sql = """
INSERT INTO patients (nik, name, age, birth_date, gender, address, phone)
VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(nik) DO UPDATE SET
name = excluded.name,
age = excluded.age,
birth_date = excluded.birth_date,
gender = excluded.gender,
address = excluded.address,
phone = excluded.phone
"""
execute(sql, (nik, name, age, birth_date, gender, address, phone))
def get_patient(nik):
"""
Mengambil data pasien berdasarkan NIK.
"""
columns = [
"nik",
"name",
"age",
"birth_date",
"gender",
"address",
"phone",
"created_at",
]
sql = """
SELECT nik, name, age, birth_date, gender, address, phone, created_at
FROM patients
WHERE nik = ?
"""
rows = execute(sql, (nik,), fetch=True, columns=columns)
if not rows:
return None
return rows[0]
============================================================
SCAN FUNCTIONS
============================================================
def add_scan_record(
patient_nik,
filename,
modality,
predicted_class,
confidence,
report_text,
original_b64=None,
heatmap_b64=None,
):
"""
Menyimpan riwayat scan pasien.
"""
sql = """
INSERT INTO scans (
patient_nik,
filename,
modality,
predicted_class,
confidence,
radiology_report,
original_image_b64,
heatmap_image_b64
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""
execute(
sql,
(
patient_nik,
filename,
modality,
predicted_class,
confidence,
report_text,
original_b64,
heatmap_b64,
),
)
def get_patient_history(nik):
"""
Mengambil riwayat scan pasien berdasarkan NIK.
"""
columns = [
"id",
"patient_nik",
"filename",
"modality",
"predicted_class",
"confidence",
"radiology_report",
"original_image_b64",
"heatmap_image_b64",
"created_at",
"name",
"age",
"gender",
"birth_date",
"address",
"phone",
]
sql = """
SELECT
s.id,
s.patient_nik,
s.filename,
s.modality,
s.predicted_class,
s.confidence,
s.radiology_report,
s.original_image_b64,
s.heatmap_image_b64,
s.created_at,
p.name,
p.age,
p.gender,
p.birth_date,
p.address,
p.phone
FROM scans s
JOIN patients p ON s.patient_nik = p.nik
WHERE s.patient_nik = ?
ORDER BY s.created_at DESC
"""
return execute(sql, (nik,), fetch=True, columns=columns)
============================================================
OPTIONAL TRAINING DATA FUNCTIONS
============================================================
def save_dataset_distribution_to_db(df):
"""
Menyimpan distribusi dataset.
Masih disediakan kalau suatu saat dipakai.
"""
execute("DELETE FROM dataset_distribution")
for _, row in df.iterrows():
execute(
"""
INSERT INTO dataset_distribution (
kelas,
sebelum_balancing,
setelah_balancing
)
VALUES (?, ?, ?)
""",
(
row["Kelas"],
int(row["Sebelum_Balancing"]),
int(row["Setelah_Balancing"]),
),
)
def save_training_history_to_db(df):
"""
Menyimpan histori training.
Masih disediakan kalau suatu saat dipakai.
"""
execute("DELETE FROM training_history")
for _, row in df.iterrows():
execute(
"""
INSERT INTO training_history (
epoch,
train_loss,
val_loss,
train_acc,
val_acc,
train_f1,
val_f1,
epoch_time_seconds
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""",
(
int(row["Epoch"]),
float(row["Train_Loss"]),
float(row["Val_Loss"]),
float(row["Train_Acc"]),
float(row["Val_Acc"]),
float(row["Train_F1"]),
float(row["Val_F1"]),
float(row["Epoch_Time_Seconds"]),
),
)
def save_test_evaluation_to_db(cm, classes):
"""
Menyimpan confusion matrix / hasil evaluasi.
Masih disediakan kalau suatu saat dipakai.
"""
execute("DELETE FROM test_evaluation_results")
for i, act_cls in enumerate(classes):
for j, pred_cls in enumerate(classes):
execute(
"""
INSERT INTO test_evaluation_results (
actual_class,
predicted_class,
count
)
VALUES (?, ?, ?)
""",
(
act_cls,
pred_cls,
int(cm[i, j]),
),
)
============================================================
AUTO INIT
============================================================
init_db()
