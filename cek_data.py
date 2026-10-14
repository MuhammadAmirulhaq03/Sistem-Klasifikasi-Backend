from build_db import BrondolanDB

db = BrondolanDB()
cursor = db.conn.cursor()

# 1. Total data keseluruhan
cursor.execute("SELECT COUNT(*) FROM brondolan")
total_semua = cursor.fetchone()[0]

# 2. Jumlah per kelas
cursor.execute("SELECT class, COUNT(*) FROM brondolan GROUP BY class")
per_kelas = dict(cursor.fetchall())

# 3. 5 Data terbaru
cursor.execute("SELECT id, id_deteksi, class, timestamp FROM brondolan ORDER BY id DESC LIMIT 5")
terbaru = cursor.fetchall()
cursor.close()

print("\n" + "="*45)
print("=== RINGKASAN DATA DATABASE (db_brondolan) ===")
print("="*45)
print(f"Total Semua Data: {total_semua} deteksi\n")

print("Rincian per Kelas:")
for k in ["segar", "busuk", "kering", "jamur"]:
    jml = per_kelas.get(k, 0)
    print(f"  - {k.capitalize():<8}: {jml} deteksi")

print("\n5 Data Terakhir:")
for r in terbaru:
    print(f"  [ID {r[0]}] #{r[1]} - {r[2].upper():<7} | {r[3]}")
print("="*45 + "\n")
