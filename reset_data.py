from build_db import BrondolanDB

print("\n=============================================")
print("  PEMBERSIHAN DATABASE (db_brondolan)")
print("=============================================")

konfirmasi = input("Apakah Anda yakin ingin MENGHAPUS SEMUA DATA deteksi? (y/n): ").strip().lower()

if konfirmasi == "y":
    db = BrondolanDB()
    cursor = db.conn.cursor()
    cursor.execute("TRUNCATE TABLE brondolan")
    db.conn.commit()
    cursor.close()
    print("\n[SUKSES] Semua data deteksi telah dikosongkan!")
    print("[INFO] ID deteksi sekarang akan dimulai kembali dari #1.")
    print("=============================================\n")
else:
    print("\n[BATAL] Penghapusan data dibatalkan.")
    print("=============================================\n")
