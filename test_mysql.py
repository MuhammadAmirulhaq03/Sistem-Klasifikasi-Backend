from build_db import BrondolanDB

db = BrondolanDB(
    host="localhost",
    user="root",
    password="",
    database="db_brondolan"
)

db.insert_detection(
    id_deteksi=1,
    class_name="segar",
    bbox=[10,20,30,40]
)

print(db.fetch_all())