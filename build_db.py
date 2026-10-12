import mysql.connector
import json
from datetime import datetime


class BrondolanDB:

    def __init__(
        self,
        host="localhost",
        user="root",
        password="",
        database="db_brondolan"
    ):
        self.conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            autocommit=True      # [FIX] Tidak perlu commit manual, setiap write langsung committed
        )

        # cursor tetap untuk DDL saja
        self.cursor = self.conn.cursor()
        self._create_table()

        print("[INFO] MySQL connected")

    def _create_table(self):
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS brondolan (
            id INT AUTO_INCREMENT PRIMARY KEY,
            id_deteksi INT,
            class VARCHAR(50),
            boundingbox JSON,
            timestamp DATETIME
        )
        """)

    def insert_detection(
        self,
        id_deteksi,
        class_name,
        bbox,
        timestamp=None
    ):
        VALID_CLASSES = ["segar", "busuk", "kering", "jamur"]

        if class_name not in VALID_CLASSES:
            raise ValueError(f"Invalid class_name: {class_name}")

        if timestamp is None:
            timestamp = datetime.now()

        sql = """
        INSERT INTO brondolan (id_deteksi, class, boundingbox, timestamp)
        VALUES (%s, %s, %s, %s)
        """
        values = (
            id_deteksi,
            class_name,
            json.dumps(bbox),
            timestamp
        )

        # [FIX] Cursor baru untuk setiap insert — hindari state cursor lama
        cursor = self.conn.cursor()
        cursor.execute(sql, values)
        cursor.close()


    def fetch_last_n(self, n=10):
        # [FIX] Fresh cursor + commit flush sebelum SELECT
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM brondolan ORDER BY id DESC LIMIT %s", (n,)
        )
        result = cursor.fetchall()
        cursor.close()
        return result

    def fetch_all(self):
        # [FIX] Fresh cursor tiap kali dipanggil
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM brondolan ORDER BY id ASC")
        result = cursor.fetchall()
        cursor.close()
        return result

    def clear_all(self):
        cursor = self.conn.cursor()
        cursor.execute("TRUNCATE TABLE brondolan")
        cursor.close()
        print("[INFO] Data berhasil dihapus")

    def close(self):
        self.cursor.close()
        self.conn.close()
        print("[INFO] Database closed")