import os
import time
import json
import mysql.connector
from datetime import datetime


class BrondolanDB:

    def __init__(
        self,
        host=None,
        user=None,
        password=None,
        database=None
    ):
        self.host = host or os.getenv("DB_HOST", "localhost")
        self.user = user or os.getenv("DB_USER", "root")
        self.password = password if password is not None else os.getenv("DB_PASSWORD", "")
        self.database = database or os.getenv("DB_NAME", "db_brondolan")

        # Retry loop jika database masih booting (misal saat docker-compose start)
        connected = False
        for attempt in range(1, 15):
            try:
                self.conn = mysql.connector.connect(
                    host=self.host,
                    user=self.user,
                    password=self.password,
                    database=self.database,
                    autocommit=True
                )
                connected = True
                break
            except Exception as e:
                time.sleep(1.5)

        if not connected:
            raise ConnectionError(f"Gagal terhubung ke database MySQL ({self.host})")

        self.cursor = self.conn.cursor()
        self._create_table()

        print(f"[INFO] MySQL connected ({self.host})")

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