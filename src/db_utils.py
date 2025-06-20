import sqlite3
from datetime import datetime


def init_db(db_path="mouth_state.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS mouth_state (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        email TEXT,
        state TEXT,
        mar REAL,
        source TEXT,
        frame_idx INTEGER,
        threshold REAL,
        detection_method TEXT,
        camera_info TEXT
    )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS users (
        email TEXT PRIMARY KEY,
        password TEXT
    )"""
    )
    conn.commit()
    conn.close()


def insert_state(
    email, state, mar, source, frame_idx, threshold, detection_method="unknown", camera_info="unknown", db_path="mouth_state.db"
):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        """INSERT INTO mouth_state (timestamp, email, state, mar, source, frame_idx, threshold, detection_method, camera_info)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (datetime.now().isoformat(), email, state, mar, source, frame_idx, threshold, detection_method, camera_info),
    )
    conn.commit()
    conn.close()


def register_user(email, password, db_path="mouth_state.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (email, password) VALUES (?, ?)", (email, password)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def check_user(email, password, db_path="mouth_state.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password))
    result = c.fetchone()
    conn.close()
    return result is not None


def user_exists(email, db_path="mouth_state.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT 1 FROM users WHERE email=?", (email,))
    result = c.fetchone()
    conn.close()
    return result is not None


def get_camera_info_from_frame(frame):
    """Get camera information from actual frame data"""
    try:
        if frame is not None:
            height, width = frame.shape[:2]
            return f"WebRTC Camera - {width}x{height}"
        else:
            return "WebRTC Camera - Unknown resolution"
    except Exception as e:
        return f"WebRTC Camera - Error: {str(e)}"
