import sqlite3
from datetime import datetime

def init_db(db_path="mouth_state.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS mouth_state (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        email TEXT,
        state TEXT,
        mar REAL,
        source TEXT,
        frame_idx INTEGER,
        threshold REAL
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        email TEXT PRIMARY KEY,
        password TEXT
    )''')
    conn.commit()
    conn.close()

def insert_state(email, state, mar, source, frame_idx, threshold, db_path="mouth_state.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''INSERT INTO mouth_state (timestamp, email, state, mar, source, frame_idx, threshold)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (datetime.now().isoformat(), email, state, mar, source, frame_idx, threshold))
    conn.commit()
    conn.close()

def register_user(email, password, db_path="mouth_state.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (email, password) VALUES (?, ?)', (email, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def check_user(email, password, db_path="mouth_state.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE email=? AND password=?', (email, password))
    result = c.fetchone()
    conn.close()
    return result is not None

def user_exists(email, db_path="mouth_state.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT 1 FROM users WHERE email=?', (email,))
    result = c.fetchone()
    conn.close()
    return result is not None 