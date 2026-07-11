"""
database.py
SQLite database layer for the Churn Intelligence app.

Tables:
- users: registered accounts
- predictions: history of every churn prediction made, tied to a user
- activity_logs: audit trail of key user actions (login, predict, upload, etc.)
- password_reset_tokens: one-time tokens for the forgot-password flow

NOTE ON DATABASE CHOICE: SQLite is used here instead of MySQL because it
requires zero server setup (it's a single file on disk, built into Python).
The SQL in this file is standard ANSI SQL -- to switch to MySQL later,
you would only need to:
  1. pip install pymysql (or mysql-connector-python)
  2. Replace sqlite3.connect(...) with a MySQL connection
  3. Change "INTEGER PRIMARY KEY AUTOINCREMENT" to "INT AUTO_INCREMENT PRIMARY KEY"
The table structure and queries stay almost identical.
"""

import sqlite3
import secrets
from datetime import datetime, timedelta

DB_PATH = "database.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets us access columns by name
    return conn


def init_db():
    """Create tables if they don't exist yet. Safe to call on every app startup."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            customer_name TEXT,
            frequency REAL NOT NULL,
            monetary REAL NOT NULL,
            avg_order_value REAL NOT NULL,
            tenure_days REAL NOT NULL,
            churn_probability REAL NOT NULL,
            risk_level TEXT NOT NULL,
            clv_estimate REAL NOT NULL,
            value_tier TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS activity_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action TEXT NOT NULL,
            details TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS password_reset_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token TEXT UNIQUE NOT NULL,
            expires_at TEXT NOT NULL,
            used INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    conn.commit()

    # Migration: add customer_name column to predictions if upgrading from
    # an older version of this database that didn't have it yet
    cur.execute("PRAGMA table_info(predictions)")
    existing_cols = [row["name"] for row in cur.fetchall()]
    if "customer_name" not in existing_cols:
        cur.execute("ALTER TABLE predictions ADD COLUMN customer_name TEXT")
        conn.commit()

    # Seed the original demo admin account if it doesn't exist yet, so
    # existing demo instructions (admin@churnapp.com / Admin@123) still work
    cur.execute("SELECT id FROM users WHERE email = ?", ("admin@churnapp.com",))
    if cur.fetchone() is None:
        from werkzeug.security import generate_password_hash
        cur.execute(
            "INSERT INTO users (name, email, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
            ("Admin", "admin@churnapp.com", generate_password_hash("Admin@123"), "admin", datetime.now().isoformat())
        )
        conn.commit()

    conn.close()


def create_user(name, email, password_hash):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (name, email, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
        (name, email, password_hash, "user", datetime.now().isoformat())
    )
    conn.commit()
    user_id = cur.lastrowid
    conn.close()
    return user_id


def get_user_by_email(email):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE email = ?", (email,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def email_exists(email):
    return get_user_by_email(email) is not None


def save_prediction(user_id, frequency, monetary, avg_order_value, tenure_days,
                     churn_probability, risk_level, clv_estimate, value_tier, customer_name=None):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO predictions
        (user_id, customer_name, frequency, monetary, avg_order_value, tenure_days,
         churn_probability, risk_level, clv_estimate, value_tier, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, customer_name, frequency, monetary, avg_order_value, tenure_days,
          churn_probability, risk_level, clv_estimate, value_tier, datetime.now().isoformat()))
    conn.commit()
    pred_id = cur.lastrowid
    conn.close()
    return pred_id


def get_predictions_for_user(user_id, limit=200, risk_filter=None, search=None):
    """Fetch prediction history for a user, optionally filtered by risk
    level and/or a search term matched against the customer name."""
    conn = get_connection()
    cur = conn.cursor()

    query = "SELECT * FROM predictions WHERE user_id = ?"
    params = [user_id]

    if risk_filter:
        query += " AND risk_level = ?"
        params.append(risk_filter)

    if search:
        query += " AND (customer_name LIKE ? OR CAST(id AS TEXT) LIKE ?)"
        like_term = f"%{search}%"
        params.extend([like_term, like_term])

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_prediction_stats_for_user(user_id):
    """Aggregate counts used on the profile page."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) as total FROM predictions WHERE user_id = ?", (user_id,))
    total = cur.fetchone()["total"]
    cur.execute("SELECT risk_level, COUNT(*) as cnt FROM predictions WHERE user_id = ? GROUP BY risk_level", (user_id,))
    by_risk = {row["risk_level"]: row["cnt"] for row in cur.fetchall()}
    conn.close()
    return {"total": total, "by_risk": by_risk}


# ---------------------------------------------------------------------------
# Activity logs
# ---------------------------------------------------------------------------

def log_activity(user_id, action, details=None):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO activity_logs (user_id, action, details, created_at) VALUES (?, ?, ?, ?)",
        (user_id, action, details, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def get_all_activity_logs(limit=200):
    """Admin-only: view activity across all users, most recent first."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT activity_logs.*, users.name as user_name, users.email as user_email
        FROM activity_logs
        LEFT JOIN users ON activity_logs.user_id = users.id
        ORDER BY activity_logs.created_at DESC
        LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Password reset
# ---------------------------------------------------------------------------

def create_reset_token(user_id, valid_minutes=30):
    token = secrets.token_urlsafe(32)
    expires_at = (datetime.now() + timedelta(minutes=valid_minutes)).isoformat()
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO password_reset_tokens (user_id, token, expires_at, used, created_at) VALUES (?, ?, ?, 0, ?)",
        (user_id, token, expires_at, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    return token


def validate_reset_token(token):
    """Returns the user_id if the token is valid and unused, else None."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM password_reset_tokens WHERE token = ?", (token,))
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    if row["used"]:
        return None
    if datetime.fromisoformat(row["expires_at"]) < datetime.now():
        return None
    return row["user_id"]


def mark_token_used(token):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE password_reset_tokens SET used = 1 WHERE token = ?", (token,))
    conn.commit()
    conn.close()


def update_password(user_id, new_password_hash):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_password_hash, user_id))
    conn.commit()
    conn.close()


def get_user_by_id(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None
