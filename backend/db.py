"""
db.py
-----
SQLite storage for essay history. Replaces the old browser-only
localStorage approach with real server-side persistence -- data now
survives clearing the browser, works across devices, and is an actual
database (not just a JS object dumped into the browser), which is what
the FYP synopsis's "Database Management" course-relevance claim needs
to be true.

SQLite specifically (not Postgres/MySQL): zero setup, a single file,
no separate server process to install/run -- appropriate for a project
this size and a student running it locally on Windows via VS Code.
"""

import sqlite3
import os
import json
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "history.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            essay TEXT NOT NULL,
            score INTEGER NOT NULL,
            baseline_score INTEGER,
            transformer_score INTEGER,
            summary TEXT,
            word_count INTEGER,
            feedback_json TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_entry(essay, score, baseline_score, transformer_score, summary, word_count, feedback):
    conn = get_connection()
    cur = conn.execute(
        """INSERT INTO history
           (essay, score, baseline_score, transformer_score, summary, word_count, feedback_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (essay, score, baseline_score, transformer_score, summary, word_count,
         json.dumps(feedback), datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    entry_id = cur.lastrowid
    conn.close()
    return entry_id


def list_entries(limit=10):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [
        {
            "id": r["id"],
            "essay": r["essay"],
            "score": r["score"],
            "baseline_score": r["baseline_score"],
            "transformer_score": r["transformer_score"],
            "summary": r["summary"],
            "word_count": r["word_count"],
            "feedback": json.loads(r["feedback_json"]) if r["feedback_json"] else [],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


def delete_entry(entry_id):
    conn = get_connection()
    conn.execute("DELETE FROM history WHERE id = ?", (entry_id,))
    conn.commit()
    conn.close()


def clear_all():
    conn = get_connection()
    conn.execute("DELETE FROM history")
    conn.commit()
    conn.close()