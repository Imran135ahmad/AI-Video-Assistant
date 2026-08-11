"""
Backend for the AI Video/Meeting Assistant web UI.

Wraps your existing pipeline (utils.Audio_processor, core.transcriber,
core.summarize, core.extractor, core.rag_engine) as an HTTP API, plus
username/password auth so jobs and history are per-user.

Run:
    pip install flask flask-cors werkzeug python-dotenv
    python backend.py
Then open http://localhost:5000 (Flask serves index.html itself).

Set FLASK_SECRET_KEY in your .env before running this for real. If it's
not set, a random key is generated at startup — every server restart
invalidates all logged-in sessions. Fine for local testing, not fine for
anything you expect to stay logged into.

READ THIS BEFORE CALLING IT "PRODUCTION READY" — it isn't:
- Auth is session-cookie based with no CSRF token (SameSite=Lax gives
  partial protection, not full). No email verification, no password
  reset flow, no login rate-limiting — someone can brute-force a
  password with a script and nothing here stops them.
- SQLite (app_data.db) is a single file with no concurrent-write
  story beyond SQLite's own file locking. Under real concurrent load
  you'll see "database is locked" errors. Fine for a handful of users,
  not fine at scale — move to Postgres before that matters.
- Job state (JOBS dict) is still in-memory. Job history (who ran what,
  when, status) now persists in SQLite, but the actual transcript/
  rag_chain for a finished job disappears on restart. History entries
  from a previous server run will show as "unavailable" if you try to
  reopen them after a restart — I'm not faking data that isn't there.
- No queue/concurrency limiter on job threads — same caveat as before.
- No HTTPS here. Session cookies over plain HTTP are sniffable on an
  untrusted network. Put this behind HTTPS before it leaves localhost.
"""

import os
import re
import uuid
import secrets
import sqlite3
import tempfile
import threading
import traceback
from datetime import datetime, timezone
from functools import wraps

from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

from dotenv import load_dotenv
from utils.Audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

app = Flask(__name__, static_folder=".", static_url_path="")

SECRET_KEY = os.environ.get("FLASK_SECRET_KEY")
if not SECRET_KEY:
    print("WARNING: FLASK_SECRET_KEY not set in .env — using a random key "
          "for this run only. Sessions will not survive a restart.")
    SECRET_KEY = secrets.token_hex(32)
app.secret_key = SECRET_KEY
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")

# supports_credentials is required because the frontend sends the session
# cookie with fetch(..., {credentials: "include"}). If you ever serve the
# frontend from a different origin than this API, you must list that exact
# origin below — a wildcard ("*") origin does not work with credentials,
# browsers will silently drop the cookie.
CORS(app, supports_credentials=True, origins=["http://localhost:5000"])

UPLOAD_DIR = tempfile.mkdtemp(prefix="ai_assistant_uploads_")
ALLOWED_EXT = {"mp4", "mp3", "wav", "m4a"}
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_data.db")

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ────────────────────────────────────────────────────────────────────────────
# DATABASE
# ────────────────────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    db = get_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS job_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT UNIQUE NOT NULL,
            user_id INTEGER NOT NULL,
            title TEXT,
            source TEXT,
            language TEXT,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    db.commit()
    db.close()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def insert_job_history(job_id, user_id, source, language, status):
    db = get_db()
    db.execute(
        "INSERT INTO job_history (job_id, user_id, title, source, language, status, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (job_id, user_id, None, source, language, status, now_iso()),
    )
    db.commit()
    db.close()


def update_job_history(job_id, **fields):
    if not fields:
        return
    db = get_db()
    cols = ", ".join(f"{k} = ?" for k in fields)
    db.execute(f"UPDATE job_history SET {cols} WHERE job_id = ?",
               (*fields.values(), job_id))
    db.commit()
    db.close()


# ────────────────────────────────────────────────────────────────────────────
# AUTH HELPERS
# ────────────────────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "authentication required"}), 401
        return f(*args, **kwargs)
    return wrapper


@app.route("/api/auth/signup", methods=["POST"])
def signup():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if len(username) < 3:
        return jsonify({"error": "username must be at least 3 characters"}), 400
    if not EMAIL_RE.match(email):
        return jsonify({"error": "enter a valid email address"}), 400
    if len(password) < 8:
        return jsonify({"error": "password must be at least 8 characters"}), 400

    db = get_db()
    try:
        existing = db.execute(
            "SELECT id FROM users WHERE username = ? OR email = ?", (username, email)
        ).fetchone()
        if existing:
            return jsonify({"error": "username or email already registered"}), 409

        pw_hash = generate_password_hash(password)
        cur = db.execute(
            "INSERT INTO users (username, email, password_hash, created_at) VALUES (?,?,?,?)",
            (username, email, pw_hash, now_iso()),
        )
        db.commit()
        user_id = cur.lastrowid
    finally:
        db.close()

    session["user_id"] = user_id
    session["username"] = username
    return jsonify({"username": username, "email": email}), 201


@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    db = get_db()
    try:
        row = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    finally:
        db.close()

    # Same error for "no such user" and "wrong password" on purpose —
    # distinguishing them tells an attacker which usernames exist.
    if not row or not check_password_hash(row["password_hash"], password):
        return jsonify({"error": "invalid username or password"}), 401

    session["user_id"] = row["id"]
    session["username"] = row["username"]
    return jsonify({"username": row["username"], "email": row["email"]})


@app.route("/api/auth/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True})


@app.route("/api/auth/me", methods=["GET"])
def me():
    if "user_id" not in session:
        return jsonify({"error": "not authenticated"}), 401
    return jsonify({"username": session["username"]})


# ────────────────────────────────────────────────────────────────────────────
# JOB PIPELINE (background thread + polling, same as before)
# ────────────────────────────────────────────────────────────────────────────

# job_id -> {status, stage, progress, result, error, rag_chain, user_id}
JOBS = {}
JOBS_LOCK = threading.Lock()

STAGES = [
    ("extracting", 10, "Extracting audio"),
    ("transcribing", 45, "Transcribing"),
    ("analyzing", 80, "Summarizing and extracting insights"),
    ("indexing", 100, "Building chat index"),
]


def _set_job(job_id, **kwargs):
    with JOBS_LOCK:
        JOBS[job_id].update(kwargs)


def _run_job(job_id, source, language):
    try:
        _set_job(job_id, status="running", stage=STAGES[0][2], progress=STAGES[0][1])
        chunks = process_input(source)

        _set_job(job_id, stage=STAGES[1][2], progress=STAGES[1][1])
        transcript = transcribe_all(chunks, language=language)

        _set_job(job_id, stage=STAGES[2][2], progress=STAGES[2][1])
        title = generate_title(transcript)
        summary = summarize(transcript)
        action_items = extract_action_items(transcript)
        decisions = extract_key_decisions(transcript)
        questions = extract_questions(transcript)

        _set_job(job_id, stage=STAGES[3][2], progress=STAGES[3][1])
        rag_chain = build_rag_chain(transcript)

        result = {
            "title": title,
            "summary": summary,
            "transcript": transcript,
            "action_item": action_items,
            "key_decision": decisions,
            "open_question": questions,
        }
        _set_job(job_id, status="done", stage="Done", progress=100,
                 result=result, rag_chain=rag_chain, error=None)
        update_job_history(job_id, status="done", title=title)

    except Exception as e:
        traceback.print_exc()
        _set_job(job_id, status="error", error=str(e))
        update_job_history(job_id, status="error")


@app.route("/")
def root():
    return send_from_directory(".", "index.html")


@app.route("/api/jobs", methods=["POST"])
@login_required
def create_job():
    language = request.form.get("language", "english")
    source_type = request.form.get("source_type")
    user_id = session["user_id"]

    if source_type == "youtube":
        source = request.form.get("youtube_url", "").strip()
        if not source:
            return jsonify({"error": "youtube_url is required"}), 400

    elif source_type == "file":
        f = request.files.get("file")
        if not f or f.filename == "":
            return jsonify({"error": "file is required"}), 400
        ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
        if ext not in ALLOWED_EXT:
            return jsonify({"error": f"unsupported file type: .{ext}"}), 400
        filename = secure_filename(f.filename)
        save_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{filename}")
        f.save(save_path)
        source = save_path

    else:
        return jsonify({"error": "source_type must be 'youtube' or 'file'"}), 400

    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued",
            "stage": "Queued",
            "progress": 0,
            "result": None,
            "rag_chain": None,
            "error": None,
            "user_id": user_id,
        }
    insert_job_history(job_id, user_id, source, language, "queued")

    thread = threading.Thread(target=_run_job, args=(job_id, source, language), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id}), 202


def _get_owned_job_or_error(job_id):
    """Returns (job, error_response). error_response is None if access is fine."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return None, (jsonify({"error": "unknown job_id"}), 404)
    if job["user_id"] != session.get("user_id"):
        return None, (jsonify({"error": "you do not have access to this job"}), 403)
    return job, None


@app.route("/api/jobs/<job_id>", methods=["GET"])
@login_required
def job_status(job_id):
    job, err = _get_owned_job_or_error(job_id)
    if err:
        return err
    return jsonify({
        "status": job["status"],
        "stage": job["stage"],
        "progress": job["progress"],
        "result": job["result"],
        "error": job["error"],
    })


@app.route("/api/jobs/<job_id>/chat", methods=["POST"])
@login_required
def job_chat(job_id):
    job, err = _get_owned_job_or_error(job_id)
    if err:
        return err
    if job["status"] != "done" or job["rag_chain"] is None:
        return jsonify({"error": "job is not ready for chat yet"}), 409

    question = (request.get_json(silent=True) or {}).get("question", "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    try:
        answer = ask_question(job["rag_chain"], question)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"answer": answer})


@app.route("/api/history", methods=["GET"])
@login_required
def history():
    db = get_db()
    try:
        rows = db.execute(
            "SELECT job_id, title, source, language, status, created_at "
            "FROM job_history WHERE user_id = ? ORDER BY created_at DESC",
            (session["user_id"],),
        ).fetchall()
    finally:
        db.close()

    with JOBS_LOCK:
        available_ids = set(JOBS.keys())

    return jsonify([
        {
            "job_id": r["job_id"],
            "title": r["title"],
            "source": r["source"],
            "language": r["language"],
            "status": r["status"],
            "created_at": r["created_at"],
            # False means the server restarted since this job ran — the
            # transcript/chat data for it no longer exists anywhere.
            "still_available": r["job_id"] in available_ids,
        }
        for r in rows
    ])


import webbrowser
from threading import Timer

def open_browser():
    webbrowser.open("http://localhost:5000")

if __name__ == "__main__":
    init_db()
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        Timer(1.0, open_browser).start()
    app.run(debug=True, port=5000, use_reloader=False)