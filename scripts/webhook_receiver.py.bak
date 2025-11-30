# -*- coding: utf-8 -*-
"""
webhook_receiver.py

Flask webhook receiver to accept CSV uploads / URLs / embedded base64 / raw CSV text,
save them into data/incoming/ and trigger the ingest pipeline.

Safety defaults:
 - If required packages are missing the script prints a friendly message and exits.
 - By default the triggered ingest will run in DRY mode (no writes). Set env INGEST_DRY=0
   to perform real ingest when you're confident.

Where to place:
  scripts/webhook_receiver.py

How to run:
  # activate your venv, then:
  python scripts/webhook_receiver.py

See README-like guidance at the bottom of this file.
"""
import base64
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

# ---- Defensive imports ----
_missing = []
try:
    from flask import Flask, request, jsonify
except Exception:
    _missing.append("flask")
try:
    import requests
except Exception:
    _missing.append("requests")
try:
    from dotenv import load_dotenv
except Exception:
    _missing.append("python-dotenv")

if _missing:
    msg = (
        "Missing Python packages required by webhook_receiver.py: "
        + ", ".join(_missing)
        + ".\nInstall them into your current environment, e.g.:\n\n"
        + "  pip install -r requirements.txt\n\n"
        + "Then activate your venv and re-run this script."
    )
    print(msg, file=sys.stderr)
    sys.exit(1)
# ----------------------------

# Load environment variables if present
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_PATH1 = os.path.join(PROJECT_ROOT, "config", "secrets.env")
ENV_PATH2 = os.path.join(PROJECT_ROOT, ".env")
if os.path.exists(ENV_PATH1):
    load_dotenv(ENV_PATH1)
elif os.path.exists(ENV_PATH2):
    load_dotenv(ENV_PATH2)

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", os.getenv("API_KEY", "replace-me"))

# Incoming directory (create if missing)
INCOMING_DIR = os.getenv("INCOMING_DIR", os.path.join(PROJECT_ROOT, "data", "incoming"))
os.makedirs(INCOMING_DIR, exist_ok=True)

# Ingest script path detection (search a few common locations)
def _find_ingest_script() -> str:
    candidates = [
        os.path.join(PROJECT_ROOT, "scripts", "ingest_and_prepare.py"),
        os.path.join(PROJECT_ROOT, "scripts", "ingest_and_prepare", "ingest_and_prepare.py"),
        os.path.join(PROJECT_ROOT, "ingest_and_prepare.py"),
        os.path.join(PROJECT_ROOT, "bahrain_agent", "ingest_and_prepare.py"),
        os.path.join(PROJECT_ROOT, "scripts", "ingest.py"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # fallback: user can set INGEST_SCRIPT env var
    env = os.getenv("INGEST_SCRIPT")
    if env and os.path.exists(env):
        return env
    # return first candidate (even if missing) so code shows intended path
    return candidates[0]

INGEST_SCRIPT = _find_ingest_script()

# Default behavior: run ingest in dry mode unless INGEST_DRY=0
INGEST_DRY = os.getenv("INGEST_DRY", "1").strip() not in ("0", "false", "False", "no", "NO")

# App
app = Flask(__name__)

# ---------- Helpers ----------

def authorized(req) -> bool:
    """Check header X-API-KEY or query param api_key"""
    key = req.headers.get("X-API-KEY") or req.args.get("api_key")
    if not key:
        return False
    return key == WEBHOOK_SECRET

def safe_filename(filename: str) -> str:
    name = os.path.basename(filename)
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    return f"{ts}_{name}"

def is_likely_csv_bytes(b: bytes) -> bool:
    try:
        s = b.decode("utf-8", errors="ignore")
    except Exception:
        return False
    if s.count("\n") < 1:
        return False
    lines = s.splitlines()[:10]
    comma_lines = sum(1 for L in lines if "," in L)
    tab_lines = sum(1 for L in lines if "\t" in L)
    if comma_lines + tab_lines >= 1:
        return True
    if re.search(r"year|governorate|nationalit|students|teachers|occupation", s, re.I):
        return True
    return False

def save_bytes_to_incoming(b: bytes, suggested_name: str) -> str:
    fname = safe_filename(suggested_name)
    dst = os.path.join(INCOMING_DIR, fname)
    with open(dst, "wb") as fh:
        fh.write(b)
    return dst

def download_url(url: str, headers: Dict[str, str] = None, timeout: int = 60) -> Tuple[str, str]:
    headers = headers or {}
    # accept file:// as local copy for testing
    if url.startswith("file://"):
        local = url[7:]
        if not os.path.exists(local):
            return ("", f"Local path not found: {local}")
        try:
            dst = os.path.join(INCOMING_DIR, safe_filename(local))
            shutil.copy2(local, dst)
            return (dst, "")
        except Exception as e:
            return ("", str(e))

    try:
        r = requests.get(url, headers=headers, timeout=timeout, stream=True)
        r.raise_for_status()
        cd = r.headers.get("content-disposition")
        if cd and "filename=" in cd:
            suggested = cd.split("filename=")[-1].strip().strip('"')
        else:
            suggested = os.path.basename(url.split("?")[0]) or "download.csv"
        content = r.content
        if not is_likely_csv_bytes(content):
            dst = save_bytes_to_incoming(content, suggested)
            return (dst, "warning:not_csv_content")
        dst = save_bytes_to_incoming(content, suggested)
        return (dst, "")
    except Exception as e:
        return ("", str(e))

def extract_candidates(obj: Any) -> List[Tuple[str, Any]]:
    candidates = []
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(v, str):
                    s = v.strip()
                    if s.startswith("http://") or s.startswith("https://") or s.startswith("file://"):
                        candidates.append(("url", s))
                        continue
                walk(v)
        elif isinstance(x, list):
            for item in x:
                walk(item)
        elif isinstance(x, str):
            s = x.strip()
            if s.startswith("http://") or s.startswith("https://") or s.startswith("file://"):
                candidates.append(("url", s))
                return
            if len(s) > 200 and re.fullmatch(r"[A-Za-z0-9+/=\n\r]+", s.replace("\n", "")):
                candidates.append(("base64", s))
                return
            if "\n" in s and ("," in s or "\t" in s):
                candidates.append(("text", s))
                return
    walk(obj)
    return candidates

def trigger_ingest_async(saved_paths: List[str] = None):
    """Run ingest in background thread. By default runs with --dry unless INGEST_DRY is set to false."""
    saved_paths = saved_paths or []

    def _worker():
        try:
            cmd = [sys.executable, INGEST_SCRIPT, "--run"]
            if INGEST_DRY:
                cmd.append("--dry")
            # pass saved paths in environment so ingest can optionally use them
            env = os.environ.copy()
            if saved_paths:
                env["LAST_SAVED_PATHS"] = ";".join(saved_paths)
            # run non-blocking with check=True to capture failure
            subprocess.run(cmd, check=True, env=env)
            app.logger.info("Ingest runner finished.")
        except Exception as e:
            app.logger.error("Ingest failed: %s", e)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

# ---------- Flask endpoints ----------

def health():
    return jsonify({"status": "ok", "incoming_dir": INCOMING_DIR, "ingest_script": INGEST_SCRIPT, "ingest_dry": INGEST_DRY})

@app.route("/ingest", methods=["POST"])
def ingest_endpoint():
    if not authorized(request):
        return jsonify({"error": "unauthorized"}), 401

    saved = []
    errors = []

    # 1) multipart file uploads
    if request.files:
        for name, file_storage in request.files.items():
            try:
                raw = file_storage.read()
                if not raw:
                    errors.append({"field": name, "error": "empty file"})
                    continue
                suggested = getattr(file_storage, "filename", f"upload_{name}.csv")
                path = save_bytes_to_incoming(raw, suggested)
                saved.append({"source": f"form:{name}", "path": path})
            except Exception as e:
                errors.append({"field": name, "error": str(e)})

    # 2) JSON body or raw text
    content_type = request.headers.get("Content-Type", "")
    if content_type.startswith("application/json") or content_type.startswith("text/"):
        payload = request.get_json(silent=True)
        if payload is None and request.data:
            payload = {"_raw_text": request.get_data(as_text=True)}
        if payload is not None:
            candidates = extract_candidates(payload)
            for ctype, val in candidates:
                if ctype == "url":
                    path, err = download_url(val)
                    if path:
                        saved.append({"source": val, "path": path})
                    else:
                        errors.append({"source": val, "error": err})
                elif ctype == "base64":
                    try:
                        b = base64.b64decode(val)
                        if not is_likely_csv_bytes(b):
                            p = save_bytes_to_incoming(b, "embedded_base64.dat")
                            saved.append({"source": "embedded_base64", "path": p, "note": "maybe_not_csv"})
                        else:
                            p = save_bytes_to_incoming(b, "embedded.csv")
                            saved.append({"source": "embedded_base64", "path": p})
                    except Exception as e:
                        errors.append({"source": "embedded_base64", "error": str(e)})
                elif ctype == "text":
                    try:
                        b = val.encode("utf-8")
                        if not is_likely_csv_bytes(b):
                            p = save_bytes_to_incoming(b, "embedded_text.dat")
                            saved.append({"source": "embedded_text", "path": p, "note": "maybe_not_csv"})
                        else:
                            p = save_bytes_to_incoming(b, "embedded.csv")
                            saved.append({"source": "embedded_text", "path": p})
                    except Exception as e:
                        errors.append({"source": "embedded_text", "error": str(e)})

    # 3) form fields and query params
    all_values = []
    try:
        for k, v in dict(request.form).items():
            all_values.append((k, v))
    except Exception:
        pass
    for k, v in dict(request.args).items():
        all_values.append((k, v))

    for k, v in all_values:
        if isinstance(v, list):
            v = v[0]
        if not isinstance(v, str):
            continue
        s = v.strip()
        if s.startswith("http://") or s.startswith("https://") or s.startswith("file://"):
            path, err = download_url(s)
            if path:
                saved.append({"source": f"param:{k}", "path": path})
            else:
                errors.append({"source": f"param:{k}", "error": err})
        if len(s) > 200 and re.fullmatch(r"[A-Za-z0-9+/=\n\r]+", s.replace("\n", "")):
            try:
                b = base64.b64decode(s)
                p = save_bytes_to_incoming(b, f"param_{k}.bin")
                saved.append({"source": f"param_base64:{k}", "path": p})
            except Exception as e:
                errors.append({"source": f"param_base64:{k}", "error": str(e)})

    # Trigger ingest only if we saved anything
    if saved:
        saved_paths = [it["path"] for it in saved if "path" in it]
        trigger_ingest_async(saved_paths)

    return jsonify({"saved": saved, "errors": errors})

@app.route("/notify", methods=["POST"])
def notify_endpoint():
    if not authorized(request):
        return jsonify({"error": "unauthorized"}), 401
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "invalid json or empty payload"}), 400
    candidates = extract_candidates(payload)
    saved = []
    errors = []
    for ctype, val in candidates:
        if ctype == "url":
            p, err = download_url(val)
            if p:
                saved.append({"source": val, "path": p})
            else:
                errors.append({"source": val, "error": err})
    if saved:
        trigger_ingest_async([it["path"] for it in saved])
    return jsonify({"saved": saved, "errors": errors, "found_candidates": candidates})

@app.route("/admin/list", methods=["GET"])
def admin_list():
    if not authorized(request):
        return jsonify({"error": "unauthorized"}), 401
    files = []
    for name in sorted(os.listdir(INCOMING_DIR), reverse=True)[:200]:
        p = os.path.join(INCOMING_DIR, name)
        try:
            stat = os.stat(p)
            files.append({"name": name, "path": p, "size": stat.st_size, "mtime": stat.st_mtime})
        except Exception:
            continue
    return jsonify({"incoming": files})

if __name__ == "__main__":
    host = os.getenv("WEBHOOK_HOST", "127.0.0.1")
    port = int(os.getenv("WEBHOOK_PORT", "5000"))
    print("="*80)
    print("webhook_receiver starting")
    print(f"INCOMING_DIR = {INCOMING_DIR}")
    print(f"INGEST_SCRIPT = {INGEST_SCRIPT}")
    print(f"INGEST_DRY = {INGEST_DRY}  (set env INGEST_DRY=0 to allow writes)")
    print("="*80)
    app.run(host=host, port=port)
