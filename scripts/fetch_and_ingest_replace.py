#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_and_ingest_replace.py

Robust downloader that saves CSVs into data/incoming/ and optionally runs the
ingest pipeline (scripts/ingest_and_prepare.py).

Safe defaults included so it won't unexpectedly write masters unless you opt in.
Includes replace-duplicate logic with backups.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, unquote

import requests

# -------------------- CONFIG & PATHS --------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INCOMING_DIR = os.path.join(PROJECT_ROOT, "data", "incoming")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
CONFIG_DEFAULT = os.path.join(PROJECT_ROOT, "config", "endpoints.json")

# Try to detect ingest script in common places, fallback to env var
def _find_ingest_script() -> str:
    candidates = [
        os.path.join(PROJECT_ROOT, "scripts", "ingest_and_prepare.py"),
        os.path.join(PROJECT_ROOT, "ingest_and_prepare.py"),
        os.path.join(PROJECT_ROOT, "bahrain_agent", "ingest_and_prepare.py"),
        os.path.join(PROJECT_ROOT, "scripts", "ingest.py"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    env = os.getenv("INGEST_SCRIPT")
    if env and os.path.exists(env):
        return env
    return candidates[0]  # may not exist, but shows intended path

INGEST_SCRIPT = _find_ingest_script()

os.makedirs(INCOMING_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

LOGFILE = os.path.join(LOG_DIR, "fetch_and_ingest.log")

# -------------------- DOWNLOAD CONTROL --------------------
MAX_RETRIES = int(os.getenv("FETCH_MAX_RETRIES", "3"))
RETRY_BACKOFF = int(os.getenv("FETCH_RETRY_BACKOFF", "5"))  # seconds initial backoff
CHUNK_SIZE = int(os.getenv("FETCH_CHUNK_SIZE", "8192"))
# maximum bytes to accept for a download (default 50 MB)
MAX_DOWNLOAD_BYTES = int(os.getenv("FETCH_MAX_DOWNLOAD_BYTES", str(50 * 1024 * 1024)))
# Allowed content types (simple contains-match)
ALLOWED_CONTENT_TYPES = ["text/csv", "application/csv", "application/octet-stream", "text/plain"]

# Folder to save suspicious/failed downloads for manual review
FAILED_DIR = os.path.join(PROJECT_ROOT, "data", "incoming_failed")
os.makedirs(FAILED_DIR, exist_ok=True)

# -------------------- BEHAVIOR FLAGS --------------------
# Set REPLACE_DUPLICATES=0 to keep original behavior (do not replace duplicates).
# Default is to REPLACE duplicates and create backups of the old file.
REPLACE_DUPLICATES = os.getenv("REPLACE_DUPLICATES", "1") not in ("0", "false", "False")

# -------------------- UTILITIES --------------------


def log(msg: str):
    ts = datetime.utcnow().isoformat()
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(LOGFILE, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def safe_filename_from_url(url: str) -> str:
    p = urlparse(url)
    name = os.path.basename(unquote(p.path)) or "download"
    host = p.hostname or "local"
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    h = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
    fname = f"{host}_{ts}_{h}_{name}"
    return fname.replace("..", "").replace("/", "_")


def md5_of_file(path: str, block_size: int = 65536) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


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
    if any(k in s.lower() for k in ("year", "governorate", "nationalit", "students", "teachers", "occupation")):
        return True
    return False


# -------------------- DOWNLOAD LOGIC --------------------


def _is_csv_like_from_headers(headers: dict) -> bool:
    ctype = headers.get("content-type", "").lower()
    for allowed in ALLOWED_CONTENT_TYPES:
        if allowed in ctype:
            return True
    return False


def download_url_to_incoming(url: str, headers: Optional[Dict[str, str]] = None) -> Tuple[Optional[str], dict]:
    headers = headers or {}
    status = {"status": "failed", "message": "", "url": url}
    # Handle local file copy (file:///)
    if url.startswith("file://"):
        try:
            local_path = url[7:]
            if not os.path.exists(local_path):
                status["message"] = f"Local file not found: {local_path}"
                return None, status
            dest_name = safe_filename_from_url(local_path)
            dest_path = os.path.join(INCOMING_DIR, dest_name)
            shutil.copy2(local_path, dest_path)
            status.update({"status": "downloaded", "path": dest_path})
            log(f"Copied local file {local_path} -> {dest_path}")
            return dest_path, status
        except Exception as e:
            status["message"] = str(e)
            return None, status

    last_exc = None
    backoff = RETRY_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        tmp_path = None
        try:
            resp = requests.get(url, headers=headers, stream=True, timeout=30)
            if resp.status_code != 200:
                status["message"] = f"HTTP {resp.status_code}"
                log(f"Download failed {url} with HTTP {resp.status_code}")
                last_exc = Exception(status["message"])
                if attempt < MAX_RETRIES:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                else:
                    return None, status

            headers_resp = dict(resp.headers)
            csv_by_header = _is_csv_like_from_headers(headers_resp)

            tmp_path = os.path.join(
                INCOMING_DIR,
                f".tmp_download_{int(time.time())}_{hashlib.md5(url.encode()).hexdigest()[:6]}",
            )
            total = 0
            with open(tmp_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        fh.write(chunk)
                        total += len(chunk)
                        if total > MAX_DOWNLOAD_BYTES:
                            fh.close()
                            try:
                                os.remove(tmp_path)
                            except Exception:
                                pass
                            status["message"] = "file_too_large"
                            log(f"Aborting {url}: exceeds MAX_DOWNLOAD_BYTES ({MAX_DOWNLOAD_BYTES})")
                            return None, status

            # validate content quickly
            with open(tmp_path, "rb") as fh:
                sample = fh.read(8192)
            csv_like = csv_by_header or is_likely_csv_bytes(sample)
            if not csv_like:
                # move to failed dir for inspection
                failed_name = safe_filename_from_url(url) + ".maybe_not_csv"
                failed_path = os.path.join(FAILED_DIR, failed_name)
                shutil.move(tmp_path, failed_path)
                status.update({"status": "skipped", "message": "not_csv", "path": failed_path})
                log(f"Downloaded content not CSV-like: saved to {failed_path}")
                return failed_path, status

            # final friendly name
            fname = os.path.basename(unquote(urlparse(url).path)) or "download.csv"
            dest_name = f"{hashlib.md5(url.encode('utf-8')).hexdigest()[:8]}_{int(time.time())}_{fname}"
            dest_path = os.path.join(INCOMING_DIR, dest_name)
            shutil.move(tmp_path, dest_path)
            tmp_path = None

            # dedupe by md5 — if identical content already present, optionally replace existing file
            try:
                cur_md5 = md5_of_file(dest_path)
                if REPLACE_DUPLICATES:
                    for other in os.listdir(INCOMING_DIR):
                        if other == os.path.basename(dest_path):
                            continue
                        op = os.path.join(INCOMING_DIR, other)
                        if not os.path.isfile(op):
                            continue
                        try:
                            if md5_of_file(op) == cur_md5:
                                # create a small backup of the existing file for safety
                                bak_name = None
                                try:
                                    bak_name = f"{op}.bak_{int(time.time())}"
                                    shutil.copy2(op, bak_name)
                                    log(f"Backup created for existing file: {bak_name}")
                                except Exception as e:
                                    log(f"Warning: failed to create backup for {op}: {e}")

                                # attempt atomic replace
                                try:
                                    try:
                                        os.replace(dest_path, op)
                                        log(f"Atomic replace succeeded for {op}")
                                    except Exception:
                                        shutil.move(dest_path, op)
                                        log(
                                            f"Replaced existing file {op} with newly "
                                            f"downloaded file for URL {url}"
                                        )
                                except Exception as e:
                                    # fallback: copy then delete dest_path
                                    try:
                                        shutil.copy2(dest_path, op)
                                        os.remove(dest_path)
                                        log(f"Fallback replace (copy) succeeded for {op}")
                                    except Exception as e2:
                                        log(f"Failed to replace existing file {op}: {e} / {e2}")
                                        status.update(
                                            {
                                                "status": "failed_replace",
                                                "message": str(e2),
                                                "path": dest_path,
                                            }
                                        )
                                        return dest_path, status

                                status.update(
                                    {
                                        "status": "replaced",
                                        "message": f"replaced:{op}",
                                        "path": op,
                                        "backup": bak_name,
                                    }
                                )
                                return op, status
                        except Exception as inner_e:
                            log(f"Warning while checking candidate file {op}: {inner_e}")
                            continue
                # if not replacing, fall through to duplicate-removal logic below
            except Exception as e:
                log(f"Warning: replace-dedupe block failed: {e}")
                pass

            # dedupe by md5 (original behavior if not replaced)
            try:
                cur_md5 = md5_of_file(dest_path)
                for other in os.listdir(INCOMING_DIR):
                    if other == os.path.basename(dest_path):
                        continue
                    op = os.path.join(INCOMING_DIR, other)
                    if not os.path.isfile(op):
                        continue
                    try:
                        if md5_of_file(op) == cur_md5:
                            os.remove(dest_path)
                            status.update(
                                {
                                    "status": "duplicate",
                                    "message": f"duplicate_of:{op}",
                                    "kept": op,
                                    "path": op,
                                }
                            )
                            log(f"Duplicate found for {url}; kept existing {op}")
                            return op, status
                    except Exception:
                        continue
            except Exception:
                pass

            status.update({"status": "downloaded", "path": dest_path})
            log(f"Successfully downloaded {url} -> {dest_path}")
            return dest_path, status

        except Exception as e:
            last_exc = e
            status["message"] = str(e)
            log(f"Attempt {attempt} failed for {url}: {e}")
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            if attempt < MAX_RETRIES:
                time.sleep(backoff)
                backoff *= 2
                continue
            else:
                return None, status

    return None, {"status": "failed", "message": str(last_exc), "url": url}


# -------------------- ENDPOINT HANDLING --------------------


def normalize_endpoint(entry) -> Tuple[str, Dict[str, str]]:
    if isinstance(entry, str):
        return entry, {}
    if isinstance(entry, dict):
        return entry.get("url"), entry.get("headers", {})
    raise ValueError("Invalid endpoint entry: must be string or object")


def extract_urls_from_json_payload(payload) -> List[str]:
    urls = []

    def walk(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)
        elif isinstance(obj, str):
            if obj.lower().endswith(".csv") or obj.startswith("http") or obj.startswith("file://"):
                urls.append(obj)

    walk(payload)
    return urls


# -------------------- MAIN WORKFLOW --------------------


def run_ingest_script(dry: bool = False):
    if not os.path.exists(INGEST_SCRIPT):
        raise FileNotFoundError(f"Ingest script not found at: {INGEST_SCRIPT}")
    cmd = [sys.executable, INGEST_SCRIPT, "--run"]
    if dry:
        cmd.append("--dry")
    log(f"Running ingest script: {' '.join(cmd)}")
    p = subprocess.run(cmd, capture_output=True, text=True)
    log(f"Ingest stdout: {p.stdout.strip()[:2000]}")
    if p.stderr:
        log(f"Ingest stderr: {p.stderr.strip()[:2000]}")
    if p.returncode != 0:
        raise RuntimeError(f"Ingest script returned code {p.returncode}")


def run_endpoints(endpoints: List, no_ingest: bool = False, dry_ingest: bool = True) -> List[Dict]:
    results = []
    for entry in endpoints:
        try:
            url, headers = normalize_endpoint(entry)
            if not url:
                log(f"Skipping invalid endpoint entry: {entry}")
                results.append({"endpoint": entry, "status": "invalid_entry"})
                continue

            log(f"Fetching: {url}")
            path, status = download_url_to_incoming(url, headers=headers)
            rec = {"endpoint": url, "status": status.get("status"), "message": status.get("message")}
            if status.get("status") == "downloaded" and path:
                rec["path"] = path
                results.append(rec)
                log(f"Downloaded and saved: {path}")
            elif status.get("status") in ("duplicate", "skipped", "replaced", "failed_replace"):
                rec.update({"path": status.get("path"), "note": status.get("message")})
                if "backup" in status:
                    rec["backup"] = status.get("backup")
                results.append(rec)
                log(f"Endpoint result (not new): {rec}")
            else:
                rec["error"] = status.get("message")
                results.append(rec)
                log(f"Endpoint failed or skipped: {rec}")
                continue

        except Exception as e:
            log(f"Unexpected error handling endpoint {entry}: {e}")
            results.append({"endpoint": str(entry), "status": "error", "error": str(e)})
            continue

    if not no_ingest:
        any_downloaded = any(r.get("status") in ("downloaded", "replaced") for r in results)
        if any_downloaded:
            try:
                run_ingest_script(dry=dry_ingest)
            except Exception as e:
                log(f"Ingest script failed after downloads: {e}")
        else:
            log("No suitable downloads found; skipping ingest run")

    return results


# -------------------- CONFIG LOADING & CLI --------------------


def load_config(path: str) -> List:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    endpoints = cfg.get("endpoints") or cfg.get("urls") or []
    return endpoints


def parse_args():
    p = argparse.ArgumentParser(description="Fetch CSVs from endpoints and optionally run ingest.")
    p.add_argument("--run", action="store_true", help="Execute fetch (and optional ingest)")
    p.add_argument("--config", type=str, default=CONFIG_DEFAULT, help="Path to endpoints JSON config")
    p.add_argument("--no-ingest", action="store_true", help="Do not run ingest script after downloads")
    p.add_argument(
        "--dry",
        action="store_true",
        help="Dry-run mode: run fetch but run ingest in dry mode (or skip it)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not args.run:
        print(
            "No action specified. Use --run to fetch endpoints. Example:\n"
            "  python scripts/fetch_and_ingest_replace.py --run"
        )
        return
    try:
        endpoints = load_config(args.config)
    except Exception as e:
        log(f"Failed to load config {args.config}: {e}")
        return
    dry_ingest = args.dry
    log(f"Starting fetch_and_ingest_replace (endpoints: {len(endpoints)}), dry_ingest={dry_ingest}")
    results = run_endpoints(endpoints, no_ingest=args.no_ingest, dry_ingest=dry_ingest)
    log("Fetch summary: " + json.dumps(results, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
