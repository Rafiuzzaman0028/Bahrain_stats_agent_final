#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio app: chat + independent assistant-driven chart + upload-to-incoming + reload/fetch.

- Chart stays on the right inside a collapsible accordion.
- Upload & Inject and Reload Endpoints & Fetch are in collapsible sections (left).
- Upload saves CSV to data/incoming/ and triggers ingest: dry -> real in background.
- Reload endpoints validates config/endpoints.json then runs fetch_and_ingest_replace.py in background:
    1) --dry
    2) --run  (only after dry completes)
- Uses a lock and writes logs to logs/ to avoid creating stray files in project root.
- Non-destructive: relies on existing scripts in scripts/*.py and data/incoming/.
"""
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import shlex
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import plotly.express as px
import gradio as gr

# Logging
LOG = logging.getLogger("bahrain_agent_ui")
logging.basicConfig(level=logging.INFO)

# Paths / config
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.getenv("BAHRAIN_DATA_PATH", os.path.join(PROJECT_ROOT, "data", "bahrain_master"))
INCOMING_DIR = os.path.join(PROJECT_ROOT, "data", "incoming")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "endpoints.json")

os.makedirs(INCOMING_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

CHART_CACHE_TTL = int(os.getenv("CHART_CACHE_TTL_SECONDS", "30"))
DEFAULT_USE_LLM = True

# Scripts (the app will prefer fetch_and_ingest_replace.py)
INGEST_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "ingest_and_prepare.py")
FETCH_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "fetch_and_ingest_replace.py")

# --- Agent imports (keep your existing behaviour) ---
from bahrain_agent.agent import BahrainStatsAgent
from bahrain_agent.nlu_router import route_and_answer  # optional LLM-refinement wrapper

agent = BahrainStatsAgent(data_path=DATA_PATH)
LOG.info("Loading BahrainStatsAgent with data path: %s", DATA_PATH)

# --- Chart safety + metric mapping (keeps your original mapping + auto-discovery if present) ---
ALLOWED_CHART_TYPES = {"bar", "line"}
ALLOWED_X = {"governorate", "year"}
ALLOWED_Y = {"households", "population", "students", "teachers", "units", "count"}
MAX_GROUPS = 50
MAX_JSON_CHARS = 4000

METRIC_TO_FILES = {
    "households": (["households.parquet", "master.parquet"], ["households.csv", "master.csv"]),
    "population": (["population_density.parquet", "master.parquet"], ["population_density.csv", "master.csv"]),
    "students": (["students.parquet", "master.parquet"], ["students.csv", "master.csv"]),
    "teachers": (["teachers.parquet", "master.parquet"], ["teachers.csv", "master.csv"]),
    "units": (["housing_units.parquet", "master.parquet"], ["housing_units.csv", "master.csv"]),
    "count": (["master.parquet"], ["master.csv"]),
}

# small in-process cache
_chart_cache: Dict[str, Dict] = {}

# Helper: read first table available
def _read_table_try(parquets: list, csvs: list) -> pd.DataFrame:
    for p in parquets:
        pth = os.path.join(DATA_PATH, p)
        if os.path.exists(pth):
            try:
                return pd.read_parquet(pth)
            except Exception as e:
                LOG.warning("Failed to read parquet %s: %s", pth, e)
    for c in csvs:
        cpth = os.path.join(DATA_PATH, c)
        if os.path.exists(cpth):
            try:
                return pd.read_csv(cpth)
            except Exception as e:
                LOG.warning("Failed to read csv %s: %s", cpth, e)
    return pd.DataFrame()

# Chart spec parsing & inference (unchanged behaviour)
def parse_chart_spec_from_text(text: str) -> Optional[dict]:
    if not text:
        return None
    m = re.search(r"```JSON_CHART\s*([\s\S]{1,%d}?)\s*```" % MAX_JSON_CHARS, text, flags=re.I)
    candidate = None
    if m:
        candidate = m.group(1).strip()
    else:
        m2 = re.search(r"(\{[\s\S]{10,1000}\})", text)
        if m2:
            candidate = m2.group(1).strip()
    if not candidate:
        return None
    try:
        spec = json.loads(candidate)
    except Exception:
        try:
            spec = json.loads(candidate.replace("'", '"'))
        except Exception:
            return None
    if not isinstance(spec, dict):
        return None
    typ = spec.get("type", "bar")
    if typ not in ALLOWED_CHART_TYPES:
        typ = "bar"
    x = spec.get("x")
    y = spec.get("y")
    if x not in ALLOWED_X or y not in ALLOWED_Y:
        return None
    filters = spec.get("filters", {}) or {}
    limit = spec.get("limit", 20)
    try:
        limit = int(limit)
    except Exception:
        limit = 20
    limit = max(1, min(limit, MAX_GROUPS))
    return {"type": typ, "x": x, "y": y, "filters": filters, "limit": limit}

def infer_spec_from_text_nl(text: str) -> Tuple[Optional[dict], float]:
    if not text:
        return None, 0.0
    nl = text.lower()
    metric = None
    for m in ALLOWED_Y:
        if m in nl:
            metric = m
            break
    x = None
    if "governorate" in nl or "region" in nl or "area" in nl:
        x = "governorate"
    elif "year" in nl or re.search(r"\b20[0-9]{2}\b", nl):
        x = "year"
    if not metric or not x:
        return None, 0.0
    years = re.findall(r"\b(19[7-9]\d|20[0-9]{2})\b", nl)
    filters = {}
    if years:
        yrs = sorted([int(y) for y in years])
        if len(yrs) == 1:
            filters["year"] = yrs[0]
        elif len(yrs) >= 2:
            filters["year"] = {"from": yrs[0], "to": yrs[-1]}
    conf = 0.5
    if "by governorate" in nl or "by area" in nl:
        conf += 0.3
    if years:
        conf += 0.15
    conf = min(conf, 0.95)
    spec = {"type": "bar" if x == "governorate" else "line", "x": x, "y": metric, "filters": filters, "limit": 20}
    return spec, conf

def build_chart_from_spec(spec: dict):
    if not spec:
        return None
    x = spec["x"]; y = spec["y"]; typ = spec.get("type", "bar"); filters = spec.get("filters", {}) or {}; limit = spec.get("limit", 20)
    parquets, csvs = METRIC_TO_FILES.get(y, (["master.parquet"], ["master.csv"]))
    df = _read_table_try(parquets=parquets, csvs=csvs)
    if df.empty:
        sample = pd.DataFrame({x: ["-"], y: [0]})
        fig = px.bar(sample, x=x, y=y, title=f"{y} by {x} (no data)")
        fig.update_layout(autosize=True, margin=dict(l=30, r=10, t=40, b=40))
        return fig
    cols_map = {c.strip().lower(): c for c in df.columns}
    def find_col(wanted):
        lw = (wanted or "").strip().lower()
        if lw in cols_map:
            return cols_map[lw]
        for k, real in cols_map.items():
            if lw and (lw in k or k in lw):
                return real
        return None
    x_col = find_col(x); y_col = find_col(y)
    if not x_col or not y_col:
        LOG.info("Schema mismatch: could not find columns for x=%s y=%s in data. Available columns: %s", x, y, list(df.columns))
        sample = pd.DataFrame({x: ["-"], y: [0]})
        fig = px.bar(sample, x=x, y=y, title=f"{y} by {x} (schema mismatch)")
        fig.update_layout(autosize=True, margin=dict(l=30, r=10, t=40, b=40))
        return fig
    sub = df[[x_col, y_col]].copy()
    sub[x_col] = sub[x_col].astype(str).str.strip()
    sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce").fillna(0)
    year_filter = filters.get("year")
    if year_filter is not None:
        year_col = find_col("year")
        if year_col and year_col in df.columns:
            try:
                if isinstance(year_filter, dict):
                    fr = year_filter.get("from"); to = year_filter.get("to")
                    if fr is not None:
                        sub = sub[df[year_col].astype(float) >= float(fr)]
                    if to is not None:
                        sub = sub[df[year_col].astype(float) <= float(to)]
                else:
                    sub = sub[df[year_col].astype(float) == float(year_filter)]
            except Exception:
                LOG.debug("Year filter could not be applied; continuing without it")
        else:
            LOG.debug("Year filter requested but no year column found; skipping")
    try:
        agg = sub.groupby(x_col, dropna=True)[y_col].sum().reset_index()
        agg = agg.sort_values(by=y_col, ascending=False).head(limit)
        agg.columns = [x, y]
    except Exception as e:
        LOG.exception("Aggregation error: %s", e)
        agg = pd.DataFrame({x: ["-"], y: [0]})
    try:
        fig = px.bar(agg, x=x, y=y, title=f"{y} by {x}") if typ == "bar" else px.line(agg, x=x, y=y, title=f"{y} by {x}")
        fig.update_layout(autosize=True, margin=dict(l=30, r=10, t=46, b=40))
        try:
            if "height" in fig.layout:
                del fig.layout["height"]
        except Exception:
            pass
        LOG.info("Built chart for spec x=%s y=%s (rows=%d)", x, y, len(agg))
        return fig
    except Exception as e:
        LOG.exception("Plot build error: %s", e)
        fallback = pd.DataFrame({x: ["-"], y: [0]})
        fig = px.bar(fallback, x=x, y=y, title=f"{y} by {x} (error)")
        fig.update_layout(autosize=True)
        return fig

def get_chart_for_spec_cached(spec: dict):
    key = json.dumps(spec, sort_keys=True)
    now = time.time()
    cached = _chart_cache.get(key)
    if cached and (now - cached.get("ts", 0) < CHART_CACHE_TTL):
        return cached["fig"]
    fig = build_chart_from_spec(spec)
    _chart_cache[key] = {"ts": now, "fig": fig}
    return fig

# --- Upload helpers ---
def safe_incoming_filename(original_name: str) -> str:
    name = os.path.basename(original_name)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    h = hashlib.md5((name + stamp).encode("utf-8")).hexdigest()[:6]
    safe = name.replace(" ", "_")
    return f"{stamp}_{h}_{safe}"

def _run_cmd_and_log(cmd: list, logfile: str):
    try:
        with open(logfile, "a", encoding="utf-8") as fh:
            fh.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] RUN: {' '.join(shlex.quote(p) for p in cmd)}\n")
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out = p.stdout or ""
        with open(logfile, "a", encoding="utf-8") as fh:
            fh.write(out + "\n")
        return p.returncode, out
    except Exception as e:
        with open(logfile, "a", encoding="utf-8") as fh:
            fh.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] EXCEPTION: {e}\n")
        LOG.exception("Exception while running command: %s", cmd)
        return 255, str(e)

def trigger_ingest_dry_then_real_async():
    logfile = os.path.join(LOG_DIR, "manual_upload_ingest.log")
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    def worker():
        # dry run first
        if os.path.exists(INGEST_SCRIPT):
            dry_cmd = [sys.executable, INGEST_SCRIPT, "--run", "--dry"]
            _run_cmd_and_log(dry_cmd, logfile)
            # real run after dry completes
            real_cmd = [sys.executable, INGEST_SCRIPT, "--run"]
            _run_cmd_and_log(real_cmd, logfile)
        else:
            with open(logfile, "a", encoding="utf-8") as fh:
                fh.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INGEST_SCRIPT not found: {INGEST_SCRIPT}\n")
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    LOG.info("Started ingest dry->real background thread (logs -> %s)", logfile)
    return True

def handle_upload_and_inject(file_obj, run_ingest_checkbox: bool):
    if not file_obj:
        return "No file provided."
    try:
        if isinstance(file_obj, dict) and "name" in file_obj and "tmp_path" in file_obj:
            orig_name = file_obj["name"]
            src_path = file_obj["tmp_path"]
        elif isinstance(file_obj, (str, Path)):
            src_path = str(file_obj)
            orig_name = os.path.basename(src_path)
        else:
            orig_name = getattr(file_obj, "name", "upload.csv")
            tmp_local = os.path.join(INCOMING_DIR, ".temp_upload")
            with open(tmp_local, "wb") as fh:
                shutil.copyfileobj(file_obj, fh)
            src_path = tmp_local
    except Exception as e:
        LOG.exception("Failed to interpret uploaded file object: %s", e)
        return f"Upload failed: {e}"
    if not orig_name.lower().endswith(".csv"):
        return "Only CSV files are accepted. Please upload a .csv file."
    dest_name = safe_incoming_filename(orig_name)
    dest_path = os.path.join(INCOMING_DIR, dest_name)
    try:
        shutil.copy2(src_path, dest_path)
    except Exception as e:
        LOG.exception("Failed to save upload to incoming: %s", e)
        return f"Failed to save file: {e}"
    LOG.info("Saved uploaded file to %s", dest_path)
    # Trigger ingest sequence in background (dry -> real)
    try:
        trigger_ingest_dry_then_real_async()
        msg = f"Uploaded and saved as {dest_name}. Ingest (dry->real) started; check logs/manual_upload_ingest.log"
    except Exception as e:
        LOG.exception("Failed to start ingest sequence: %s", e)
        msg = f"Uploaded and saved as {dest_name}. Failed to trigger ingest: {e}"
    return msg

# ---------- lock + fetch/reload ----------
ingest_thread_lock = threading.Lock()
INGEST_LOCKFILE = os.path.join(PROJECT_ROOT, "tmp", "ingest_running.lock")
os.makedirs(os.path.dirname(INGEST_LOCKFILE), exist_ok=True)

def _is_ingest_running() -> bool:
    if ingest_thread_lock.locked():
        return True
    return os.path.exists(INGEST_LOCKFILE)

def _acquire_ingest_lock():
    acquired = ingest_thread_lock.acquire(blocking=False)
    if not acquired:
        return False
    try:
        open(INGEST_LOCKFILE, "w", encoding="utf-8").write(str(time.time()))
    except Exception:
        pass
    return True

def _release_ingest_lock():
    try:
        if os.path.exists(INGEST_LOCKFILE):
            os.remove(INGEST_LOCKFILE)
    except Exception:
        pass
    try:
        if ingest_thread_lock.locked():
            ingest_thread_lock.release()
    except Exception:
        pass

def reload_endpoints() -> Tuple[bool, str]:
    if not os.path.exists(CONFIG_PATH):
        return False, f"Config not found: {CONFIG_PATH}"
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        endpoints = cfg.get("endpoints") or cfg.get("urls") or []
        n = len(endpoints) if isinstance(endpoints, list) else 0
        bad = []
        for i, e in enumerate(endpoints):
            if isinstance(e, str):
                continue
            if isinstance(e, dict) and "url" in e:
                continue
            bad.append(i)
        if bad:
            return False, f"Config loaded but {len(bad)} invalid entries (indexes {bad})"
        return True, f"Config loaded ok — {n} endpoints."
    except json.JSONDecodeError as je:
        return False, f"Config JSON parse error: {je}"
    except Exception as ex:
        return False, f"Failed to read config: {ex}"

def start_fetch_with_reload(run_real_after_dry: bool = True) -> str:
    if _is_ingest_running():
        return "A fetch/ingest is already running. Please wait for it to finish."
    ok, msg = reload_endpoints()
    if not ok:
        return f"Endpoints reload failed: {msg}"
    if not _acquire_ingest_lock():
        return "Failed to acquire run lock; another process may be starting."
    logfile = os.path.join(LOG_DIR, "fetch_and_ingest_manual.log")
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    def worker():
        try:
            # dry fetch
            if os.path.exists(FETCH_SCRIPT):
                dry_cmd = [sys.executable, FETCH_SCRIPT, "--run", "--dry"]
                _run_cmd_and_log(dry_cmd, logfile)
                # optional real fetch
                if run_real_after_dry:
                    real_cmd = [sys.executable, FETCH_SCRIPT, "--run"]
                    _run_cmd_and_log(real_cmd, logfile)
            else:
                with open(logfile, "a", encoding="utf-8") as fh:
                    fh.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] FETCH_SCRIPT not found: {FETCH_SCRIPT}\n")
            with open(logfile, "a", encoding="utf-8") as fh:
                fh.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Fetch+Ingest sequence finished\n")
        finally:
            _release_ingest_lock()
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    LOG.info("Started fetch+ingest (reload) background thread (logs -> %s)", logfile)
    return f"{msg} — Fetch+Ingest sequence started (logs: {os.path.relpath(logfile, PROJECT_ROOT)})"

# ----- Chat submit (unchanged) -----
def submit_message(message: str, history: list, use_llm: bool):
    if not message or not message.strip():
        return history or [], "", None, "Last updated: -"
    history = history or []
    user_msg = {"role": "user", "content": message.strip()}
    history.append(user_msg)
    try:
        if use_llm:
            answer = route_and_answer(agent, message.strip(), use_llm=True)
        else:
            answer = agent.answer_question(message.strip())
    except Exception as e:
        LOG.exception("Error calling agent:")
        answer = f"Error producing answer: {e}"
    assistant_msg = {"role": "assistant", "content": answer}
    history.append(assistant_msg)
    LOG.debug("Assistant answer: %s", answer)
    spec = parse_chart_spec_from_text(answer)
    confidence = 1.0 if spec else 0.0
    if not spec:
        inferred, conf = infer_spec_from_text_nl(answer)
        if inferred and conf >= 0.4:
            spec = inferred
            confidence = conf
    if not spec:
        inferred_user, conf2 = infer_spec_from_text_nl(message)
        if inferred_user and conf2 >= 0.45:
            spec = inferred_user
            confidence = conf2
    LOG.info("Chart spec chosen: %s (confidence=%.2f)", spec, confidence)
    if spec:
        try:
            fig = get_chart_for_spec_cached(spec)
            last_updated = f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} (conf={confidence:.2f})"
        except Exception as e:
            LOG.exception("Chart build failed: %s", e)
            fig = None
            last_updated = "Last updated: -"
    else:
        fig = None
        last_updated = "Last updated: -"
    return history, "", fig, last_updated

def clear_history():
    return []

# ----- Gradio UI (with accordions) -----
with gr.Blocks(title="BH Bahrain Statistical AI Agent") as demo:
    gr.Markdown("## BH Bahrain Statistical AI Agent\nAsk about labour, households, population density, housing, segmentation etc.")
    with gr.Row():
        llm_checkbox = gr.Checkbox(value=DEFAULT_USE_LLM, label="Use LLM (ChatGPT) refinement")
    with gr.Row():
        # left column: chat + controls
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat", elem_id="chatbot")
            txt = gr.Textbox(placeholder="Type your question...", show_label=False)
            with gr.Row():
                send_btn = gr.Button("Send")
                clear_btn = gr.Button("Clear")

            # COLLAPSIBLE: Upload & Inject
            with gr.Accordion("Upload CSV & Inject (click to open)", open=False):
                upload_file = gr.File(label="Upload CSV (browse or drop)", file_count="single", file_types=[".csv"])
                run_ingest_checkbox = gr.Checkbox(value=True, label="(auto) Run ingest after upload")
                upload_btn = gr.Button("Upload & Inject")
                upload_status = gr.Textbox(value="", interactive=False, label="Upload status")
                gr.Markdown("- Uploaded files are saved to `data/incoming/` and an ingest (dry -> real) starts in background. Logs → `logs/manual_upload_ingest.log`")

            # COLLAPSIBLE: Reload endpoints & Fetch
            with gr.Accordion("Reload Endpoints & Fetch (click to open)", open=False):
                reload_fetch_checkbox = gr.Checkbox(value=True, label="Run real fetch after dry (auto)")
                reload_fetch_btn = gr.Button("Reload endpoints & Fetch")
                reload_fetch_status = gr.Textbox(value="", interactive=False, label="Reload & Fetch status")
                gr.Markdown("- Validates , then runs `scripts/fetch_and_ingest.py --run --dry` followed by `--run` in background Logs `")

        # right column: chart inside an accordion (dropdown)
        with gr.Column(scale=1, min_width=320):
            with gr.Accordion("Chart (independently driven by assistant answer)", open=False):
                chart_plot = gr.Plot(label="Chart")
                last_updated = gr.Text(value="Last updated: -", interactive=False)

    state = gr.State(value=[])

    # actions wiring
    send_btn.click(fn=submit_message, inputs=[txt, state, llm_checkbox],
                   outputs=[chatbot, txt, chart_plot, last_updated], queue=True).then(
        lambda h: h, inputs=[chatbot], outputs=[state]
    )
    txt.submit(fn=submit_message, inputs=[txt, state, llm_checkbox],
               outputs=[chatbot, txt, chart_plot, last_updated], queue=True).then(
        lambda h: h, inputs=[chatbot], outputs=[state]
    )
    clear_btn.click(fn=clear_history, inputs=None, outputs=[chatbot, state])

    upload_btn.click(fn=handle_upload_and_inject, inputs=[upload_file, run_ingest_checkbox], outputs=[upload_status])

    def _reload_then_fetch_ui(run_real_flag: bool):
        return start_fetch_with_reload(run_real_after_dry=bool(run_real_flag))

    reload_fetch_btn.click(fn=_reload_then_fetch_ui, inputs=[reload_fetch_checkbox], outputs=[reload_fetch_status])

    # default chart
    def _default_chart():
        default_spec = {"type": "bar", "x": "governorate", "y": "households", "limit": 20}
        fig = get_chart_for_spec_cached(default_spec)
        return fig, f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}"
    demo.load(fn=_default_chart, inputs=None, outputs=[chart_plot, last_updated])

if __name__ == "__main__":
    port = int(os.getenv("GRADIO_PORT", "7860"))
    demo.launch(server_name="127.0.0.1", server_port=port, share=False)
