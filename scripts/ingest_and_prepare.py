# -*- coding: utf-8 -*-
"""
ingest_and_prepare.py

Conservative ingestion pipeline:
 - Reads CSVs from data/incoming/
 - Attempts to map incoming columns to canonical master schemas
 - Merges into master CSVs under data/bahrain_master/ (or alternate folders)
 - Writes metadata JSON to a metadata/ folder (next to masters)
 - Optionally updates an in-memory DataRepository (best-effort; non-fatal)
 - Supports config/schemas.json for adding new masters/hints without editing code

Usage:
  # Dry run (no writes)
  python scripts/ingest_and_prepare.py --run --dry

  # Real run (write masters)
  python scripts/ingest_and_prepare.py --run

  # Watch (requires watchdog)
  python scripts/ingest_and_prepare.py --watch
"""
import argparse
import json
import os
import re
import shutil
import time
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

# optional filesystem watcher
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except Exception:
    WATCHDOG_AVAILABLE = False

# -------------------- PROJECT PATHS --------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INCOMING_DIR = os.path.join(PROJECT_ROOT, "data", "incoming")
PROCESSED_DIR = os.path.join(INCOMING_DIR, "processed")
MASTER_DIR = os.path.join(PROJECT_ROOT, "data", "bahrain_master")
BACKUP_DIR = os.path.join(PROJECT_ROOT, "data", "bahrain_master_backups")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "schemas.json")

os.makedirs(INCOMING_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
# MASTER_DIR may be reassigned below if alt path found
os.makedirs(BACKUP_DIR, exist_ok=True)

# -------------------- DEFAULT SCHEMAS / HINTS (can be extended by config/schemas.json) ----
DEFAULT_MASTER_FILES = {
    "labour_master.csv": {
        "required": ["year", "nationality", "sex", "total_workers"],
        "schema": ["year", "nationality", "sex", "total_workers"],
    },
    "occupation_workers.csv": {
        "required": ["year", "main_occupation", "nationality", "occupation_workers"],
        "schema": ["year", "main_occupation", "nationality", "occupation_workers"],
    },
    "households.csv": {
        "required": ["year", "governorate", "nationality", "households"],
        "schema": ["year", "governorate", "nationality", "households"],
    },
    "population_density.csv": {
        "required": ["year", "governorate", "population", "density"],
        "schema": ["year", "governorate", "population", "density"],
    },
    "housing_units.csv": {
        "required": ["year", "governorate", "housing_type", "units"],
        "schema": ["year", "governorate", "housing_type", "units"],
    },
    "students.csv": {
        "required": ["year", "sector", "level", "students"],
        "schema": ["year", "sector", "level", "students"],
    },
    "teachers.csv": {
        "required": ["year", "school_type", "level", "teachers"],
        "schema": ["year", "school_type", "level", "teachers"],
    },
    "higher_education.csv": {
        "required": ["year", "sector", "academic_programme", "students"],
        "schema": ["year", "sector", "academic_programme", "students"],
    },
    # Additional masters for the domestic / mobility domain:
    "domestic_permits.csv": {
        "required": ["year", "permit_id", "status"],
        "schema": ["year", "period", "permit_id", "status", "job_role", "nationality", "employer_id", "area_name", "avg_salary", "recruitment_fee", "notes"]
    },
    "employer_households.csv": {
        "required": ["year", "household_id", "num_domestic_workers"],
        "schema": ["year", "household_id", "num_domestic_workers", "avg_workers_per_household", "area_name"]
    },
    "work_permit_applications.csv": {
        "required": ["year", "application_id", "status"],
        "schema": ["year", "application_id", "status", "job_role", "nationality", "area_name", "submitted_at"]
    },
    "permit_renewals.csv": {
        "required": ["year", "permit_id", "renewal_date"],
        "schema": ["year", "permit_id", "renewal_date", "renewed", "area_name", "nationality"]
    },
    "workforce_distribution.csv": {
        "required": ["year", "sector", "job_role", "count"],
        "schema": ["year", "sector", "job_role", "count", "area_name", "company_type"]
    },
    "salary_fee_aggregates.csv": {
        "required": ["year", "job_role"],
        "schema": ["year", "job_role", "avg_salary", "median_salary", "salary_bracket_low", "salary_bracket_high", "avg_recruitment_fee"]
    },
    "footfall.csv": {
        "required": ["date", "area_name", "footfall_visits"],
        "schema": ["date", "period", "area_name", "area_code", "footfall_visits", "footfall_unique_devices", "avg_dwell_time_seconds", "movement_direction"]
    },
    "mobility.csv": {
        "required": ["date", "area_code", "footfall_visits"],
        "schema": ["date", "period", "area_name", "area_code", "footfall_visits", "footfall_unique_devices", "avg_dwell_time_seconds", "origin_area_code", "destination_area_code"]
    },
}

# sensible default regex hints
DEFAULT_COLUMN_HINTS = {
    r"^year$": "year",
    r"^yr$": "year",
    r"^academic_year$": "year",
    r"governorate|gov_name|region": "governorate",
    r"nationalit(y|ies)|nat$|nat_name": "nationality",
    r"sex|gender": "sex",
    r"total_?workers|workers_total|number_of_workers|no_of_workers|workers": "total_workers",
    r"occupation|main_occupation|job_title|occupation_name": "main_occupation",
    r"occupation_?workers|occupation_workers|workers_in_occupation": "occupation_workers",
    r"household|households|no_of_households|number_of_households": "households",
    r"population(?!_density)|pop_total": "population",
    r"density|population_density": "density",
    r"units|no_of_units|number_of_units|housing_unit|housing_units": "units",
    r"housing_type|unit_type|type_of_unit": "housing_type",
    r"student|students|no_of_students|number_of_students": "students",
    r"teacher|teachers|no_of_teachers|number_of_teachers": "teachers",
    r"school_type|sector": "sector",
    r"academic_programme|programme|program|major": "academic_programme",
    r"level|education_level|grade": "level",
    r"permit_id|id_permit|domestic_id": "permit_id",
    r"permit_?status|status": "status",
    r"job|job_role|occupation|role|position": "job_role",
    r"employer|employer_id|employer_name": "employer_id",
    r"salary|avg_salary|monthly_salary|wage|basic_salary": "avg_salary",
    r"recruitment_fee|agency_fee|rec_fee": "recruitment_fee",
    r"household_id|hh_id": "household_id",
    r"num_domestic_workers|domestic_workers_count|num_workers_per_household": "num_domestic_workers",
    r"application_id|app_id": "application_id",
    r"renewal_date|date_renewal": "renewal_date",
    r"renewed|is_renewed": "renewed",
    r"count|num|workers_count|employee_count": "count",
    r"median_salary|salary_median": "median_salary",
    r"salary_bracket_low|salary_min": "salary_bracket_low",
    r"salary_bracket_high|salary_max": "salary_bracket_high",
    r"footfall|visits|visitor_count": "footfall_visits",
    r"unique_devices|devices|device_count": "footfall_unique_devices",
    r"dwell_time|avg_dwell|dwell_seconds": "avg_dwell_time_seconds",
    r"origin_area_code|from_area": "origin_area_code",
    r"destination_area_code|to_area": "destination_area_code",
    r"area_code|areaid": "area_code",
}

# default filename -> master mapping (can be extended via config/schemas.json)
DEFAULT_FILENAME_KEYWORDS = {
    "labour": "labour_master.csv",
    "workers": "labour_master.csv",
    "occupation": "occupation_workers.csv",
    "household": "households.csv",
    "population": "population_density.csv",
    "density": "population_density.csv",
    "housing": "housing_units.csv",
    "student": "students.csv",
    "teacher": "teachers.csv",
    "higher": "higher_education.csv",
    "domestic": "domestic_permits.csv",
    "permit": "domestic_permits.csv",
    "permits": "domestic_permits.csv",
    "employer_household": "employer_households.csv",
    "application": "work_permit_applications.csv",
    "renewal": "permit_renewals.csv",
    "workforce": "workforce_distribution.csv",
    "salary": "salary_fee_aggregates.csv",
    "footfall": "footfall.csv",
    "mobility": "mobility.csv",
}

# populate runtime maps from the defaults (so config merge below can operate safely)
#from copy import deepcopy
MASTER_FILES = deepcopy(DEFAULT_MASTER_FILES)
COLUMN_HINTS = deepcopy(DEFAULT_COLUMN_HINTS)
FILENAME_KEYWORDS = deepcopy(DEFAULT_FILENAME_KEYWORDS)


# --------------------
# Optional runtime schema overrides (non-destructive)
# Loads config/schemas.json if present and merges into the in-memory maps.
# This is intentionally defensive: it will only log warnings and will not raise.
# --------------------
SCHEMAS_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "schemas.json")
try:
    if os.path.exists(SCHEMAS_CONFIG_PATH):
        with open(SCHEMAS_CONFIG_PATH, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        # merge masters: add or update entries without destroying defaults
        cfg_masters = cfg.get("masters", {})
        for k, v in cfg_masters.items():
            if k in MASTER_FILES:
                MASTER_FILES[k].update({**MASTER_FILES.get(k, {}), **v})
            else:
                MASTER_FILES[k] = v

        # merge filename keywords (config overrides)
        cfg_keywords = cfg.get("filename_keywords", {})
        for kk, vv in cfg_keywords.items():
            FILENAME_KEYWORDS[kk] = vv

        # merge hints: convert list -> regex pattern and add to COLUMN_HINTS
        cfg_hints = cfg.get("hints", {})
        for canon, patterns in cfg_hints.items():
            if not isinstance(patterns, (list, tuple)):
                continue
            safe_patterns = [re.escape(p) for p in patterns if isinstance(p, str) and p.strip()]
            if safe_patterns:
                pat = r"(" + "|".join(safe_patterns) + r")"
                # config takes precedence (non-destructive)
                COLUMN_HINTS[pat] = canon

        print(f"[ingest] Loaded schema overrides from {SCHEMAS_CONFIG_PATH}")
except Exception as e:
    print(f"[ingest] Warning: failed to load config/schemas.json: {e}")


# we'll populate these from defaults + optional config file
# MASTER_FILES = deepcopy(DEFAULT_MASTER_FILES)
# COLUMN_HINTS = deepcopy(DEFAULT_COLUMN_HINTS)
# FILENAME_KEYWORDS = deepcopy(DEFAULT_FILENAME_KEYWORDS)

# -------------------- Config loader (optional) --------------------
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            _cfg = json.load(fh)
        # Merge masters
        for mname, meta in _cfg.get("masters", {}).items():
            if mname not in MASTER_FILES:
                MASTER_FILES[mname] = {"required": meta.get("required", []), "schema": meta.get("schema", [])}
            else:
                # extend schema without duplicates
                existing = MASTER_FILES[mname].get("schema", [])
                for c in meta.get("schema", []):
                    if c not in existing:
                        existing.append(c)
                MASTER_FILES[mname]["schema"] = existing
                # extend required if provided
                existing_req = set(MASTER_FILES[mname].get("required", []))
                for r in meta.get("required", []):
                    existing_req.add(r)
                MASTER_FILES[mname]["required"] = list(existing_req)
        # Merge filename keywords
        for k, v in _cfg.get("filename_keywords", {}).items():
            if k not in FILENAME_KEYWORDS:
                FILENAME_KEYWORDS[k] = v
        # Merge hints (simple approach: treat each hint list as variants and build a small regex)
        for canonical, variants in _cfg.get("hints", {}).items():
            if isinstance(variants, list) and len(variants) > 0:
                safe = "|".join([re.escape(v) for v in variants])
                key = safe
                if key not in COLUMN_HINTS:
                    COLUMN_HINTS[key] = canonical
    except Exception:
        print("[ingest] Warning: failed to load config/schemas.json (continuing with defaults).")

# -------------------- Alternate master path detection & metadata dir --------------------
# Check a few likely alternate directories (non-fatal)
_alt_candidates = [
    os.path.join(PROJECT_ROOT, "bahrain_agent", "data", "__Bahrain_master"),
    os.path.join(PROJECT_ROOT, "bahrain_agent", "data", "Bahrain_master"),
    os.path.join(PROJECT_ROOT, "bahrain_stats_agent", "data", "__Bahrain_master"),
    os.path.join(PROJECT_ROOT, "bahrain_stats_agent", "data", "Bahrain_master"),
    os.path.join(PROJECT_ROOT, "bahrain_stats_agent", "data", "bahrain_master"),
    os.path.join(PROJECT_ROOT, "bahrain_stat_agent", "data", "Bahrain_master"),
]
for _alt in _alt_candidates:
    if os.path.exists(_alt) and not os.path.exists(MASTER_DIR):
        MASTER_DIR = _alt
        break

os.makedirs(MASTER_DIR, exist_ok=True)
METADATA_DIR = os.path.abspath(os.path.join(os.path.dirname(MASTER_DIR), "metadata"))
os.makedirs(METADATA_DIR, exist_ok=True)

# Try to import DataRepository (best-effort; non-fatal)
try:
    from bahrain_agent.data_layer import DataRepository
    _HAS_REPO = True
except Exception:
    DataRepository = None
    _HAS_REPO = False

# -------------------- Helper functions --------------------
def normalize_col_name(c: str) -> str:
    if c is None:
        return c
    c2 = re.sub(r"\s+", "_", c.strip().lower())
    c2 = re.sub(r"[^\w_]", "", c2)
    return c2

def smart_read_csv(path: str) -> pd.DataFrame:
    """Read CSV trying common encodings; return empty DataFrame on failure."""
    for enc in (None, "utf-8-sig", "latin-1"):
        try:
            if enc:
                return pd.read_csv(path, encoding=enc)
            return pd.read_csv(path)
        except Exception:
            continue
    try:
        return pd.read_csv(path, engine="python", encoding="latin-1")
    except Exception:
        return pd.DataFrame()

def coerce_numeric_series(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.str.replace(",", "", regex=False)
        s = s.str.replace(" ", "", regex=False)
    return pd.to_numeric(s, errors="coerce")

def map_columns_by_hints(cols: List[str]) -> Dict[str, str]:
    mapping = {}
    for col in cols:
        nc = normalize_col_name(col)
        for pat, can in COLUMN_HINTS.items():
            try:
                if re.search(pat, nc):
                    mapping[col] = can
                    break
            except re.error:
                # if pat is raw literal or malformed, do substring match
                if pat.lower() in nc:
                    mapping[col] = can
                    break
    return mapping

def detect_target_by_filename(fname: str) -> Optional[str]:
    n = fname.lower()
    for k, t in FILENAME_KEYWORDS.items():
        if k in n:
            return t
    return None

def detect_target_by_columns(cols: List[str]) -> Optional[str]:
    best = None
    best_score = 0
    norm_cols = [normalize_col_name(c) for c in cols]
    for master_fname, meta in MASTER_FILES.items():
        score = 0
        for req in meta.get("required", []):
            if req in norm_cols:
                score += 2
            else:
                for pat, can in COLUMN_HINTS.items():
                    if can == req:
                        for nc in norm_cols:
                            try:
                                if re.search(pat, nc):
                                    score += 1
                                    break
                            except re.error:
                                if pat.lower() in nc:
                                    score += 1
                                    break
        if score > best_score:
            best_score = score
            best = master_fname
    return best if best_score >= 2 else None

def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    norm_map = {c: normalize_col_name(c) for c in df.columns}
    df.rename(columns=norm_map, inplace=True)
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    return df

def map_to_master_schema(df: pd.DataFrame, target_master: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Return (mapped_df, mapping) where mapped_df matches the canonical schema of target_master."""
    if df.empty:
        return df, {}
    incoming_cols = list(df.columns)
    mapping: Dict[str, str] = {}
    # direct matches by normalized name
    norm_to_incoming = {normalize_col_name(c): c for c in incoming_cols}
    for can in MASTER_FILES[target_master]["schema"]:
        if can in norm_to_incoming:
            mapping[norm_to_incoming[can]] = can
    # hints
    hints = map_columns_by_hints(incoming_cols)
    for inc, can in hints.items():
        if inc not in mapping and can in MASTER_FILES[target_master]["schema"]:
            mapping[inc] = can
    # fuzzy substring fallback
    for inc in incoming_cols:
        if inc in mapping:
            continue
        nin = normalize_col_name(inc)
        for can in MASTER_FILES[target_master]["schema"]:
            if can in nin or nin in can:
                mapping[inc] = can
                break
    # build output DataFrame
    out = pd.DataFrame()
    for inc, can in mapping.items():
        out[can] = df[inc]
    # ensure all canonical columns present
    for can in MASTER_FILES[target_master]["schema"]:
        if can not in out.columns:
            out[can] = pd.NA
    # normalize year where present
    if "year" in out.columns:
        def extract_year_value(x):
            if pd.isna(x):
                return pd.NA
            s = str(x)
            m = re.search(r"(19[0-9]{2}|20[0-9]{2}|2100)", s)
            if m:
                return int(m.group(0))
            try:
                return int(float(s))
            except Exception:
                return pd.NA
        out["year"] = out["year"].apply(extract_year_value)
    # coerce numeric-ish columns by simple name heuristic
    for c in out.columns:
        if re.search(r"(_?workers$|workers$|units$|population$|households$|students$|teachers$|_count$|_num$|density$|_visits$|visits$|count$|avg_salary|salary|recruitment_fee)", c):
            out[c] = coerce_numeric_series(out[c])
    for c in out.select_dtypes(include="object").columns:
        out[c] = out[c].str.strip()
    return out, mapping

def _write_metadata_and_update_repo(master_filename: str, df: pd.DataFrame, extra_meta: dict = None):
    """
    Write metadata JSON and attempt a best-effort update of DataRepository instance.
    This function is defensive and will not raise on failure.
    """
    try:
        meta = {
            "layer": master_filename,
            "n_rows": int(len(df)),
            "n_columns": int(df.shape[1]),
            "columns": {c: {"dtype": str(df[c].dtype)} for c in df.columns},
            "updated_at_utc": datetime.utcnow().isoformat() + "Z",
        }
        if extra_meta:
            meta.update(extra_meta)
        meta_name = os.path.splitext(master_filename)[0] + ".json"
        meta_path = os.path.join(METADATA_DIR, meta_name)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[ingest] Warning: failed to write metadata {master_filename}: {e}")

    # best-effort update DataRepository (non-fatal)
    if globals().get("_HAS_REPO", False):
        try:
            repo = DataRepository()
            attr = os.path.splitext(master_filename)[0]
            setattr(repo, attr, df)
            if hasattr(repo, "reload") and callable(getattr(repo, "reload")):
                try:
                    repo.reload()
                except Exception:
                    pass
        except Exception:
            pass

# -------------------- Merge logic --------------------
def merge_into_master(mapped_df: pd.DataFrame, master_filename: str, dry: bool = False) -> Dict[str, int]:
    """
    Merge mapped_df into the canonical master file.
    Returns simple stats dict.
    Non-destructive: backups master before overwriting.
    """
    out_path = os.path.join(MASTER_DIR, master_filename)
    stats = {"master_exists": False, "rows_incoming": len(mapped_df), "rows_before": 0, "rows_after": 0}
    # If no incoming rows -> nothing to do
    if mapped_df is None or len(mapped_df) == 0:
        return stats
    # If master doesn't exist, write directly (but still go through schema column ordering)
    if not os.path.exists(out_path):
        stats["master_exists"] = False
        if not dry:
            try:
                schema = MASTER_FILES.get(master_filename, {}).get("schema", list(mapped_df.columns))
                cols_to_write = [c for c in schema if c in mapped_df.columns] + [c for c in mapped_df.columns if c not in schema]
                mapped_df.to_csv(out_path, index=False, columns=cols_to_write)
                stats["rows_after"] = len(mapped_df)
                # metadata & best-effort repo update
                try:
                    _write_metadata_and_update_repo(os.path.basename(out_path), mapped_df, extra_meta={"rows_after": stats["rows_after"]})
                except Exception:
                    pass
            except Exception as e:
                return {"error": str(e)}
        else:
            stats["rows_after"] = len(mapped_df)
        return stats

    # Master exists -> merge
    master_df = smart_read_csv(out_path)
    master_df = standardize_dataframe(master_df)
    stats["master_exists"] = True
    stats["rows_before"] = len(master_df)
    # Ensure mapped columns present in master
    for col in mapped_df.columns:
        if col not in master_df.columns:
            master_df[col] = pd.NA
    merged = pd.concat([master_df, mapped_df], ignore_index=True, sort=False)
    # quick dedupe (full-row duplicates)
    merged = merged.drop_duplicates()
    # build key columns heuristic for dedupe by key (year + short categorical columns)
    key_cols = ["year"] if "year" in merged.columns else []
    for c in merged.columns:
        if c == "year":
            continue
        try:
            if merged[c].dtype == "object" or merged[c].dtype.name.startswith("category"):
                sample = merged[c].dropna().astype(str)
                if len(sample) > 0 and sample.map(len).mean() < 60:
                    key_cols.append(c)
        except Exception:
            continue
    if not key_cols:
        key_cols = list(merged.columns)
    merged = merged.drop_duplicates(subset=key_cols, keep="last")
    stats["rows_after"] = len(merged)
    if not dry:
        try:
            # backup
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            try:
                shutil.copy2(out_path, os.path.join(BACKUP_DIR, f"{os.path.basename(out_path)}.{ts}.bak"))
            except Exception:
                pass
            # write respecting canonical schema order where possible
            schema = MASTER_FILES.get(master_filename, {}).get("schema", list(merged.columns))
            cols_to_write = [c for c in schema if c in merged.columns] + [c for c in merged.columns if c not in schema]
            merged.to_csv(out_path, index=False, columns=cols_to_write)
            # metadata & repo update
            try:
                _write_metadata_and_update_repo(os.path.basename(out_path), merged, extra_meta={"rows_before": stats["rows_before"], "rows_incoming": stats["rows_incoming"], "rows_after": stats["rows_after"], "written_columns": cols_to_write})
            except Exception:
                pass
        except Exception as e:
            return {"error": str(e)}
    return stats

# -------------------- File processing --------------------
def process_file(path: str, dry: bool = False) -> Dict:
    fname = os.path.basename(path)
    info = {"file": fname, "detected_master": None, "mapping": None, "merge_stats": None, "error": None}
    try:
        df = smart_read_csv(path)
        if df is None or df.empty:
            info["error"] = "Empty or unreadable CSV"
            return info
        # standardize incoming frame
        df_std = standardize_dataframe(df)
        # detect master by filename first, else by columns, else fallback
        detected = detect_target_by_filename(fname)
        if not detected:
            detected = detect_target_by_columns(list(df_std.columns))
        if not detected:
            cols_join = " ".join(list(df_std.columns)).lower()
            if "worker" in cols_join or "employee" in cols_join:
                detected = "labour_master.csv"
            else:
                # fallback to first known master
                detected = list(MASTER_FILES.keys())[0]
        info["detected_master"] = detected
        mapped_df, mapping = map_to_master_schema(df_std, detected)
        info["mapping"] = mapping
        # drop rows missing year if schema expects year (conservative)
        if "year" in mapped_df.columns:
            mapped_df = mapped_df.dropna(subset=["year"])
        # strip object columns
        for c in mapped_df.select_dtypes(include="object").columns:
            mapped_df[c] = mapped_df[c].str.strip()
        merge_stats = merge_into_master(mapped_df, detected, dry=dry)
        info["merge_stats"] = merge_stats
        # move processed file to processed/ (copy, do not remove original)
        dst = os.path.join(PROCESSED_DIR, f"{fname}.processed_{int(time.time())}")
        try:
            shutil.copy2(path, dst)
        except Exception:
            pass
    except Exception as e:
        info["error"] = str(e)
    return info

def run_once(dry: bool = True, verbose: bool = True) -> List[Dict]:
    results = []
    for name in os.listdir(INCOMING_DIR):
        if not name.lower().endswith(".csv"):
            continue
        path = os.path.join(INCOMING_DIR, name)
        if os.path.isdir(path):
            continue
        if verbose:
            print(f"[ingest] Processing: {path}")
        res = process_file(path, dry=dry)
        results.append(res)
        if verbose:
            print(json.dumps(res, indent=2, ensure_ascii=False))
    return results

class NewCSVHandler(FileSystemEventHandler):
    def __init__(self, dry: bool = False):
        super().__init__()
        self.dry = dry

    def on_created(self, event):
        if event.is_directory:
            return
        path = event.src_path
        if path.lower().endswith(".csv"):
            print(f"[ingest] Detected new CSV: {path} - processing...")
            res = process_file(path, dry=self.dry)
            print(json.dumps(res, indent=2, ensure_ascii=False))

def watch_loop(dry: bool = False):
    if not WATCHDOG_AVAILABLE:
        print("[ingest] watchdog package not installed. Install with: pip install watchdog")
        return
    event_handler = NewCSVHandler(dry=dry)
    observer = Observer()
    observer.schedule(event_handler, INCOMING_DIR, recursive=False)
    observer.start()
    print(f"[ingest] Watching directory: {INCOMING_DIR}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Ingest incoming CSVs and prepare master files.")
    p.add_argument("--run", action="store_true", help="Process all files once (non-watch).")
    p.add_argument("--dry", action="store_true", help="Dry run: do not write master files; show actions.")
    p.add_argument("--watch", action="store_true", help="Watch incoming dir and process new CSVs automatically.")
    return p.parse_args()

def main():
    args = parse_args()
    if args.run:
        dry = args.dry
        print(f"[ingest] Running ingest (dry={dry}) - incoming dir: {INCOMING_DIR}")
        res = run_once(dry=dry, verbose=True)
        print("[ingest] Done. Summary:")
        print(json.dumps(res, indent=2, ensure_ascii=False))
    elif args.watch:
        dry = False
        if not WATCHDOG_AVAILABLE:
            print("[ingest] Watch mode requested but watchdog is not available. Install: pip install watchdog")
            return
        watch_loop(dry=dry)
    else:
        print("[ingest] No action specified. Use --run or --watch. Examples:")
        print("  python scripts/ingest_and_prepare.py --run")
        print("  python scripts/ingest_and_prepare.py --run --dry")

if __name__ == "__main__":
    main()
