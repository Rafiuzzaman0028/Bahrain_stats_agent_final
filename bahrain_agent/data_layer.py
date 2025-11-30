# data_layer.py
# -*- coding: utf-8 -*-
"""
Data layer for Bahrain agent.

- Loads canonical master CSVs from a directory.
- Keeps backward-compatible attributes for the 8 core masters.
- Any additional CSVs found in the same folder are exposed under repo.extras
  so new datasets (footfall.csv, domestic_permits.csv, etc.) are available
  programmatically without changing the rest of your code.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import os
import pandas as pd

# --- configurable: core master filenames (kept for backwards compatibility) ---
CORE_MASTERS = {
    "labour_master.csv": "labour_master",
    "occupation_workers.csv": "occupation_workers",
    "households.csv": "households",
    "population_density.csv": "population_density",
    "housing_units.csv": "housing_units",
    "students.csv": "students",
    "teachers.csv": "teachers",
    "higher_education.csv": "higher_education",
}

@dataclass
class DataRepository:
    # core, kept for backward compatibility with existing code
    labour_master: pd.DataFrame = field(default_factory=pd.DataFrame)
    occupation_workers: pd.DataFrame = field(default_factory=pd.DataFrame)
    households: pd.DataFrame = field(default_factory=pd.DataFrame)
    population_density: pd.DataFrame = field(default_factory=pd.DataFrame)
    housing_units: pd.DataFrame = field(default_factory=pd.DataFrame)
    students: pd.DataFrame = field(default_factory=pd.DataFrame)
    teachers: pd.DataFrame = field(default_factory=pd.DataFrame)
    higher_education: pd.DataFrame = field(default_factory=pd.DataFrame)

    # extras for any other CSVs that show up in the master directory
    extras: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def get(self, name: str) -> pd.DataFrame:
        """
        Convenience: return a DataFrame by core attribute name or extras key.
        Examples:
           repo.get('labour_master')
           repo.get('domestic_permits')  # if present in extras
        """
        if hasattr(self, name):
            return getattr(self, name)
        return self.extras.get(name, pd.DataFrame())

# --- helpers ---

def _safe_read_csv(path: str) -> pd.DataFrame:
    """Read CSV if it exists, otherwise return empty DataFrame (non-fatal)."""
    if not os.path.exists(path):
        # guard: do not raise; print lightweight warning
        print(f"[bahrain_agent] WARNING: CSV not found: {path} (using empty DataFrame)")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[bahrain_agent] ERROR reading {path}: {e} (using empty DataFrame)")
        df = pd.DataFrame()
    return df

def _normalize_column_name(c: str) -> str:
    if not isinstance(c, str):
        return c
    c2 = c.strip().lower()
    c2 = c2.replace(" ", "_")
    # remove punctuation except underscore
    import re
    c2 = re.sub(r"[^\w_]", "", c2)
    return c2

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names: lowercase, underscores, a few common renames.
    This is intentionally conservative to avoid altering actual data values.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # strip whitespace from header values then standardize
    df = df.rename(columns={c: c.strip() for c in df.columns})
    df.columns = [_normalize_column_name(c) for c in df.columns]

    # a few safe common renames to harmonize variants
    rename_map = {}
    for col in list(df.columns):
        # governorate/gov_name variants
        if col in ("governorate_name", "gov_name"):
            rename_map[col] = "governorate"
        # nationality variants
        if col in ("nationality_group", "nat_group", "nationalityname"):
            rename_map[col] = "nationality"
        # workers totals
        if col in ("total_workers", "workers_total", "no_of_workers"):
            rename_map[col] = "total_workers"
        # household counts
        if col in ("no_of_households", "number_of_households"):
            rename_map[col] = "households"
        # housing units
        if col in ("no_of_units", "number_of_units", "housing_unit_count"):
            rename_map[col] = "units"
        # students / value variants
        if col in ("no_of_students", "number_of_students", "students_count"):
            rename_map[col] = "students"
        # teachers
        if col in ("no_of_teachers", "number_of_teachers"):
            rename_map[col] = "teachers"
        # area name alias
        if col in ("area",):
            rename_map[col] = "area_name"

        # --- additional safe mappings for variants you reported ---
        if col in ("academic_year", "academicyear", "year_academic"):
            rename_map[col] = "year"
        if col in ("school_sector", "schooltype", "sector_name"):
            rename_map[col] = "sector"
        if col in ("education_level", "level", "grade"):
            rename_map[col] = "level"
        if col in ("value", "val", "count", "students_value"):
            # caution: "value" is ambiguous; mapping to 'students' is safe for students.csv
            rename_map[col] = "students"
        if col in ("sex", "gender"):
            rename_map[col] = "sex"
        if col in ("permit_id", "id_permit", "domestic_id"):
            rename_map[col] = "permit_id"
        if col in ("employer", "employer_id", "employername"):
            rename_map[col] = "employer_id"
        if col in ("avg_salary", "salary", "monthly_salary", "wage", "basic_salary"):
            rename_map[col] = "avg_salary"
        if col in ("recruitment_fee", "agency_fee", "rec_fee"):
            rename_map[col] = "recruitment_fee"
        if col in ("household_id", "hh_id"):
            rename_map[col] = "household_id"
        if col in ("num_domestic_workers", "domestic_workers_count", "num_workers_per_household"):
            rename_map[col] = "num_domestic_workers"

    if rename_map:
        df = df.rename(columns=rename_map)

    # conservative trimming of string columns (avoid transforming floats)
    for c in df.select_dtypes(include=["object"]).columns:
        try:
            df[c] = df[c].astype(str).str.strip()
        except Exception:
            pass

    return df

# --- main loader ---

def load_all_data(base_path: str = "data/bahrain_master") -> DataRepository:
    """
    Load all CSV files from `base_path`.

    Behavior:
      - For the 8 known master filenames, assign to the dataclass attributes so
        existing code that uses repo.labour_master etc. keeps working.
      - Any other CSV found in the folder is loaded and stored under repo.extras
        with key equal to the filename without extension (normalized).
      - No files are written, no metadata changed. Missing files produce empty DataFrames.
    """
    repo = DataRepository()

    if not os.path.exists(base_path):
        print(f"[bahrain_agent] WARNING: master folder not found: {base_path} -> no data loaded")
        return repo

    # list csv files (non-recursive)
    files = [f for f in os.listdir(base_path) if f.lower().endswith(".csv")]
    # ensure deterministic order
    files.sort()

    for fname in files:
        path = os.path.join(base_path, fname)
        df = _safe_read_csv(path)
        if df.empty:
            # still attach empty DF to the relevant core attr (so code expecting attribute won't break)
            key = CORE_MASTERS.get(fname)
            if key:
                setattr(repo, key, pd.DataFrame())
            else:
                k = os.path.splitext(fname)[0]
                repo.extras[_normalize_column_name(k)] = pd.DataFrame()
            continue

        dfn = _standardize_columns(df)

        # assign to known core masters (if filename matches)
        core_key = CORE_MASTERS.get(fname)
        if core_key:
            setattr(repo, core_key, dfn)
            continue

        # otherwise put into extras with normalized key (filename without extension)
        key = os.path.splitext(fname)[0]
        key = _normalize_column_name(key)
        # ensure we don't overwrite a core-named extras key accidentally
        if hasattr(repo, key):
            # if a CSV accidentally has the same name as an attribute, put it under extras_<key>
            extras_key = f"extras_{key}"
            repo.extras[extras_key] = dfn
        else:
            repo.extras[key] = dfn

    return repo

# convenience helper
def get_repo_df(repo: DataRepository, name: str) -> pd.DataFrame:
    """
    Retrieve a DataFrame from repo by name.
    Accepts:
      - name of core attribute: 'labour_master'
      - filename-like: 'labour_master.csv' or 'labour-master'
      - extras key: 'domestic_permits' (if added later)
    """
    if name.endswith(".csv"):
        name = name[:-4]
    name_norm = _normalize_column_name(name)
    # first try core attributes
    if hasattr(repo, name_norm):
        return getattr(repo, name_norm)
    # then extras
    return repo.extras.get(name_norm, pd.DataFrame())
