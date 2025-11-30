# scripts/check_masters.py
import os
import pandas as pd
from bahrain_agent.data_layer import load_all_data

print("=== MASTER DIR INFO ===")
MASTER_DIR = os.path.abspath("data/bahrain_master")
print("MASTER_DIR:", MASTER_DIR)
print("exists:", os.path.exists(MASTER_DIR))
if os.path.exists(MASTER_DIR):
    for f in sorted(os.listdir(MASTER_DIR)):
        fp = os.path.join(MASTER_DIR, f)
        try:
            size = os.path.getsize(fp)
        except Exception:
            size = "N/A"
        print(f"- {f}  size={size} bytes")
else:
    print("No master dir found.")

print("\n=== students.csv INSPECTION ===")
students_path = os.path.abspath(os.path.join(MASTER_DIR, "students.csv"))
print("path:", students_path)
if not os.path.exists(students_path):
    print("students.csv not found")
else:
    try:
        df = pd.read_csv(students_path)
        print("rows:", len(df))
        print("columns:", list(df.columns))
        print("head:\n", df.head(5).to_string(index=False))
    except Exception as e:
        print("Failed to read students.csv:", e)

print("\n=== DataRepository LOADED ===")
try:
    repo = load_all_data(MASTER_DIR)
    layers = [a for a in dir(repo) if not a.startswith("_")]
    print("Loaded layers on repo:", layers)
    if hasattr(repo, "students"):
        df2 = getattr(repo, "students")
        try:
            print("repo.students rows:", len(df2))
            print("repo.students columns:", list(df2.columns))
            print("repo.students head:\n", df2.head(5).to_string(index=False))
        except Exception as e:
            print("Could not inspect repo.students:", e)
    else:
        print("repo has no 'students' attribute.")
except Exception as e:
    print("load_all_data failed:", e)
