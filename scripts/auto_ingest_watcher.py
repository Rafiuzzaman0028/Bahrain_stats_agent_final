# scripts/auto_ingest_watcher.py
import os
import sys
import subprocess
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INCOMING_DIR = os.path.join(PROJECT_ROOT, "data", "incoming")
INGEST_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "ingest_and_prepare.py")

class NewCSVHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        if not event.src_path.lower().endswith(".csv"):
            return
        print(f"[watcher] Detected new CSV: {event.src_path} -> running ingest (dry by default)")
        # By default run dry; set INGEST_DRY env var to 0 to allow writes
        env = os.environ.copy()
        cmd = [sys.executable, INGEST_SCRIPT, "--run"]
        # remove "--dry" to perform actual writes, or set INGEST_DRY env
        cmd.append("--dry")
        subprocess.Popen(cmd, env=env)

if __name__ == "__main__":
    os.makedirs(INCOMING_DIR, exist_ok=True)
    event_handler = NewCSVHandler()
    observer = Observer()
    observer.schedule(event_handler, INCOMING_DIR, recursive=False)
    observer.start()
    print("[watcher] Monitoring", INCOMING_DIR)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
