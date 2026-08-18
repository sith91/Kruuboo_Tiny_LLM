import os
import time
import subprocess
import sys
from pathlib import Path

# Paths
WORKSPACE = Path(__file__).resolve().parent.parent
LOCK_PATH = "/tmp/kruboo_chat_active.lock"
INTERVAL_SECONDS = 3600  # Check/Run every hour

def is_chat_active():
    return os.path.exists(LOCK_PATH)

def run_cmd(cmd, cwd):
    print(f"🚀 Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode == 0

def main():
    print("🤖 Kruboo Continual Learning Daemon Started...")
    print(f"📁 Workspace: {WORKSPACE}")
    print(f"⏳ Interval: {INTERVAL_SECONDS} seconds")
    
    while True:
        try:
            if is_chat_active():
                print("⏸️ Chat is active (lock file present). Skipping this iteration...")
            else:
                print("⚡ Starting incremental training run...")
                # Run train.py for fast incremental updates
                train_ok = run_cmd([sys.executable, "src/train.py"], WORKSPACE)
                
                if train_ok:
                    print("✅ Incremental training completed successfully. Exporting ONNX...")
                    run_cmd([sys.executable, "scripts/export_onnx.py"], WORKSPACE)
                else:
                    print("❌ Incremental training run failed.")
            
        except Exception as e:
            print(f"⚠️ Daemon Error: {e}")
            
        # Wait for the next interval cycle
        time.sleep(INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
