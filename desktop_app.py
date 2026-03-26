import webview
import threading
import subprocess
import time
import sys
import os
from pathlib import Path

PORT = 8501
URL = f"http://127.0.0.1:{PORT}"

def start_streamlit():
    base_path = Path(__file__).parent
    app_path = base_path / "app.py"
    
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(PORT),
        "--server.headless", "true",
        "--global.developmentMode", "false"
    ]
    
    return subprocess.Popen(cmd)

if __name__ == "__main__":
    proc = start_streamlit()

    time.sleep(5)

    try:
        window = webview.create_window(
            "🏀 NBA Predictor", 
            URL, 
            width=1300, 
            height=900,
            resizable=True
        )
        webview.start()
    finally:
        proc.kill()