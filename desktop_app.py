import webview
import threading
import subprocess
import time
import sys
import os

PORT = 8501
URL = f"http://127.0.0.1:{PORT}"

def start_streamlit():
    subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", 
         "--server.port", str(PORT), 
         "--server.headless", "true", 
         "--global.developmentMode", "false"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

if __name__ == "__main__":
    threading.Thread(target=start_streamlit, daemon=True).start()

    time.sleep(5)

    webview.create_window("NBA Edge Predictor Pro", URL, width=1200, height=800)
    webview.start()