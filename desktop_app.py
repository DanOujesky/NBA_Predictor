import webview
import threading
import subprocess
import time
import sys

PORT = 8501
URL = f"http://127.0.0.1:{PORT}"

def start_streamlit():
    subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(PORT)],
        stdout=None,
        stderr=None
    )

if __name__ == "__main__":
    threading.Thread(target=start_streamlit, daemon=True).start()

    time.sleep(3)

    webview.create_window("NBA Predictor", URL)

    webview.start()