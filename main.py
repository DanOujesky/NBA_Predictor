import webview
import subprocess
import time
import sys
import os
import multiprocessing
from pathlib import Path
import urllib.request

PORT = 8501
URL = f"http://127.0.0.1:{PORT}"


def get_base_path():
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(os.path.abspath(".")).resolve()


def wait_for_server(timeout=30):
    """Čeká dokud Streamlit server neběží"""
    start = time.time()

    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(URL):
                return True
        except:
            time.sleep(0.5)

    return False


def start_streamlit():
    base_path = get_base_path()
    app_path = base_path / "app.py"

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(PORT),
        "--server.headless", "true",
        "--server.address", "127.0.0.1"
    ]

    env = os.environ.copy()

    return subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()

    if not os.environ.get("STREAMLIT_RUNNING"):
        os.environ["STREAMLIT_RUNNING"] = "true"

        proc = start_streamlit()

        if not wait_for_server():
            print("Streamlit server se nepodařilo spustit")
            proc.kill()
            sys.exit(1)

        try:
            window = webview.create_window(
                "NBA Predictor",
                URL,
                width=1300,
                height=900,
                resizable=True
            )
            webview.start()
        finally:
            proc.kill()