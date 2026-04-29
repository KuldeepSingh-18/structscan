"""
StructScan — Run Script
Place in: structural-damage-detector/ (the inner folder with backend/)
"""
import argparse
import os
import socket
import threading
import time
import webbrowser

# Suppress TensorFlow verbose warnings BEFORE any imports
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import uvicorn


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",       default="0.0.0.0")
    parser.add_argument("--port",       type=int, default=8000)
    parser.add_argument("--reload",     action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    local_ip   = get_local_ip()
    pc_url     = f"http://127.0.0.1:{args.port}"
    mobile_url = f"http://{local_ip}:{args.port}"

    print(f"""
╔══════════════════════════════════════════════════════╗
║   StructScan — Early Structural Damage Detection     ║
╠══════════════════════════════════════════════════════╣
║  This PC   : {pc_url:<39}║
║  Mobile QR : {mobile_url:<39}║
║  Scan QR code on Live Camera page (same WiFi)        ║
║  Press Ctrl+C to stop                                ║
╚══════════════════════════════════════════════════════╝
""")

    if not args.no_browser:
        def _open():
            time.sleep(2.0)
            webbrowser.open(pc_url)
        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="warning",
    )
