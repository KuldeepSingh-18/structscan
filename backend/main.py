"""
StructScan — FastAPI Backend v4
"""
import base64
import io
import json
import os
import socket
import tempfile
import traceback
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.alert import AlertSystem
from backend.model import DamageDetector

app = FastAPI(title="StructScan", version="4.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

detector  = DamageDetector()
alert_sys = AlertSystem()


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def frame_to_b64(frame_bgr: np.ndarray, quality: int = 80) -> str:
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")


def build_response(result: dict, quality: int = 80) -> dict:
    alert         = alert_sys.generate_alert(result)
    annotated_b64 = frame_to_b64(result.pop("annotated_frame"), quality)
    result.pop("zones", [])
    return {**result, **alert, "annotated_frame": annotated_b64}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return HTMLResponse(
        (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    )


@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "demo_mode":    detector.demo_mode,
        "model_loaded": detector.model is not None,
        "opencv":       cv2.__version__,
        "local_ip":     get_local_ip(),
    }


@app.get("/network-url")
async def network_url(port: int = 8000):
    ip = get_local_ip()
    return {"url": f"http://{ip}:{port}", "ip": ip, "port": port}


@app.get("/qr")
async def get_qr(port: int = 8000):
    """
    Generate QR code PNG for the local WiFi URL.
    Mobile devices scan this to connect over the same WiFi.
    """
    try:
        import qrcode

        ip  = get_local_ip()
        url = f"http://{ip}:{port}"

        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=8,
            border=3,
        )
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        return Response(
            content=buf.read(),
            media_type="image/png",
            headers={"X-Network-URL": url, "Cache-Control": "no-cache"},
        )

    except ImportError:
        # qrcode not installed — return a simple SVG placeholder with the URL
        ip  = get_local_ip()
        url = f"http://{ip}:{port}"
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="200" height="60">
          <rect width="200" height="60" fill="white" stroke="#ccc"/>
          <text x="10" y="20" font-size="11" font-family="monospace">QR not available</text>
          <text x="10" y="40" font-size="10" font-family="monospace" fill="#333">{url}</text>
          <text x="10" y="55" font-size="9" font-family="monospace" fill="#888">pip install qrcode pillow</text>
        </svg>"""
        return Response(content=svg, media_type="image/svg+xml")

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        data  = await file.read()
        nparr = np.frombuffer(data, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse(
                {"error": "Cannot decode image — try JPG or PNG"},
                status_code=400,
            )
        result = detector.analyze(img)
        return JSONResponse(build_response(result))
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    tmp_path = None
    cap      = None
    try:
        data = await file.read()
        if not data:
            return JSONResponse({"error": "Empty file"}, status_code=400)

        print(f"📹 Video: {file.filename} | {len(data)/1048576:.1f} MB")

        fname  = (file.filename or "video.mp4").lower()
        suffix = Path(fname).suffix
        if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}:
            suffix = ".mp4"

        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
        except Exception:
            os.close(fd)
            raise

        for backend in [cv2.CAP_ANY, cv2.CAP_FFMPEG, cv2.CAP_MSMF]:
            c = cv2.VideoCapture(tmp_path, backend)
            if c.isOpened():
                ret, test = c.read()
                if ret and test is not None:
                    cap = c
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    break
                c.release()

        if cap is None:
            return JSONResponse(
                {"error": "Cannot open video. Use MP4 (H.264) format."},
                status_code=400,
            )

        fps_v        = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration     = round(total_frames / max(fps_v, 1), 1)
        sample_every = max(int(fps_v * 2), 20)

        frame_results = []
        frame_idx     = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_every == 0:
                h, w = frame.shape[:2]
                if h > 480:
                    frame = cv2.resize(frame, (int(w * 480 / h), 480))
                result = detector.analyze(frame)
                fr = build_response(result, quality=70)
                frame_results.append({
                    "frame_index":   frame_idx,
                    "timestamp_sec": round(frame_idx / max(fps_v, 1), 1),
                    **fr,
                })
            frame_idx += 1

        cap.release()
        cap = None

        return JSONResponse({
            "frames":             frame_results,
            "total_frames":       frame_idx,
            "frames_analyzed":    len(frame_results),
            "video_duration_sec": duration,
            "max_severity":       max(
                (f.get("severity_score", 0) for f in frame_results), default=0
            ),
            "cracked_frames": sum(
                1 for f in frame_results if f.get("label") == "Cracked"
            ),
        })

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": f"Video error: {str(e)}"}, status_code=500)
    finally:
        if cap:
            try:
                cap.release()
            except Exception:
                pass
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@app.websocket("/ws/camera")
async def websocket_camera(websocket: WebSocket):
    await websocket.accept()
    print("📡 Camera connected")
    try:
        while True:
            raw = await websocket.receive_text()
            if "," in raw:
                raw = raw.split(",", 1)[1]
            img_bytes = base64.b64decode(raw)
            nparr     = np.frombuffer(img_bytes, np.uint8)
            frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                await websocket.send_text(json.dumps({"error": "Bad frame"}))
                continue

            # Resize for speed — live frames don't need to be full res
            lh, lw = frame.shape[:2]
            if lw > 480:
                scale = 480 / lw
                frame = cv2.resize(frame, (480, int(lh * scale)))

            result   = detector.analyze(frame, live=True)
            response = build_response(result, quality=75)
            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        print("📡 Camera disconnected")
    except Exception as e:
        print(f"❌ WS error: {e}")
