"""
Language Detector - Backend Server  (fixed for Python 3.14)
Run:  python server.py
Then open app.html in your browser.
"""

import os, sys, json, time, threading, tempfile, warnings, logging
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ── dependency check ──────────────────────────────────────────────────────────
def check_and_install():
    missing = []
    for pkg in ["whisper", "pandas", "requests", "openpyxl"]:
        try: __import__(pkg)
        except ImportError: missing.append(pkg)
    if missing:
        print(f"\n📦 Installing: {', '.join(missing)} ...")
        os.system(f"{sys.executable} -m pip install openai-whisper pandas openpyxl requests --quiet")
        print("✅ Installed!\n")

check_and_install()

import whisper, pandas as pd, requests

# ── globals ───────────────────────────────────────────────────────────────────
state = {
    "status": "idle",
    "total": 0, "done": 0, "errors": 0,
    "current_url": "", "current_lang": "",
    "lang_counts": {}, "model_name": "base",
    "model": None, "df": None, "filepath": None,
    "start_time": None, "log": [],
}
state_lock = threading.Lock()

LANGUAGE_NAMES = {
    "hi":"Hindi","en":"English","bn":"Bengali","te":"Telugu","mr":"Marathi",
    "ta":"Tamil","ur":"Urdu","gu":"Gujarati","kn":"Kannada","pa":"Punjabi",
    "ml":"Malayalam","or":"Odia","as":"Assamese","ne":"Nepali","zh":"Chinese",
    "ar":"Arabic","fr":"French","de":"German","es":"Spanish","pt":"Portuguese",
    "ru":"Russian","ja":"Japanese","ko":"Korean","it":"Italian","tr":"Turkish",
    "id":"Indonesian","th":"Thai","vi":"Vietnamese","ms":"Malay","fa":"Persian",
}

def lang_name(code):
    return LANGUAGE_NAMES.get((code or "").lower(), (code or "Unknown").upper())

def log(msg):
    with state_lock:
        state["log"].append(msg)
        if len(state["log"]) > 300:
            state["log"] = state["log"][-300:]
    print(msg)

# ── smart file reader ─────────────────────────────────────────────────────────
def read_file(filepath):
    """Read CSV or Excel by sniffing magic bytes — ignores extension."""
    with open(filepath, "rb") as f:
        magic = f.read(8)

    is_excel = (
        magic[:4] == b"PK\x03\x04" or
        magic[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
    )

    if is_excel:
        log("   Format detected: Excel")
        try:
            return pd.read_excel(filepath, header=None, engine="openpyxl")
        except Exception:
            return pd.read_excel(filepath, header=None, engine="xlrd")
    else:
        log("   Format detected: CSV")
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                return pd.read_csv(filepath, header=None, encoding=enc)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(filepath, header=None, encoding="latin-1")

# ── audio download ────────────────────────────────────────────────────────────
def download_audio(url, timeout=30):
    try:
        url = str(url).strip()
        if not url or url.lower() in ("nan","none",""): return None, "Empty URL"
        r = requests.get(url, timeout=timeout,
                         headers={"User-Agent": "LangDetector/1.0"}, stream=True)
        r.raise_for_status()
        ct = r.headers.get("Content-Type", "")
        ext = ".mp3"
        for fmt in ["wav","ogg","mp4","m4a","flac","webm"]:
            if fmt in ct or url.lower().endswith(f".{fmt}"):
                ext = f".{fmt}"; break
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        for chunk in r.iter_content(8192): tmp.write(chunk)
        tmp.close()
        return tmp.name, None
    except requests.exceptions.Timeout: return None, "Timeout"
    except requests.exceptions.HTTPError as e: return None, f"HTTP {e.response.status_code}"
    except Exception as e: return None, str(e)[:50]

# ── language detection ────────────────────────────────────────────────────────
def detect_lang(path, model):
    try:
        audio = whisper.load_audio(path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        _, probs = model.detect_language(mel)
        code = max(probs, key=probs.get)
        return lang_name(code), round(probs[code] * 100, 1), code
    except Exception as e:
        return "Error", 0.0, "err"

# ── main processing thread ────────────────────────────────────────────────────
def run_detection(filepath, model_name, resume):
    with state_lock:
        state["status"] = "loading_model"
        state["filepath"] = filepath
        state["start_time"] = time.time()
        state["done"] = 0; state["errors"] = 0
        state["lang_counts"] = {}; state["log"] = []

    log(f"📂 Loading file: {filepath}")
    try:
        df = read_file(filepath)
        while df.shape[1] < 3:
            df[df.shape[1]] = ""
        df.columns = list(range(df.shape[1]))
        log(f"   Total rows in file: {len(df):,}")
    except Exception as e:
        with state_lock:
            state["status"] = "error"
            state["log"].append(f"❌ File read error: {e}")
        return

    # Build list of rows to process
    rows = []
    skipped = 0
    empty = 0
    for i, row in df.iterrows():
        url = str(row[0]).strip()
        existing = str(row[1]).strip() if pd.notna(row[1]) else ""
        if not url or url.lower() in ("nan", "none", ""):
            empty += 1
            continue
        if resume and existing and existing.lower() not in ("nan", "none", ""):
            skipped += 1
            continue
        rows.append((i, url))

    log(f"   To process: {len(rows):,} | Already done: {skipped:,} | Empty rows: {empty:,}")

    with state_lock:
        state["total"] = len(rows)
        state["df"] = df

    if len(rows) == 0:
        with state_lock:
            state["status"] = "done"
            state["log"].append("✅ Nothing to process — all rows already have results!")
        return

    log(f"🤖 Loading Whisper '{model_name}' model (first run downloads it)...")

    try:
        model = whisper.load_model(model_name)
    except Exception as e:
        with state_lock:
            state["status"] = "error"
            state["log"].append(f"❌ Model load failed: {e}")
        return

    with state_lock:
        state["status"] = "processing"
    log("✅ Model ready! Processing recordings...\n")

    # Detect file extension for saving
    with open(filepath, "rb") as f:
        magic = f.read(8)
    is_excel = magic[:4] == b"PK\x03\x04" or magic[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

    def save_progress():
        try:
            if is_excel:
                df.to_excel(filepath, header=False, index=False, engine="openpyxl")
            else:
                df.to_csv(filepath, header=False, index=False)
        except Exception as e:
            log(f"⚠️ Save error: {e}")

    for i, url in rows:
        with state_lock:
            if state["status"] == "idle":
                break
            state["current_url"] = url

        tmp, err = download_audio(url)
        if not tmp:
            with state_lock:
                df.at[i, 1] = f"FAILED: {err}"
                df.at[i, 2] = "0%"
                state["errors"] += 1
                state["done"] += 1
            log(f"  ❌ Row {i+1}: {err}")
            continue

        try:
            lname, conf, code = detect_lang(tmp, model)
        finally:
            try: os.unlink(tmp)
            except: pass

        with state_lock:
            df.at[i, 1] = lname
            df.at[i, 2] = f"{conf}%"
            state["current_lang"] = lname
            state["done"] += 1
            state["lang_counts"][lname] = state["lang_counts"].get(lname, 0) + 1

        log(f"  ✅ Row {i+1}: {lname} ({conf}%)")

        if state["done"] % 100 == 0:
            save_progress()
            log(f"  💾 Auto-saved at {state['done']} rows")

    save_progress()

    with state_lock:
        state["status"] = "done"
        elapsed = time.time() - state["start_time"]
        state["log"].append(f"\n🏁 Done! {state['done']} rows in {elapsed/60:.1f} min. File saved.")

    log(f"\n🏁 Finished! File saved: {filepath}")

# ── HTTP request handler ──────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass

    def send_json(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if urlparse(self.path).path == "/status":
            with state_lock:
                elapsed = time.time() - state["start_time"] if state["start_time"] else 0
                done = state["done"]; total = state["total"]
                rate = done / elapsed if elapsed > 0 and done > 0 else 0
                eta = (total - done) / rate if rate > 0 and total > done else 0
                self.send_json({
                    "status": state["status"],
                    "total": total, "done": done, "errors": state["errors"],
                    "current_url": state["current_url"][-70:],
                    "current_lang": state["current_lang"],
                    "lang_counts": state["lang_counts"],
                    "pct": round(done / total * 100, 1) if total > 0 else 0,
                    "elapsed_min": round(elapsed / 60, 1),
                    "eta_min": round(eta / 60, 1),
                    "rate_per_min": round(rate * 60, 1),
                    "log": state["log"][-40:],
                    "filepath": state["filepath"] or "",
                })
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/start":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            fp = body.get("filepath", "").strip()
            model = body.get("model", "base")
            resume = body.get("resume", True)
            if not fp or not os.path.exists(fp):
                self.send_json({"error": f"File not found: {fp}"}, 400); return
            if state["status"] == "processing":
                self.send_json({"error": "Already running"}); return
            t = threading.Thread(target=run_detection, args=(fp, model, resume), daemon=True)
            t.start()
            self.send_json({"ok": True})

        elif path == "/stop":
            with state_lock:
                state["status"] = "idle"
                state["log"].append("⏹️ Stopped by user.")
            self.send_json({"ok": True})

        elif path == "/upload":
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                self.send_json({"error": "Expected multipart/form-data"}, 400); return

            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)

            filename = "uploaded_file"
            file_content = None

            try:
                # Parse boundary
                boundary = None
                for seg in content_type.split(";"):
                    seg = seg.strip()
                    if seg.lower().startswith("boundary="):
                        boundary = seg[9:].strip().strip('"').encode()
                        break
                if not boundary:
                    self.send_json({"error": "No boundary found"}, 400); return

                # Split and parse parts
                for section in raw.split(b"--" + boundary):
                    if b"Content-Disposition" not in section:
                        continue
                    if b"filename=" not in section:
                        continue
                    # Split on first double newline (headers vs body)
                    sep = b"\r\n\r\n" if b"\r\n\r\n" in section else b"\n\n"
                    if sep not in section:
                        continue
                    hdr_bytes, body = section.split(sep, 1)
                    hdr = hdr_bytes.decode("utf-8", errors="ignore")
                    # Extract filename
                    for part in hdr.split(";"):
                        part = part.strip()
                        if part.lower().startswith("filename="):
                            filename = part[9:].strip().strip('"').strip("'")
                            break
                    # Strip trailing boundary junk
                    body = body.rstrip(b"\r\n-")
                    if body:
                        file_content = body
                    break

            except Exception as e:
                self.send_json({"error": f"Upload parse error: {e}"}, 400); return

            if not file_content:
                self.send_json({"error": "No file data received"}, 400); return

            # Ensure filename has correct extension based on file magic bytes
            magic = file_content[:8]
            safe = os.path.basename(filename).replace("..", "").replace("/","").replace("\\","")
            # If the browser stripped the extension, detect and add it
            if "." not in safe:
                if magic[:4] == b"PK\x03\x04":
                    safe += ".xlsx"
                else:
                    safe += ".csv"

            save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), safe)
            with open(save_path, "wb") as f:
                f.write(file_content)

            log(f"📁 Uploaded: {safe} ({len(file_content):,} bytes)")
            self.send_json({"ok": True, "filepath": save_path, "filename": safe})

        else:
            self.send_response(404); self.end_headers()


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PORT = 7845
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"\n{'='*55}")
    print(f"  🎙️  Language Detector Server")
    print(f"{'='*55}")
    print(f"  ✅ Server running on http://127.0.0.1:{PORT}")
    print(f"  📂 Now open  app.html  in your browser")
    print(f"  🛑 Press Ctrl+C to stop")
    print(f"{'='*55}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
