# 🌐 Call Language Detector

An internal tool that detects the spoken language in bulk call recordings — helping analyse regional language distribution across customer interactions.

---

## 📸 What It Does

Upload a CSV/Excel file of call recording URLs → select a Whisper model → run detection → download the same file with a **Language** column filled in for every row.

---

## ✨ Features

- 📁 **Drag & drop file upload** — supports `.csv` and `.xlsx`
- 🤖 **Multiple Whisper model options** — choose speed vs. accuracy
- ▶️ **Start / Stop processing** — safely pause and resume anytime
- 📊 **Real-time progress bar** — with row count and percentage
- 🌍 **Language distribution chart** — live breakdown as results come in
- 📜 **Colour-coded live log** — success (green), error (red), info (blue), warning (yellow)
- ⚠️ **Server health check** — warns if the Python backend isn't running
- 💾 **Output saved automatically** — completed file written back to disk

---

## 🤖 Model Options

| Model | Size | Speed | Recommended For |
|---|---|---|---|
| `tiny` | ~39 MB | Fastest | Quick test runs only |
| `base` | ~74 MB | Fast | Hindi & English · **Recommended** ✅ |
| `small` | ~244 MB | ~2× slower | Better accuracy |
| `medium` | ~769 MB | ~5× slower | Highest accuracy |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Vanilla HTML, CSS, JavaScript |
| Backend | Python (local server on port `7845`) |
| AI Model | OpenAI Whisper |

---

## ⚙️ Setup

### 1. Install Python dependencies
```bash
pip install openai-whisper flask pandas openpyxl
```

### 2. Start the backend server
```bash
python server.py
```
The server runs at `http://127.0.0.1:7845`. Keep this terminal open while using the app.

### 3. Open the frontend
Open `index.html` in any browser. The app checks the server automatically on load.

---

## 📄 Input File Format

Your CSV or Excel file must have:
- **Column A** — Call recording URLs (publicly accessible or local paths the server can reach)
- **Column B** — Will be populated with the detected language

---

## 📁 File Structure

```
call-language-detector/
├── index.html      ← frontend UI
└── server.py       ← Python backend (Whisper + Flask)
```

---

## 💡 Use Case

Operations and QA teams use this to:
- Understand which languages customers are calling in
- Plan call centre staffing by language/region
- Route calls to the right language-trained consultants
- Generate regional language reports for leadership

---

*Built for internal use · Rachit Tiwari*
