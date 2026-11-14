# SignBridge AI — Backend Service

The Python FastAPI backend service for **SignBridge AI**, providing API endpoints for video ingestion, MediaPipe feature extraction, PyTorch sequence inference, confidence calibration, and bilingual (English/Hindi) translation services.

---

## 1. Directory Structure

```text
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI application factory & lifespan context
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py               # Pydantic Settings & environment variables
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       └── health.py           # Health and readiness check endpoint
│   ├── services/                   # Business logic (video processing, translators)
│   ├── schemas/                    # Pydantic request/response schemas
│   │   ├── __init__.py
│   │   └── health.py
│   └── ml/                         # Machine learning models & inference runners
├── tests/
│   ├── __init__.py
│   └── test_health.py              # Automated test suite for health & CORS
├── requirements.txt                # Initial production & testing dependencies
├── .env.example                    # Environment variable template
└── README.md
```

---

## 2. Setup & Virtual Environment

### Prerequisites
- Python 3.10+ (tested on Python 3.12.10)

### Create Virtual Environment
```bash
# Navigate to the backend directory
cd backend

# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Activate on Linux/macOS
source .venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 3. Configuration

Copy `.env.example` to `.env` to customize runtime settings:

```bash
cp .env.example .env
```

Key environment configurations:
- `ALLOWED_ORIGINS`: Allowed CORS origins for the frontend (defaults to `http://localhost:3000`, `http://localhost:5173`).
- `MAX_VIDEO_DURATION_SEC`: Maximum allowed video duration (default: `8.0`).
- `TARGET_SAMPLE_FPS`: Sampling rate for frame processing (default: `20`).
- `CALIBRATED_CONFIDENCE_THRESHOLD`: Decision threshold for abstention gating (default: `0.50`).
- `MIN_LANDMARK_DETECTION_RATIO`: Hand landmark quality gate (default: `0.60`).

---

## 4. Running the Development Server

Start the FastAPI application with Uvicorn:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Once running:
- **Interactive OpenAPI Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Health Check Endpoint:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) or [http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)

---

## 5. Running Automated Tests

Run the test suite using `pytest`:

```bash
pytest
```

To run with verbose output:

```bash
pytest -v
```

