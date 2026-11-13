# SignBridge AI: Backend Architecture & Implementation Plan

**Document ID:** `03-backend-plan`  
**Status:** Planning & API Blueprint  
**Primary Reference:** `docs/project-blueprint.pdf` (Sections 7, 8, 11, 12, 14, 18)  
**Last Updated:** 2026-11-13  

---

## 1. Overview & Objectives

The backend service for **SignBridge AI** is built using **Python 3.10+** and **FastAPI**. It acts as the core gateway and orchestration engine responsible for:
- Providing secure, validated REST endpoints for web clients.
- Receiving and validating raw video payloads (WebM / MP4) with zero persistent storage.
- Preprocessing video frames (downsampling, FPS adjustment, coordinate normalization).
- Executing visual landmark extraction (MediaPipe Hands / Holistic).
- Feeding temporal feature tensors to trained PyTorch models for sentence recognition.
- Generating bilingual outputs (English text and Hindi translation).
- Managing domain context without introducing false predictions.
- Enforcing strict privacy boundaries (ephemeral processing and zero disk leak).

---

## 2. Project Directory Structure

The `backend/` directory will follow an idiomatic, clean layered FastAPI architecture:

```text
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI application factory & lifespan
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py               # Pydantic Settings (env vars, thresholds)
│   │   └── security.py             # CORS, rate limits, request headers
│   ├── api/
│   │   ├── __init__.py
│   │   ├── router.py               # Main API router mounting all v1 routes
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── health.py           # Health check & system status
│   │       ├── domains.py          # Domain list and supported phrase metadata
│   │       └── inference.py        # Video ingestion and translation route
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── health.py               # Pydantic schemas for health & diagnostics
│   │   ├── domains.py              # Schemas for domain definitions
│   │   └── inference.py            # Input validation & translation responses
│   └── services/
│       ├── __init__.py
│       ├── video_processor.py      # OpenCV video decoding and frame sampling
│       ├── feature_extractor.py    # MediaPipe landmark extraction (Hands/Pose)
│       ├── model_runner.py         # PyTorch inference wrapper (BiLSTM/GRU)
│       └── translator.py           # English to Hindi translation service
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Pytest fixtures & mock clients
│   ├── test_health.py              # Health check test cases
│   ├── test_domains.py             # Domain endpoint test cases
│   ├── test_video_processing.py    # Frame extraction & corruption handling
│   └── test_inference.py           # Mock inference and validation tests
├── requirements.txt                # Production dependencies
├── requirements-dev.txt            # Development & testing dependencies
├── Dockerfile                      # Container definition for reproducible runtime
└── .env.example                    # Template for environment configuration
```

---

## 3. Configuration & Environment Variables

All configuration is managed through Pydantic's `BaseSettings` in `app/core/config.py`:

```python
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    PROJECT_NAME: str = "SignBridge AI Backend"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False
    
    # CORS Configuration
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",  # Vite default dev server
        "http://127.0.0.1:5173"
    ]
    
    # Video & Inference Parameters
    MAX_VIDEO_SIZE_MB: int = 25
    MAX_VIDEO_DURATION_SEC: float = 12.0
    SAMPLE_FPS: int = 25
    SEQUENCE_LENGTH: int = 60  # Fixed sequence length for temporal model
    CONFIDENCE_THRESHOLD: float = 0.55  # Below this, model abstains
    
    # Model Weights Path
    MODEL_WEIGHTS_PATH: str = "models/weights/baseline_isl.pth"
    DEVICE: str = "cpu"  # "cuda" if GPU is available
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
```

---

## 4. API Endpoint Specifications

### 4.1. Health & Readiness (`/api/v1/health`)
- **Method:** `GET`
- **Purpose:** System readiness, model loading verification, and uptime check.
- **Response Example (`200 OK`):**
  ```json
  {
    "status": "healthy",
    "version": "1.0.0",
    "model_loaded": true,
    "device": "cpu",
    "supported_domains": 7
  }
  ```

### 4.2. Domains Metadata (`/api/v1/domains`)
- **Method:** `GET`
- **Purpose:** Returns the seven domains and their metadata for frontend synchronization.
- **Response Example (`200 OK`):**
  ```json
  {
    "domains": [
      {
        "id": "education",
        "name": "Education",
        "validated_scope": "Basic classroom inquiries"
      },
      {
        "id": "healthcare",
        "name": "Healthcare",
        "validated_scope": "Emergency symptom reporting"
      }
    ]
  }
  ```

### 4.3. Video Translation Inference (`/api/v1/inference/translate`)
- **Method:** `POST`
- **Consumes:** `multipart/form-data`
- **Parameters:**
  - `video: UploadFile`: Binary video file (WebM or MP4, max 25 MB).
  - `domain: str`: One of the 7 supported domain IDs.
  - `target_language: str`: Target language for translation (`"en"` or `"hi"`).

#### Success Response (`200 OK`)
```json
{
  "status": "success",
  "domain": "healthcare",
  "english_text": "I need help finding the doctor.",
  "hindi_text": "मुझे डॉक्टर को खोजने में मदद चाहिए।",
  "confidence": 0.89,
  "abstained": false,
  "inference_latency_ms": 284,
  "metadata": {
    "frames_processed": 75,
    "landmarks_detected_ratio": 0.96
  }
}
```

#### Abstention Response (`200 OK`)
When confidence is below threshold ($\tau < 0.55$) or hands were occluded:
```json
{
  "status": "abstained",
  "domain": "healthcare",
  "english_text": "",
  "hindi_text": "",
  "confidence": 0.31,
  "abstained": true,
  "guidance": "Sign movement was unclear or poorly framed. Please repeat the sign with hands in full view.",
  "metadata": {
    "frames_processed": 50,
    "landmarks_detected_ratio": 0.38
  }
}
```

---

## 5. Video Ingestion & Preprocessing Pipeline

Video files recorded in the browser come in varying resolutions, encodings (VP8/VP9/H264 in WebM), and frame rates. The preprocessing pipeline standardizes these inputs into uniform tensor sequences:

```mermaid
flowchart LR
    Upload[Uploaded WebM/MP4 Bytes] --> InMem[In-Memory Buffer / Ephemeral Stream]
    InMem --> CV[OpenCV VideoCapture]
    CV --> Sample[Uniform Frame Sampling @ 25 FPS]
    Sample --> Quality[Quality Check: Blur & Illumination]
    Quality --> Norm[Resize & Coordinate Normalization]
    Norm --> MP[MediaPipe Landmark Extractor]
    MP --> Tensor[Normalized Feature Tensor [B, T, D]]
```

### Preprocessing Steps:
1. **Format Validation:** Verify that MIME type is `video/webm` or `video/mp4` and magic bytes correspond to valid video containers.
2. **OpenCV Ingestion:** Read frames sequentially. If total frames $<15$ (too short) or $>300$ (over 12 seconds), reject with `400 Bad Request`.
3. **Temporal Sampling:** Resample frames to fixed 25 FPS using linear frame index interpolation.
4. **Coordinate Normalization:** Rather than feeding raw pixel grids into large CNNs, pass frames to MediaPipe to extract landmark coordinates:
   - Left Hand: 21 landmarks $\times 3$ coordinates $(x, y, z) = 63$ values.
   - Right Hand: 21 landmarks $\times 3$ coordinates $(x, y, z) = 63$ values.
   - Upper Pose (Shoulders, elbows, wrists, nose): 10 landmarks $\times 3 = 30$ values.
   - **Total Feature Dimension per frame ($D$):** $156$ floating-point values.
5. **Sequence Padding / Truncation:** Pad with zeros or sample to standard sequence length $T = 60$. Resulting model input shape: `(1, 60, 156)`.

---

## 6. Machine Learning Inference Integration

### Model Lifecycle & Lifespan Management
FastAPI's `lifespan` context manager handles loading the model into memory **once at startup** rather than on each request:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
import torch

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model weights
    model_path = settings.MODEL_WEIGHTS_PATH
    if os.path.exists(model_path):
        model = ISLTemporalModel(input_dim=156, hidden_dim=128, num_classes=NUM_CLASSES)
        model.load_state_dict(torch.load(model_path, map_location=settings.DEVICE))
        model.eval()
        ml_models["isl_model"] = model
    else:
        # Graceful placeholder mode during early development
        ml_models["isl_model"] = None
        print("WARNING: Model weights not found. Inference will report 503 until model is trained.")
        
    yield
    
    # Shutdown: Clean up memory
    ml_models.clear()
```

### Abstention & Confidence Thresholding
The inference pipeline applies strict confidence gating:
$$\text{Confidence} = \max_{k} \text{Softmax}(z)_k$$
- If $\text{Confidence} \ge \tau$ (e.g., $0.55$): Return predicted sentence.
- If $\text{Confidence} < \tau$: **Abstain** and trigger repeat prompt.
- **Never fabricate** a sentence or return a random top-1 guess when uncertain.

---

## 7. Privacy & Ephemeral Video Processing

To honor the core privacy requirements of the blueprint:
- **No Video Persistence:** Video files are received into memory streams (`io.BytesIO`) or temporary directories (`tempfile.NamedTemporaryFile`).
- **Immediate Cleanup:** Ingestion functions wrap video operations in `try ... finally` blocks to ensure that any temporary files on disk are deleted immediately after frame extraction:
  ```python
  temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
  try:
      temp_file.write(await video.read())
      temp_file.flush()
      # Run OpenCV extraction...
  finally:
      temp_file.close()
      if os.path.exists(temp_file.name):
          os.remove(temp_file.name)
  ```
- **No Private Logging:** Raw frames, user faces, and video streams are never logged to console or stored in database tables.

---

## 8. Backend Testing Strategy

Comprehensive automated tests will be executed with **`pytest`**:

| Test Suite | File | Focus Areas |
| :--- | :--- | :--- |
| **Health & Config** | `tests/test_health.py` | Verify `/health` returns 200, checks model load status, CORS headers. |
| **Domain Metadata** | `tests/test_domains.py` | Verify all 7 domains are returned with required fields. |
| **Video Validation** | `tests/test_video_processing.py` | Rejection of non-video files, handling corrupted video streams, handling oversized files. |
| **Inference Logic** | `tests/test_inference.py` | Mock model inference, confidence thresholding, verified abstention behavior on low confidence. |
| **Privacy Compliance** | `tests/test_privacy.py` | Verify that no leftover temporary video files exist in `/tmp` after inference calls. |

