# SignBridge AI

> An AI-powered Indian Sign Language (ISL) to Text and Speech Translation Platform.

SignBridge AI is an assistive communication platform designed to bridge the communication gap between Indian Sign Language (ISL) users and individuals who do not understand sign language. By leveraging computer vision and temporal machine learning models, SignBridge AI translates continuous ISL signing gestures captured via camera into meaningful sentence-level text and natural spoken audio in both English and Hindi.

---

## 🌟 Key Features

- **Camera-Based Signing Input**: Direct browser-based camera capture without requiring specialized sensory gloves or hardware.
- **Visual Feature Extraction**: Landmark detection (hands, pose, and facial expressions) to capture spatial and temporal nuances of ISL.
- **Sentence-Level Translation**: Moves beyond isolated gesture/alphabet recognition toward continuous sentence interpretation.
- **Bilingual Output (English & Hindi)**: Text and speech synthesis with clear language selection.
- **Human-in-the-Loop Review**: Users can review and edit recognized sentences before triggering audio playback to ensure accuracy.
- **Confidence & Uncertainty Handling**: Abstains and prompts for repetition when video framing, lighting, or model confidence is insufficient.
- **Privacy First**: Temporary in-memory frame processing with no default storage or logging of raw video feeds.

---

## 🏛️ Seven Communication Domains

To organize vocabulary and conversational context, SignBridge AI provides dedicated interaction modules across seven settings:

1. **Education**: Classroom questions, explanations, attendance, and learning support.
2. **Healthcare**: Describing symptoms, expressing urgent needs, appointments, and assistance.
3. **Banking**: Counter inquiries, account queries, and transactional guidance.
4. **Workplaces**: Introductions, meeting coordination, and workplace requests.
5. **Travel**: Directions, ticketing, transport navigation, and accommodation.
6. **Government & Public Services**: Civic forms, documentation assistance, and service desk interactions.
7. **Everyday Conversations**: Greetings, family discussions, time, feelings, and social interactions.

> **Note**: While the platform interface introduces all seven domains, model recognition capabilities are strictly validated against genuine, trained ISL datasets for specific vocabularies and contexts.

---

## 🛠️ Technology Stack

- **Frontend**: React, Vite, Tailwind CSS / Modern CSS, Web Speech API (SpeechSynthesis)
- **Backend API**: Python, FastAPI, Uvicorn
- **Computer Vision & ML**: OpenCV, MediaPipe, PyTorch
- **Experimentation & Research**: Jupyter Notebooks, Google Colab
- **Version Control**: Git & GitHub

---

## 📁 Repository Structure

```text
SignBridge-AI/
├── docs/
│   └── project-blueprint.pdf    # Complete project overview & implementation blueprint
├── frontend/                    # React + Vite web client (UI, camera capture, speech)
├── backend/                     # FastAPI inference service & API endpoints
├── notebooks/                   # ML exploration, dataset preprocessing & training
├── README.md                    # Project documentation
└── .gitignore                   # Git exclusion rules
```

---

## 🛡️ Responsible AI & Ethical Boundaries

SignBridge AI is developed as an assistive communication prototype. It is **not** a certified substitute for qualified human sign language interpreters, particularly in high-stakes legal, medical, or financial scenarios. Transparent uncertainty reporting and human confirmation are core design principles of this platform.

