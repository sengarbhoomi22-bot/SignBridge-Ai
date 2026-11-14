# SignBridge AI: Frontend Technical Audit & Architecture Review

**Audit Date:** 2026-11-14  
**Auditor:** Senior Full-Stack Software Engineer (Review Role)  
**Target Repository:** `SignBridge-Ai/frontend/`  
**Primary References:**  
- `docs/project-blueprint.pdf` (Sections 1, 3, 5, 6, 7, 8, 11, 14, 16)  
- `docs/implementation-plan/01-master-plan.md`  
- `docs/implementation-plan/02-frontend-plan.md`  
- `docs/implementation-plan/03-backend-plan.md`  
- `docs/implementation-plan/05-translation-and-speech-plan.md`  
- `docs/implementation-plan/06-integration-testing-deployment.md`  
- `frontend/reference/signbridge_ai_warm_ivory_rich_burgundy_edition.html`

---

## 1. Executive Summary

SignBridge AI aims to bridge Indian Sign Language (ISL) communication with hearing individuals through camera-based signing capture, temporal machine learning recognition, bilingual English/Hindi sentence translation, and confirmed speech synthesis across seven everyday domains.

The frontend currently implemented in `frontend/` represents a high-fidelity **editorial landing page and interactive prototype walkthrough** ported from `frontend/reference/signbridge_ai_warm_ivory_rich_burgundy_edition.html`. It demonstrates the aesthetic vision, core brand values (consent, dignity, and transparency), 3D WebGL visuals, on-demand camera access, and browser text-to-speech.

However, from an application architecture standpoint, **the frontend is currently an interactive presentation shell with a mock walkthrough modal rather than a production-ready client for continuous video inference**. Core engineering modules specified in `02-frontend-plan.md`—such as the video recording pipeline (`MediaRecorder`), binary video transmission to FastAPI, decoupled non-destructive bilingual editing buffers, calibrated confidence visual tiers, and quality-based abstention messaging—are either placeholders or not yet built.

---

## 2. Feature Implementation Status Matrix

The following table evaluates the 9 key functional requirements specified in the user review request against the current codebase:

| Functional Requirement | Implementation Status | Implementation Reality in Codebase |
| :--- | :---: | :--- |
| **1. Seven Domains** | **Partially Implemented** | All 7 domains (Education, Healthcare, Banking, Workplaces, Travel, Government, Everyday) are present in `constants.js`, rendered in `DomainShowcase.jsx`, and selectable via `<select>` in `WalkthroughModal.jsx`. However, domain-specific phrase dictionaries with `validated: true/false` status badges (`SuggestedPhrases.jsx`) are not implemented. |
| **2. Camera Access & Live Preview** | **Implemented (Preview Only)** | Implemented in `WalkthroughModal.jsx` using `navigator.mediaDevices.getUserMedia`. Displays live video feed, upper-body framing guide, active status labels, and strictly terminates media tracks upon stopping or closing the modal. |
| **3. ISL Signing Input Interface** | **Missing (Placeholder)** | Only live video playback is present. There is **no recording mechanism** (`MediaRecorder`), no 2–6s duration countdown/timer, no 8s hard cutoff, no clip buffering, and no frame-capture pipeline. |
| **4. English & Hindi Selection** | **Partially Implemented** | Modal contains English and Hindi toggle buttons that swap predefined sample strings from `CONTEXT_SAMPLES`. However, the decoupled, independent dual-buffer architecture (`editedEnglishText` vs `editedHindiText`) planned in `02-frontend-plan.md` is absent. |
| **5. Translation Result Display** | **Missing (Mocked in UI)** | No `ResultCard.jsx` exists. Translation output is simulated by pre-populating the modal's `<textarea>` with static hardcoded dictionary strings (`CONTEXT_SAMPLES`). No API data is consumed. |
| **6. Translation Review & Editing** | **Partially Implemented** | The user can freely edit the text inside the modal's single `<textarea>`. However, there is no "Regenerate Hindi from English" action, no overwrite protection modal, and no directional editing safety. |
| **7. Text-to-Speech Controls** | **Implemented** | Built in `WalkthroughModal.jsx` using browser `window.speechSynthesis`. Binds `en-IN` / `hi-IN` voices, sets measured rate (`0.96`), displays an animated acoustic waveform visualizer during playback, and enforces explicit confirmation before audio plays. |
| **8. Loading, Error & Low-Confidence States** | **Partially Implemented** | Basic camera permission error handling and empty-text validation are present. However, **API loading skeletons, network retry handlers, calibrated confidence gauges (High/Moderate/Abstain), and abstention diagnostics (`UncertaintyMessage.jsx`) are completely missing.** |
| **9. Responsive Design & Accessibility** | **Implemented** | Fully responsive Tailwind layout across mobile, tablet, and desktop. Focus outlines (`focus-visible`) are active. Semantic HTML and ARIA labels are used. `prefers-reduced-motion` is honored in both CSS and JavaScript. |

---

## 3. Deep-Dive Subsystem Audit

### 3.1. Domain Management & Scope Transparency
- **Blueprint Expectation:** The system must support seven domains to structure vocabulary without artificially forcing the ML model to predict in that domain. Each domain should display sample phrases labeled as either *Validated Model Phrase* (empirically trained) or *Future Scope* (`02-frontend-plan.md` §4).
- **Current State:**
  - `frontend/src/utils/constants.js` defines `DOMAIN_WORLDS` (7 items) with titles, tags, colors, icons, and descriptions.
  - `DomainShowcase.jsx` provides an interactive selector and display card with an illustrative sentence.
  - `WalkthroughModal.jsx` synchronizes context selection with sample sentences in `CONTEXT_SAMPLES`.
- **Gaps & Deficiencies:**
  - The domain selector acts purely as a content filter for static sentences.
  - The planned `SuggestedPhrases.jsx` component with `validated: boolean` tags is not built.
  - There is no domain-specific disclaimer banner when high-stakes domains (Healthcare or Banking) are selected, as required by Section 4 of `02-frontend-plan.md`.

### 3.2. Camera & Video Ingestion Subsystem
- **Blueprint Expectation:** A robust camera management lifecycle (`CameraManager.jsx`, `VideoPreview.jsx`, `CaptureControls.jsx`) managing permissions, live framing feedback, a 2.0s–6.0s recording window with a hard 8.0s cutoff, and producing a validated `video/webm` blob for multipart API transmission (`02-frontend-plan.md` §5).
- **Current State:**
  - Located entirely inside `WalkthroughModal.jsx`.
  - Requests `{ video: { facingMode: 'user' }, audio: false }`.
  - Attaches stream to `<video ref={videoRef} autoPlay playsInline muted>`.
  - Dashed framing guide overlay: *"Keep hands & upper posture inside framing guide"*.
  - Strict resource cleanup via `stream.getTracks().forEach(track => track.stop())` on component unmount, manual stop, or modal dismiss (Escape key, backdrop, or close icon).
- **Gaps & Deficiencies:**
  - **No `MediaRecorder` integration:** The application cannot record video clips or generate files.
  - **No Temporal Timer:** No countdown timer or visual progress bar indicating minimum (1.0s), recommended (2–6s), or maximum (8.0s) signing duration.
  - **No Quality Check:** No client-side validation of camera resolution or lighting conditions.

### 3.3. Bilingual Review & Non-Destructive Editing Subsystem
- **Blueprint Expectation:** Independent state buffers (`editedEnglishText` and `editedHindiText`). Editing English activates an explicit *"Regenerate Hindi from English"* action with an overwrite confirmation dialog if Hindi was customized. Direct edits in Hindi never overwrite or mutate the English buffer (`02-frontend-plan.md` §6, `05-translation-and-speech-plan.md` §4).
- **Current State:**
  - In `WalkthroughModal.jsx`, there is only **one shared text state**: `const [sentence, setSentence] = useState('')`.
  - Switching between English and Hindi calls `handleLangSelect()`, which replaces `sentence` with `CONTEXT_SAMPLES[domain][langCode]`.
- **Gaps & Deficiencies:**
  - **Destructive State Overwrite:** If a user edits English text and then clicks "हिन्दी", their manual edits are immediately replaced by the static sample Hindi sentence.
  - **No Re-Translation Action:** There is no mechanism to trigger backend re-translation of custom English text to Hindi (`POST /api/v1/inference/retranslate`).
  - **No Parallel Display:** English and Hindi cannot be inspected simultaneously; the user can only view one language at a time.

### 3.4. Speech Playback & Native Voice Subsystem
- **Blueprint Expectation:** Client-side speech synthesis via Web Speech API (`SpeechSynthesisUtterance`), binding native `en-US`/`en-IN` and `hi-IN` voices, measured rate (0.95–0.96), explicit user confirmation gate, and a fallback "Copy Text" button if Hindi voice is missing (`02-frontend-plan.md` §8).
- **Current State:**
  - Implemented cleanly in `WalkthroughModal.jsx` (`handleSpeak`).
  - Sets `utterance.lang = selectedLang` (`en-IN` or `hi-IN`).
  - Discovers voices and binds matching voice via prefix matching.
  - Sets rate to `0.96`.
  - Cancels previous utterances (`speechSynthesis.cancel()`) before queueing.
  - Enforces explicit user confirmation: audio only plays when the user clicks *"Confirm & Speak"*.
  - Displays animated acoustic wave bars (`svg-wave-bar`) while audio is playing (`onstart` / `onend`).
- **Gaps & Deficiencies:**
  - If a device lacks Hindi voices (e.g. Linux or older Windows installations), there is no UI fallback notice or "Copy Text" button to copy the unvocalized Devanagari script to clipboard.
  - Voice pitch and rate controls (`SpeechControls.jsx`) are not exposed to the user.

### 3.5. 3D WebGL Background & Visual Sensory Layer
- **Blueprint Expectation:** High-performance, non-intrusive 3D visual metaphor representing spatial signing, temporal alignment, and voice synthesis, with fallback for low-end devices and reduced motion (`02-frontend-plan.md` §2).
- **Current State:**
  - Implemented in `ThreeDVisual.jsx` using **Three.js** with custom GLSL shaders (`silkVertexShader`, `silkFragmentShader`).
  - Three Fresnel ribbons with dynamic morphing driven by `stageIndex` (0: Gesture, 1: Intent, 2: Text, 3: Speech).
  - 360 spline-following communication signal particles.
  - Mouse pointer parallax and scroll-choreographed trajectories.
  - Proper memory management: all geometries, materials, event listeners, and render loops are disposed of on unmount.
  - Fallback: renders `#webgl-fallback` with styled ambient gradient orbs if WebGL context creation fails.
  - Reduced Motion: disables continuous rotation, wave animation, and particle movement when `prefers-reduced-motion` is active.

---

## 4. Architecture, State Management & Code Quality

### 4.1. Directory Structure Comparison
The actual directory structure differs significantly from the planned architecture in `02-frontend-plan.md`:

```
Planned Architecture (02-frontend-plan.md)     Current Actual Structure
------------------------------------------     ------------------------
src/                                           src/
├── components/                                ├── components/
│   ├── common/ (Header, Footer, Modal)        │   ├── common/ (KineticHeading)
│   ├── domain/ (Selector, Header, Phrases)    │   ├── Navbar.jsx
│   ├── camera/ (Manager, Preview, Controls)   │   ├── Hero.jsx
│   ├── review/ (ResultCard, SentenceEditor)   │   ├── VisualJourney.jsx
│   └── feedback/ (Confidence, ErrorBoundary)  │   ├── Story.jsx
├── hooks/                                     │   ├── Workflow.jsx
│   ├── useCamera.js                           │   ├── DomainShowcase.jsx
│   ├── useMediaRecorder.js                    │   ├── TrustPrivacy.jsx
│   ├── useSpeechSynthesis.js                  │   ├── FinalCTA.jsx
│   └── useTranslationAPI.js                   │   ├── Footer.jsx
├── services/                                  │   ├── ThreeDVisual.jsx
│   └── api.js                                 │   └── WalkthroughModal.jsx
└── utils/                                     └── utils/
    ├── constants.js                               └── constants.js
    └── speechUtils.js
```

### 4.2. Routing & Navigation
- Currently, the application has **no client-side router** (`react-router-dom` is not installed).
- Navigation is entirely hash-based anchor scrolling (`#hero`, `#visual-journey`, `#story`, `#how-it-works`, `#domains`, `#trust`).
- The entire application lives on a single landing page. There is no route separation between the informational marketing page (`/`) and an active full-screen translation workspace (`/translate` or `/workspace`).

### 4.3. State Management
- State is managed via local `useState` hooks inside `App.jsx` and passed down via props.
- There is no global state management (Context API, Zustand, or Redux). While sufficient for the landing page and self-contained walkthrough modal, this will become brittle when state must be shared across recording timers, WebSocket feeds, and dual-buffer text editors.

---

## 5. Technical Bugs, UX Inconsistencies & Risks

1. **State Loss on Language Toggle:** In `WalkthroughModal.jsx`, changing language immediately overwrites whatever the user typed in the sentence textarea with the default sample string for that language.
2. **Modal Keyboard Focus Trap Missing:** While the modal handles `Escape` key dismiss and backdrop clicks, focus is not trapped within the `<dialog>` container. A user navigating via `Tab` can tab out of the modal into the underlying page elements.
3. **No ARIA Live Regions on Status Updates:** Camera status messages (`cameraStatus`) and speech playback status (`speechStatus`) update dynamically in the DOM, but lack `aria-live="polite"`. Screen reader users will not be informed of state changes.
4. **Vite Chunk Size Warning:** During production build (`npm run build`), Vite emits a warning:
   ```
   dist/assets/index-StFSJ5HS.js  683.41 kB │ gzip: 182.48 kB
   (!) Some chunks are larger than 500 kB after minification.
   ```
   This is primarily due to bundling `three` and `lucide-react` into a single monolithic bundle. Code-splitting via `build.rollupOptions.output.manualChunks` or dynamic imports will be necessary for mobile performance.
5. **No Network Error Boundaries:** A failure in Web Speech or WebGL does not crash React, but there is no top-level `<ErrorBoundary>` component to catch rendering errors in child components.

---

## 6. Backend Integration Gap Analysis

To connect the frontend to the planned FastAPI backend (`03-backend-plan.md` and `06-integration-testing-deployment.md`), the following layers must be implemented:

```
[Camera Video Feed]
        │
        ▼
[MediaRecorder Pipeline] ───► Captures WebM Blob (2.0s - 8.0s cap)
        │
        ▼
[API Client Service] ───────► POST /api/v1/inference/translate (multipart/form-data)
        │
        ▼
[Inference Response Handler]
        ├─► Success (200 OK) ──► Populate ResultCard (English + Hindi + Calibrated Confidence)
        └─► Abstained (200 OK) ─► Display UncertaintyMessage (Low landmark coverage guidance)
        │
        ▼
[Dual-Buffer Editor] ───────► POST /api/v1/inference/retranslate (on manual English edits)
        │
        ▼
[Confirmed Speech Playback] ─► Speaks user-confirmed text aloud
```

### Specific Missing Artifacts:
1. **API Client (`frontend/src/services/api.js`):**
   - Need standard fetch client configured with `import.meta.env.VITE_API_BASE_URL`.
   - Method `submitSignVideoForTranslation(videoBlob, domain, targetLang)`.
   - Method `retranslateEnglishText(englishText, domain)`.
2. **MediaRecorder Hook (`frontend/src/hooks/useMediaRecorder.js`):**
   - Must handle `MediaRecorder` setup with `mimeType: 'video/webm;codecs=vp8,opus'` or `video/mp4`.
   - Automatic 8.0-second timeout stop.
   - Slice assembly on `dataavailable`.
3. **Confidence Tier & Abstention Components:**
   - Visual badges for High ($\ge 0.75$), Moderate ($0.50 - 0.74$), and Abstain ($< 0.50$).
   - Dedicated UI for handling abstention responses (`status: "abstained"`, showing `reason` and `guidance` from FastAPI).

---

## 7. Recommended Next Steps

Before initiating FastAPI backend development, the team should execute the following frontend enhancements in order:

### Phase 1: Real Media Recording & File Assembly
1. Create `src/hooks/useMediaRecorder.js` to record actual video blobs from the existing camera stream.
2. Add a visual recording timer (0.0s to 8.0s) with 2–6s recommended range indicators and a hard 8s auto-stop.
3. Test recording playback and blob generation locally.

### Phase 2: API Client & Environment Configuration
1. Create `frontend/.env.development` with `VITE_API_BASE_URL=http://localhost:8000`.
2. Implement `src/services/api.js` with typed request/response handlers matching the FastAPI schemas defined in `03-backend-plan.md` §3.
3. Create mock API service or MSW (Mock Service Worker) to test HTTP network responses before the real PyTorch model is deployed.

### Phase 3: Decoupled Dual-Buffer Sentence Editor
1. Refactor the review area in `WalkthroughModal.jsx` (or extract to `SentenceEditor.jsx`) into separate English and Hindi state buffers.
2. Add the *"Regenerate Hindi"* action with conflict confirmation dialog.
3. Implement the three-tier confidence badge (High / Moderate / Abstain).

### Phase 4: Production Optimization & Accessibility
1. Configure Vite `manualChunks` in `vite.config.js` to split `three` into its own chunk, eliminating the 683 kB single-bundle warning.
2. Add focus trapping to modal dialogs.
3. Add `aria-live` regions for status announcements.

---

## 8. Honest Assessment of Frontend Readiness

### Verdict: **Prototype Ready / Production Backend Integration Pending**

- **What is completely ready:**
  - The design, brand identity, color scheme, typography, and responsive presentation are **100% faithful** to the reference design.
  - The 3D WebGL background is complete, responsive, performant, and memory-safe.
  - Camera permissions and live framing preview work reliably.
  - Browser-based bilingual text-to-speech works accurately with explicit user confirmation.
  - The production build passes with zero compilation errors (`vite build` in 6.37s).

- **What is NOT ready:**
  - The frontend **cannot currently communicate with the backend** because no HTTP client, video recording pipeline, or API response state handlers exist.
  - The translation output in the UI is currently hardcoded mock data.
  - The dual-buffer non-destructive editing workflow is not yet implemented.

**Conclusion:** The frontend has successfully completed the *Foundation & Design Prototype* stage. To connect to the FastAPI backend, the immediate next technical milestone must be implementing the `MediaRecorder` capture pipeline and the `api.js` client service.

