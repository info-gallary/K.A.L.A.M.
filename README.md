# Project KALAM — Atmospheric Nowcasting & Visualization

National Runner-up (2nd Place), ISRO Bhartiya Antariksh Hackathon 2025. Built by Team DOMinators.

Project Kalam is our hackathon-winning system for near-term atmospheric forecasting and visualization. It combines a Python-based inference pipeline, a Node.js/Express API with WebSocket live logs, Cloudflare R2 storage, MongoDB persistence, and a modern React + Vite + Tailwind UI for map-based exploration, animation, and model report views.

This repository contains the full hackathon project that secured 2nd place at the national level.

**Live Focus**
- **Nowcasting:** Predicts short-term satellite frames (T+1, T+2, T+3)
- **Visualization:** Rich, fast UI for bands, timelines, and animations
- **Operational UX:** File uploads to R2, report views, and WebSocket streaming of Python model logs

**Tech Stack**
- **Frontend:** React 19, Vite, Tailwind CSS, react-router, lucide-react
- **Backend:** Node.js, Express, WebSocket (`ws`), Multer, MongoDB (Mongoose)
- **Storage:** Cloudflare R2 via AWS SDK v3
- **ML/Inference:** Python pipeline with PyTorch and ONNX

---

**Monorepo Layout**
- `frontend/`: React/Vite app (UI, animations, map visualization, reports)
- `backend/`: Express API + WebSocket server + R2 uploads + MongoDB
- `Python-Backend/`: Training/inference utilities and artifacts used by backend

---

**Quick Start**
- **Prerequisites:** Node.js 18+, Python 3.10+, MongoDB (local or Atlas), Cloudflare R2 bucket + credentials
- **Clone:** `git clone <this-repo>` and `cd` into it

Backend
- `cd backend`
- `npm install`
- Create `.env` (see below)
- `npm run dev` (runs Express on `:3000` and WebSocket on `:3001`)

Frontend
- `cd frontend`
- `npm install`
- `npm run dev` (Vite dev server on `:5173`)

Open the app at `http://localhost:5173`.

---

**Environment Variables (backend/.env)**
- **Server**
  - `PORT=3000`
  - `NODE_ENV=development`
- **Database**
  - `MONGO_URL=mongodb://localhost:27017/project-kalam` (or Atlas URI)
- **Cloudflare R2**
  - `R2_ENDPOINT=https://<your-account-id>.r2.cloudflarestorage.com`
  - `R2_ACCESS_KEY_ID=<your-access-key>`
  - `R2_SECRET_ACCESS_KEY=<your-secret-key>`
  - `R2_BUCKET_NAME=<your-bucket>`
  - `R2_PUBLIC_URL=https://pub-<id>.r2.dev/<your-bucket>` (or your custom/public base URL)

On startup, the backend verifies R2 configuration and logs a basic health summary.

---

**Local Paths To Update (very important)**
There are a few Windows paths hard-coded for local development during the hackathon. Update these to match your environment.

- Python script used during model validation logs
  - `backend/controllers/modelTest.controller.js:20`
    - `const pythonScriptPath = "d:\\Hackathon\\ISRO\\pre_final\\test1.py";`

- Static serving of predicted images (so the UI can fetch PNGs)
  - `backend/app.js:28`
    - `const testOutputPath = "D:\\Hackathon\\ISRO\\pre_final\\inference_outputs\\test";`
  - `backend/controllers/modelTest.controller.js:255, 445`
    - same `testOutputPath` used for locating sequences and performance.json

Set these to your local inference outputs directory and Python entry script.

---

**Run Flow**
- Start backend (`:3000`) and WebSocket server (`:3001`).
- Start frontend (`:5173`).
- In the UI:
  - Use “Chase The Cloud” to animate frames and request predictions for T+1..T+3.
  - Use “Test Model” to stream Python logs via WebSocket.
  - Use “Visualize On Map” to explore overlays (if configured).

---

**API Overview (backend)**
- Base URL: `http://localhost:3000`

Core
- `GET /` — Service info + advertised endpoints
- `GET /api/prediction-images/<relative-path>` — Serves predicted images from your `testOutputPath`

R2 Uploads (`/api/v1`)
- `POST /upload` — Upload a single file (`field: file`). Validates type/size and stores to R2, persists metadata to MongoDB.
- `POST /test-upload` — Multer sanity check. Returns details of the processed file.
- `GET /health` — R2 configuration/health snapshot.

Model Test & Predictions (`/api/v1`)
- `POST /folder-path` — Triggers Python validation job; logs stream over WebSocket `ws://localhost:3001`.
- `POST /predict-frames` — Body: `{ timeWindow: number[4], selectedDirectory: string, bands: string[], windowSize: number }`. Returns predicted frames metadata + performance aggregates.
- `GET /available-sequences` — Lists available `sequence_****` folders under `testOutputPath`.

---

**Frontend Highlights**
- Multi-band timeline with keyboard navigation and single-cycle animation.
- Predicted vs ground-truth frame browsing for T+1..T+3 with metrics display.
- Directory selection using the File System Access API.
- Toaster notifications, dark/light theme toggle, and polished UI components.

Key Entrypoints
- `frontend/src/App.jsx:1` — Routes: `/` (landing), `/test`, `/overlay-clouds`, `/satellite-animation`
- `frontend/src/pages/SatelliteAnimationPage.jsx:1` — Main nowcasting UI
- `frontend/src/components/ModelTestAndTerminalPreview.jsx:1` — WebSocket log streaming UI
- `frontend/src/libs/axios.js:1` — Base API URL (`http://localhost:3000/api/v1`)

---

**Project Structure**
- `backend/app.js:1` — Express setup, static files, WebSocket server, and routes
- `backend/routes/r2upload.routes.js:1` — Upload/health/test endpoints
- `backend/controllers/r2upload.controller.js:1` — R2 upload implementation and MongoDB persistence
- `backend/routes/modelTest.routes.js:1` — Model log + prediction routes
- `backend/controllers/modelTest.controller.js:1` — Python spawn, predictions assembly, performance parsing
- `backend/config/r2.config.js:1` — R2 client and connectivity test
- `backend/utils/db.js:1` — Mongo connection helper (uses `MONGO_URL`)
- `backend/models/File.model.js:1` — File schema with helpful indexes and virtuals
- `Python-Backend/*` — Training/inference scripts and artifacts

---

**Troubleshooting**
- **Images not loading:** Update `testOutputPath` in `backend/app.js:28` and controller references.
- **WebSocket not connecting:** Ensure `ws://localhost:3001` is reachable and not firewalled.
- **Uploads failing:** Verify `.env` R2 variables and bucket permissions; check `GET /api/v1/health`.
- **Mongo errors:** Confirm `MONGO_URL` and MongoDB is running.
- **Python errors:** Fix `pythonScriptPath` and Python env; run the script manually to verify.

---

**Acknowledgements**
- ISRO Bhartiya Antariksh Hackathon 2025 — National Runner-up (2nd Place)
- Gratitude to mentors, organizers, and the open-source community.

---

**License**
- MIT — see `LICENSE` for details.
