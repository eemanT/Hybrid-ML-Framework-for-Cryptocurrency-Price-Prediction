# NeuralPredict v2 — AI Crypto Terminal

Multi-page React frontend + FastAPI backend.

## Pages
- **Home** — Landing page with live ticker, feature cards, 3D coin widgets
- **Dashboard** — Full prediction terminal with chart, forecast, explainability, metrics
- **Model** — Architecture diagram, specs, Kaggle benchmark results

## Run

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev   # → http://localhost:5173
```

Vite proxies all /api/* calls to localhost:8000 automatically.
