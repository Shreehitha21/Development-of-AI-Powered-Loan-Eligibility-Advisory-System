## AI Credit Underwriting System (FastAPI + Vanilla JS)

This project is a fully runnable, real-time AI Credit Underwriting system with:

- FastAPI backend (JWT auth, WebSockets, OCR, dynamic training on Kaggle datasets, predictions, PDF reports)
- SQLite database (default; can switch to PostgreSQL)
- Frontend HTML/CSS/JS (auth, file uploads, training progress, predictions, voice chatbot)
- OCR via Tesseract/EasyOCR (open-source)
- Model training with scikit-learn (LogReg, RF, GB, SVM, SGD), SHAP importance, simple fairness metric
- PDF report generation via ReportLab

### Folder Structure

```
workspace/
  backend/
    main.py              # FastAPI app
    models.py            # SQLAlchemy models + DB init
    schemas.py           # Pydantic models
    auth.py              # JWT, password hashing
    ws.py                # WebSocket manager
    ml/
      training.py        # Kaggle loader, preprocess, training, predict
    ocr/
      ocr.py             # Tesseract-based OCR parsers
    reports/
      report.py          # ReportLab PDF generator
    data/                # Place CSVs here (optional cache)
    models/              # Saved models
    reports_out/         # Generated PDFs
  frontend/
    index.html
    styles.css
    script.js
  requirements.txt
  README.md
```

### Prerequisites

- Python 3.10+
- Tesseract OCR installed on your system
  - Ubuntu/Debian: `sudo apt update && sudo apt install -y tesseract-ocr libtesseract-dev`
  - macOS (brew): `brew install tesseract`
- (Optional) Kaggle API credentials
  - Get an API token from your Kaggle account settings
  - Place `kaggle.json` in `~/.kaggle/kaggle.json` and set permissions `chmod 600 ~/.kaggle/kaggle.json`

### Local Setup (VS Code or Terminal)

1) Install dependencies

```bash
cd /workspace
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Start the backend

```bash
export APP_SECRET_KEY=dev_secret_key
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

3) Open the frontend

Open `http://localhost:8000/` in your browser. The backend serves the frontend statically from `/frontend` and root `/`.

### Usage Walkthrough

- Register a user (checkbox Admin for training permissions)
- Login; the app will open the dashboard
- Create an application (enter name, DOB, Aadhaar number)
- Upload documents (choose `Aadhaar` or `Salary Slip` and upload an image)
  - OCR is run immediately; Aadhaar basic format validation is attempted
  - Upload triggers an optional lightweight auto-training (logreg + rf)
- For full training, as an Admin, provide Kaggle `dataset_slug` (optional) and `target_column` (e.g., `Loan_Status`) and click Start Training
  - Progress and metrics appear in real-time over WebSockets
- Click Predict to use the best saved model on your application features (OCR-extracted + optional manual data)
- Click Download PDF Report to get a report containing application data and extracted fields
- Try the voice chatbot (Web Speech API in Chrome)

### Datasets

- Place CSVs in `backend/data/` named `loan_train.csv` or `application_train.csv`. The trainer will pick the largest CSV if none provided.
- Or use Kaggle dataset slugs like:
  - `nikhil7280/loan-predication` (Loan Prediction)
  - `mlg-ulb/creditcardfraud` (not exactly underwriting, but for demo)
  - `shubh0799/loan-prediction-dataset` (examples)

If you provide a slug, the Kaggle API will unzip files into `backend/data/` and the trainer will pick the largest CSV.

### Database

- Default is SQLite at `backend/db.sqlite3`.
- To switch to Postgres, update `DATABASE_URL` in `backend/models.py` and install `psycopg[binary]`.

### Authentication

- JWT bearer tokens via `/api/auth/login` using `OAuth2PasswordRequestForm` (email as `username`).
- Set a strong `APP_SECRET_KEY` in production.

### WebSockets

- Connect to `ws://<host>/ws` and immediately send `{ "token": "<jwt>" }`.
- Events: `training_progress`, `training_completed`, `training_failed`, `document_status`.

### OCR

- Uses `pytesseract`. Ensure Tesseract is installed and available on PATH.
- Aadhaar parsing: naive regex for number, heuristics for name/DOB.
- Salary slip parsing: simple regexes for fields like Gross/Net/Basic.

### Models and Explainability

- Trains Logistic Regression, Random Forest, Gradient Boosting, SVM, and SGD (log loss).
- Picks best by ROC AUC; artifacts saved to `backend/models/`.
- SHAP importance is computed best-effort; for unsupported models, a fallback normalization is used.
- Simple fairness metric (demographic parity difference) is included when a binary sensitive attribute is available.

### PDF Reports

- Generated into `backend/reports_out/` using ReportLab.

### Deployment

Heroku/Render/Railway (example commands):

1) Create app and set env vars
   - `APP_SECRET_KEY` to a strong secret
   - For Kaggle, set `KAGGLE_USERNAME` and `KAGGLE_KEY` as env vars, or mount `kaggle.json`

2) Install system packages
   - Tesseract required. On Debian-based stacks, add buildpack or apt instruction.

3) Run with

```bash
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

Frontend can be served by the same backend (already mounted) or separately via static hosting; if separate, point the API base to the deployed backend origin.

### Testing Tips

- Start backend and open `/`.
- Register and login.
- Create an application.
- Upload an Aadhaar image (any text image containing a 12-digit pattern ### #### #### works for demo).
- Start training with a dataset present in `backend/data/` and `target_column` like `Loan_Status` or `TARGET`.
- Predict and download the report.

### Notes

- This demo is educational and not a production underwriting system.
- Improve OCR templates and fairness analysis for real use cases.

# Development-of-AI-Powered-Loan-Eligibility-Advisory-System