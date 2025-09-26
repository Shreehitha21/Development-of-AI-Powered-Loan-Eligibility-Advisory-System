from __future__ import annotations

import io
import json
import os
import threading
from datetime import timedelta
from typing import Optional, Dict, Any

import aiofiles
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from .models import init_db, SessionLocal, User, Application, Document, TrainingJob, ModelArtifact
from .schemas import (
    UserCreate,
    UserOut,
    Token,
    LoginRequest,
    ApplicationCreate,
    ApplicationOut,
    DocumentOut,
    TrainingRequest,
    TrainingStatus,
    PredictRequest,
    PredictResponse,
)
from .auth import get_db, get_current_user, require_admin, get_password_hash, verify_password, create_access_token
from .ws import ConnectionManager

# ML / OCR / Reports modules (implemented in subsequent files)
from .ml.training import train_models_background, predict_with_best
from .ocr.ocr import process_document
from .reports.report import generate_application_report


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


app = FastAPI(title="AI Credit Underwriting System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

manager = ConnectionManager()


@app.on_event("startup")
def on_startup():
    init_db()

# Serve frontend
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")


@app.get("/")
def index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"message": "Backend is running"}


@app.post("/api/auth/register", response_model=UserOut)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == user_in.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(email=user_in.email, hashed_password=get_password_hash(user_in.password), is_admin=user_in.is_admin)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.post("/api/auth/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    token = create_access_token({"sub": str(user.id)}, expires_delta=timedelta(minutes=60 * 24))
    return {"access_token": token, "token_type": "bearer"}


@app.post("/api/applications", response_model=ApplicationOut)
def create_application(body: ApplicationCreate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    app_row = Application(user_id=user.id, applicant_name=body.applicant_name, dob=body.dob, aadhaar_number=body.aadhaar_number)
    db.add(app_row)
    db.commit()
    db.refresh(app_row)
    return app_row


@app.post("/api/upload", response_model=DocumentOut)
async def upload_document(
    application_id: int = Form(...),
    doc_type: str = Form(...),
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    app_row = db.get(Application, application_id)
    if not app_row or app_row.user_id != user.id:
        raise HTTPException(status_code=404, detail="Application not found")

    user_dir = os.path.join(UPLOAD_DIR, f"user_{user.id}", f"app_{application_id}")
    os.makedirs(user_dir, exist_ok=True)
    dest_path = os.path.join(user_dir, file.filename)
    async with aiofiles.open(dest_path, "wb") as out:
        content = await file.read()
        await out.write(content)

    extracted, status = process_document(dest_path, doc_type)

    doc = Document(application_id=application_id, doc_type=doc_type, filename=dest_path, extracted_json=json.dumps(extracted), status=status)
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # Notify via websockets
    import asyncio

    asyncio.create_task(manager.send_to_user(user.id, {"event": "document_status", "doc_id": doc.id, "status": doc.status, "extracted": extracted}))

    # Optional: auto-trigger lightweight retraining if dataset is available
    try:
        job = TrainingJob(status="queued", progress=0.0, dataset_slug=None, target_column=None)
        db.add(job)
        db.commit()
        db.refresh(job)
        t = threading.Thread(target=train_models_background, args=(job.id, ["logreg", "rf"], manager), daemon=True)
        t.start()
    except Exception:
        pass

    return DocumentOut(
        id=doc.id,
        application_id=doc.application_id,
        doc_type=doc.doc_type,
        filename=os.path.basename(doc.filename),
        status=doc.status,
        extracted=extracted,
        created_at=doc.created_at,
    )


@app.post("/api/training/start", response_model=TrainingStatus)
def start_training(req: TrainingRequest, user: User = Depends(require_admin), db: Session = Depends(get_db)):
    job = TrainingJob(status="queued", progress=0.0, dataset_slug=req.dataset_slug, target_column=req.target_column)
    db.add(job)
    db.commit()
    db.refresh(job)

    thread = threading.Thread(
        target=train_models_background,
        args=(job.id, req.algorithms or ["logreg", "rf", "gb", "svm", "sgd"], manager),
        daemon=True,
    )
    thread.start()
    return TrainingStatus(id=job.id, status=job.status, progress=job.progress, logs=job.logs, best_model_path=job.best_model_path)


@app.get("/api/training/status/{job_id}", response_model=TrainingStatus)
def training_status(job_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    job = db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return TrainingStatus(id=job.id, status=job.status, progress=job.progress, logs=job.logs, best_model_path=job.best_model_path)


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    features: Dict[str, Any] = {}
    if req.application_id:
        app_row = db.get(Application, req.application_id)
        if not app_row:
            raise HTTPException(status_code=404, detail="Application not found")
        # gather features from documents if available
        docs = db.query(Document).filter(Document.application_id == app_row.id).all()
        for d in docs:
            features.update(json.loads(d.extracted_json or "{}"))
        features.update({"aadhaar_present": 1 if app_row.aadhaar_number else 0})
    if req.features:
        features.update(req.features)
    if not features:
        raise HTTPException(status_code=400, detail="No features provided")

    result = predict_with_best(features)
    return PredictResponse(**result)


@app.get("/api/report/{application_id}.pdf")
def get_report(application_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    app_row = db.get(Application, application_id)
    if not app_row or app_row.user_id != user.id and not user.is_admin:
        raise HTTPException(status_code=404, detail="Application not found")
    docs = db.query(Document).filter(Document.application_id == application_id).all()
    extracted = {d.doc_type: json.loads(d.extracted_json or "{}") for d in docs}
    report_path = generate_application_report(app_row, extracted)
    return FileResponse(report_path, media_type="application/pdf", filename=f"application_{application_id}_report.pdf")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        # Expecting a JSON message with {"token": "..."} to authenticate
        init_msg = await websocket.receive_json()
        token = init_msg.get("token")
        if not token:
            await websocket.close()
            return
        # decode token manually to fetch user id, token is not reused after WS upgrades
        from jose import jwt
        from .auth import SECRET_KEY, ALGORITHM
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload["sub"])
        await manager.connect(user_id, websocket)
        while True:
            await websocket.receive_text()  # keep alive; client may send pings
    except WebSocketDisconnect:
        try:
            manager.disconnect(user_id, websocket)
        except Exception:
            pass
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass
