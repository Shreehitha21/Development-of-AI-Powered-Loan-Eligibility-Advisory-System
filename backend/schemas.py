from __future__ import annotations

from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, EmailStr


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    is_admin: bool = False


class UserOut(BaseModel):
    id: int
    email: EmailStr
    is_admin: bool
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class ApplicationCreate(BaseModel):
    applicant_name: str
    dob: Optional[str] = None
    aadhaar_number: Optional[str] = None


class ApplicationOut(BaseModel):
    id: int
    user_id: int
    applicant_name: str
    dob: Optional[str]
    aadhaar_number: Optional[str]
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class DocumentOut(BaseModel):
    id: int
    application_id: int
    doc_type: str
    filename: str
    status: str
    extracted: Dict[str, Any]
    created_at: datetime


class TrainingRequest(BaseModel):
    dataset_slug: Optional[str] = None
    target_column: Optional[str] = None
    algorithms: Optional[List[str]] = None  # ["logreg","rf","gb","svm","sgd"]


class TrainingStatus(BaseModel):
    id: int
    status: str
    progress: float
    logs: str
    best_model_path: Optional[str]


class PredictRequest(BaseModel):
    application_id: Optional[int] = None
    features: Optional[Dict[str, Any]] = None


class PredictResponse(BaseModel):
    probability: float
    label: int
    explanation: Dict[str, float]
    model_name: str
    fairness: Optional[Dict[str, Any]] = None
