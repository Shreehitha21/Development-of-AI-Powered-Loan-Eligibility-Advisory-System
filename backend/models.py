from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    Boolean,
    ForeignKey,
    Text,
    Float,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship


DATABASE_URL = "sqlite:///backend/db.sqlite3"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    future=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    applications = relationship("Application", back_populates="user")
    notifications = relationship("Notification", back_populates="user")


class Application(Base):
    __tablename__ = "applications"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    applicant_name = Column(String, nullable=False)
    dob = Column(String, nullable=True)
    aadhaar_number = Column(String, nullable=True)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="applications")
    documents = relationship("Document", back_populates="application")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(Integer, ForeignKey("applications.id"), nullable=False)
    doc_type = Column(String, nullable=False)  # aadhaar | salary_slip
    filename = Column(String, nullable=False)
    extracted_json = Column(Text, default="{}")
    status = Column(String, default="uploaded")
    created_at = Column(DateTime, default=datetime.utcnow)

    application = relationship("Application", back_populates="documents")

    def get_extracted(self) -> dict:
        try:
            return json.loads(self.extracted_json or "{}")
        except Exception:
            return {}


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, index=True)
    status = Column(String, default="queued")
    progress = Column(Float, default=0.0)
    logs = Column(Text, default="")
    dataset_slug = Column(String, nullable=True)
    target_column = Column(String, nullable=True)
    best_model_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ModelArtifact(Base):
    __tablename__ = "model_artifacts"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, nullable=False)
    path = Column(String, nullable=False)
    metrics_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow)

    def get_metrics(self) -> dict:
        try:
            return json.loads(self.metrics_json or "{}")
        except Exception:
            return {}


class Notification(Base):
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    type = Column(String, default="info")
    message = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="notifications")


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
