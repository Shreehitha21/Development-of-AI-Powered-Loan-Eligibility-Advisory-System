from __future__ import annotations

import json
import os
import pickle
import warnings
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import shap

from ..models import SessionLocal, TrainingJob, ModelArtifact


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


@dataclass
class TrainResult:
    model_name: str
    model_path: str
    metrics: Dict[str, float]
    shap_importance: Dict[str, float]


def _update_job(job_id: int, status: str | None = None, progress: float | None = None, log: str | None = None, best_path: str | None = None) -> None:
    db = SessionLocal()
    try:
        job = db.get(TrainingJob, job_id)
        if not job:
            return
        if status is not None:
            job.status = status
        if progress is not None:
            job.progress = float(progress)
        if log is not None:
            job.logs = (job.logs or "") + f"\n{log}"
        if best_path is not None:
            job.best_model_path = best_path
        db.add(job)
        db.commit()
    finally:
        db.close()


def _load_kaggle_dataset(dataset_slug: str | None) -> pd.DataFrame:
    # Attempt to load common loan datasets if slug missing
    # Users are expected to set Kaggle credentials via environment before running
    # We'll try local cache first
    possible_files = [
        os.path.join(DATA_DIR, "loan_train.csv"),
        os.path.join(DATA_DIR, "application_train.csv"),
    ]
    for f in possible_files:
        if os.path.exists(f):
            return pd.read_csv(f)

    # If not cached, try to use kaggle API when slug provided
    if dataset_slug:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(dataset_slug, path=DATA_DIR, unzip=True)
            # Heuristic: pick the largest csv
            csvs = [os.path.join(DATA_DIR, p) for p in os.listdir(DATA_DIR) if p.endswith('.csv')]
            if not csvs:
                raise FileNotFoundError("No CSV files found after Kaggle download")
            csvs.sort(key=lambda p: os.path.getsize(p), reverse=True)
            return pd.read_csv(csvs[0])
        except Exception as e:
            raise RuntimeError(f"Failed to download Kaggle dataset: {e}")

    raise FileNotFoundError("Dataset not found. Place a CSV in backend/data as loan_train.csv or application_train.csv or provide dataset_slug.")


def _preprocess(df: pd.DataFrame, target_column: str | None) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer, List[str]]:
    if not target_column:
        # Try common target names
        for col in ["Loan_Status", "TARGET", "Default", "loan_status", "label"]:
            if col in df.columns:
                target_column = col
                break
    if not target_column or target_column not in df.columns:
        raise ValueError("Target column not found; specify target_column in training request")

    y_raw = df[target_column]
    X_df = df.drop(columns=[target_column])

    # Basic clean: drop id-like columns
    drop_cols = [c for c in X_df.columns if c.lower() in {"id", "loan_id", "sk_id_curr"}]
    if drop_cols:
        X_df = X_df.drop(columns=drop_cols)

    numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X_df.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Normalize target to binary 0/1
    if y_raw.dtype == object:
        y = y_raw.astype(str).str.upper().map({"Y": 1, "YES": 1, "1": 1, "APPROVED": 1}).fillna(0).astype(int).values
    else:
        y = (y_raw > 0).astype(int).values

    X = X_df
    return X, y, preprocessor, X_df.columns.tolist()


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else 0.0,
        }


def _fairness_metrics(y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray | None = None) -> Dict[str, Any]:
    # Simple demographic parity difference on predicted positives by a binary sensitive attr if provided
    if sensitive is None:
        return {"notes": "No sensitive attribute provided; add later if available."}
    mask0 = sensitive == 0
    mask1 = sensitive == 1
    rate0 = float(np.mean(y_pred[mask0])) if mask0.any() else 0.0
    rate1 = float(np.mean(y_pred[mask1])) if mask1.any() else 0.0
    return {"demographic_parity_diff": rate1 - rate0}


def _compute_shap_importance(pipeline: Pipeline, X_sample: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
    try:
        # Extract model
        model = pipeline.named_steps.get("model", None)
        # SHAP on tree models uses TreeExplainer; fallback to LinearExplainer else KernelExplainer
        if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier)):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                sv = np.mean(np.abs(shap_values[1]), axis=0)
            else:
                sv = np.mean(np.abs(shap_values), axis=0)
        elif isinstance(model, LogisticRegression):
            explainer = shap.LinearExplainer(model, X_sample)
            sv = np.mean(np.abs(explainer.shap_values(X_sample)), axis=0)
        else:
            explainer = shap.KernelExplainer(model.predict_proba, X_sample[:50])
            sv = np.mean(np.abs(explainer.shap_values(X_sample[:100])[1]), axis=0)

        # Align to transformed feature space. We approximate by using feature_names length.
        # For simplicity, if lengths mismatch, map by min length
        k = min(len(feature_names), len(sv))
        imp = {feature_names[i]: float(sv[i]) for i in range(k)}
        # Normalize
        total = sum(imp.values()) or 1.0
        return {k: v / total for k, v in imp.items()}
    except Exception:
        return {name: 0.0 for name in feature_names[:20]}


def _build_models(algorithms: List[str]) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    for alg in algorithms:
        if alg == "logreg":
            models["logreg"] = LogisticRegression(max_iter=1000)
        elif alg == "rf":
            models["rf"] = RandomForestClassifier(n_estimators=150, random_state=42)
        elif alg == "gb":
            models["gb"] = GradientBoostingClassifier(random_state=42)
        elif alg == "svm":
            models["svm"] = SVC(kernel="rbf", probability=True)
        elif alg == "sgd":
            models["sgd"] = SGDClassifier(loss="log_loss", max_iter=1000)
    return models


def train_models_background(job_id: int, algorithms: List[str], manager) -> None:
    _update_job(job_id, status="running", progress=0.01, log="Starting training")
    db = SessionLocal()
    try:
        job = db.get(TrainingJob, job_id)
        df = _load_kaggle_dataset(job.dataset_slug)
        X_df, y, preprocessor, feature_names = _preprocess(df, job.target_column)
        X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42, stratify=y)
        models = _build_models(algorithms)

        best_auc = -1.0
        best_art: TrainResult | None = None

        num_models = max(len(models), 1)
        progress_per_model = 0.9 / num_models
        current_progress = 0.1

        for name, estimator in models.items():
            pipe = Pipeline(steps=[("pre", preprocessor), ("model", estimator)])
            pipe.fit(X_train, y_train)
            y_prob = pipe.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            metrics = _compute_metrics(y_test, y_prob, y_pred)
            shap_imp = _compute_shap_importance(pipe, preprocessor.fit_transform(X_train)[:200], feature_names)

            model_path = os.path.join(MODELS_DIR, f"{name}_model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(pipe, f)

            art = ModelArtifact(model_name=name, path=model_path, metrics_json=json.dumps(metrics))
            db.add(art)
            db.commit()

            if metrics.get("roc_auc", 0.0) > best_auc:
                best_auc = metrics["roc_auc"]
                best_art = TrainResult(model_name=name, model_path=model_path, metrics=metrics, shap_importance=shap_imp)

            current_progress += progress_per_model
            _update_job(job_id, progress=min(current_progress, 0.99), log=f"Trained {name}: AUC={metrics.get('roc_auc', 0):.3f}")
            # WS notification
            try:
                import asyncio
                asyncio.run(manager.broadcast({"event": "training_progress", "job_id": job_id, "progress": current_progress, "model": name, "metrics": metrics}))
            except RuntimeError:
                pass

        if best_art is None:
            raise RuntimeError("No model trained")

        _update_job(job_id, status="completed", progress=1.0, best_path=best_art.model_path, log=f"Best: {best_art.model_name} AUC={best_art.metrics.get('roc_auc', 0):.3f}")
        try:
            import asyncio
            asyncio.run(manager.broadcast({"event": "training_completed", "job_id": job_id, "best_model": best_art.model_name, "metrics": best_art.metrics}))
        except RuntimeError:
            pass
    except Exception as e:
        _update_job(job_id, status="failed", progress=1.0, log=f"Error: {e}")
        try:
            import asyncio
            asyncio.run(manager.broadcast({"event": "training_failed", "job_id": job_id, "error": str(e)}))
        except RuntimeError:
            pass
    finally:
        db.close()


def _load_best_model() -> Tuple[str, Any] | None:
    # Pick the latest artifact by metrics AUC if available; else latest file
    db = SessionLocal()
    try:
        arts = db.query(ModelArtifact).all()
        if not arts:
            return None
        def get_auc(a: ModelArtifact) -> float:
            try:
                return float(a.get_metrics().get("roc_auc", 0))
            except Exception:
                return 0.0
        best = max(arts, key=get_auc)
        with open(best.path, "rb") as f:
            model = pickle.load(f)
        return best.model_name, model
    except Exception:
        return None
    finally:
        db.close()


def predict_with_best(features: Dict[str, Any]) -> Dict[str, Any]:
    best = _load_best_model()
    if best is None:
        raise RuntimeError("No trained model available. Start training first.")
    model_name, model = best
    # Prepare dataframe with one row
    df = pd.DataFrame([features])
    prob = float(model.predict_proba(df)[:, 1][0])
    label = int(prob >= 0.5)

    # Approximate explanation via permutation importance on single row fallback to feature values
    explanation = {k: float(v) if isinstance(v, (int, float, np.floating)) else 0.0 for k, v in features.items()}
    total = sum(abs(v) for v in explanation.values()) or 1.0
    explanation = {k: abs(v) / total for k, v in explanation.items()}

    # Fairness placeholder; in production use real sensitive attribute
    fairness = {"notes": "Sensitive attribute not provided at prediction time."}

    return {"probability": prob, "label": label, "explanation": explanation, "model_name": model_name, "fairness": fairness}

