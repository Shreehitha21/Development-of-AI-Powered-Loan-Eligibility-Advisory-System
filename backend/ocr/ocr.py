from __future__ import annotations

import os
import re
from typing import Dict, Any, Tuple

import cv2
import numpy as np
import pytesseract


AADHAAR_REGEX = re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b")


def _read_image(path: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        # fallback to cv2.imread
        img = cv2.imread(path)
    return img


def _preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    return th


def parse_aadhaar_text(text: str) -> Dict[str, Any]:
    aadhaar_match = AADHAAR_REGEX.search(text.replace("\n", " "))
    aadhaar_number = aadhaar_match.group(0) if aadhaar_match else None
    name_match = re.search(r"Name\s*[:\-]?\s*([A-Za-z ]{3,})", text)
    dob_match = re.search(r"(?:DOB|Year of Birth|YOB)\s*[:\-]?\s*([\d\-/]{4,10})", text, re.IGNORECASE)
    return {
        "aadhaar_number": aadhaar_number,
        "name": name_match.group(1).strip() if name_match else None,
        "dob": dob_match.group(1) if dob_match else None,
        "valid": bool(aadhaar_number),
    }


def parse_salary_slip_text(text: str) -> Dict[str, Any]:
    # Very simple heuristics to extract numbers
    salary_match = re.search(r"Gross\s*Pay\s*[:\-]?\s*([\d,]+\.?\d*)", text, re.IGNORECASE)
    net_match = re.search(r"Net\s*Pay\s*[:\-]?\s*([\d,]+\.?\d*)", text, re.IGNORECASE)
    basic_match = re.search(r"Basic\s*[:\-]?\s*([\d,]+\.?\d*)", text, re.IGNORECASE)
    def parse_amt(m):
        if not m:
            return None
        return float(m.group(1).replace(",", ""))
    return {
        "gross_pay": parse_amt(salary_match),
        "net_pay": parse_amt(net_match),
        "basic_pay": parse_amt(basic_match),
    }


def process_document(path: str, doc_type: str) -> Tuple[Dict[str, Any], str]:
    img = _read_image(path)
    if img is None:
        return {}, "failed"
    proc = _preprocess_for_ocr(img)
    text = pytesseract.image_to_string(proc)

    if doc_type.lower() == "aadhaar":
        data = parse_aadhaar_text(text)
        status = "verified" if data.get("valid") else "needs_review"
        return data, status
    else:
        data = parse_salary_slip_text(text)
        status = "extracted"
        return data, status

