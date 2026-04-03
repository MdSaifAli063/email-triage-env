"""
Graders for the Email Triage environment.

Each grader takes an action dict and a ground_truth dict and returns a
float score in [0.0, 1.0].  Scores are deterministic and reproducible.

Task 1 — classify-urgency   (easy)
    Weights: urgency 0.6, category 0.4
    Binary per field — either correct or not.

Task 2 — classify-and-extract  (medium)
    Weights: urgency 0.25, category 0.25, sender_name 0.20,
             deadline 0.15, sentiment 0.15
    Partial credit on fuzzy string matches.

Task 3 — full-triage   (hard)
    Weights: urgency 0.15, category 0.15, sender_name 0.10,
             deadline 0.10, sentiment 0.10, reply 0.40
    Reply scored on: keyword coverage, tone alignment, length, greeting.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def _fuzzy_date_match(predicted: Optional[str], expected: Optional[str]) -> float:
    """Return 1.0 for exact match, 0.5 for partial (same month/year), 0.0 otherwise."""
    if expected is None:
        return 1.0 if predicted is None or predicted == "" else 0.5
    if predicted is None or predicted == "":
        return 0.0
    p = _normalize(predicted)
    e = _normalize(str(expected))
    if p == e:
        return 1.0
    # Accept "today" / "end of day" / same date substring
    date_numbers = re.findall(r"\d{4}-\d{2}-\d{2}", e)
    if date_numbers:
        target = date_numbers[0]
        if target in p or target[:7] in p:   # match year-month
            return 0.5
    if any(w in p for w in ["today", "end of day", "eod"]):
        if any(w in e for w in ["today", "end of day", "eod"]):
            return 1.0
    return 0.0


def _name_match(predicted: Optional[str], expected: Optional[str]) -> float:
    if expected is None:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    p_parts = set(_normalize(predicted).split())
    e_parts = set(_normalize(str(expected)).split())
    if not e_parts:
        return 1.0
    overlap = p_parts & e_parts
    return len(overlap) / len(e_parts)


def _reply_score(reply_body: Optional[str], ground_truth: Dict[str, Any]) -> float:
    """
    Score a reply draft on four dimensions:
      1. Keyword coverage  (0.0–0.50)
      2. Length adequacy   (0.0–0.20)
      3. Has greeting      (0.0–0.15)
      4. Tone alignment    (0.0–0.15)
    """
    if not reply_body:
        return 0.0

    body = reply_body.lower()
    score = 0.0

    # 1. Keyword coverage
    keywords = ground_truth.get("reply_must_contain", [])
    if keywords:
        hits = sum(1 for kw in keywords if kw.lower() in body)
        score += 0.50 * (hits / len(keywords))
    else:
        score += 0.50  # spam — no reply expected, full marks if empty

    # 2. Length: 50–400 chars is good
    length = len(reply_body.strip())
    if 50 <= length <= 400:
        score += 0.20
    elif 20 <= length < 50 or 400 < length <= 600:
        score += 0.10

    # 3. Greeting present
    greeting_patterns = [r"dear\b", r"hi\b", r"hello\b", r"good morning\b", r"thank you"]
    if any(re.search(p, body) for p in greeting_patterns):
        score += 0.15

    # 4. Tone alignment (coarse heuristic)
    tone = ground_truth.get("reply_tone", "")
    if tone == "none":
        score += 0.15 if (not reply_body or len(reply_body.strip()) < 10) else 0.0
    elif "apologetic" in tone or "de-escalation" in tone:
        if any(w in body for w in ["sorry", "apologize", "apologies", "understand"]):
            score += 0.15
    elif "empathetic" in tone:
        if any(w in body for w in ["understand", "concern", "confidential", "support"]):
            score += 0.15
    elif "urgent" in tone:
        if any(w in body for w in ["immediately", "right away", "urgent", "asap", "today"]):
            score += 0.15
    else:
        score += 0.10  # partial credit for any other tone

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Task graders
# ---------------------------------------------------------------------------

def grade_task1(action: Dict[str, Any], ground_truth: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Task 1: classify-urgency (easy)
    Returns (score, breakdown)
    """
    urgency_ok = action.get("urgency") == ground_truth["urgency"]
    category_ok = action.get("category") == ground_truth["category"]

    breakdown = {
        "urgency": 1.0 if urgency_ok else 0.0,
        "category": 1.0 if category_ok else 0.0,
    }
    score = 0.6 * breakdown["urgency"] + 0.4 * breakdown["category"]
    return round(score, 4), breakdown


def grade_task2(action: Dict[str, Any], ground_truth: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Task 2: classify-and-extract (medium)
    Returns (score, breakdown)
    """
    urgency_ok = float(action.get("urgency") == ground_truth["urgency"])
    category_ok = float(action.get("category") == ground_truth["category"])
    name_sc = _name_match(action.get("sender_name"), ground_truth.get("sender_name"))
    date_sc = _fuzzy_date_match(action.get("deadline"), ground_truth.get("deadline"))
    sentiment_ok = float(action.get("sentiment") == ground_truth.get("sentiment"))

    breakdown = {
        "urgency": urgency_ok,
        "category": category_ok,
        "sender_name": name_sc,
        "deadline": date_sc,
        "sentiment": sentiment_ok,
    }
    score = (
        0.25 * urgency_ok
        + 0.25 * category_ok
        + 0.20 * name_sc
        + 0.15 * date_sc
        + 0.15 * sentiment_ok
    )
    return round(score, 4), breakdown


def grade_task3(action: Dict[str, Any], ground_truth: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Task 3: full-triage (hard)
    Returns (score, breakdown)
    """
    urgency_ok = float(action.get("urgency") == ground_truth["urgency"])
    category_ok = float(action.get("category") == ground_truth["category"])
    name_sc = _name_match(action.get("sender_name"), ground_truth.get("sender_name"))
    date_sc = _fuzzy_date_match(action.get("deadline"), ground_truth.get("deadline"))
    sentiment_ok = float(action.get("sentiment") == ground_truth.get("sentiment"))
    reply_sc = _reply_score(action.get("reply_body"), ground_truth)

    breakdown = {
        "urgency": urgency_ok,
        "category": category_ok,
        "sender_name": name_sc,
        "deadline": date_sc,
        "sentiment": sentiment_ok,
        "reply": reply_sc,
    }
    score = (
        0.15 * urgency_ok
        + 0.15 * category_ok
        + 0.10 * name_sc
        + 0.10 * date_sc
        + 0.10 * sentiment_ok
        + 0.40 * reply_sc
    )
    return round(score, 4), breakdown


GRADERS = {
    "classify-urgency": grade_task1,
    "classify-and-extract": grade_task2,
    "full-triage": grade_task3,
}


def grade(task_name: str, action: Dict[str, Any], ground_truth: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    grader = GRADERS.get(task_name)
    if grader is None:
        raise ValueError(f"Unknown task: {task_name}")
    return grader(action, ground_truth)