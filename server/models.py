"""
Typed Pydantic models for the Email Triage OpenEnv environment.
Defines Observation, Action, and all supporting data structures.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class UrgencyLevel(str, Enum):
    URGENT = "urgent"
    NORMAL = "normal"
    LOW = "low"
    SPAM = "spam"


class Category(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    SALES = "sales"
    HR = "hr"
    LEGAL = "legal"
    GENERAL = "general"
    SPAM = "spam"


class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    ANGRY = "angry"


# ---------------------------------------------------------------------------
# Email data (part of Observation)
# ---------------------------------------------------------------------------

class Email(BaseModel):
    id: str
    subject: str
    sender: str
    sender_domain: str
    body: str
    received_at: str                # ISO-8601 string
    has_attachment: bool = False
    thread_length: int = 1          # number of prior messages in thread


# ---------------------------------------------------------------------------
# Observation — what the agent sees each step
# ---------------------------------------------------------------------------

class EmailTriageObservation(BaseModel):
    email: Email
    task_name: str
    step: int
    max_steps: int
    # Feedback from previous action (empty on first step)
    last_feedback: str = ""
    # Fields already confirmed correct this episode (for multi-step tasks)
    confirmed_fields: List[str] = Field(default_factory=list)
    # Task-specific context hints
    context_hint: str = ""


# ---------------------------------------------------------------------------
# Action — what the agent submits
# ---------------------------------------------------------------------------

class EmailTriageAction(BaseModel):
    # Task 1: classification only
    urgency: Optional[UrgencyLevel] = None
    category: Optional[Category] = None

    # Task 2: classification + entity extraction
    sender_name: Optional[str] = None          # extracted from body/signature
    deadline: Optional[str] = None             # ISO date or natural string e.g. "2024-06-30"
    sentiment: Optional[SentimentLabel] = None

    # Task 3: classification + extraction + reply draft
    reply_subject: Optional[str] = None
    reply_body: Optional[str] = None

    # Free-form reasoning (not graded, logged for analysis)
    reasoning: Optional[str] = None


# ---------------------------------------------------------------------------
# Step result returned by env.step()
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: EmailTriageObservation
    reward: float = Field(ge=0.0, le=1.0)
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# State — full internal state returned by env.state()
# ---------------------------------------------------------------------------

class EnvState(BaseModel):
    task_name: str
    episode_id: str
    step: int
    max_steps: int
    email: Email
    cumulative_reward: float
    done: bool
    ground_truth: Dict[str, Any]       # hidden from agent via API
    action_history: List[Dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Reset result
# ---------------------------------------------------------------------------

class ResetResult(BaseModel):
    observation: EmailTriageObservation
    info: Dict[str, Any] = Field(default_factory=dict)