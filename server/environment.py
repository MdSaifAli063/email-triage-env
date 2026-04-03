"""
Core Email Triage Environment logic.
Implements reset(), step(), and state() following the OpenEnv spec.
"""

from __future__ import annotations

import random
import uuid
from typing import Any, Dict, List, Optional

from dataset import EMAILS, get_emails_for_task
from graders import grade
from models import (
    Email,
    EmailTriageAction,
    EmailTriageObservation,
    EnvState,
    ResetResult,
    StepResult,
)

# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

TASK_CONFIG = {
    "classify-urgency": {
        "description": "Classify the email's urgency level and category.",
        "max_steps": 1,
        "context_hint": (
            "You must provide: urgency (urgent/normal/low/spam) and "
            "category (billing/technical/sales/hr/legal/general/spam)."
        ),
    },
    "classify-and-extract": {
        "description": (
            "Classify urgency/category AND extract: sender_name, deadline (if any), sentiment."
        ),
        "max_steps": 2,
        "context_hint": (
            "You must provide: urgency, category, sender_name (from signature), "
            "deadline (ISO date or natural language, null if none), sentiment."
        ),
    },
    "full-triage": {
        "description": (
            "Full triage: classify, extract entities, AND draft a professional reply."
        ),
        "max_steps": 3,
        "context_hint": (
            "You must provide: urgency, category, sender_name, deadline, sentiment, "
            "reply_subject, and reply_body (a professional reply to the email)."
        ),
    },
}

VALID_TASKS = list(TASK_CONFIG.keys())


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class EmailTriageEnv:
    """
    Stateful email triage environment.
    One instance is kept per session (managed by FastAPI lifespan).
    """

    def __init__(self) -> None:
        self._state: Optional[EnvState] = None
        self._email_pool: List[Dict[str, Any]] = []
        self._email_index: int = 0
        self._rng = random.Random()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_name: str = "classify-urgency", seed: Optional[int] = None) -> ResetResult:
        if task_name not in TASK_CONFIG:
            raise ValueError(
                f"Unknown task '{task_name}'. Valid tasks: {VALID_TASKS}"
            )

        if seed is not None:
            self._rng.seed(seed)

        cfg = TASK_CONFIG[task_name]
        self._email_pool = get_emails_for_task(task_name)
        self._rng.shuffle(self._email_pool)
        self._email_index = 0

        email_data = self._email_pool[self._email_index]
        email = Email(**email_data["email"])
        ground_truth = email_data["ground_truth"]

        self._state = EnvState(
            task_name=task_name,
            episode_id=str(uuid.uuid4()),
            step=0,
            max_steps=cfg["max_steps"],
            email=email,
            cumulative_reward=0.0,
            done=False,
            ground_truth=ground_truth,
            action_history=[],
        )

        obs = self._build_observation(last_feedback="", confirmed_fields=[])
        return ResetResult(
            observation=obs,
            info={
                "task": task_name,
                "episode_id": self._state.episode_id,
                "description": cfg["description"],
            },
        )

    def step(self, action: EmailTriageAction) -> StepResult:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._state.step += 1
        action_dict = action.model_dump(exclude_none=False)

        # Grade this action
        score, breakdown = grade(
            self._state.task_name,
            action_dict,
            self._state.ground_truth,
        )

        # Reward = score this step (with step penalty for multi-step waste)
        step_penalty = 0.0
        if self._state.step > 1 and score < 0.3:
            step_penalty = 0.05  # small penalty for a bad retry
        reward = max(0.0, round(score - step_penalty, 4))

        self._state.cumulative_reward += reward
        self._state.action_history.append(
            {"step": self._state.step, "action": action_dict, "score": score, "breakdown": breakdown}
        )

        # Determine done
        done = (
            self._state.step >= self._state.max_steps
            or score >= 0.95   # perfect — end early
        )
        self._state.done = done

        # Build feedback
        feedback = self._build_feedback(breakdown, score)
        confirmed = [k for k, v in breakdown.items() if v >= 0.9]

        obs = self._build_observation(last_feedback=feedback, confirmed_fields=confirmed)

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={
                "score": score,
                "breakdown": breakdown,
                "step": self._state.step,
                "episode_id": self._state.episode_id,
            },
        )

    def state(self) -> EnvState:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        last_feedback: str,
        confirmed_fields: List[str],
    ) -> EmailTriageObservation:
        assert self._state is not None
        cfg = TASK_CONFIG[self._state.task_name]
        return EmailTriageObservation(
            email=self._state.email,
            task_name=self._state.task_name,
            step=self._state.step,
            max_steps=self._state.max_steps,
            last_feedback=last_feedback,
            confirmed_fields=confirmed_fields,
            context_hint=cfg["context_hint"],
        )

    def _build_feedback(self, breakdown: Dict[str, float], total: float) -> str:
        lines = [f"Score: {total:.2f}"]
        for field, val in breakdown.items():
            status = "✓" if val >= 0.9 else ("~" if val >= 0.5 else "✗")
            lines.append(f"  {status} {field}: {val:.2f}")
        return "\n".join(lines)