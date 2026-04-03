"""
FastAPI server for the Email Triage OpenEnv environment.
Exposes the standard OpenEnv HTTP endpoints:
  POST /reset  — start new episode
  POST /step   — submit one action
  GET  /state  — inspect current state
  GET  /health — liveness probe
  GET  /tasks  — list available tasks
"""

from __future__ import annotations

import sys
import os

# Ensure server/ directory is on the path when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from environment import EmailTriageEnv, TASK_CONFIG, VALID_TASKS
from models import EmailTriageAction, EnvState, ResetResult, StepResult


# ---------------------------------------------------------------------------
# Lifespan — single env instance shared across requests
# ---------------------------------------------------------------------------

env: EmailTriageEnv = EmailTriageEnv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global env
    env = EmailTriageEnv()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Email Triage OpenEnv",
    description=(
        "An OpenEnv-compliant environment for training and evaluating AI agents "
        "on real-world email triage tasks: classification, entity extraction, and reply drafting."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: Optional[str] = "classify-urgency"
    seed: Optional[int] = None


class StepRequest(BaseModel):
    urgency: Optional[str] = None
    category: Optional[str] = None
    sender_name: Optional[str] = None
    deadline: Optional[str] = None
    sentiment: Optional[str] = None
    reply_subject: Optional[str] = None
    reply_body: Optional[str] = None
    reasoning: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": "email-triage-openenv"}


@app.get("/tasks")
async def list_tasks() -> Dict[str, Any]:
    return {
        "tasks": [
            {
                "name": name,
                "difficulty": {"classify-urgency": "easy", "classify-and-extract": "medium", "full-triage": "hard"}[name],
                "description": cfg["description"],
                "max_steps": cfg["max_steps"],
            }
            for name, cfg in TASK_CONFIG.items()
        ]
    }


@app.post("/reset")
async def reset(request: ResetRequest = ResetRequest()) -> Dict[str, Any]:
    try:
        task = request.task or "classify-urgency"
        result: ResetResult = env.reset(task_name=task, seed=request.seed)
        return result.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")


@app.post("/step")
async def step(request: StepRequest) -> Dict[str, Any]:
    try:
        action = EmailTriageAction(
            urgency=request.urgency,
            category=request.category,
            sender_name=request.sender_name,
            deadline=request.deadline,
            sentiment=request.sentiment,
            reply_subject=request.reply_subject,
            reply_body=request.reply_body,
            reasoning=request.reasoning,
        )
        result: StepResult = env.step(action)
        return result.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {e}")


@app.get("/state")
async def state() -> Dict[str, Any]:
    try:
        s: EnvState = env.state()
        # Mask ground truth for fairness — agents should not see it via /state
        data = s.model_dump()
        data["ground_truth"] = "*** hidden ***"
        return data
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))