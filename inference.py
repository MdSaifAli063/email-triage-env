"""
Inference Script — Email Triage OpenEnv
=======================================
Runs a language model agent against all three email triage tasks and emits
structured stdout logs in the mandatory [START] / [STEP] / [END] format.

Environment variables:
  API_BASE_URL   LLM endpoint  (default: https://router.huggingface.co/v1)
  MODEL_NAME     Model ID      (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       API key
  API_KEY        Alternative API key env var
  ENV_BASE_URL   OpenEnv server URL (default: http://localhost:7860)
  TASK           Single task to run (default: runs all 3)
  SEED           Random seed for reproducibility (default: 42)
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860").rstrip("/")
BENCHMARK = "email-triage-openenv"
SEED = int(os.getenv("SEED", "42"))

TASKS = ["classify-urgency", "classify-and-extract", "full-triage"]
if os.getenv("TASK"):
    TASKS = [os.getenv("TASK")]

MAX_STEPS_PER_TASK = 3
TEMPERATURE = 0.2
MAX_TOKENS = 600

SUCCESS_THRESHOLD = {
    "classify-urgency": 0.6,
    "classify-and-extract": 0.7,
    "full-triage": 0.75,
}

# ---------------------------------------------------------------------------
# Stdout logging (mandatory format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# OpenEnv HTTP client
# ---------------------------------------------------------------------------

def env_reset(task: str, seed: int) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task": task, "seed": seed},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_BASE_URL}/step",
        json=action,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert email triage assistant.
Given an email, you must analyze it and return a JSON object with your assessment.

ALWAYS respond with a valid JSON object — no markdown fences, no extra text.

The JSON fields you may include:
  urgency       : "urgent" | "normal" | "low" | "spam"
  category      : "billing" | "technical" | "sales" | "hr" | "legal" | "general" | "spam"
  sender_name   : string (extracted from signature) or null
  deadline      : ISO date "YYYY-MM-DD" or natural language or null
  sentiment     : "positive" | "neutral" | "negative" | "angry"
  reply_subject : string (subject line for your reply)
  reply_body    : string (full reply body, professional tone, 50-300 chars)
  reasoning     : string (brief explanation of your choices)

Classification rules:
- urgent: server outages, security incidents, legal deadlines, angry customers with threats
- normal: standard business requests requiring timely response
- low: no hard deadline, routine administrative
- spam: unsolicited marketing, phishing, prize claims

Always extract sender_name from the email signature if present.
Always extract deadline if explicitly mentioned (invoice due dates, meeting requests, etc.).
""").strip()


def build_user_prompt(obs: Dict[str, Any]) -> str:
    email = obs["email"]
    task = obs["task_name"]
    hint = obs.get("context_hint", "")
    feedback = obs.get("last_feedback", "")
    confirmed = obs.get("confirmed_fields", [])

    prompt_parts = [
        f"Task: {task}",
        f"Instruction: {hint}",
        "",
        f"From: {email['sender']}",
        f"Subject: {email['subject']}",
        f"Received: {email['received_at']}",
        f"Has attachment: {email['has_attachment']}",
        "",
        "Body:",
        email["body"],
    ]

    if feedback:
        prompt_parts += ["", f"Previous feedback:\n{feedback}"]
    if confirmed:
        prompt_parts += [f"Already confirmed correct: {', '.join(confirmed)}"]

    prompt_parts += ["", "Respond with JSON only."]
    return "\n".join(prompt_parts)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def call_model(client: OpenAI, obs: Dict[str, Any]) -> Dict[str, Any]:
    user_prompt = build_user_prompt(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "{}").strip()
        # Strip markdown fences if the model adds them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON parse error: {e} | raw: {raw[:200]}", flush=True)
        return {}
    except Exception as e:
        print(f"[DEBUG] Model call failed: {e}", flush=True)
        return {}


def action_to_log_str(action: Dict[str, Any]) -> str:
    """Compact single-line representation of the action for [STEP] logging."""
    parts = []
    for key in ["urgency", "category", "sender_name", "deadline", "sentiment"]:
        val = action.get(key)
        if val:
            parts.append(f"{key}={val}")
    if action.get("reply_body"):
        parts.append(f"reply={len(action['reply_body'])}chars")
    return "|".join(parts) if parts else "empty_action"


# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------

def run_episode(client: OpenAI, task: str) -> Dict[str, Any]:
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_data = env_reset(task, seed=SEED)
        obs = reset_data["observation"]

        for step in range(1, MAX_STEPS_PER_TASK + 1):
            action = call_model(client, obs)

            error_msg = None
            try:
                result = env_step(action)
            except Exception as e:
                error_msg = str(e)
                log_step(step=step, action="error", reward=0.0, done=True, error=error_msg)
                break

            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            info = result.get("info", {})
            score = info.get("score", reward)

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_to_log_str(action),
                reward=reward,
                done=done,
                error=error_msg,
            )

            if done:
                break

            # Update observation for next step
            obs = result.get("observation", obs)

        threshold = SUCCESS_THRESHOLD.get(task, 0.6)
        final_score = sum(rewards) / max(len(rewards), 1)
        success = final_score >= threshold

    except Exception as e:
        print(f"[DEBUG] Episode failed: {e}", flush=True)

    log_end(success=success, steps=steps_taken, score=final_score if rewards else 0.0, rewards=rewards)

    return {
        "task": task,
        "success": success,
        "steps": steps_taken,
        "score": final_score if rewards else 0.0,
        "rewards": rewards,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy")

    # Quick health check
    try:
        resp = requests.get(f"{ENV_BASE_URL}/health", timeout=10)
        if resp.status_code != 200:
            print(f"[DEBUG] Health check failed: {resp.status_code}", flush=True)
            sys.exit(1)
    except Exception as e:
        print(f"[DEBUG] Cannot reach environment server at {ENV_BASE_URL}: {e}", flush=True)
        sys.exit(1)

    results = []
    for task in TASKS:
        result = run_episode(client, task)
        results.append(result)

    # Summary
    print("\n[SUMMARY]", flush=True)
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        print(
            f"  {status} | {r['task']:<25} | score={r['score']:.3f} | steps={r['steps']}",
            flush=True,
        )

    overall = sum(r["score"] for r in results) / len(results) if results else 0.0
    print(f"  OVERALL avg score: {overall:.3f}", flush=True)


if __name__ == "__main__":
    main()