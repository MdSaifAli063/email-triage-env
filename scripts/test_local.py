#!/usr/bin/env python3
"""
Local test script — verifies the environment works end-to-end without Docker.
Run this before building the Docker image.

Usage:
  # In one terminal:
  uvicorn server.main:app --port 7860

  # In another terminal:
  python scripts/test_local.py
"""

import json
import sys
import requests

BASE = "http://localhost:7860"


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "✓ PASS" if condition else "✗ FAIL"
    print(f"  {status}  {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        sys.exit(1)


def run_task(task: str) -> float:
    print(f"\n--- Task: {task} ---")

    # Reset
    r = requests.post(f"{BASE}/reset", json={"task": task, "seed": 42}, timeout=10)
    check("reset() returns 200", r.status_code == 200)
    data = r.json()
    check("observation present", "observation" in data)
    obs = data["observation"]
    check("email in observation", "email" in obs)
    check("task_name matches", obs["task_name"] == task)

    # Build a test action (reasonable but not perfect)
    action = {
        "urgency": "urgent",
        "category": "technical",
        "sender_name": "Test Sender",
        "deadline": "2024-06-30",
        "sentiment": "neutral",
        "reply_subject": "Re: Your email",
        "reply_body": "Dear team, thank you for reaching out. We are investigating the issue and will escalate immediately.",
    }

    # Step
    r = requests.post(f"{BASE}/step", json=action, timeout=10)
    check("step() returns 200", r.status_code == 200)
    result = r.json()
    check("reward in [0,1]", 0.0 <= result["reward"] <= 1.0, f"reward={result['reward']}")
    check("done is bool", isinstance(result["done"], bool))

    # State
    r = requests.get(f"{BASE}/state", timeout=10)
    check("state() returns 200", r.status_code == 200)
    state = r.json()
    check("ground_truth masked", state["ground_truth"] == "*** hidden ***")

    score = result["info"].get("score", result["reward"])
    print(f"  Score: {score:.4f} | Reward: {result['reward']:.4f}")
    return score


def main() -> None:
    print("=" * 50)
    print("Email Triage OpenEnv — Local Test")
    print("=" * 50)

    # Health
    r = requests.get(f"{BASE}/health", timeout=5)
    check("health endpoint", r.status_code == 200)

    # Tasks list
    r = requests.get(f"{BASE}/tasks", timeout=5)
    check("tasks endpoint", r.status_code == 200)
    tasks_data = r.json()
    check("3 tasks defined", len(tasks_data["tasks"]) == 3)

    # Run each task
    scores = {}
    for task in ["classify-urgency", "classify-and-extract", "full-triage"]:
        scores[task] = run_task(task)

    # Invalid action test
    print("\n--- Error handling ---")
    r = requests.post(f"{BASE}/reset", json={"task": "nonexistent"}, timeout=10)
    check("invalid task returns 400", r.status_code == 400)

    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("Scores:")
    for task, score in scores.items():
        print(f"  {task}: {score:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()