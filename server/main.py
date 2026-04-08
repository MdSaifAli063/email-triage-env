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
from fastapi.responses import HTMLResponse
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


FRONTEND_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Email Triage OpenEnv</title>
  <style>
    :root {
      --bg: #f4efe6;
      --panel: rgba(255, 251, 245, 0.94);
      --panel-strong: #fffdf9;
      --text: #1f2933;
      --muted: #6a7280;
      --line: #d8c8b8;
      --accent: #0f766e;
      --accent-strong: #115e59;
      --accent-soft: #d8f3ef;
      --danger: #b42318;
      --shadow: 0 18px 50px rgba(89, 66, 38, 0.12);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: "Segoe UI", "Trebuchet MS", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.14), transparent 26%),
        radial-gradient(circle at bottom right, rgba(180, 35, 24, 0.1), transparent 24%),
        linear-gradient(135deg, #f8f5ef 0%, #f1eadf 100%);
      min-height: 100vh;
    }

    .shell {
      max-width: 1240px;
      margin: 0 auto;
      padding: 28px 18px 40px;
    }

    .hero {
      display: grid;
      gap: 10px;
      margin-bottom: 20px;
    }

    .eyebrow {
      width: fit-content;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(15, 118, 110, 0.1);
      color: var(--accent-strong);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    h1 {
      margin: 0;
      font-size: clamp(32px, 4vw, 52px);
      line-height: 0.95;
      letter-spacing: -0.03em;
    }

    .hero p {
      margin: 0;
      max-width: 820px;
      color: var(--muted);
      font-size: 16px;
    }

    .grid {
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      gap: 18px;
    }

    .card {
      background: var(--panel);
      border: 1px solid rgba(216, 200, 184, 0.8);
      border-radius: 22px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }

    .card-inner {
      padding: 20px;
    }

    .section-title {
      margin: 0 0 12px;
      font-size: 18px;
    }

    .stack { display: grid; gap: 12px; }

    .field {
      display: grid;
      gap: 6px;
    }

    .field label {
      font-size: 13px;
      font-weight: 700;
      color: #344054;
    }

    input, select, textarea, button {
      font: inherit;
    }

    input, select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: var(--panel-strong);
      padding: 11px 12px;
      color: var(--text);
    }

    textarea {
      min-height: 94px;
      resize: vertical;
    }

    .actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }

    button {
      border: 0;
      border-radius: 14px;
      padding: 12px 14px;
      cursor: pointer;
      font-weight: 700;
      transition: transform 120ms ease, opacity 120ms ease, background 120ms ease;
    }

    button:hover { transform: translateY(-1px); }
    button:disabled { opacity: 0.55; cursor: not-allowed; transform: none; }

    .primary {
      background: linear-gradient(135deg, var(--accent) 0%, #15803d 100%);
      color: white;
    }

    .secondary {
      background: #efe4d6;
      color: #5f3b20;
    }

    .status {
      padding: 12px 14px;
      border-radius: 14px;
      background: var(--accent-soft);
      color: var(--accent-strong);
      font-size: 14px;
      white-space: pre-wrap;
    }

    .status.error {
      background: #fde7e4;
      color: var(--danger);
    }

    .pill-row, .stats {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .pill, .stat {
      border-radius: 999px;
      padding: 8px 12px;
      background: #f6ece0;
      border: 1px solid #ecd6bd;
      font-size: 13px;
    }

    .email-box, .result-box {
      background: rgba(255, 253, 249, 0.9);
      border: 1px solid rgba(216, 200, 184, 0.65);
      border-radius: 18px;
      padding: 18px;
    }

    .email-meta {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }

    .meta-item span {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 3px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .email-body, pre {
      white-space: pre-wrap;
      word-break: break-word;
      margin: 0;
      font-family: Consolas, "Courier New", monospace;
      font-size: 13px;
      line-height: 1.55;
    }

    .two-up {
      display: grid;
      grid-template-columns: 1.05fr 0.95fr;
      gap: 18px;
    }

    .muted {
      color: var(--muted);
      font-size: 13px;
    }

    @media (max-width: 980px) {
      .grid, .two-up {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="eyebrow">Built-In Frontend</div>
      <h1>Email Triage OpenEnv</h1>
      <p>Run the environment manually from the browser, inspect each email, submit actions, and watch reward feedback update live. This UI is served directly from the FastAPI app.</p>
    </section>

    <section class="grid">
      <aside class="card">
        <div class="card-inner stack">
          <div>
            <h2 class="section-title">Session Control</h2>
            <div class="field">
              <label for="task">Task</label>
              <select id="task"></select>
            </div>
            <div class="field">
              <label for="seed">Seed</label>
              <input id="seed" type="number" value="42">
            </div>
            <div class="actions">
              <button id="resetBtn" class="primary">Reset Episode</button>
              <button id="stateBtn" class="secondary">Refresh State</button>
            </div>
          </div>

          <div>
            <h2 class="section-title">Action Payload</h2>
            <div class="stack">
              <div class="field"><label for="urgency">Urgency</label><select id="urgency"><option value="">None</option><option>urgent</option><option>normal</option><option>low</option><option>spam</option></select></div>
              <div class="field"><label for="category">Category</label><select id="category"><option value="">None</option><option>billing</option><option>technical</option><option>sales</option><option>hr</option><option>legal</option><option>general</option><option>spam</option></select></div>
              <div class="field"><label for="sender_name">Sender Name</label><input id="sender_name" placeholder="Marcus Lee"></div>
              <div class="field"><label for="deadline">Deadline</label><input id="deadline" placeholder="2024-06-30 or today"></div>
              <div class="field"><label for="sentiment">Sentiment</label><select id="sentiment"><option value="">None</option><option>positive</option><option>neutral</option><option>negative</option><option>angry</option></select></div>
              <div class="field"><label for="reply_subject">Reply Subject</label><input id="reply_subject" placeholder="Re: Production server down"></div>
              <div class="field"><label for="reply_body">Reply Body</label><textarea id="reply_body" placeholder="Professional reply text..."></textarea></div>
              <div class="field"><label for="reasoning">Reasoning</label><textarea id="reasoning" placeholder="Optional explanation for your choices"></textarea></div>
              <button id="submitBtn" class="primary">Submit Step</button>
            </div>
          </div>

          <div id="status" class="status">Loading tasks...</div>
        </div>
      </aside>

      <main class="stack">
        <div class="card">
          <div class="card-inner stack">
            <div class="stats" id="stats"></div>
            <div class="two-up">
              <div class="email-box">
                <h2 class="section-title">Current Email</h2>
                <div class="email-meta" id="emailMeta"></div>
                <pre class="email-body" id="emailBody">Reset an episode to load an email.</pre>
              </div>
              <div class="result-box stack">
                <div>
                  <h2 class="section-title">Feedback</h2>
                  <div class="pill-row" id="confirmed"></div>
                  <pre id="feedback">No step submitted yet.</pre>
                </div>
                <div>
                  <h2 class="section-title">Latest Result</h2>
                  <pre id="resultJson">{}</pre>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </section>
  </div>

  <script>
    const state = {
      observation: null,
      latestResult: null,
      tasks: [],
    };

    const ids = [
      "task", "seed", "urgency", "category", "sender_name", "deadline",
      "sentiment", "reply_subject", "reply_body", "reasoning",
      "status", "stats", "emailMeta", "emailBody", "confirmed",
      "feedback", "resultJson", "resetBtn", "submitBtn", "stateBtn"
    ];
    const el = Object.fromEntries(ids.map((id) => [id, document.getElementById(id)]));

    function setStatus(message, isError = false) {
      el.status.textContent = message;
      el.status.className = isError ? "status error" : "status";
    }

    function collectAction() {
      const payload = {};
      ["urgency", "category", "sender_name", "deadline", "sentiment", "reply_subject", "reply_body", "reasoning"]
        .forEach((key) => {
          const value = el[key].value.trim();
          if (value) payload[key] = value;
        });
      return payload;
    }

    function populateTasks(tasks) {
      el.task.innerHTML = "";
      tasks.forEach((task) => {
        const option = document.createElement("option");
        option.value = task.name;
        option.textContent = task.name + " (" + task.difficulty + ")";
        el.task.appendChild(option);
      });
    }

    function renderStats(obs, latestResult) {
      const pieces = [];
      if (obs) {
        pieces.push("task: " + obs.task_name);
        pieces.push("step: " + obs.step + " / " + obs.max_steps);
      }
      if (latestResult) {
        pieces.push("reward: " + latestResult.reward);
        pieces.push("done: " + latestResult.done);
        if (latestResult.info && latestResult.info.score !== undefined) {
          pieces.push("score: " + latestResult.info.score);
        }
      }
      if (!pieces.length) {
        el.stats.innerHTML = '<div class="stat">No active episode</div>';
        return;
      }
      el.stats.innerHTML = pieces.map((piece) => '<div class="stat">' + piece + '</div>').join("");
    }

    function renderObservation(obs) {
      state.observation = obs;
      renderStats(state.observation, state.latestResult);

      if (!obs) {
        el.emailMeta.innerHTML = "";
        el.emailBody.textContent = "Reset an episode to load an email.";
        el.feedback.textContent = "No feedback yet.";
        el.confirmed.innerHTML = "";
        return;
      }

      const email = obs.email;
      el.emailMeta.innerHTML = [
        ["From", email.sender],
        ["Subject", email.subject],
        ["Received", email.received_at],
        ["Attachment", String(email.has_attachment)]
      ].map(([label, value]) => '<div class="meta-item"><span>' + label + '</span><strong>' + value + '</strong></div>').join("");

      el.emailBody.textContent = email.body;
      el.feedback.textContent = obs.last_feedback || "No feedback yet.";
      el.confirmed.innerHTML = (obs.confirmed_fields || []).length
        ? obs.confirmed_fields.map((item) => '<div class="pill">' + item + '</div>').join("")
        : '<div class="muted">No confirmed fields yet.</div>';
    }

    function renderResult(result) {
      state.latestResult = result;
      el.resultJson.textContent = JSON.stringify(result || {}, null, 2);
      renderStats(state.observation, state.latestResult);
    }

    async function api(path, options = {}) {
      const response = await fetch(path, {
        headers: { "Content-Type": "application/json" },
        ...options,
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.detail || JSON.stringify(data) || "Request failed");
      }
      return data;
    }

    async function loadTasks() {
      const data = await api("/tasks");
      state.tasks = data.tasks || [];
      populateTasks(state.tasks);
      setStatus("Tasks loaded. Reset an episode to begin.");
    }

    async function resetEpisode() {
      setStatus("Resetting episode...");
      renderResult(null);
      const payload = { task: el.task.value, seed: Number(el.seed.value || 42) };
      const data = await api("/reset", { method: "POST", body: JSON.stringify(payload) });
      renderObservation(data.observation);
      setStatus("Episode ready. Review the email and submit your action.");
    }

    async function submitStep() {
      setStatus("Submitting action...");
      const result = await api("/step", { method: "POST", body: JSON.stringify(collectAction()) });
      renderResult(result);
      renderObservation(result.observation);
      setStatus(result.done ? "Episode finished. Reset to try another email." : "Step graded. You can refine and submit again.");
    }

    async function refreshState() {
      setStatus("Refreshing state...");
      const data = await api("/state");
      const syntheticObservation = state.observation ? {
        ...state.observation,
        step: data.step,
        max_steps: data.max_steps
      } : null;
      renderObservation(syntheticObservation);
      setStatus("State refreshed.");
    }

    el.resetBtn.addEventListener("click", async () => {
      try { await resetEpisode(); } catch (error) { setStatus(error.message, true); }
    });

    el.submitBtn.addEventListener("click", async () => {
      try { await submitStep(); } catch (error) { setStatus(error.message, true); }
    });

    el.stateBtn.addEventListener("click", async () => {
      try { await refreshState(); } catch (error) { setStatus(error.message, true); }
    });

    loadTasks().catch((error) => setStatus(error.message, true));
  </script>
</body>
</html>
"""


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

@app.get("/", response_class=HTMLResponse)
async def frontend() -> HTMLResponse:
    return HTMLResponse(FRONTEND_HTML)

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
