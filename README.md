---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - agent-benchmark
  - email-triage
  - nlp
---

# 📧 Email Triage OpenEnv

An **OpenEnv-compliant** benchmark environment for training and evaluating AI agents on real-world **email triage** — the task of classifying, prioritizing, extracting information from, and replying to emails. This is one of the most common and high-value tasks performed by knowledge workers every day.

---

## 🎯 Why Email Triage?

Every organization processes hundreds to thousands of emails daily. Automating triage:
- Reduces response time for urgent issues
- Ensures consistent prioritization across teams
- Frees humans for higher-value work

This environment models the **full triage pipeline** end-to-end, giving RL agents and LLMs a realistic, multi-dimensional benchmark.

---

## 🗂️ Environment Overview

| Property | Value |
|---|---|
| **Tasks** | 3 (easy → medium → hard) |
| **Action type** | Structured JSON (classification + extraction + text generation) |
| **Observation type** | Structured email + step context |
| **Reward range** | 0.0 – 1.0 (continuous, partial credit) |
| **Max steps/episode** | 1–3 (depends on task) |
| **Framework** | FastAPI + Pydantic |
| **Port** | 7860 |

---

## 📋 Tasks

### Task 1: `classify-urgency` — **Easy**
**Objective:** Classify the email's urgency level and business category.

| Field | Options |
|---|---|
| `urgency` | `urgent`, `normal`, `low`, `spam` |
| `category` | `billing`, `technical`, `sales`, `hr`, `legal`, `general`, `spam` |

**Reward weights:** urgency 60% + category 40%
**Max steps:** 1 | **Success threshold:** ≥ 0.60

---

### Task 2: `classify-and-extract` — **Medium**
**Objective:** Classify + extract structured entities from the email body.

| Field | Description |
|---|---|
| `urgency` | As above |
| `category` | As above |
| `sender_name` | Extracted from email signature |
| `deadline` | ISO date or natural language (null if none) |
| `sentiment` | `positive`, `neutral`, `negative`, `angry` |

**Reward weights:** urgency 25% + category 25% + sender_name 20% + deadline 15% + sentiment 15%
**Max steps:** 2 | **Success threshold:** ≥ 0.70

---

### Task 3: `full-triage` — **Hard**
**Objective:** Complete triage pipeline — classify, extract, AND draft a professional reply.

Includes all fields from Task 2, plus:

| Field | Description |
|---|---|
| `reply_subject` | Subject line for the reply |
| `reply_body` | Full reply body (50–300 chars, professional tone) |

**Reward weights:** classification 15% + extraction 35% + reply quality 40%

Reply quality scored on: keyword coverage · tone alignment · length · greeting presence

**Max steps:** 3 | **Success threshold:** ≥ 0.75

---

## 🔌 API Endpoints

### `POST /reset`
Start a new episode.
```json
{ "task": "classify-urgency", "seed": 42 }
```

### `POST /step`
Submit an action.
```json
{
  "urgency": "urgent",
  "category": "technical",
  "sender_name": "Marcus Lee",
  "deadline": null,
  "sentiment": "angry",
  "reply_subject": "Re: Production server down",
  "reply_body": "Dear Marcus, We are investigating immediately and escalating to our on-call team. Updates in 15 minutes."
}
```

### `GET /state`
Returns current environment state (ground truth masked).

### `GET /health`
Liveness probe — returns `{"status": "ok"}`.

### `GET /tasks`
Lists all available tasks with metadata.

---

## 🏆 Reward Function

The reward function provides **continuous partial credit** throughout the episode:

```
Task 1 reward = 0.6 × urgency_correct + 0.4 × category_correct

Task 2 reward = 0.25 × urgency + 0.25 × category
              + 0.20 × name_match + 0.15 × date_match + 0.15 × sentiment

Task 3 reward = 0.15 × urgency + 0.15 × category
              + 0.10 × name + 0.10 × date + 0.10 × sentiment
              + 0.40 × reply_quality

reply_quality = 0.50 × keyword_coverage
              + 0.20 × length_adequacy
              + 0.15 × has_greeting
              + 0.15 × tone_alignment
```

**Penalties:** −0.05 for a bad retry step (score < 0.3 on step 2+)

---

## 🚀 Setup & Usage

### Option 1: Docker (recommended)

```bash
# Build
docker build -t email-triage-openenv .

# Run
docker run -p 7860:7860 email-triage-openenv

# Test
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "classify-urgency", "seed": 42}'
```

### Option 2: Local Python

```bash
pip install -r requirements.txt
uvicorn server.main:app --host 0.0.0.0 --port 7860
```

---

## 🤖 Running the Baseline Inference Script

```bash
# Install inference dependencies
pip install -r requirements-inference.txt

# Set credentials
export HF_TOKEN=your_huggingface_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export ENV_BASE_URL=http://localhost:7860

# Run all 3 tasks
python inference.py

# Run a single task
TASK=classify-urgency python inference.py
```

### Expected stdout format:
```
[START] task=classify-urgency env=email-triage-openenv model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=urgency=urgent|category=technical reward=1.00 done=true error=null
[END] success=true steps=1 score=1.000 rewards=1.00

[START] task=classify-and-extract env=email-triage-openenv model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=urgency=urgent|category=sales|sender_name=James Whitfield|deadline=2024-07-15|sentiment=neutral reward=0.90 done=false error=null
[END] success=true steps=1 score=0.900 rewards=0.90
```

---

## 📊 Baseline Scores

Baseline measured with `Qwen/Qwen2.5-72B-Instruct`, seed=42:

| Task | Difficulty | Avg Score | Success Rate |
|---|---|---|---|
| classify-urgency | Easy | ~0.85 | ~95% |
| classify-and-extract | Medium | ~0.72 | ~80% |
| full-triage | Hard | ~0.58 | ~55% |

---

## 📁 Project Structure

```
email-triage-env/
├── inference.py              # Baseline agent script (MANDATORY)
├── openenv.yaml              # OpenEnv spec metadata
├── Dockerfile                # Container definition
├── requirements.txt          # Server dependencies
├── requirements-inference.txt
├── README.md
└── server/
    ├── main.py               # FastAPI app + endpoints
    ├── environment.py        # Core reset/step/state logic
    ├── models.py             # Pydantic typed models
    ├── graders.py            # Task grading functions
    └── dataset.py            # Email dataset + ground truth
```

---

## ✅ Pre-Submission Checklist

- [x] HF Space deploys and responds to `/reset` with HTTP 200
- [x] OpenEnv spec compliance (`openenv.yaml`, typed models, all endpoints)
- [x] `docker build` succeeds
- [x] `inference.py` in root, uses OpenAI client
- [x] Mandatory `[START]`/`[STEP]`/`[END]` stdout format
- [x] 3+ tasks with deterministic graders scoring 0.0–1.0
- [x] Meaningful partial-credit reward function
- [x] README with full documentation