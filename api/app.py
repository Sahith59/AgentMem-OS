import os
from pathlib import Path

# Load .env before any other imports so API keys are available
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_file, override=False)

import asyncio
import json as _json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

from agentmem_os.storage.store import ConversationStore
from agentmem_os.storage.manager import StorageManager
from agentmem_os.llm.adapters import UniversalAdapter
from agentmem_os.db.engine import get_session
from agentmem_os.db.models import Session as DBSession, Turn, Summary, ProceduralPattern, KnowledgeGraphNode, KnowledgeGraphEdge

app = FastAPI(title="AgentMem OS API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared singletons
store = ConversationStore()
adapter = UniversalAdapter()
manager = StorageManager()

_WEB_DIR = Path(__file__).parent.parent / "web"

# Map deprecated / shorthand model names → current LiteLLM identifiers
_MODEL_ALIASES: dict[str, str] = {
    "claude-3-haiku-20240307":        "anthropic/claude-haiku-4-5-20251001",
    "claude-3-sonnet-20240229":        "anthropic/claude-sonnet-4-6",
    "claude-3-opus-20240229":          "anthropic/claude-opus-4-7",
    "claude-haiku":                    "anthropic/claude-haiku-4-5-20251001",
    "claude-sonnet":                   "anthropic/claude-sonnet-4-6",
    "claude-haiku-4-5":               "anthropic/claude-haiku-4-5-20251001",
    "gpt-4o-mini":                    "openai/gpt-4o-mini",
    "gemini-1.5-flash":               "gemini/gemini-1.5-flash",
    "groq/llama-3.1-8b-instant":      "groq/llama-3.1-8b-instant",  # keep as-is
}


def _normalise_model(model: str) -> str:
    return _MODEL_ALIASES.get(model, model)


# ─────────────────────────────────────────────────────────────────────────────
# Request models
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message: str
    model: str = "anthropic/claude-haiku-4-5-20251001"

class CompareRequest(BaseModel):
    message: str
    agentmem_session: str = "agentmem-demo"
    plain_session: str = "plain-demo"
    model: str = "anthropic/claude-haiku-4-5-20251001"

class BranchRequest(BaseModel):
    parent_id: str
    branch_name: str

class ForgetRequest(BaseModel):
    session_id: str
    about: str   # topic to forget, e.g. "python" or "Google"


# ─────────────────────────────────────────────────────────────────────────────
# Existing endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    # Normalise legacy model strings → current equivalents
    model = _normalise_model(req.model)
    store.get_or_create_session(req.session_id, name=req.session_id, model=model)
    store.save_turn(req.session_id, "user", req.message)
    try:
        response_text = adapter.send_message(req.session_id, req.message, model=model)
        store.save_turn(req.session_id, "assistant", response_text)
        return {"session_id": req.session_id, "reply": response_text}
    except Exception as e:
        err_msg = str(e)
        # Return 200 with error text so the UI can display it instead of crashing
        return {"session_id": req.session_id, "reply": f"[AgentMem OS error — {err_msg[:200]}]", "error": err_msg}

@app.get("/history/{session_id}")
async def get_history(session_id: str, limit: int = 50):
    turns = store.get_history(session_id, last_n=limit)
    return {"turns": turns}

@app.get("/branch/list/{session_id}")
async def list_branches(session_id: str):
    branches = store.list_branches(session_id)
    return [{"id": b.session_id, "parent": b.parent_session_id, "name": b.name, "tokens": b.total_tokens} for b in branches]

@app.post("/branch/create")
async def create_branch(req: BranchRequest):
    try:
        child = store.create_branch(req.parent_id, req.branch_name)
        return {"status": "success", "new_session_id": child.session_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/storage/status")
async def storage_status():
    return {
        "active_path": manager.active_path,
        "is_fallback": manager.is_fallback_active()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Conflict Detection — explicit forget + conflict status
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/memory/forget")
async def forget_topic(req: ForgetRequest):
    """
    Soft-delete all active user turns in session_id whose content is
    topically similar to `about`. Returns the number of turns forgotten.
    """
    from agentmem_os.memory.conflict_detector import forget_about
    n = forget_about(req.session_id, req.about)
    return {"status": "ok", "turns_forgotten": n, "topic": req.about}


@app.get("/memory/conflicts/{session_id}")
async def list_conflicts(session_id: str):
    """
    Return all soft-deleted turns (contradicted facts) for a session.
    Useful for debugging and for the Memory Inspector UI.
    """
    db = get_session()
    try:
        from agentmem_os.db.models import Turn as TurnModel
        inactive = (
            db.query(TurnModel)
            .filter(
                TurnModel.session_id == session_id,
                TurnModel.is_active == False,
            )
            .order_by(TurnModel.id.asc())
            .all()
        )
        return {
            "session_id": session_id,
            "contradicted_turns": [
                {
                    "id": t.id,
                    "content": t.content,
                    "contradicted_by": t.contradicted_by,
                    "created_at": str(t.created_at),
                }
                for t in inactive
            ],
            "count": len(inactive),
        }
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# Demo: Split-screen comparison
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/demo/compare")
async def compare_chat(req: CompareRequest):
    """
    Send the same message to both AgentMem OS (with memory) and a plain LLM
    (stateless, no memory). Returns both responses for split-screen display.
    Never returns HTTP 500 — errors are surfaced as reply text.

    Uses raw sqlite3 for ALL DB operations (same as simulation) to fully bypass
    the SQLAlchemy StaticPool shared connection — any flush failure on the shared
    connection would poison all concurrent sessions through the pool.
    """
    import litellm
    import sqlite3 as _sqlite3
    import datetime as _dt
    from agentmem_os.db.engine import DB_PATH as _DB_PATH

    model = _normalise_model(req.model)
    agentmem_reply = "[No response]"
    plain_reply = "[No response]"
    agentmem_tokens = 0
    turn_count = 0
    summary_count = 0
    _now = _dt.datetime.utcnow().isoformat()

    def _ensure_session(db: _sqlite3.Connection, session_id: str, label: str) -> None:
        exists = db.execute(
            "SELECT 1 FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        if not exists:
            db.execute(
                "INSERT INTO sessions (session_id, name, model, created_at, total_tokens) "
                "VALUES (?,?,?,?,0)",
                (session_id, label, model, _now),
            )
            db.commit()

    def _save_turn(db: _sqlite3.Connection, session_id: str, role: str, content: str) -> None:
        imp = _score_importance(role, content)
        db.execute(
            "INSERT INTO turns (session_id, role, content, token_count, created_at, importance_score) "
            "VALUES (?,?,?,?,?,?)",
            (session_id, role, content, len(content.split()), _now, imp),
        )
        db.commit()

    def _get_history(db: _sqlite3.Connection, session_id: str, limit: int = 100) -> list:
        # is_active=1 filters out soft-deleted (contradicted) turns
        rows = db.execute(
            "SELECT role, content FROM turns WHERE session_id=? AND is_active=1 ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        return [{"role": r, "content": c} for r, c in rows[-limit:]]

    _db = _sqlite3.connect(_DB_PATH)
    try:
        # ── AgentMem OS side ─────────────────────────────────────────────
        # Uses ALL stored turns — demonstrates persistent memory across the session.
        _ensure_session(_db, req.agentmem_session, "AgentMem OS Demo")
        _save_turn(_db, req.agentmem_session, "user", req.message)

        agentmem_history = _get_history(_db, req.agentmem_session, limit=100)
        agentmem_tokens = sum(len(t["content"].split()) for t in agentmem_history)

        # Anthropic requires the message list to start with a user turn.
        # When a user turn is soft-deleted (contradicted), the next turn may be
        # an assistant turn — trim leading assistant turns rather than wiping
        # all context, which would lose the Go preference established later.
        while agentmem_history and agentmem_history[0]["role"] != "user":
            agentmem_history.pop(0)
        if not agentmem_history:
            agentmem_history = [{"role": "user", "content": req.message}]

        try:
            _am_resp = litellm.completion(model=model, messages=agentmem_history)
            agentmem_reply = _am_resp.choices[0].message.content
        except Exception as e:
            agentmem_reply = f"[LLM error: {str(e)[:200]}]"

        _save_turn(_db, req.agentmem_session, "assistant", agentmem_reply)

        # ── Background tasks: KG ingestion + conflict detection ─────────
        # compare_chat uses raw sqlite3 (bypasses store.save_turn), so both
        # background tasks must be triggered manually here.
        import threading as _threading
        from agentmem_os.memory.conflict_detector import ConflictDetector as _CD

        # Get the new user turn ID for conflict detection
        _new_turn_id = _db.execute(
            "SELECT id FROM turns WHERE session_id=? AND role='user' ORDER BY id DESC LIMIT 1",
            (req.agentmem_session,)
        ).fetchone()
        _new_turn_id = _new_turn_id[0] if _new_turn_id else None

        def _run_background():
            try:
                store._ingest_kg(req.agentmem_session, None, req.message)
                store._ingest_kg(req.agentmem_session, None, agentmem_reply)
            except Exception:
                pass
            if _new_turn_id:
                try:
                    # Pass db_path so detector uses raw sqlite3 — avoids
                    # StaticPool WAL snapshot issue with cross-connection commits
                    _CD().check_and_resolve(
                        req.agentmem_session, _new_turn_id, req.message,
                        db_path=_DB_PATH,
                    )
                except Exception:
                    pass
        _threading.Thread(target=_run_background, daemon=True).start()

        # ── Plain LLM side — bounded sliding window (last 20 turns) ─────
        # The plain LLM gets only the most recent context window — how most
        # real chatbots with no memory system work in practice.
        _ensure_session(_db, req.plain_session, "Plain LLM Demo")
        _save_turn(_db, req.plain_session, "user", req.message)

        plain_history = _get_history(_db, req.plain_session, limit=20)
        if not plain_history or plain_history[0]["role"] != "user":
            plain_history = [{"role": "user", "content": req.message}]

        try:
            _pl_resp = litellm.completion(model=model, messages=plain_history)
            plain_reply = _pl_resp.choices[0].message.content
        except Exception as e:
            plain_reply = f"[LLM error: {str(e)[:200]}]"

        _save_turn(_db, req.plain_session, "assistant", plain_reply)

        # Stats
        turn_count = _db.execute(
            "SELECT COUNT(*) FROM turns WHERE session_id=?", (req.agentmem_session,)
        ).fetchone()[0]

    except Exception as e:
        if agentmem_reply == "[No response]":
            agentmem_reply = f"[Error: {str(e)[:200]}]"
    finally:
        _db.close()

    # Summary count — read-only fresh session, safe to use SQLAlchemy here
    db = get_session()
    try:
        summary_count = db.query(Summary).filter(Summary.session_id == req.agentmem_session).count()
    except Exception:
        pass
    finally:
        db.close()

    return {
        "agentmem": {
            "reply": agentmem_reply,
            "context_tokens": agentmem_tokens,
            "turn_count": turn_count,
            "summary_count": summary_count,
        },
        "plain": {
            "reply": plain_reply,
            "context_tokens": len(req.message) // 4,
        },
        "message": req.message,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Shared: importance scorer (used by simulation seeding + compare_chat)
# ─────────────────────────────────────────────────────────────────────────────

import re as _re

def _score_importance(role: str, content: str, turn_index: int = 99) -> float:
    """
    Rule-based importance score [0.0, 1.0] for a conversation turn.

    Components:
      - Base: 0.25
      - Named-entity density: up to +0.25  (capitalised words as proxy)
      - Early turn bonus: +0.20 for first 20 turns (fact-planting window)
      - Length bonus: +0.10 for >80 chars
      - Personal-fact signal: +0.20 if user turn contains identity keywords
    """
    score = 0.25

    words = content.split()
    if words:
        entities = sum(1 for w in words if w and w[0].isupper() and len(w) > 2)
        score += min(entities / len(words), 0.5) * 0.5   # up to +0.25

    if turn_index < 20:
        score += 0.20

    if len(content) > 80:
        score += 0.10

    if role == "user" and _re.search(
        r"\b(my name|i am|i work|i live|i study|i use|i prefer|i graduated|before i|i was at)\b",
        content, _re.I
    ):
        score += 0.20

    return round(min(score, 1.0), 3)


# ─────────────────────────────────────────────────────────────────────────────
# Demo: Simulation — automated multi-turn stress test with recall scoring
# ─────────────────────────────────────────────────────────────────────────────

# ── Simulation scenarios ──────────────────────────────────────────────────────
# Each turn is (user_msg, canned_assistant_ack).
# All 40 turns are fast-seeded via save_turn() — no per-turn LLM call.
# Only 2 LLM calls happen: AgentMem recall + Plain stateless recall.
_SIM_SCENARIOS = {
    "default": {
        "persona": "Jordan Lee — ML engineer at Stripe (4 yrs), San Francisco, Stanford CS, PyTorch over TF, fraud detection, hates JavaScript, reads DDIA by Kleppmann, rock climber, team of 8, open-source data-validation side project, NeurIPS attendee, Python-first",
        "turns": [
            ("Hi! My name is Jordan Lee. I'm a machine learning engineer at Stripe.",
             "Nice to meet you, Jordan! Machine learning at Stripe sounds like fascinating and high-stakes work."),
            ("I've been working at Stripe for about 4 years now.",
             "Four years is solid tenure — you must know the codebase and culture deeply by now."),
            ("My main project is building fraud detection models for Stripe's payment processing pipeline.",
             "Fraud detection is one of the most consequential ML applications — high precision-recall tradeoffs at scale."),
            ("I strongly prefer PyTorch over TensorFlow for all my deep learning work.",
             "PyTorch's dynamic graph and Pythonic API make research iteration much faster."),
            ("One thing I genuinely dislike is JavaScript — I just cannot get into it.",
             "Ha, a lot of ML and backend engineers feel the same way about JavaScript."),
            ("I'm currently reading 'Designing Data-Intensive Applications' by Martin Kleppmann.",
             "That book is a classic — the chapters on replication and consensus are especially dense but rewarding."),
            ("I'm based in San Francisco, near the Mission District.",
             "Great neighborhood — central, good food scene, easy access to the rest of the city."),
            ("I did my Computer Science degree at Stanford University.",
             "Stanford CS is a strong program — lots of ML research output from there."),
            ("Outside of work I love rock climbing — I try to get to the gym twice a week.",
             "Rock climbing is great for both physical and mental focus — good counterbalance to desk work."),
            ("My team has about 8 engineers total — a mix of ML engineers and data engineers.",
             "Eight is a good team size: small enough to move fast, large enough to specialize."),
            ("Python is my primary language. I've dabbled in Go and Scala but always come back to Python.",
             "Python really is the lingua franca of ML — the ecosystem is just unmatched."),
            ("I'm working on a small open-source library for data validation as a side project.",
             "Open-source work is a great way to build a portfolio and get community feedback."),
            ("I try to attend NeurIPS every year — it's the highlight of my conference calendar.",
             "NeurIPS is intense but the density of cutting-edge research is unmatched."),
            ("The biggest challenge right now is handling class imbalance in fraud datasets — fraud events are rare.",
             "Class imbalance is tricky. Have you tried focal loss, weighted sampling, or oversampling with SMOTE?"),
            ("I've been experimenting with SMOTE and custom loss functions for the imbalance problem.",
             "SMOTE can amplify noise — a custom focal loss often generalises better for rare events."),
            ("What do you think about using transformer-based models for tabular fraud data?",
             "Transformers on tabular data are promising but often outperformed by well-tuned GBMs like XGBoost or LightGBM."),
            ("We use XGBoost as a baseline and it's genuinely hard to beat.",
             "XGBoost on fraud is a tough baseline — feature engineering really matters there."),
            ("Feature engineering is probably 70% of the work in our pipeline.",
             "Exactly — garbage in, garbage out, especially with imbalanced classes and adversarial signals."),
            ("We do real-time scoring with latency under 50ms — it's a hard infrastructure constraint.",
             "Sub-50ms ML inference is challenging — ONNX export and model quantisation are common solutions."),
            ("Yes, we export models to ONNX for the serving layer — cuts latency significantly.",
             "ONNX is great for that. Stripe's transaction volume must put serious pressure on that serving pipeline."),
            ("We process millions of transactions per day so even 1ms overhead compounds fast.",
             "At that scale, latency is a first-class feature of the model, not just a serving concern."),
            ("I've been thinking about adding a graph neural network layer to capture merchant relationship signals.",
             "GNNs for fraud detection are interesting — the transaction graph can reveal collusion and ring fraud."),
            ("Connected fraudsters often share device IDs, IP ranges, or merchant accounts.",
             "Graph features like shared-attribute connected components can be very powerful discrimination signals."),
            ("We're also exploring federated learning to train across regions without moving PII.",
             "Federated learning for payments is increasingly important — regulatory pressure is only going up."),
            ("GDPR and CCPA compliance makes centralised training complicated for us.",
             "That's a real constraint — federated training plus differential privacy is the standard approach now."),
            ("How do you approach model explainability for fraud decisions that affect customers?",
             "SHAP values are the industry standard — particularly for showing which features drove a decline decision."),
            ("We use SHAP but explaining to non-technical stakeholders is still difficult.",
             "Summary plots and natural language templates often help bridge the technical-business gap."),
            ("Our ML platform is built on top of Kubeflow and MLflow.",
             "Good combination — Kubeflow for pipeline orchestration, MLflow for experiment tracking and model registry."),
            ("I want to improve our retraining pipeline — we currently retrain weekly.",
             "Weekly retraining may lag behind fast-evolving fraud patterns — triggered retraining on distribution shift is worth exploring."),
            ("Distribution shift detection is on my roadmap for next quarter.",
             "PSI and KS tests are common drift detectors — real-time monitoring dashboards alongside are essential."),
            ("Do you have any recommendations for papers on fraud detection specifically?",
             "The 'Graph Neural Networks for Fraud Detection in E-Commerce' paper and LinkedIn's economic graph work are both worth reading."),
            ("I'll check those out — my reading list is already long with Kleppmann's book.",
             "Kleppmann is worth a slow read — the replication chapter especially rewards re-reading."),
            ("What's your take on LLM-based anomaly detection for payment fraud?",
             "Interesting but often overkill for structured tabular data — traditional ML is hard to beat on cost-accuracy tradeoff."),
            ("That matches my intuition — LLMs shine on unstructured text, not payment rows.",
             "Though LLMs are useful for generating synthetic adversarial fraud scenarios for robustness testing."),
            ("That's actually clever — synthetic adversarial data for red-teaming our own model.",
             "Worth a short spike — it could also help with data augmentation for rare fraud sub-types."),
            ("I might pitch that to the team next sprint — seems high value.",
             "A one-week spike with a clear success metric is a low-risk way to validate it."),
            ("What's your take on the CAP theorem trade-offs for a real-time scoring service?",
             "For fraud scoring, availability over strict consistency makes sense — a brief stale model beats a timeout."),
            ("That's how we've architected it — eventual consistency with a circuit breaker pattern.",
             "Circuit breakers are essential at Stripe's scale. Good defensive design for a high-availability system."),
            ("Thanks for the great conversation — this has been really helpful for thinking through the architecture.",
             "Happy to help, Jordan — you're working on some genuinely hard and important ML problems."),
            ("One last thing — I also mentor two junior ML engineers on the team.",
             "Mentorship compounds across the team — that investment pays off well beyond your own direct contributions."),
            # ── Turns 41–100: deeper dive into Jordan Lee's background ────────────
            ("Before Stripe, I worked at Lyft as a data scientist for about two years.",
             "Lyft is a great place to build production ML instincts — the ride-matching and pricing problems are non-trivial."),
            ("At Lyft I focused on ETA prediction models — very different from fraud but the infrastructure lessons transferred.",
             "ETA models are interesting because you have dense temporal signals and real-time feedback loops."),
            ("Moving from Lyft to Stripe felt like a step up in both scale and stakes — Stripe processes trillions annually.",
             "Financial infrastructure is a uniquely high-consequence domain — errors cost real money and erode user trust instantly."),
            ("One win I'm really proud of this year: we reduced our false positive rate by 18% without touching recall.",
             "An 18% FP reduction is a significant business impact — fewer legitimate transactions declined means higher revenue."),
            ("We achieved it through better calibration — we were using a threshold tuned on old data distributions.",
             "Calibration drift is easy to overlook — periodic threshold re-tuning on recent data is a simple but high-value practice."),
            ("Our entire ML infrastructure runs on GCP — we use BigQuery for feature storage and Vertex AI for batch scoring jobs.",
             "GCP and BigQuery are a natural pairing for ML workloads — the integration with Dataflow and Pub/Sub is seamless."),
            ("We also use Google Cloud Spanner for the global transaction metadata store — strong consistency across regions.",
             "Spanner's globally consistent distributed transactions are rare — most systems compromise consistency for partition tolerance."),
            ("For model serving we deploy gRPC microservices on GKE with dedicated node pools for inference latency.",
             "GKE dedicated node pools prevent noisy neighbour problems at inference time — smart isolation strategy."),
            ("My manager is Alex — technically sharp, very outcome-focused, gives the team a lot of autonomy.",
             "Good managers define the outcomes clearly and trust engineers on the approach — sounds like Alex does that well."),
            ("Alex is actively pushing me toward the staff engineer promotion track — I'm building a promo packet right now.",
             "Staff promos are about scope and cross-team impact, not just code quality — the narrative matters as much as the work."),
            ("My favourite IDE is VS Code — I use Vim keybindings everywhere, even in the terminal.",
             "VS Code with Vim keybindings is a popular combination — you get the ecosystem breadth with the editing efficiency."),
            ("I've customised VS Code heavily — custom snippets for PyTorch boilerplate and SQL templates for BigQuery.",
             "Custom snippets are a high-leverage investment — a few hours of setup pays off every day in reduced friction."),
            ("We do A/B testing for every model update — shadow mode first, then 1% traffic, then full rollout.",
             "Shadow mode before live traffic is a good safety practice — you catch silent failures before they affect users."),
            ("One thing I've noticed is that A/B test results in fraud detection can be gamed by sophisticated fraudsters.",
             "That's an underappreciated problem — adversarial agents can learn the exploration policy and exploit the test window."),
            ("We use Kafka for our real-time feature pipeline — events flow from the payment processor into feature stores within 200ms.",
             "200ms end-to-end is tight — Kafka's low-latency guarantees and partition-level ordering make it the right choice here."),
            ("The Kafka pipeline also feeds our monitoring dashboards — we alert on feature distribution shifts in real time.",
             "Real-time drift monitoring is significantly more valuable than batch reports — you catch problems in minutes not days."),
            ("Outside work I'm planning a climbing trip to Yosemite in August — trying to do El Capitan's beginner routes.",
             "Yosemite climbing is incredible — El Capitan beginner routes like Lurking Fear are still committing multi-day objectives."),
            ("I've been training for altitude — the approach hikes at Yosemite are deceptively demanding.",
             "Approach fitness is as important as technical climbing skill for big walls — good to be systematic about it."),
            ("My morning routine starts with a 5km run before logging on — it genuinely helps my focus for the day.",
             "Morning exercise has strong evidence for cognitive performance — particularly for sustained attention and working memory."),
            ("I've been consistent with it for about eight months now — it's become non-negotiable for me.",
             "Eight months is past the habit formation threshold — that kind of consistency compounds into a real performance edge."),
            ("I keep a paper notebook for system design sketches — I find digital tools too slow for first-pass architecture thinking.",
             "There's real cognitive research behind analog sketching — the slower medium forces more deliberate spatial thinking."),
            ("We hold weekly architecture reviews within the ML platform team — anyone can bring a design doc.",
             "Open architecture reviews build shared context and catch blind spots early — much cheaper than post-hoc refactoring."),
            ("I wrote a design doc last month on triggered retraining — got some great pushback that improved the final design.",
             "Good pushback on a design doc is a gift — it's much easier to revise a doc than to refactor a deployed system."),
            ("My favourite ML paper is still Attention Is All You Need — the simplicity of the core idea is what makes it elegant.",
             "'Attention Is All You Need' is a landmark — removing recurrence while preserving sequence modelling was a paradigm shift."),
            ("I re-read it about once a year — I always notice something I missed before.",
             "Re-reading foundational papers is a high-value habit — your mental model evolves and you find new layers."),
            ("I'm also a big fan of the original XGBoost paper by Chen and Guestrin — extremely practical and well-written.",
             "The XGBoost paper is notable for bridging theory and engineering elegantly — rare in systems ML work."),
            ("For model versioning we use MLflow — every training run gets logged with hyperparams, metrics, and dataset hash.",
             "Dataset hashing for reproducibility is crucial — without it you can't reliably compare runs across data refreshes."),
            ("We treat the MLflow model registry as the source of truth — nothing goes to production that isn't registered.",
             "A model registry as the production gate is a good governance pattern — it forces an explicit promotion step."),
            ("My mentorship style is Socratic — I ask questions rather than give answers to force the engineer to think through trade-offs.",
             "Socratic mentorship builds durable understanding — telling someone the answer skips the productive struggle."),
            ("Both my mentees are ramping up on the fraud pipeline — one is now leading a feature engineering project solo.",
             "Giving mentees project ownership is the fastest accelerant — autonomy with support is the ideal gradient."),
            ("I'm working on improving our containerisation story — currently Docker images are rebuilt from scratch too often.",
             "Layer caching and multi-stage Docker builds can cut image build times dramatically — worth a focused sprint."),
            ("We're moving toward Nix-based reproducible builds for the ML environment — Python dependency hell is real.",
             "Nix for ML environments is gaining traction — the reproducibility guarantees are worth the learning curve."),
            ("Our team retrospectives are monthly and asynchronous — we use a shared Notion doc instead of a live meeting.",
             "Async retros reduce social pressure to agree and tend to surface more honest feedback in my experience."),
            ("I run a monthly ML reading group within the broader engineering org — attendance has grown to about 25 people.",
             "Growing a reading group to 25 is impressive — that kind of knowledge-sharing culture is hard to build organically."),
            ("Last month we read the Netflix recommendation system paper — the discussion on exploration vs exploitation was lively.",
             "Exploration-exploitation trade-offs resonate across domains — fraud, recommendations, and pricing all share that tension."),
            ("I'm presenting our fraud GNN work at an internal Stripe tech talk next month.",
             "Internal tech talks are high-leverage for a staff promo packet — they build visibility and cross-team influence."),
            ("The talk will cover graph construction, message passing, and how we handle the cold-start problem for new merchants.",
             "Merchant cold start is a hard graph ML problem — you have no history to initialise node embeddings from."),
            ("I've been learning Rust on the side — not for work yet, but I want to understand the performance primitives.",
             "Rust's ownership model is a genuinely different mental model from Python — the investment pays off in systems intuition."),
            ("My goal for next quarter is to get the triggered retraining pipeline into production and reduce drift lag to under 4 hours.",
             "A sub-4-hour drift response loop is an aggressive target — achievable if the feature pipeline is already real-time."),
            ("I also want to publish a short technical blog post on the calibration fix that drove our 18% FP improvement.",
             "A blog post on a concrete measurable win is excellent portfolio material and positions you as a practitioner-researcher."),
            ("Long term I'd love to move into ML research — maybe a research engineer role at a lab after a few more years at Stripe.",
             "The path from industry MLE to research engineer is well-travelled — a conference paper or two accelerates it significantly."),
            ("I'm considering applying to NeurIPS with a short paper on our federated learning experiments.",
             "NeurIPS has a Systems and Applications track that rewards practitioner-scale experiments — that could be a good fit."),
            ("The federated experiments are still early — I want at least 6 months of production data before writing anything up.",
             "Six months of production data is a reasonable bar — it shows the approach is not just a research artefact."),
            ("One thing I underestimated starting this role was how much time goes into stakeholder communication.",
             "Communication overhead grows superlinearly with team size and scope — it's a core staff engineer skill, not overhead."),
            ("I've been working on my writing — clearer design docs, shorter emails, more structured tech talks.",
             "Investing in written communication has asymmetric returns — a well-written doc can influence dozens of engineers at once."),
            ("That's probably the most underrated skill in ML engineering — clear technical writing.",
             "Agreed — the best ML engineers I've seen are always excellent writers. The thinking and the writing compound each other."),
            ("Anyway, I think that covers most of what I've been working on. It's been a great conversation.",
             "Jordan, it's been genuinely fascinating — you're working on hard problems with real rigour. Good luck with the promo packet."),
            ("One more thing — I also use Weights & Biases for experiment tracking on personal projects outside Stripe.",
             "W&B is excellent for personal or research projects — richer visualisations than MLflow for deep learning experiments."),
            ("I'm a big believer in reproducibility — every experiment I run gets a full config hash and random seed logged.",
             "Reproducibility discipline is rare and valuable — it makes ablation studies credible and collaboration frictionless."),
            ("I've given a talk at PyBay before — a local Python conference in San Francisco — on feature engineering for fraud.",
             "PyBay is a great community conference — the audience mix of practitioners and researchers makes for sharp Q&A."),
            ("I genuinely enjoy the teaching aspect — the questions from the audience always sharpen my own thinking.",
             "Teaching is one of the fastest ways to find the holes in your own understanding — it surfaces unexamined assumptions."),
            ("I'm also in a small ML study group with engineers from Google and OpenAI — we meet every other week online.",
             "Cross-company study groups are valuable — you get exposure to different engineering cultures and problem framings."),
            ("We've been working through 'The Elements of Statistical Learning' together — very rigorous, very slow going.",
             "ESL is a dense read — the regularisation and ensemble chapters reward the slow pace though."),
            ("I think my biggest weakness professionally is sometimes going too deep into implementation before validating assumptions.",
             "That's a common trap for strong engineers — the cure is forcing a design doc before any code, even a short one."),
            ("I'm actively working on that — I now block out time to write a 1-pager before starting any new initiative.",
             "Forcing the 1-pager is a great habit — it also creates a paper trail of the original intent for future reference."),
            ("For personal finance I keep things simple — index funds and a Roth IRA, nothing fancy.",
             "Simple financial strategies compound well over time — the cognitive overhead of active management rarely pays off."),
            ("I try to cook at home most nights — it's part of how I decompress after a heavy engineering day.",
             "Cooking as a decompression ritual makes sense — it's tactile, bounded, and the feedback loop is immediate."),
            ("I'm vegetarian — have been for about three years now. Makes the SF restaurant scene very navigable.",
             "SF is one of the best cities for vegetarian food — the density of quality options is exceptional."),
            ("My partner is a product manager at a healthcare startup — very different world from payments infrastructure.",
             "That cross-domain perspective is genuinely useful — product managers often have sharper user intuition than engineers."),
            ("That about wraps it up — I appreciate the depth of this conversation. Really useful for organising my thoughts.",
             "Glad it was useful, Jordan. You have a clear head for systems and people — that combination scales well in this field."),
        ],
        "recall_prompt": (
            "This is a memory recall test. Based purely on what I've told you in our conversation, "
            "give me a comprehensive profile of who I am. Include: my full name, current employer, role, "
            "years of experience at current company, previous company, current project, framework preferences, "
            "what I dislike, what book I'm reading, where I live, where I studied, my hobbies, my team size, "
            "my primary programming language, any side projects, which conferences I attend, my manager's name, "
            "my favourite IDE, cloud provider, and any career achievements you remember."
        ),
        "facts": [
            {"label": "Name",           "keywords": ["jordan", "lee"]},
            {"label": "Company",        "keywords": ["stripe"]},
            {"label": "Role",           "keywords": ["machine learning", "ml engineer"]},
            {"label": "Experience",     "keywords": ["4 years", "four years"]},
            {"label": "Project",        "keywords": ["fraud"]},
            {"label": "Framework",      "keywords": ["pytorch"]},
            {"label": "Dislike",        "keywords": ["javascript"]},
            {"label": "Current Book",   "keywords": ["designing data", "kleppmann"]},
            {"label": "Location",       "keywords": ["san francisco", "mission"]},
            {"label": "Education",      "keywords": ["stanford"]},
            {"label": "Hobby",          "keywords": ["rock climbing", "climbing"]},
            {"label": "Language",       "keywords": ["python"]},
            {"label": "Team Size",      "keywords": ["8 engineers", "team of 8", "eight engineer"]},
            {"label": "Side Project",   "keywords": ["data validation", "open-source", "open source"]},
            {"label": "Conference",     "keywords": ["neurips", "nips"]},
            {"label": "Prev Company",   "keywords": ["lyft"]},
            {"label": "Achievement",    "keywords": ["18%", "false positive", "false-positive"]},
            {"label": "Cloud",          "keywords": ["gcp", "google cloud"]},
            {"label": "Manager",        "keywords": ["alex"]},
            {"label": "IDE",            "keywords": ["vs code", "vscode"]},
        ],
    }
}


class SimRequest(BaseModel):
    scenario: str = "default"
    model: str = "anthropic/claude-haiku-4-5-20251001"
    session_id: str = "sim-session"


@app.post("/demo/simulate")
async def run_simulation(req: SimRequest):
    """
    Streaming simulation: seeds 100 turns via raw SQLite INSERT (no per-turn LLM call),
    then fires 2 real LLM calls for AgentMem vs Plain recall test.
    Yields NDJSON — one JSON line per event so the UI updates in real time.
    """
    model = _normalise_model(req.model)
    scenario = _SIM_SCENARIOS.get(req.scenario, _SIM_SCENARIOS["default"])
    plain_session = f"{req.session_id}-plain"

    async def _generate():
        import sqlite3 as _sqlite3
        import litellm as _litellm
        import datetime as _dt

        from db.engine import DB_PATH as _DB_PATH

        # ── Wipe + seed via raw sqlite3 — fully bypasses SQLAlchemy ──────────
        # SQLAlchemy connection pooling causes cross-transaction snapshot issues
        # that silently defeat ORM-level DELETEs. Raw sqlite3 has none of that.
        _db = _sqlite3.connect(_DB_PATH)
        try:
            for _sid in [req.session_id, plain_session]:
                _db.execute("DELETE FROM turns    WHERE session_id=?", (_sid,))
                _db.execute("DELETE FROM summaries WHERE session_id=?", (_sid,))
                _db.execute("DELETE FROM sessions  WHERE session_id=?", (_sid,))
            _db.commit()

            _now = _dt.datetime.utcnow().isoformat()
            for _sid, _label in [
                (req.session_id, "AgentMem Simulation"),
                (plain_session,  "Plain LLM Simulation"),
            ]:
                _db.execute(
                    "INSERT INTO sessions (session_id, name, model, created_at, total_tokens) "
                    "VALUES (?,?,?,?,0)",
                    (_sid, _label, model, _now),
                )
            _db.commit()
        except Exception as e:
            _db.close()
            yield (_json.dumps({"type": "error", "message": f"Wipe/session error: {e!r}"[:200]}) + "\n").encode()
            return

        turns = scenario["turns"]
        total = len(turns)

        # ── Fast-seed all 100 turns — raw INSERT, no LLM calls ──────────────
        # importance_score is computed per-turn (entity density + recency + signals)
        for i, (user_msg, assistant_ack) in enumerate(turns):
            for _sid, _role, _content in [
                (req.session_id, "user",      user_msg),
                (req.session_id, "assistant", assistant_ack),
                (plain_session,  "user",      user_msg),
                (plain_session,  "assistant", assistant_ack),
            ]:
                _imp = _score_importance(_role, _content, turn_index=i)
                try:
                    _db.execute(
                        "INSERT INTO turns (session_id, role, content, token_count, created_at, importance_score) "
                        "VALUES (?,?,?,?,?,?)",
                        (_sid, _role, _content, len(_content.split()), _now, _imp),
                    )
                except Exception:
                    pass
            _db.commit()

            progress = round(92 * (i + 1) / total)
            yield (_json.dumps({
                "type": "turn",
                "index": i,
                "total": total,
                "user": user_msg,
                "assistant": assistant_ack,
                "progress": progress,
            }) + "\n").encode()

        # ── Recall test via LLM ───────────────────────────────────────────
        recall_q = scenario["recall_prompt"]
        yield (_json.dumps({"type": "recall_start", "question": recall_q, "progress": 92}) + "\n").encode()

        mem_recall = "[error — LLM unavailable]"
        plain_recall = "[error — LLM unavailable]"

        # AgentMem recall: full stored history → LLM.
        # Read via same raw sqlite3 connection (already committed) — no pool issues.
        try:
            mem_rows = _db.execute(
                "SELECT role, content FROM turns WHERE session_id=? ORDER BY id ASC",
                (req.session_id,),
            ).fetchall()
            messages = [{"role": r, "content": c} for r, c in mem_rows]
            messages.append({"role": "user", "content": recall_q})
            mem_result = await asyncio.to_thread(
                _litellm.completion, model=model, messages=messages,
            )
            mem_recall = mem_result.choices[0].message.content
        except Exception as e:
            mem_recall = f"[AgentMem error: {str(e)[:120]}]"

        yield (_json.dumps({"type": "agentmem_recall", "text": mem_recall, "progress": 96}) + "\n").encode()

        # Plain LLM recall: last 8 turn-pairs (16 rows) — simulates a typical
        # chatbot with a bounded sliding window (the baseline most real apps use).
        try:
            plain_rows = _db.execute(
                "SELECT role, content FROM turns WHERE session_id=? ORDER BY id DESC LIMIT 16",
                (plain_session,),
            ).fetchall()
            plain_rows = list(reversed(plain_rows))
            plain_messages = [{"role": r, "content": c} for r, c in plain_rows]
            plain_messages.append({"role": "user", "content": recall_q})
            pr = await asyncio.to_thread(
                _litellm.completion, model=model, messages=plain_messages,
            )
            plain_recall = pr.choices[0].message.content
        except Exception as e:
            plain_recall = f"[Plain LLM error: {str(e)[:120]}]"

        # ── Score ─────────────────────────────────────────────────────────
        def _score(reply: str) -> list:
            rl = reply.lower()
            return [
                {"label": f["label"], "recalled": any(kw in rl for kw in f["keywords"])}
                for f in scenario["facts"]
            ]

        mem_score = _score(mem_recall)
        plain_score = _score(plain_recall)
        fact_count = len(scenario["facts"])
        mem_pct = round(100 * sum(1 for f in mem_score if f["recalled"]) / fact_count)
        plain_pct = round(100 * sum(1 for f in plain_score if f["recalled"]) / fact_count)

        yield (_json.dumps({
            "type": "done",
            "persona": scenario["persona"],
            "recall": {"question": recall_q, "agentmem": mem_recall, "plain": plain_recall},
            "score": {
                "agentmem": {"pct": mem_pct, "facts": mem_score},
                "plain":    {"pct": plain_pct, "facts": plain_score},
            },
            "progress": 100,
        }) + "\n").encode()

        try:
            _db.close()
        except Exception:
            pass

    return StreamingResponse(_generate(), media_type="application/x-ndjson")


# ─────────────────────────────────────────────────────────────────────────────
# Demo: Reset session (wipe turns + summaries so demo starts fresh)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/demo/reset")
async def reset_demo(agentmem_session: str = "agentmem-demo", plain_session: str = "plain-demo"):
    db = get_session()
    try:
        for sid in [agentmem_session, plain_session]:
            db.query(Turn).filter(Turn.session_id == sid).delete()
            db.query(Summary).filter(Summary.session_id == sid).delete()

        # Clear the entire global (agent_id IS NULL) KG namespace so no cross-session
        # entity bleed. All demo sessions share this namespace.
        db.query(KnowledgeGraphEdge).filter(KnowledgeGraphEdge.source_id.in_(
            db.query(KnowledgeGraphNode.id).filter(KnowledgeGraphNode.agent_id == None).scalar_subquery()
        )).delete(synchronize_session=False)
        db.query(KnowledgeGraphEdge).filter(KnowledgeGraphEdge.target_id.in_(
            db.query(KnowledgeGraphNode.id).filter(KnowledgeGraphNode.agent_id == None).scalar_subquery()
        )).delete(synchronize_session=False)
        db.query(KnowledgeGraphNode).filter(KnowledgeGraphNode.agent_id == None).delete()
        db.commit()
        # Reset the in-memory KG singleton so it reloads from DB
        if store._kg is not None:
            store._kg = None
        # Expire the shared store's SQLAlchemy identity map so stale cached
        # Turn objects from deleted sessions don't resurface on next query.
        try:
            store.db.expire_all()
        except Exception:
            pass
        return {"status": "reset", "sessions": [agentmem_session, plain_session]}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# Demo: Benchmark results (ablation + head-to-head)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/demo/benchmarks")
async def get_benchmarks():
    """
    Return all benchmark results: ablation, head-to-head, and E2E eval harness.

    Response shape for ablation/head_to_head/e2e/baseline is unchanged
    (backward compatible with the existing demo frontend) — each key still
    holds the raw JSON contents directly. A new `_provenance` key describes
    which of those came from benchmarks/deprecated_proxy_sim/ — self-contained
    architecture simulations, not real integrations with the real AgentMem OS
    package or with the named competitor systems (see LAUNCH_ROADMAP.md
    Phase 2) — versus tests/test_e2e_claude.py, which does exercise the real
    package. Not yet surfaced in the UI; available for a future update.
    """
    bench_dir = Path(__file__).parent.parent / "benchmarks"
    sim_dir = bench_dir / "deprecated_proxy_sim"
    result = {}
    provenance = {}
    for key, fname, subdir, simulated in [
        ("ablation",     "ablation_results.json",      sim_dir,   True),
        ("head_to_head", "head_to_head_results.json",  sim_dir,   True),
        ("baseline",     "baseline_comparison.json",   sim_dir,   True),
        ("e2e",          "latest_report.json",         bench_dir, False),
    ]:
        p = subdir / fname
        if p.exists():
            with open(p) as f:
                result[key] = _json.load(f)
            provenance[key] = {
                "simulated": simulated,
                "note": ("Self-contained architecture simulation, not a real "
                         "integration — see LAUNCH_ROADMAP.md Phase 2."
                         if simulated else
                         "Real end-to-end run against the actual AgentMem OS package."),
            }
        else:
            result[key] = None
    result["_provenance"] = provenance
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Demo: Memory Inspector
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/demo/memory/{session_id}")
async def memory_inspector(session_id: str):
    """
    Return a structured snapshot of all 4 memory tiers for a session.
    Powers the Memory Inspector panel in the demo UI.
    """
    db = get_session()
    try:
        sess = db.query(DBSession).filter(DBSession.session_id == session_id).first()
        if not sess:
            return {"error": f"Session '{session_id}' not found"}

        # Tier 1: Working memory (Redis — last N turns in-memory)
        recent_turns = store.get_history(session_id, last_n=5)

        # Tier 2: Episodic — all raw turns in SQLite
        all_turns = db.query(Turn).filter(Turn.session_id == session_id).all()
        episodic = [
            {
                "id": t.id,
                "role": t.role,
                "content": t.content[:120] + "…" if len(t.content) > 120 else t.content,
                "importance": round(t.importance_score or 0.0, 3),
                "tokens": t.token_count or 0,
            }
            for t in all_turns
        ]

        # Tier 3: Semantic — DBSCAN summaries
        summaries = db.query(Summary).filter(Summary.session_id == session_id).all()
        semantic = [
            {
                "id": s.id,
                "level": s.abstraction_level,
                "level_name": {1: "Episode", 2: "Pattern", 3: "Principle"}.get(s.abstraction_level, "?"),
                "content": s.content[:180] + "…" if len(s.content) > 180 else s.content,
                "cluster_id": s.cluster_id,
                "turn_range": s.turn_range,
            }
            for s in summaries
        ]

        # Tier 4: Procedural patterns
        patterns = (
            db.query(ProceduralPattern)
            .filter(ProceduralPattern.source_sessions.like(f"%{session_id}%"))
            .order_by(ProceduralPattern.confidence.desc())
            .limit(10)
            .all()
        )
        procedural = [
            {
                "trigger": p.trigger,
                "action": p.action,
                "confidence": round(p.confidence, 3),
                "support_count": p.support_count,
            }
            for p in patterns
        ]

        # Token budget breakdown
        total_budget = int(128000 * 0.60)
        used_episodic = sum(t.get("tokens", 0) for t in episodic)
        used_semantic = sum(len(s["content"]) // 4 for s in semantic)

        return {
            "session_id": session_id,
            "session_name": sess.name,
            "total_tokens": sess.total_tokens or 0,
            "tiers": {
                "working": {
                    "label": "Tier 1 — Working Memory (Redis)",
                    "turns": recent_turns[-5:],
                    "count": len(recent_turns),
                    "budget_pct": 40,
                },
                "episodic": {
                    "label": "Tier 2 — Episodic Memory (SQLite)",
                    "turns": episodic,
                    "count": len(episodic),
                    "tokens_used": used_episodic,
                    "budget_pct": 25,
                },
                "semantic": {
                    "label": "Tier 3 — Semantic Memory (ChromaDB + DBSCAN)",
                    "summaries": semantic,
                    "count": len(semantic),
                    "tokens_used": used_semantic,
                    "budget_pct": 20,
                },
                "procedural": {
                    "label": "Tier 4 — Procedural Memory (Pattern Mining)",
                    "patterns": procedural,
                    "count": len(procedural),
                    "budget_pct": 3,
                },
            },
            "budget": {
                "total": total_budget,
                "used": min(total_budget, used_episodic + used_semantic),
                "pct": min(100, int((used_episodic + used_semantic) / total_budget * 100)),
            }
        }
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# Demo: Knowledge Graph (D3-ready)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/demo/kg/{session_id}")
async def knowledge_graph_data(session_id: str):
    """
    Return KG nodes and edges in D3.js force-graph format.
    KnowledgeGraphEdge links via source_id/target_id (FKs to kg_nodes.id).
    """
    from agentmem_os.db.models import KnowledgeGraphNode, KnowledgeGraphEdge

    db = get_session()
    try:
        # Filter to the requested session only — the KG is global but the demo
        # should only show what THIS session's conversation produced.
        nodes_q = (
            db.query(KnowledgeGraphNode)
            .filter(KnowledgeGraphNode.session_id == session_id)
            .order_by(KnowledgeGraphNode.mention_count.desc())
            .limit(80)
            .all()
        )

        # Build lookup: node DB id → entity text (used to resolve edge endpoints)
        id_to_text = {n.id: n.entity_text for n in nodes_q}
        node_id_set = set(id_to_text.keys())

        nodes = [
            {
                "id": n.entity_text,
                "type": n.entity_type,
                "count": n.mention_count,
                "session": n.session_id,
            }
            for n in nodes_q
        ]

        # Only fetch edges where both endpoints are in our node set
        edges_q = (
            db.query(KnowledgeGraphEdge)
            .filter(
                KnowledgeGraphEdge.source_id.in_(node_id_set),
                KnowledgeGraphEdge.target_id.in_(node_id_set),
            )
            .order_by(KnowledgeGraphEdge.weight.desc())
            .limit(300)
            .all()
        )

        edges = []
        seen = set()
        for e in edges_q:
            src = id_to_text.get(e.source_id)
            tgt = id_to_text.get(e.target_id)
            if src and tgt and src != tgt:
                key = tuple(sorted([src, tgt]))
                if key not in seen:
                    seen.add(key)
                    edges.append({"source": src, "target": tgt, "weight": e.weight})

        return {"nodes": nodes, "edges": edges}
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# Serve static web demo
# ─────────────────────────────────────────────────────────────────────────────

_NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}


@app.get("/demo")
async def serve_demo():
    demo_file = _WEB_DIR / "demo.html"
    target = demo_file if demo_file.exists() else _WEB_DIR / "index.html"
    return FileResponse(target, headers=_NO_CACHE_HEADERS)


@app.get("/classic")
async def serve_classic():
    return FileResponse(_WEB_DIR / "index.html", headers=_NO_CACHE_HEADERS)


@app.get("/")
async def serve_index():
    demo_file = _WEB_DIR / "demo.html"
    target = demo_file if demo_file.exists() else _WEB_DIR / "index.html"
    return FileResponse(target, headers=_NO_CACHE_HEADERS)


if __name__ == "__main__":
    uvicorn.run("agentmem_os.api.app:app", host="0.0.0.0", port=8000, reload=True)
