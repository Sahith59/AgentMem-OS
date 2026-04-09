"""
AgentMem OS — Procedural Memory (Tier 4)
==========================================
Novel Algorithm #4: Extracts reusable behavioral patterns from episodic sequences.

Biological analogy:
  Procedural memory in humans stores HOW to do things — riding a bike,
  touch-typing, recognizing warning signs. It's compiled from repeated episodes
  into automatic behavior. This tier does the same for LLM agents.

What gets stored:
  "When user mentions X → agent should do Y"
  "Pattern: error report → always ask for stack trace before proposing fix"
  "Pattern: new feature request → clarify acceptance criteria first"

How it works:
  1. Sequence Mining   — sliding window over (user_turn, assistant_turn) pairs
  2. Pattern Detection — find recurring trigger-action pairs across sessions
  3. LLM Abstraction  — generalize specific instances into reusable templates
  4. Confidence Score  — patterns gain confidence as more episodes support them
  5. Retrieval         — at context time, inject relevant patterns into Tier-4 slot

Research contribution:
  First formulation of procedural memory for LLM agents using sequence mining
  over conversation pairs. Equivalent to imitation learning from self-play.
"""

import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

from loguru import logger


# ──────────────────────────────────────────────────────────────────────────────
# Trigger Classifier
# Used to categorize user turns into trigger types without needing an LLM
# ──────────────────────────────────────────────────────────────────────────────

TRIGGER_PATTERNS = {
    # Use non-terminated boundary so inflected forms match:
    # crash → crashes, crashing | fail → failed, failing | break → broken, breaks
    "bug_report":       r'\b(bug|error|crash\w*|fail\w*|exception|broken|not work\w*)',
    "feature_request":  r'\b(add|implement\w*|build\w*|create\w*|feature|want|need|would like)',
    "question":         r'\b(how|what|why|when|where|explain\w*|tell me|help\b)',
    "code_review":      r'\b(review\w*|check\w*|look at|feedback|improve\w*|refactor\w*)',
    "debugging":        r'\b(debug\w*|trace\w*|stack trace|why is|what is wrong)',
    "planning":         r'\b(plan\w*|design\w*|architect\w*|roadmap|structure|outline\w*)',
    "clarification":    r'\b(clarif\w*|unclear|confus\w*|rephrase\w*)',
}

def classify_trigger(text: str) -> str:
    """Classify a user turn into a trigger category."""
    text_lower = text.lower()
    scores = {}
    for category, pattern in TRIGGER_PATTERNS.items():
        matches = len(re.findall(pattern, text_lower))
        if matches:
            scores[category] = matches
    if not scores:
        return "general"
    return max(scores, key=scores.get)


# ──────────────────────────────────────────────────────────────────────────────
# Action Summarizer
# Extracts what the assistant DID (not what it said) from a response
# ──────────────────────────────────────────────────────────────────────────────

ACTION_PATTERNS = [
    (r'\basked for\b.{5,60}',       "asked for clarification"),
    (r'\bprovided\b.{5,60}',        "provided information"),
    (r'\bsuggest\w*.{5,60}',         "suggested solution"),
    (r'\bexplain\w*.{5,60}',        "explained concept"),
    (r'\bwrote\b.{5,60}code',       "wrote code"),
    (r'\brefactored\b.{5,60}',      "refactored code"),
    (r'\bdiagnosed\b.{5,60}',       "diagnosed issue"),
    (r'\bcreated\b.{5,60}',         "created artifact"),
    (r'\breviewed\b.{5,60}',        "reviewed content"),
    (r'\blisted\b.{5,60}',          "listed items"),
    (r'```',                         "provided code block"),
]

def extract_action(assistant_response: str) -> str:
    """Extract the primary action taken in an assistant response."""
    response_lower = assistant_response.lower()
    for pattern, label in ACTION_PATTERNS:
        if re.search(pattern, response_lower):
            return label
    if len(assistant_response) > 500:
        return "gave detailed response"
    return "responded"


# ──────────────────────────────────────────────────────────────────────────────
# ProceduralMemory: The main class
# ──────────────────────────────────────────────────────────────────────────────

class ProceduralMemory:
    """
    Mines conversation history for reusable behavioral patterns
    and persists them as procedural memory (Tier 4).

    Usage:
        pm = ProceduralMemory(get_db_session, summarizer)
        # After accumulating turns:
        pm.mine_patterns(session_id, agent_id)
        # At context time:
        pm.get_relevant_patterns(query, agent_id, top_k=3)
    """

    # Minimum episode count before a pattern is saved
    MIN_SUPPORT_COUNT = 2
    # Minimum confidence to include in context
    MIN_CONTEXT_CONFIDENCE = 0.4

    def __init__(self, get_db_session, summarizer=None):
        self.get_db = get_db_session
        self.summarizer = summarizer

    # ──────────────────────────────────────────────────────────────────────────
    # Pattern Mining
    # ──────────────────────────────────────────────────────────────────────────

    def mine_patterns(
        self,
        session_id: str,
        agent_id: Optional[str] = None,
    ) -> int:
        """
        Mine (trigger, action) patterns from a session's turns.
        New patterns are saved; existing patterns get their support_count incremented.
        Returns number of patterns upserted.
        """
        db = self.get_db()
        try:
            from memnai.db.models import Turn, ProceduralPattern

            turns = (
                db.query(Turn)
                .filter(Turn.session_id == session_id)
                .order_by(Turn.id.asc())
                .all()
            )

            if len(turns) < 4:
                return 0

            # Build (user, assistant) pairs via sliding window
            pairs = []
            for i in range(len(turns) - 1):
                if turns[i].role == "user" and turns[i+1].role == "assistant":
                    pairs.append((turns[i].content, turns[i+1].content))

            if not pairs:
                return 0

            # Count (trigger_type, action_type) co-occurrences
            combo_counts: Counter = Counter()
            combo_examples: Dict[Tuple, List] = defaultdict(list)

            for user_text, assistant_text in pairs:
                trigger_type = classify_trigger(user_text)
                action_type  = extract_action(assistant_text)
                key = (trigger_type, action_type)
                combo_counts[key] += 1
                combo_examples[key].append((user_text[:200], assistant_text[:200]))

            # Save patterns that meet minimum support
            saved = 0
            for (trigger_type, action_type), count in combo_counts.items():
                if count < self.MIN_SUPPORT_COUNT:
                    continue

                # Generate human-readable pattern
                full_pattern = self._generate_pattern_text(
                    trigger_type, action_type,
                    combo_examples[(trigger_type, action_type)][:3]
                )

                # Confidence = count / total pairs (capped at 0.95)
                confidence = min(0.95, count / max(1, len(pairs)))

                # Check if pattern already exists
                existing = (
                    db.query(ProceduralPattern)
                    .filter(
                        ProceduralPattern.agent_id == agent_id,
                        ProceduralPattern.trigger == trigger_type,
                        ProceduralPattern.action == action_type,
                    )
                    .first()
                )

                if existing:
                    existing.support_count += count
                    existing.confidence = min(0.95, existing.confidence + 0.05)
                    # Add session to sources if not already there
                    sources = set((existing.source_sessions or "").split(","))
                    sources.add(session_id)
                    existing.source_sessions = ",".join(filter(None, sources))
                else:
                    pattern = ProceduralPattern(
                        agent_id=agent_id,
                        trigger=trigger_type,
                        action=action_type,
                        full_pattern=full_pattern,
                        confidence=confidence,
                        support_count=count,
                        source_sessions=session_id,
                        is_global=False,
                    )
                    db.add(pattern)

                saved += 1

            db.commit()
            logger.info(
                f"[ProceduralMemory] Mined {saved} patterns from session={session_id}"
            )
            return saved

        except Exception as e:
            logger.error(f"[ProceduralMemory] mine_patterns failed: {e}")
            db.rollback()
            return 0
        finally:
            db.close()

    # ──────────────────────────────────────────────────────────────────────────
    # Pattern Retrieval
    # ──────────────────────────────────────────────────────────────────────────

    def get_relevant_patterns(
        self,
        query: str,
        agent_id: Optional[str] = None,
        top_k: int = 3,
    ) -> str:
        """
        Find procedural patterns most relevant to the current user query.
        Returns serialized text for injection into the context window.
        """
        trigger_type = classify_trigger(query)

        db = self.get_db()
        try:
            from memnai.db.models import ProceduralPattern

            # Match by trigger type; order by confidence × support_count
            patterns = (
                db.query(ProceduralPattern)
                .filter(
                    ProceduralPattern.agent_id == agent_id,
                    ProceduralPattern.trigger == trigger_type,
                    ProceduralPattern.confidence >= self.MIN_CONTEXT_CONFIDENCE,
                )
                .order_by(
                    (ProceduralPattern.confidence * ProceduralPattern.support_count).desc()
                )
                .limit(top_k)
                .all()
            )

            if not patterns:
                # Fallback: get globally applicable patterns
                patterns = (
                    db.query(ProceduralPattern)
                    .filter(
                        ProceduralPattern.is_global == True,
                        ProceduralPattern.confidence >= self.MIN_CONTEXT_CONFIDENCE,
                    )
                    .order_by(ProceduralPattern.confidence.desc())
                    .limit(top_k)
                    .all()
                )

            if not patterns:
                return ""

            # Update last_used_at
            for p in patterns:
                p.last_used_at = datetime.utcnow()
            db.commit()

            return self._serialize_patterns(patterns)

        except Exception as e:
            logger.warning(f"[ProceduralMemory] get_relevant_patterns failed: {e}")
            return ""
        finally:
            db.close()

    # ──────────────────────────────────────────────────────────────────────────
    # Serialization
    # ──────────────────────────────────────────────────────────────────────────

    def _serialize_patterns(self, patterns) -> str:
        """Convert pattern list to compact context string."""
        lines = ["[BEHAVIORAL PATTERNS]"]
        for p in patterns:
            confidence_pct = int(p.confidence * 100)
            lines.append(
                f"• {p.full_pattern} "
                f"(confidence={confidence_pct}%, seen {p.support_count}×)"
            )
        return "\n".join(lines)

    def _generate_pattern_text(
        self,
        trigger_type: str,
        action_type: str,
        examples: List[Tuple[str, str]],
    ) -> str:
        """
        Generate a human-readable behavioral pattern description.
        Uses an LLM if available, otherwise uses template.
        """
        if self.summarizer:
            try:
                prompt = (
                    f"Generalize this behavioral pattern into one concise rule "
                    f"(max 15 words):\n"
                    f"Trigger type: {trigger_type}\n"
                    f"Action type: {action_type}\n"
                    f"Examples:\n"
                    + "\n".join(
                        f"  User: {ex[0][:100]}\n  Agent: {ex[1][:100]}"
                        for ex in examples
                    )
                    + "\n\nRule:"
                )
                # Use a single-turn LLM call for pattern generalization
                result, _ = self.summarizer.compress(
                    [{"role": "user", "content": prompt}]
                )
                return result.strip()
            except Exception:
                pass

        # Template fallback
        trigger_labels = {
            "bug_report": "user reports a bug",
            "feature_request": "user requests a feature",
            "question": "user asks a question",
            "code_review": "user asks for code review",
            "debugging": "user is debugging",
            "planning": "user needs planning help",
            "clarification": "there is ambiguity",
            "general": "user sends a message",
        }
        return (
            f"When {trigger_labels.get(trigger_type, trigger_type)} → "
            f"{action_type}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Batch operations
    # ──────────────────────────────────────────────────────────────────────────

    def mine_all_sessions(self, agent_id: Optional[str] = None) -> Dict[str, int]:
        """Mine patterns from all sessions for an agent. Returns {session_id: patterns_saved}."""
        db = self.get_db()
        try:
            from memnai.db.models import Session
            sessions = db.query(Session).filter(
                Session.agent_id == agent_id,
                Session.is_archived == False,
            ).all()
            session_ids = [s.session_id for s in sessions]
        finally:
            db.close()

        results = {}
        for sid in session_ids:
            results[sid] = self.mine_patterns(sid, agent_id)
        return results

    def promote_to_global(self, pattern_id: int) -> bool:
        """Promote a high-confidence pattern to be shared across all agents."""
        db = self.get_db()
        try:
            from memnai.db.models import ProceduralPattern
            p = db.query(ProceduralPattern).filter(ProceduralPattern.id == pattern_id).first()
            if p and p.confidence >= 0.8:
                p.is_global = True
                db.commit()
                return True
            return False
        except Exception:
            db.rollback()
            return False
        finally:
            db.close()
