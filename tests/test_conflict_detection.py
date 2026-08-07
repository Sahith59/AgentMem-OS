"""
Phase 1 Stress Test — Conflict Detection
==========================================
10 true contradictions  → all must be detected   (no false negatives)
 5 false-positive traps → none must be flagged    (no false positives)
 3 explicit forget tests → topic-based deletion works

Run: python3 tests/test_conflict_detection.py
"""

# DB-path guard (G3 S4-R1 Ma8, REPLACED in R2 — the first version was
# injected INSIDE this docstring and never executed, proven when an
# inherited env var migrated 17 tables at its path; third incident of
# the class): this is a SCRIPT outside conftest protection, and
# importing the product reaches db.engine, whose import runs init_db()
# against whatever DB resolves. FORCE a scratch path before any
# agentmem_os import — inherited environment included.
import os as _os
import tempfile as _tempfile
_os.environ["AGENTMEM_OS_DB_PATH"] = _os.path.join(
    _tempfile.mkdtemp(prefix="agentmem-conflict-test-"), "t.db")

import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentmem_os.memory.conflict_detector import (
    ConflictDetector,
    forget_about,
    _has_polarity_flip,
    _entity_overlap,
    _tfidf_cosine,
)

G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[1m"; E = "\033[0m"

passed = failed = 0

def ok(label):
    global passed
    passed += 1
    print(f"  {G}✓{E}  {label}")

def fail(label, detail=""):
    global failed
    failed += 1
    print(f"  {R}✗{E}  {label}" + (f"  ({detail})" if detail else ""))

def section(title):
    print(f"\n{B}── {title} ──{E}")


# ─────────────────────────────────────────────────────────────────────────────
# Section A: Unit tests on _has_polarity_flip
# ─────────────────────────────────────────────────────────────────────────────
section("A. True contradictions — must detect polarity flip")

TRUE_CONTRADICTIONS = [
    # (old_turn, new_turn, description)
    ("I love Python and use it for everything.",
     "I hate Python now, I only use Go.",
     "preference flip: love→hate Python"),

    ("I prefer PyTorch over TensorFlow.",
     "I stopped using PyTorch, TensorFlow is better.",
     "framework preference reversal"),

    ("I work at Google as a software engineer.",
     "I left Google and joined Stripe last month.",
     "employment change"),

    ("I live in San Francisco near the Mission.",
     "I moved away from San Francisco to New York.",
     "location change"),

    ("I study at Stanford University.",
     "I left Stanford and transferred to MIT.",
     "education change"),

    ("I enjoy rock climbing every weekend.",
     "I hate rock climbing, I quit after my injury.",
     "hobby sentiment flip"),

    ("JavaScript is my favourite language.",
     "I avoid JavaScript completely now, it's terrible.",
     "language preference flip"),

    ("I love working at Stripe — great culture.",
     "Stripe is a terrible place to work, I hate it.",
     "workplace sentiment flip"),

    ("I use TensorFlow for all my deep learning work.",
     "I don't use TensorFlow anymore, I switched to JAX.",
     "framework switch with negation in new"),

    ("I really like using Vim for coding.",
     "I can't stand Vim, I switched to VS Code.",
     "editor preference flip"),
]

for old, new, desc in TRUE_CONTRADICTIONS:
    flipped, category = _has_polarity_flip(old, new)
    if flipped:
        ok(desc)
    else:
        fail(desc, f"not detected (category would be ?)")


# ─────────────────────────────────────────────────────────────────────────────
# Section B: False-positive traps — must NOT detect contradiction
# ─────────────────────────────────────────────────────────────────────────────
section("B. False-positive traps — must NOT flag as contradiction")

FALSE_POSITIVES = [
    # (old_turn, new_turn, description)
    ("I love Python.",
     "I love Go too.",
     "two positive preferences, no conflict"),

    ("I work at Stripe.",
     "I used to work at Google before Stripe.",
     "past employment mentioned, current unchanged"),

    ("I like coffee in the morning.",
     "I hate getting up early in the morning.",
     "same context word (morning), different subjects"),

    ("Python is great for ML.",
     "JavaScript is great for web.",
     "positive claims about unrelated subjects"),

    ("I moved to San Francisco.",
     "San Francisco has great weather.",
     "related entity, no polarity flip"),
]

for old, new, desc in FALSE_POSITIVES:
    flipped, category = _has_polarity_flip(old, new)
    if not flipped:
        ok(desc)
    else:
        fail(desc, f"wrongly flagged as {category}")


# ─────────────────────────────────────────────────────────────────────────────
# Section C: Entity overlap gate
# ─────────────────────────────────────────────────────────────────────────────
section("C. Entity overlap gate — fast pre-filter")

OVERLAP_CASES = [
    ("I work at Stripe.", "I left Stripe.", True,  "Stripe appears in both"),
    ("I love Python.",    "I hate Python.", True,  "Python appears in both"),
    ("I love coffee.",    "I hate Python.", False, "no shared entity"),
    ("Good morning!",     "I hate Mondays.", False, "no entities at all"),
]

for a, b, expected, desc in OVERLAP_CASES:
    result = _entity_overlap(a, b)
    if result == expected:
        ok(desc)
    else:
        fail(desc, f"got {result}, expected {expected}")


# ─────────────────────────────────────────────────────────────────────────────
# Section D: TF-IDF cosine sanity
# ─────────────────────────────────────────────────────────────────────────────
section("D. TF-IDF cosine similarity sanity checks")

COSINE_CASES = [
    ("I love Python programming.",
     "Python is my favourite programming language.",
     0.20, "similar Python sentences should score > 0.20"),

    ("I love Python.",
     "I enjoy eating pizza.",
     None, "unrelated sentences should score < similar ones"),
]

sim_python = _tfidf_cosine(
    "I love Python programming.",
    "Python is my favourite programming language."
)
sim_pizza = _tfidf_cosine("I love Python.", "I enjoy eating pizza.")

if sim_python > 0.15:
    ok(f"related sentences score {sim_python:.3f} (> 0.15)")
else:
    fail(f"related sentences score too low: {sim_python:.3f}")

if sim_python > sim_pizza:
    ok(f"related > unrelated ({sim_python:.3f} > {sim_pizza:.3f})")
else:
    fail(f"cosine ordering wrong: {sim_python:.3f} vs {sim_pizza:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Section E: ConflictDetector integration (end-to-end with real DB)
# ─────────────────────────────────────────────────────────────────────────────
section("E. End-to-end DB integration test")

try:
    from agentmem_os.db.engine import get_session, init_db
    from agentmem_os.db.models import Turn, Session as DBSession
    import datetime

    init_db()  # ensures migration runs

    TEST_SESSION = f"conflict-test-{int(time.time())}"
    db = get_session()

    try:
        # Create test session
        sess = DBSession(session_id=TEST_SESSION, name="conflict test", model="test")
        db.add(sess)
        db.commit()

        def _add_turn(role, content):
            t = Turn(
                session_id=TEST_SESSION,
                role=role,
                content=content,
                token_count=len(content.split()),
                is_active=True,
            )
            db.add(t)
            db.commit()
            db.refresh(t)
            return t.id

        # Plant old fact
        old_id = _add_turn("user", "I love Python and use it for everything.")
        _add_turn("assistant", "That's great! Python is very versatile.")

        # Plant contradicting fact
        new_id = _add_turn("user", "I hate Python now, I only write Go.")

        # Run detector synchronously (wait for result)
        detector = ConflictDetector()
        n = detector.check_and_resolve(TEST_SESSION, new_id, "I hate Python now, I only write Go.")

        # Re-query to check state
        old_turn = db.query(Turn).filter(Turn.id == old_id).first()
        db.refresh(old_turn)

        if n >= 1:
            ok(f"Detector returned {n} deactivated turn(s)")
        else:
            fail(f"Detector returned 0 deactivations (expected ≥1)")

        if old_turn.is_active == False:
            ok(f"Old turn {old_id} correctly marked is_active=False")
        else:
            fail(f"Old turn {old_id} still is_active=True after contradiction")

        if old_turn.contradicted_by == new_id:
            ok(f"contradicted_by correctly points to new turn {new_id}")
        else:
            fail(f"contradicted_by={old_turn.contradicted_by}, expected {new_id}")

        # Verify new turn is still active
        new_turn = db.query(Turn).filter(Turn.id == new_id).first()
        db.refresh(new_turn)
        if new_turn.is_active:
            ok("New (superseding) turn remains is_active=True")
        else:
            fail("New turn incorrectly marked inactive")

        # Check get_history excludes the contradicted turn
        active_turns = (
            db.query(Turn)
            .filter(Turn.session_id == TEST_SESSION, Turn.is_active == True)
            .all()
        )
        contents = [t.content for t in active_turns]
        if "I love Python and use it for everything." not in contents:
            ok("get_history excludes contradicted turn")
        else:
            fail("Contradicted turn still visible in active turns")

    finally:
        # Clean up test data
        db.query(Turn).filter(Turn.session_id == TEST_SESSION).delete()
        db.query(DBSession).filter(DBSession.session_id == TEST_SESSION).delete()
        db.commit()
        db.close()

except Exception as e:
    fail(f"DB integration test crashed: {e}")
    import traceback; traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Section F: forget_about() explicit topic deletion
# ─────────────────────────────────────────────────────────────────────────────
section("F. forget_about() — explicit topic-based deletion")

try:
    from agentmem_os.db.engine import get_session, init_db
    from agentmem_os.db.models import Turn, Session as DBSession

    FORGET_SESSION = f"forget-test-{int(time.time())}"
    db2 = get_session()

    try:
        sess2 = DBSession(session_id=FORGET_SESSION, name="forget test", model="test")
        db2.add(sess2)
        db2.commit()

        def _add2(content):
            t = Turn(
                session_id=FORGET_SESSION,
                role="user",
                content=content,
                token_count=len(content.split()),
                is_active=True,
            )
            db2.add(t)
            db2.commit()

        _add2("I love Python and use it for ML projects.")
        _add2("My primary language is Python.")
        _add2("I enjoy coffee every morning.")

        n = forget_about(FORGET_SESSION, "Python programming language")

        if n >= 2:
            ok(f"forget_about deleted {n} turns (expected ≥2 about Python)")
        else:
            fail(f"forget_about only deleted {n} turns (expected ≥2)")

        # Coffee turn should survive
        coffee = db2.query(Turn).filter(
            Turn.session_id == FORGET_SESSION,
            Turn.content.like("%coffee%"),
        ).first()
        db2.refresh(coffee)
        if coffee.is_active:
            ok("Unrelated turn (coffee) not affected by forget_about(Python)")
        else:
            fail("Unrelated turn (coffee) incorrectly deleted")

    finally:
        db2.query(Turn).filter(Turn.session_id == FORGET_SESSION).delete()
        db2.query(DBSession).filter(DBSession.session_id == FORGET_SESSION).delete()
        db2.commit()
        db2.close()

except Exception as e:
    fail(f"forget_about() test crashed: {e}")
    import traceback; traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────
total = passed + failed
print(f"\n{'─'*50}")
print(f"{B}Results: {G}{passed}{E}{B}/{total} passed{E}  |  "
      f"{(R+str(failed)+E+B) if failed else str(failed)} failed{E}")
precision = passed / total if total else 0
print(f"Score: {precision*100:.1f}%")
print()

if failed == 0:
    print(f"{G}{B}✓ Phase 1 Conflict Detection — ALL TESTS PASSED{E}")
    sys.exit(0)
else:
    print(f"{R}{B}✗ Phase 1 — {failed} test(s) failed{E}")
    sys.exit(1)
