"""
Profile tier G2: project REAL preference/identity facts from the Gate C
corpus with the REAL local model, then show what an actual user's
profile looks like and what it would cost to inject.

$0 (local llama3.1). Reads a COPY of the corpus — the Gate C artifact
is never mutated by a smoke.

Reports honestly: how many facts the model refused to key, how many
keys the gates rejected, how many attributes collapsed onto one key
(the whole point), and whether the injected block fits its slice.
"""
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
_SCRATCH = Path(tempfile.mkdtemp(prefix="agentmem-profsmoke-"))
_CORPUS = _SCRATCH / "profile_smoke.db"
shutil.copy(HERE / "extracted_memories" / "gate_c_facts.db", _CORPUS)
os.environ["AGENTMEM_OS_DB_PATH"] = str(_CORPUS)
os.environ["AGENTMEM_OS_DISABLE_REDIS"] = "1"

sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent))

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 120


def main():
    from agentmem_os.db.engine import get_session
    from agentmem_os.db.models import SemanticFact
    from agentmem_os.db.profile import ProfileStore
    from agentmem_os.llm.profile_extractor import ProfileExtractor

    print(f"corpus copy: {_CORPUS}")
    db = get_session()
    try:
        tot = db.query(SemanticFact).filter(
            SemanticFact.fact_type.in_(("preference", "identity")),
            SemanticFact.superseded_by.is_(None)).count()
    finally:
        db.close()
    print(f"profileable facts in corpus: {tot} "
          f"(projecting the first {LIMIT})")

    ex = ProfileExtractor(get_session)
    rep = ex.project_scope(limit=LIMIT)
    print(f"\nprojection report: {rep}")
    keyed = rep["projected"]
    print(f"  model refused to key : {rep['skipped_by_model']} "
          f"({rep['skipped_by_model'] / max(1, rep['candidates']):.0%} — "
          f"'not a stable property' is a legitimate answer)")
    print(f"  gates rejected       : {rep['rejected']}")
    print(f"  stored               : {keyed}")

    store = ProfileStore(get_session)
    attrs = store.current(limit=1000)
    print(f"\ndistinct attributes after projection: {len(attrs)}")
    db = get_session()
    try:
        from agentmem_os.db.models import ProfileAttribute
        rows = db.query(ProfileAttribute).all()
        per_key = Counter(r.attribute_key for r in rows)
    finally:
        db.close()
    collapsed = {k: n for k, n in per_key.items() if n > 1}
    print(f"attribute keys with MULTIPLE facts (the collapse that makes "
          f"a profile a profile): {len(collapsed)}")
    for k, n in sorted(collapsed.items(), key=lambda kv: -kv[1])[:8]:
        hist = store.history(k)
        print(f"   {k}: {n} facts | current={hist[-1].value_text[:40]!r}")

    # THE ASSEMBLER'S OWN PATH — not a hand-picked subset (G3 R2 major
    # 7: the old numbers used limit=25/char_budget=4000, which is not
    # what the assembler injects, so the reported cost described
    # nothing that ships).
    from agentmem_os.llm.token_counter import TokenCounter
    from agentmem_os.llm.context_assembler import PROFILE_BUDGET_SHARE
    tc = TokenCounter()
    slice_tokens = int(4740 * PROFILE_BUDGET_SHARE)
    injected = store.current(limit=40)
    block = store.render(injected, token_budget=slice_tokens, counter=tc)
    print("\n--- AS THE ASSEMBLER INJECTS IT (limit=40, real slice) ---")
    print(block[:1800])
    n_values = sum(len(l.split(": ", 1)[1].split("; ")) for l in
                   block.split("\n") if ": " in l)
    print(f"\ninjected: {len(block.split(chr(10)))} lines / {n_values} "
          f"values | {tc.count(block)} tokens of the {slice_tokens} slice")
    print(f"REACH: {n_values} of {keyed} projected facts reach the prompt "
          f"({n_values / max(1, keyed):.0%}) — the B5 measurement")
    print(f"selection: {store.last_selection} | render: {store.last_render}")
    assert tc.count(block) <= slice_tokens, "render exceeded its slice"

    print(f"\nRESULT: {keyed} attributes stored from {rep['candidates']} "
          f"candidate facts; {len(collapsed)} keys carry history")


if __name__ == "__main__":
    main()
