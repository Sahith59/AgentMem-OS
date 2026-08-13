"""$0 standalone preflight for the FULL 500-question set.

Calls the SAME preflight functions qa_accuracy_eval.py runs before
spending (gate_c_facts_source.preflight + gate_d_profile_source.preflight)
with the same scope-map shape, so a PASS here is exactly the eval's own
gate. Exists so the corpus can be verified without launching (and
paying for) the eval itself.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

ds = json.load(open(HERE / "benchmark_cache/longmemeval_s.json"))
mems = {m["mid"] for m in ds["memories"]}
scope_by_q = {}
for q in ds["queries"]:
    keys = [k for k in q.get("scope_keys", []) if k in mems]
    if keys:
        scope_by_q[q["question"]] = keys
print(f"{len(scope_by_q)} questions with scopes "
      f"({len({s for v in scope_by_q.values() for s in v})} unique sessions)")

import gate_c_facts_source as gc
ok_c = gc.preflight(scope_by_q)
print(f"GATE C PREFLIGHT: {'PASS' if ok_c else 'FAIL'}")

import gate_d_profile_source as gd
ok_d = gd.preflight(scope_by_q)
print(f"GATE D PREFLIGHT: {'PASS' if ok_d else 'FAIL'}")

sys.exit(0 if (ok_c and ok_d) else 1)
