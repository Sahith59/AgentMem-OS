"""Diff the two audit passes (old vs new stack) and cross-reference the
actual right/wrong outcomes, per question and per category. $0."""
import json
from pathlib import Path

HERE = Path(__file__).parent
old = {r["question"]: r for r in json.load(open(HERE / "audit_dualstack_old.json"))}
new = {r["question"]: r for r in json.load(open(HERE / "audit_dualstack_new.json"))}

r1 = {r["question"]: r["correct"] for r in json.load(
    open(HERE / "qa_accuracy_longmemeval_500q_40k_r1.json"))["results"]}
ku_new = {r["question"]: r["correct"] for r in json.load(
    open(HERE / "qa_accuracy_longmemeval_ku78_fullturns.json"))["results"]}
pref_new = {r["question"]: r["correct"] for r in json.load(
    open(HERE / "qa_accuracy_longmemeval_pref30_fullturns.json"))["results"]}
new_correct = {**ku_new, **pref_new}


def mean(xs):
    xs = list(xs)
    return sum(xs) / max(1, len(xs))


for qtype in ("knowledge-update", "single-session-preference"):
    qs = [q for q in new if new[q]["type"] == qtype and q in old]
    print(f"\n=== {qtype} (n={len(qs)}) ===")
    print(f"{'metric':<38}{'OLD':>10}{'NEW':>10}")
    for label, fn in (
        ("packet chars", lambda r: r["packet_chars"]),
        ("verbatim section chars",
         lambda r: r["section_chars"].get("[SEMANTIC MEMORY]", 0)),
        ("facts section chars",
         lambda r: r["section_chars"].get("[SEMANTIC FACTS]", 0)),
        ("BREADTH: sessions in packet",
         lambda r: r["n_sessions_in_packet"]),
        ("turns in packet", lambda r: r["turns_in_packet_total"]),
        ("gold coverage (frac)",
         lambda r: r["gold_cov"] / max(1, r["gold_sessions"])),
        ("gold answer in packet (frac)",
         lambda r: 1.0 if r["gold_ans_in_packet"] else 0.0),
        ("  ...in facts section",
         lambda r: 1.0 if r["gold_ans_in_facts"] else 0.0),
        ("  ...in verbatim section",
         lambda r: 1.0 if r["gold_ans_in_verbatim"] else 0.0),
        ("[UPDATED] annotations",
         lambda r: r["updated_annotations"]),
    ):
        print(f"{label:<38}{mean(fn(old[q]) for q in qs):>10.2f}"
              f"{mean(fn(new[q]) for q in qs):>10.2f}")

    # flip-level attribution
    broke = [q for q in qs if r1.get(q) and new_correct.get(q) is False]
    fixed = [q for q in qs if r1.get(q) is False and new_correct.get(q)]
    print(f"\n  BROKE ({len(broke)}) — old->new per question:")
    for q in broke:
        o, n = old[q], new[q]
        print(f"   breadth {o['n_sessions_in_packet']}->"
              f"{n['n_sessions_in_packet']} | goldcov {o['gold_cov']}/"
              f"{o['gold_sessions']}->{n['gold_cov']}/{n['gold_sessions']}"
              f" | ans {o['gold_ans_in_packet']}->{n['gold_ans_in_packet']}"
              f" | {q[:58]}")
    print(f"  FIXED ({len(fixed)}):")
    for q in fixed:
        o, n = old[q], new[q]
        print(f"   breadth {o['n_sessions_in_packet']}->"
              f"{n['n_sessions_in_packet']} | goldcov {o['gold_cov']}/"
              f"{o['gold_sessions']}->{n['gold_cov']}/{n['gold_sessions']}"
              f" | ans {o['gold_ans_in_packet']}->{n['gold_ans_in_packet']}"
              f" | {q[:58]}")
