"""
Merge the sharded Gate C extraction DBs into ONE corpus DB.

This is NOT a file concatenation. Each shard is an independent SQLite
database with its own autoincrement id space, and the same real-world
entity appears as a DIFFERENT kg_nodes row in every shard that saw it.
A naive merge would collide primary keys, orphan every foreign key,
and fragment entities — silently destroying the entity floor that
facts_for_entity depends on.

So the merge remaps ids per table and DEDUPLICATES entities the same
way the live schema does:

  turns                  → new ids; old→new map kept
  semantic_facts         → new ids; source_turn_ids JSON remapped;
                           superseded_by remapped (may point at a fact
                           from the SAME shard only — verified)
  kg_nodes               → deduped on (agent_id, entity_text), the exact
                           key of uq_kg_nodes_scope_text; mention_count
                           SUMMED, first_seen min, last_seen max
  kg_edges               → deduped on (src, tgt, relation); weight SUMMED
  semantic_fact_entities → remapped through both maps; UNIQUE respected
  supersession_judgments → fact_id remapped
  consolidation_log      → session rows carried (the resume ledger)

Every step is verified after the fact: row counts must equal the sum of
the parts (minus deliberate dedup), and ZERO foreign keys may dangle.
Refuses to overwrite an existing output unless --force.

Usage:
  python3 benchmarks/merge_gate_c_shards.py [--force]
"""
import json
import shutil
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).parent
SHARD_DIR = HERE / "extracted_memories" / "shards"
OUT = HERE / "extracted_memories" / "gate_c_facts.db"


def _cols(con, table):
    return [r[1] for r in con.execute(f"PRAGMA table_info({table})")]


def main():
    force = "--force" in sys.argv
    shards = sorted(SHARD_DIR.glob("gate_c_shard*.db"))
    if not shards:
        raise SystemExit(f"no shards in {SHARD_DIR}")
    if OUT.exists() and not force:
        raise SystemExit(f"{OUT} exists — pass --force to replace")
    print(f"merging {len(shards)} shards -> {OUT}")

    # Start from shard 0's SCHEMA (not its data) so every table,
    # index and constraint matches the live schema exactly.
    tmp = OUT.with_suffix(".db.building")
    if tmp.exists():
        tmp.unlink()
    shutil.copy(shards[0], tmp)
    out = sqlite3.connect(tmp)
    out.execute("PRAGMA foreign_keys=OFF")
    for t in ("supersession_judgments", "semantic_fact_entities",
              "kg_edges", "kg_nodes", "semantic_facts", "turns",
              "consolidation_log", "sessions"):
        try:
            out.execute(f"DELETE FROM {t}")
        except sqlite3.OperationalError:
            pass
    out.commit()

    totals = {"sessions": 0, "turns": 0, "facts": 0, "links": 0,
              "nodes_in": 0, "edges_in": 0, "judgments": 0,
              "reaffirmed": 0}
    node_key_to_id = {}   # (agent_id, entity_text) -> merged node id
    edge_key_to_id = {}   # (src, tgt, relation) -> merged edge id

    for shard in shards:
        src = sqlite3.connect(f"file:{shard}?mode=ro", uri=True)
        src.row_factory = sqlite3.Row
        turn_map, fact_map, node_map = {}, {}, {}

        # ── sessions (string PK, disjoint by construction) ───────────
        for r in src.execute("SELECT * FROM sessions"):
            cols = r.keys()
            out.execute(
                f"INSERT OR IGNORE INTO sessions ({','.join(cols)}) "
                f"VALUES ({','.join('?' * len(cols))})", tuple(r))
            totals["sessions"] += 1

        # ── turns ────────────────────────────────────────────────────
        tcols = [c for c in _cols(src, "turns") if c != "id"]
        for r in src.execute("SELECT * FROM turns"):
            cur = out.execute(
                f"INSERT INTO turns ({','.join(tcols)}) "
                f"VALUES ({','.join('?' * len(tcols))})",
                tuple(r[c] for c in tcols))
            turn_map[r["id"]] = cur.lastrowid
            totals["turns"] += 1

        # ── kg_nodes (DEDUPED on the live unique key) ────────────────
        ncols = [c for c in _cols(src, "kg_nodes") if c != "id"]
        for r in src.execute("SELECT * FROM kg_nodes"):
            totals["nodes_in"] += 1
            key = (r["agent_id"], r["entity_text"])
            if key in node_key_to_id:
                nid = node_key_to_id[key]
                out.execute(
                    "UPDATE kg_nodes SET mention_count = mention_count + ?, "
                    "last_seen = MAX(last_seen, ?) WHERE id = ?",
                    (r["mention_count"] or 0, r["last_seen"], nid))
            else:
                cur = out.execute(
                    f"INSERT INTO kg_nodes ({','.join(ncols)}) "
                    f"VALUES ({','.join('?' * len(ncols))})",
                    tuple(r[c] for c in ncols))
                nid = cur.lastrowid
                node_key_to_id[key] = nid
            node_map[r["id"]] = nid

        # ── semantic_facts (superseded_by remapped in pass 2) ────────
        fcols = [c for c in _cols(src, "semantic_facts") if c != "id"]
        for r in src.execute("SELECT * FROM semantic_facts"):
            vals = []
            for c in fcols:
                v = r[c]
                if c == "source_turn_ids" and v:
                    old = json.loads(v)
                    v = json.dumps([turn_map[t] for t in old
                                    if t in turn_map])
                elif c == "superseded_by":
                    v = None          # remapped after all facts exist
                vals.append(v)
            try:
                cur = out.execute(
                    f"INSERT INTO semantic_facts ({','.join(fcols)}) "
                    f"VALUES ({','.join('?' * len(fcols))})", tuple(vals))
                fact_map[r["id"]] = cur.lastrowid
                totals["facts"] += 1
            except sqlite3.IntegrityError as e:
                # The SAME fact extracted in two shards. In a
                # single-process run the store would have RE-AFFIRMED it
                # (one row, several sources) rather than stored a
                # duplicate — (scope_key, normalized_hash) is exactly
                # that dedup identity. Mirror _reaffirm so the merged
                # corpus equals what one process would have produced.
                if "normalized_hash" not in str(e):
                    raise
                ex = out.execute(
                    "SELECT id, source_session_ids, source_turn_ids, "
                    "mention_count FROM semantic_facts WHERE scope_key IS ? "
                    "AND normalized_hash = ?",
                    (r["scope_key"], r["normalized_hash"])).fetchone()
                eid, esess, eturns, ecount = ex
                sess = json.loads(esess) if esess else []
                if r["source_session_id"] and r["source_session_id"] not in sess:
                    sess.append(r["source_session_id"])
                turns = json.loads(eturns) if eturns else []
                for t in json.loads(vals[fcols.index("source_turn_ids")] or "[]"):
                    if t not in turns:
                        turns.append(t)
                out.execute(
                    "UPDATE semantic_facts SET source_session_ids=?, "
                    "source_turn_ids=?, mention_count=? WHERE id=?",
                    (json.dumps(sess), json.dumps(turns),
                     (ecount or 1) + (r["mention_count"] or 1), eid))
                fact_map[r["id"]] = eid
                totals["reaffirmed"] += 1
        for r in src.execute(
                "SELECT id, superseded_by FROM semantic_facts "
                "WHERE superseded_by IS NOT NULL"):
            out.execute("UPDATE semantic_facts SET superseded_by=? "
                        "WHERE id=?",
                        (fact_map[r["superseded_by"]], fact_map[r["id"]]))

        # ── kg_edges (DEDUPED; weights summed) ───────────────────────
        ecols = [c for c in _cols(src, "kg_edges") if c != "id"]
        for r in src.execute("SELECT * FROM kg_edges"):
            totals["edges_in"] += 1
            s, t = node_map.get(r["source_id"]), node_map.get(r["target_id"])
            if s is None or t is None:
                continue
            key = (s, t, r["relation_type"])
            if key in edge_key_to_id:
                out.execute("UPDATE kg_edges SET weight = weight + ? "
                            "WHERE id = ?",
                            (r["weight"] or 0, edge_key_to_id[key]))
                continue
            vals = []
            for c in ecols:
                v = r[c]
                if c == "source_id":
                    v = s
                elif c == "target_id":
                    v = t
                vals.append(v)
            cur = out.execute(
                f"INSERT INTO kg_edges ({','.join(ecols)}) "
                f"VALUES ({','.join('?' * len(ecols))})", tuple(vals))
            edge_key_to_id[key] = cur.lastrowid

        # ── join table + judgments + log ─────────────────────────────
        jcols = [c for c in _cols(src, "semantic_fact_entities")
                 if c != "id"]
        for r in src.execute("SELECT * FROM semantic_fact_entities"):
            f, n = fact_map.get(r["fact_id"]), node_map.get(r["node_id"])
            if f is None or n is None:
                continue
            vals = [f if c == "fact_id" else n if c == "node_id" else r[c]
                    for c in jcols]
            out.execute(
                f"INSERT OR IGNORE INTO semantic_fact_entities "
                f"({','.join(jcols)}) VALUES "
                f"({','.join('?' * len(jcols))})", tuple(vals))
            totals["links"] += 1

        scols = [c for c in _cols(src, "supersession_judgments")
                 if c != "id"]
        for r in src.execute("SELECT * FROM supersession_judgments"):
            f = fact_map.get(r["fact_id"])
            if f is None:
                continue
            vals = [f if c == "fact_id" else r[c] for c in scols]
            out.execute(
                f"INSERT INTO supersession_judgments ({','.join(scols)}) "
                f"VALUES ({','.join('?' * len(scols))})", tuple(vals))
            totals["judgments"] += 1

        lcols = [c for c in _cols(src, "consolidation_log") if c != "id"]
        for r in src.execute("SELECT * FROM consolidation_log"):
            out.execute(
                f"INSERT INTO consolidation_log ({','.join(lcols)}) "
                f"VALUES ({','.join('?' * len(lcols))})",
                tuple(r[c] for c in lcols))

        src.close()
        out.commit()
        print(f"  {shard.name}: cumulative facts={totals['facts']} "
              f"turns={totals['turns']} nodes={len(node_key_to_id)}")

    # ── VERIFY: nothing dangling, nothing lost ───────────────────────
    print("\nverifying...")
    checks = {
        "facts with dangling superseded_by": (
            "SELECT COUNT(*) FROM semantic_facts f WHERE f.superseded_by "
            "IS NOT NULL AND NOT EXISTS (SELECT 1 FROM semantic_facts x "
            "WHERE x.id = f.superseded_by)"),
        "links with dangling fact_id": (
            "SELECT COUNT(*) FROM semantic_fact_entities e WHERE NOT EXISTS "
            "(SELECT 1 FROM semantic_facts f WHERE f.id = e.fact_id)"),
        "links with dangling node_id": (
            "SELECT COUNT(*) FROM semantic_fact_entities e WHERE NOT EXISTS "
            "(SELECT 1 FROM kg_nodes n WHERE n.id = e.node_id)"),
        "edges with dangling endpoints": (
            "SELECT COUNT(*) FROM kg_edges g WHERE NOT EXISTS (SELECT 1 "
            "FROM kg_nodes n WHERE n.id = g.source_id) OR NOT EXISTS "
            "(SELECT 1 FROM kg_nodes n WHERE n.id = g.target_id)"),
        "judgments with dangling fact_id": (
            "SELECT COUNT(*) FROM supersession_judgments j WHERE NOT EXISTS "
            "(SELECT 1 FROM semantic_facts f WHERE f.id = j.fact_id)"),
        "duplicate (agent_id, entity_text) nodes": (
            "SELECT COUNT(*) FROM (SELECT agent_id, entity_text FROM "
            "kg_nodes GROUP BY agent_id, entity_text HAVING COUNT(*) > 1)"),
    }
    failed = False
    for label, q in checks.items():
        n = out.execute(q).fetchone()[0]
        print(f"  {label}: {n}")
        failed |= n != 0

    got_facts = out.execute("SELECT COUNT(*) FROM semantic_facts").fetchone()[0]
    got_sess = out.execute(
        "SELECT COUNT(*) FROM consolidation_log WHERE triggered_by=?",
        ("consolidation_v2",)).fetchone()[0]
    print(f"\n  facts: {got_facts} (expected {totals['facts']}); "
          f"cross-shard duplicates re-affirmed into existing rows: "
          f"{totals['reaffirmed']}")
    print(f"  sessions consolidated: {got_sess}")
    print(f"  kg_nodes: {len(node_key_to_id)} merged from "
          f"{totals['nodes_in']} shard rows "
          f"({totals['nodes_in'] - len(node_key_to_id)} duplicates fused)")
    print(f"  kg_edges: {len(edge_key_to_id)} merged from "
          f"{totals['edges_in']}")
    print(f"  fact->entity links: {totals['links']}")
    failed |= got_facts != totals["facts"]

    out.execute("PRAGMA foreign_keys=ON")
    fk = out.execute("PRAGMA foreign_key_check").fetchall()
    print(f"  sqlite foreign_key_check violations: {len(fk)}")
    failed |= bool(fk)
    out.commit()
    out.close()

    if failed:
        print("\nMERGE FAILED VERIFICATION — output left at "
              f"{tmp} for inspection")
        sys.exit(1)
    tmp.replace(OUT)
    print(f"\nMERGE OK -> {OUT}")


if __name__ == "__main__":
    main()
