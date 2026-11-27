"""Local ECB+ evaluator fixtures; never open held-out XML files."""

import argparse
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

DEV = {2, 5, 12, 18, 21, 23, 34, 35}


def dump(folder, name, records):
    with (folder / name).open("w") as out:
        for rec in records:
            out.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")


def parse(xml, selected):
    root = ET.fromstring(xml)
    sentences = defaultdict(list)
    token_index = {}
    for token in root.findall("token"):
        sent = int(token.get("sentence"))
        token_index[token.get("t_id")] = (sent, len(sentences[sent]))
        sentences[sent].append(token.text or "")
    chains = {}
    for rel in root.find("Relations"):
        if "COREF" not in rel.tag:
            continue
        target = rel.find("target").get("m_id")
        for source in rel.findall("source"):
            mention = source.get("m_id")
            assert mention not in chains or chains[mention] == target
            chains[mention] = target
    gold = defaultdict(list)
    for mark in root.find("Markables"):
        if not mark.tag.startswith(("ACTION_", "NEG_ACTION_")):
            continue
        anchors = [token_index[a.get("t_id")] for a in mark.findall("token_anchor")]
        if not anchors:
            continue
        sent = anchors[0][0]
        assert all(s == sent for s, _ in anchors)
        if sent in selected:
            mid = mark.get("m_id")
            gold[sent].append(
                {
                    "span": sorted(i for _, i in anchors),
                    "chain": chains.get(mid, "singleton:" + mid),
                    "mention": mid,
                    "tag": mark.tag,
                }
            )
    return sentences, gold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.data / "ECBplus_coreference_sentences.csv"
    selected = defaultdict(set)
    for row in csv.DictReader(csv_path.open()):
        selected[f"{row['Topic']}_{row['File']}"].add(int(row["Sentence Number"]))
    freq, hits = Counter(), Counter()
    runtime, reference = [], []
    docmap = {}
    with zipfile.ZipFile(args.data / "ECB+.zip") as archive:
        for member in sorted(archive.namelist()):
            if not member.startswith("ECB+/") or not member.endswith(".xml"):
                continue
            doc = Path(member).stem
            topic = int(doc.split("_")[0])
            if topic >= 36:
                continue
            assert doc not in docmap.values()
            sents, gold = parse(archive.read(member), selected[doc])
            if topic not in DEV:
                for sid in selected[doc]:
                    tokens = [t.lower() for t in sents[sid]]
                    freq.update(tokens)
                    hits.update(
                        tokens[idx]
                        for idx in {m["span"][0] for m in gold[sid] if len(m["span"]) == 1}
                    )
                continue
            doc_id = hashlib.sha256(("input-audit|" + doc).encode()).hexdigest()[:16]
            docmap[doc_id] = doc
            for sid, tokens in sorted(sents.items()):
                runtime.append({"doc": doc_id, "sentence": sid, "tokens": tokens})
                if sid in selected[doc]:
                    reference.append(
                        {"doc": doc_id, "sentence": sid, "topic": topic, "mentions": gold[sid]}
                    )
            assert selected[doc] <= set(sents)
    vocab = {w for w, n in hits.items() if n >= 2 and 2 * n >= freq[w]}
    dump(args.out, "ecb_runtime.jsonl", runtime)
    dump(args.out, "ecb_gold.jsonl", reference)
    dump(
        args.out,
        "ecb_lexical_predictions.jsonl",
        (
            {
                "doc": r["doc"],
                "sentence": r["sentence"],
                "spans": [[i] for i, w in enumerate(r["tokens"]) if w.lower() in vocab],
            }
            for r in runtime
        ),
    )
    (args.out / "ecb_source_map.json").write_text(
        json.dumps(docmap, indent=2, sort_keys=True) + "\n"
    )
    paths = [
        args.data / "ECB+.zip",
        csv_path,
        args.out / "ecb_gold.jsonl",
        args.out / "ecb_runtime.jsonl",
        args.out / "ecb_lexical_predictions.jsonl",
        args.out / "ecb_source_map.json",
    ]
    print(
        json.dumps(
            {
                "hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                "documents": len(docmap),
                "full_sentences": len(runtime),
                "selected_sentences": len(reference),
                "gold_events": sum(len(r["mentions"]) for r in reference),
                "lexical_types": len(vocab),
                "reserved_topics_xml_opened": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
