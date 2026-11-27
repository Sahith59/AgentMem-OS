"""Reader for the author-selected ECB+ annotations (Cybulska and Vossen)."""
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

DEVELOPMENT = {2, 5, 12, 18, 21, 23, 34, 35}


@dataclass(frozen=True)
class Mention:
    start: int
    end: int
    sentence: int
    offsets: tuple
    words: tuple
    left: tuple
    label: str


def documents(folder):
    folder = Path(folder)
    selected = defaultdict(set)
    with (folder / "ECBplus_coreference_sentences.csv").open() as stream:
        for row in csv.DictReader(stream):
            selected[f"{row['Topic']}_{row['File']}"].add(int(row["Sentence Number"]))
    with ZipFile(folder / "ECB+.zip") as archive:
        for info in sorted(archive.namelist()):
            name = Path(info).stem
            if not info.startswith("ECB+/") or name not in selected or not info.endswith(".xml"):
                continue
            topic = int(name.split("_")[0])
            if topic > 35:
                continue
            doc = ElementTree.fromstring(archive.read(info))
            by_sentence, tokens = defaultdict(list), {}
            for tok in doc.findall("token"):
                sent = int(tok.get("sentence"))
                position = len(by_sentence[sent])
                word = tok.text or ""
                tokens[int(tok.get("t_id"))] = (sent, position, word)
                by_sentence[sent].append(word)
            chains = {}
            for rel in doc.find("Relations"):
                target = rel.find("target")
                if target is None or rel.tag not in {"INTRA_DOC_COREF", "CROSS_DOC_COREF"}:
                    continue
                chain = rel.get("note") or "target:" + target.get("m_id")
                for source in rel.findall("source"):
                    key = source.get("m_id")
                    if key in chains and chains[key] != chain:
                        raise ValueError(f"Conflicting chains in {name}")
                    chains[key] = chain
            mentions = []
            for mark in doc.find("Markables"):
                if not mark.tag.startswith(("ACTION_", "NEG_ACTION_")):
                    continue
                ids = sorted(int(x.get("t_id")) for x in mark.findall("token_anchor"))
                if not ids:
                    continue
                sent, position, _ = tokens[ids[0]]
                if sent not in selected[name] or any(tokens[i][0] != sent for i in ids):
                    continue
                mid = mark.get("m_id")
                mentions.append(Mention(ids[0], ids[-1], sent, tuple(tokens[i][1] for i in ids),
                                        tuple(tokens[i][2].casefold() for i in ids),
                                        tuple(w.casefold() for w in by_sentence[sent][max(0,position-7):position]),
                                        chains.get(mid, "singleton:" + mid)))
            yield topic, name, {s: by_sentence[s] for s in sorted(selected[name])}, sorted(mentions, key=lambda m:(m.start,m.end))
