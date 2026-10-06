"""Versioned two-arm session-opening screen. Exactly one paid run, no retries."""

import json
import time
from pathlib import Path

from . import guarded_answer_screen as prior
from .evaluator_v1 import contract as judge
from .luna_evidence_plan import canonical
from .session_opening_retrieval import Session, Turn, expand

ROOT = Path(__file__).resolve().parents[1]
ANSWER_SETTINGS = prior.ANSWER_SETTINGS
MAX_BYTES = prior.MAX_BYTES
STAGES = ("baseline", "focus", "judge_baseline", "judge_focus")
sha, file_hash = prior.sha, prior.file_hash
input_bound, reservation, receipt_cost = prior.input_bound, prior.reservation, prior.receipt_cost


def stages(case):
    arms = sorted(
        ("baseline", "focus"), key=lambda a: sha("session-opening-v1:" + case["id"] + ":" + a)
    )
    return (*arms, *("judge_" + a for a in arms))


def request(package, case, stage, outputs=None, *, upper_bound=False):
    outputs = outputs or {}
    if stage.startswith("judge_"):
        arm = stage.removeprefix("judge_")
        return judge.request(
            dict(
                type=case["type"],
                question=case["question"],
                gold=case["gold"],
                abstention=case["abstention"],
                response="x" * MAX_BYTES if upper_bound else outputs[arm],
            )
        )
    if stage not in ("baseline", "focus"):
        raise ValueError("Unknown stage")
    context = case["baseline"] if stage == "baseline" else case["candidate"]
    content = package["answer_prompt"].format(
        context=context,
        question=case["question"],
        today_line="\nToday's date is " + case["date"] + "." if case["date"] else "",
    )
    return dict(ANSWER_SETTINGS, messages=[dict(role="user", content=content)])


def code_hashes():
    paths = list(prior.code_hashes()) + [
        "benchmarks/session_opening_retrieval.py",
        "benchmarks/session_opening_screen.py",
    ]
    return {p: file_hash(ROOT / p) for p in paths}


def verify_sources(package):
    """Evaluator-side provenance checks; labels never enter retrieval or generation."""
    from datetime import datetime

    record = package["source_dataset"]
    source = package["source_package"]
    if (
        file_hash(record["path"]) != record["sha256"]
        or file_hash(source["path"]) != source["sha256"]
    ):
        raise ValueError("Changed source artifact")
    data = json.loads(Path(record["path"]).read_text())
    lookup = {r["question_id"]: r for r in data}
    if len(lookup) != len(data):
        raise ValueError("Duplicate source question")
    original = json.loads(Path(source["path"]).read_text())
    if [c["id"] for c in original["cases"]] != [c["id"] for c in package["cases"]] or package[
        "answer_prompt"
    ] != original["answer_prompt"]:
        raise ValueError("Changed population or prompt")
    for case, old in zip(package["cases"], original["cases"], strict=True):
        row = lookup[case["id"]]
        runtime = old["runtime"]
        if (
            case["question"] != runtime["question"]
            or case["question"] != row["question"]
            or case["date"] != runtime["question_date"]
            or case["date"] != row["question_date"]
            or case["baseline"] != runtime["packet"]
            or any(case[k] != old[k] for k in ("gold", "type", "abstention"))
        ):
            raise ValueError("Changed original case")

        def parse(text):
            return datetime.strptime(text, "%Y/%m/%d (%a) %H:%M")

        projected = [
            dict(observed_at=date, turns=[dict(role=t["role"], text=t["content"]) for t in turns])
            for date, turns in zip(row["haystack_dates"], row["haystack_sessions"], strict=True)
            if parse(date) <= parse(case["date"])
        ]
        if case["sessions"] != projected:
            raise ValueError("Changed source projection or date cutoff")


def validate(package):
    import importlib.metadata

    if package["sklearn_version"] != importlib.metadata.version("scikit-learn"):
        raise ValueError("Changed retrieval dependency version")
    verify_sources(package)
    if package["code_sha256"] != code_hashes():
        raise ValueError("Changed implementation")
    if (
        package["answer_settings"] != ANSWER_SETTINGS
        or package["judge_settings"] != judge.SETTINGS
        or package["judge_rates"] != judge.RATES
    ):
        raise ValueError("Changed models or rates")
    cases = package["cases"]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Invalid population")
    total = 0
    for case in cases:
        sessions = tuple(
            Session(s["observed_at"], tuple(Turn(**t) for t in s["turns"]))
            for s in case["sessions"]
        )
        candidate, report = expand(case["question"], case["baseline"], sessions)
        if candidate != case["candidate"] or report != case["retrieval"]:
            raise ValueError("Changed source expansion")
        for stage in stages(case):
            req = request(package, case, stage, upper_bound=True)
            cost = reservation(req)
            if (
                input_bound(req) >= 272000
                or cost != package["request_bounds_nusd"][case["id"]][stage]
            ):
                raise ValueError("Invalid request bound")
            total += cost
    if (
        package["maximum_attempts"] != 4 * len(cases)
        or package["maximum_reservation_nusd"] != total
    ):
        raise ValueError("Invalid total bounds")
    return sha(canonical(package))


def accept(case, stage, response):
    if (
        response.get("finish_reason") != "stop"
        or not response.get("text")
        or len(response["text"].encode()) > MAX_BYTES
    ):
        raise ValueError("Incomplete or oversized output")
    if stage.startswith("judge_"):
        judge.verdict(response["text"])


def verify(package, state, *, complete=False):
    if state["binding"]["package_sha256"] != validate(package):
        raise ValueError("Changed package binding")
    jobs, expected, receipts, rows = state["jobs"], set(), set(), []
    reserved = usage = 0
    unreconciled = []
    tokens = {stage: dict(input=0, output=0) for stage in STAGES}
    for case in package["cases"]:
        outputs, grades = {}, {}
        for stage in stages(case):
            key = case["id"] + ":" + stage
            expected.add(key)
            job = jobs.get(key)
            if job is None:
                continue
            req = request(package, case, stage, outputs)
            bound = package["request_bounds_nusd"][case["id"]][stage]
            if job["request_sha256"] != sha(canonical(req)) or job["reservation_nusd"] != bound:
                raise ValueError("Changed job")
            reserved += bound
            response = job.get("response")
            if response and "usage_upper_nusd" in job:
                cost = receipt_cost(req, response)
                if response["id"] in receipts or job["usage_upper_nusd"] != cost or cost > bound:
                    raise ValueError("Invalid or duplicate receipt")
                receipts.add(response["id"])
                usage += cost
                tokens[stage]["input"] += response["usage"]["prompt_tokens"]
                tokens[stage]["output"] += response["usage"]["completion_tokens"]
            else:
                unreconciled.append(key)
            if job["status"] == "complete":
                if not response:
                    raise ValueError("Missing response")
                accept(case, stage, response)
                outputs[stage] = response["text"]
                if stage.startswith("judge_"):
                    grades[stage.removeprefix("judge_")] = judge.verdict(response["text"])
            elif complete:
                raise ValueError("Incomplete run")
        rows.append(
            dict(id=case["id"], grades=grades, retrieval_status=case["retrieval"]["status"])
        )
    if (
        set(jobs) - expected
        or reserved != state["reserved_nusd"]
        or reserved > state["binding"]["budget_nusd"]
    ):
        raise ValueError("Invalid ledger")
    if complete and len(jobs) != package["maximum_attempts"]:
        raise ValueError("Missing attempts")
    paired = [r for r in rows if len(r["grades"]) == 2]
    gains = sum(r["grades"]["focus"] and not r["grades"]["baseline"] for r in paired)
    losses = sum(r["grades"]["baseline"] and not r["grades"]["focus"] for r in paired)
    passed = len(paired) == len(rows) and gains - losses >= 3 and losses <= 1
    return dict(
        status="PASS_EXPLORATORY_SCREEN" if passed else "FAIL_OR_INCOMPLETE",
        attempted=len(jobs),
        completed=sum(j["status"] == "complete" for j in jobs.values()),
        complete_pairs=len(paired),
        correct={a: sum(r["grades"][a] for r in paired) for a in ("baseline", "focus")},
        gains=gains,
        losses=losses,
        rows=rows,
        reserved_nusd=reserved,
        usage_upper_nusd=usage,
        unreconciled_usage_attempts=unreconciled,
        tokens_by_stage=tokens,
        full500_accuracy="NOT_MEASURED",
        mode=state["binding"]["mode"],
    )


def run(package, directory, approval, *, mode="paid", provider=None):
    import fcntl

    from .english_screen.runner import OpenAIProvider, atomic, error_diagnostics

    identity = validate(package)
    directory = Path(directory).resolve()
    if (
        mode not in ("paid", "offline-test")
        or approval.get("approved") is not True
        or approval.get("mode") != mode
        or approval.get("package_sha256") != identity
        or approval.get("maximum_attempts") != package["maximum_attempts"]
        or type(approval.get("budget_nusd")) is not int
        or approval["budget_nusd"] < package["maximum_reservation_nusd"]
        or not approval.get("authorization_text")
    ):
        raise ValueError("Exact approval required")
    if mode == "paid" and (
        provider is not None or approval.get("output_directory") != str(directory)
    ):
        raise ValueError("Real provider and bound directory required")
    if mode == "offline-test" and provider is None:
        raise ValueError("Fake provider required")
    directory.mkdir(parents=True, exist_ok=True)
    binding = dict(
        package_sha256=identity,
        approval_sha256=sha(canonical(approval)),
        budget_nusd=approval["budget_nusd"],
        mode=mode,
    )
    with (directory / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = directory / "checkpoint.json"
        state = (
            json.loads(path.read_text())
            if path.exists()
            else dict(binding=binding, jobs={}, reserved_nusd=0)
        )
        if state["binding"] != binding:
            raise ValueError("Changed approval binding")
        verify(package, state)
        if any(j["status"] != "complete" for j in state["jobs"].values()):
            raise ValueError("Unresolved attempt; no retry")
        for case in package["cases"]:
            outputs = {}
            for stage in stages(case):
                key = case["id"] + ":" + stage
                if key in state["jobs"]:
                    outputs[stage] = state["jobs"][key]["response"]["text"]
                    continue
                req = request(package, case, stage, outputs)
                bound = package["request_bounds_nusd"][case["id"]][stage]
                if (
                    reservation(req) > bound
                    or state["reserved_nusd"] + bound > approval["budget_nusd"]
                ):
                    raise ValueError("Budget stop before dispatch")
                job = dict(
                    status="pending", request_sha256=sha(canonical(req)), reservation_nusd=bound
                )
                state["jobs"][key] = job
                state["reserved_nusd"] += bound
                atomic(path, state)
                started = time.monotonic()
                try:
                    if provider is None:
                        provider = OpenAIProvider()
                    response = provider(req)
                    job["response"] = response
                    job["usage_upper_nusd"] = receipt_cost(req, response)
                    accept(case, stage, response)
                    job["status"] = "complete"
                    outputs[stage] = response["text"]
                except Exception as error:
                    job.update(
                        status="error",
                        elapsed_seconds=time.monotonic() - started,
                        error_class=type(error).__name__,
                        error_diagnostics=error_diagnostics(error),
                    )
                    atomic(path, state)
                    raise RuntimeError("Attempt failed; preserved; no retry") from None
                job["elapsed_seconds"] = time.monotonic() - started
                atomic(path, state)
        result = verify(package, state, complete=True)
        atomic(directory / "summary.json", result)
        return result
