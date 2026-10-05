"""Offline construction of a three-arm, fixed-model answer experiment.

Inference requires exact approval: preparation does not authorize spending. The package
contains exact first requests and deterministic dependent-request construction.
"""

import hashlib
import json
import time
from pathlib import Path

from .evaluator_v1 import contract as judge
from .evidence_focus import FocusInput, SourceTurn, validate_input
from .guarded_focus import apply_guarded
from .luna_evidence_plan import SETTINGS, canonical
from .luna_evidence_plan_v3 import plan_request

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("baseline", "revision", "focus")
STAGES = (
    "plan",
    "draft",
    "baseline",
    "revision",
    "focus",
    "judge_baseline",
    "judge_revision",
    "judge_focus",
)
ANSWER_SETTINGS = {"model": "gpt-5.6-luna", "max_completion_tokens": 4200}
MAX_BYTES = 8192
SEED = "guarded-focus-outcome-screen-v1"


def stages(case):
    # Counterbalance arm order deterministically without outcomes or labels.
    order = sorted(ARMS, key=lambda arm: sha(SEED + ":" + case["id"] + ":" + arm))
    return ("plan", "draft", *order, *("judge_" + arm for arm in order))


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime(case):
    v = case["runtime"]
    value = FocusInput(
        v["question"], v["question_date"], v["packet"], tuple(SourceTurn(**t) for t in v["turns"])
    )
    validate_input(value)
    return value


def request(package, case, stage, outputs=None, *, upper_bound=False):
    outputs = outputs or {}
    value = runtime(case)
    if stage == "plan":
        return plan_request(value)
    if stage.startswith("judge_"):
        arm = stage.removeprefix("judge_")
        answer = "x" * MAX_BYTES if upper_bound else outputs[arm]
        return judge.request(
            dict(
                type=case["type"],
                question=value.question,
                gold=case["gold"],
                response=answer,
                abstention=case["abstention"],
            )
        )
    context = value.packet
    if stage == "focus":
        if upper_bound:
            context += "x" * 16000  # <=4,000 extra Unicode code points, <=16k UTF-8 bytes
        else:
            context, _ = apply_guarded(value, outputs["plan"])
    content = package["answer_prompt"].format(
        context=context,
        question=value.question,
        today_line=("\nToday's date is " + value.question_date + ".")
        if value.question_date
        else "",
    )
    if stage == "revision":
        draft = "x" * MAX_BYTES if upper_bound else outputs["draft"]
        # Draft is explicitly untrusted model output, not original evidence.
        return dict(
            ANSWER_SETTINGS,
            messages=[
                {"role": "user", "content": content},
                {"role": "assistant", "content": draft},
                {
                    "role": "user",
                    "content": (
                        "Check the draft against the original conversation above. "
                        "Correct unsupported claims or omissions. Return only the final "
                        "answer; the draft is not evidence."
                    ),
                },
            ],
        )
    if stage not in ("draft", "baseline", "focus"):
        raise ValueError("Unknown stage")
    return dict(
        SETTINGS if stage == "draft" else ANSWER_SETTINGS,
        messages=[{"role": "user", "content": content}],
    )


def input_bound(req):
    return (
        sum(len(m["content"].encode()) for m in req["messages"])
        + len(canonical(req.get("response_format", {})).encode())
        + 2048
    )


def reservation(req):
    rates = (
        judge.RATES
        if req["model"] == judge.SETTINGS["model"]
        else {"cache_write": 250, "output": 1200}
    )
    return input_bound(req) * rates["cache_write"] + req["max_completion_tokens"] * rates["output"]


def code_hashes():
    paths = (
        "benchmarks/guarded_answer_screen.py",
        "benchmarks/guarded_focus.py",
        "benchmarks/evidence_focus.py",
        "benchmarks/luna_evidence_plan.py",
        "benchmarks/luna_evidence_plan_v2.py",
        "benchmarks/luna_evidence_plan_v3.py",
        "benchmarks/evaluator_v1/contract.py",
        "benchmarks/evaluator_v1/verify_focus.py",
        "benchmarks/lme_judge.py",
        "benchmarks/model_policy.py",
        "benchmarks/english_screen/runner.py",
    )
    return {p: file_hash(ROOT / p) for p in paths}


def build(source_path, focus_report_path, count=32):
    source = json.loads(Path(source_path).read_text())
    report = json.loads(Path(focus_report_path).read_text())
    if len(source["cases"]) != 500 or len(report["rows"]) != 500:
        raise ValueError("Full500 source population required")
    if source["settings"]["generate"] != ANSWER_SETTINGS:
        raise ValueError("Changed baseline answerer settings")
    if file_hash(source_path) != report["sources"]["package"]["sha256"]:
        raise ValueError("Focus source mismatch")
    from .evaluator_v1.verify_focus import verify as verify_sources

    if len({c["id"] for c in source["cases"]}) != 500 or {c["id"] for c in source["cases"]} != {
        r["id"] for r in report["rows"]
    }:
        raise ValueError("Duplicate or incomplete source population")
    source_verification = verify_sources(Path(focus_report_path).parent)
    rows = {r["id"]: r for r in report["rows"]}
    ordered = sorted(source["cases"], key=lambda c: sha(SEED + ":" + c["id"]))
    package = dict(
        schema="guarded-answer-screen-v1",
        seed=SEED,
        cases=[],
        answer_prompt=source["prompt"],
        answer_settings=ANSWER_SETTINGS,
        planner_settings=SETTINGS,
        judge_settings=judge.SETTINGS,
        judge_rates=judge.RATES,
        code_sha256=code_hashes(),
        source_verification=source_verification,
        source_sha256=file_hash(source_path),
        focus_report_sha256=file_hash(focus_report_path),
        selection="sha256(seed:id), no outcome labels",
        status="OFFLINE_READY_NOT_AUTHORIZED",
        gate={
            "minimum_net_gain_vs_each_control": 3,
            "maximum_losses_vs_each_control": 1,
            "minimum_applied_focus": 8,
            "interpretation": "exploratory only; no significance/generalization claim",
        },
    )
    if type(count) is not int or not 1 <= count <= 500:
        raise ValueError("Invalid sample count")
    for c in ordered[:count]:
        record = rows[c["id"]]["input_file"]
        if file_hash(record["path"]) != record["sha256"]:
            raise ValueError("Changed runtime input")
        value = json.loads(Path(record["path"]).read_text())
        if (
            value["packet"] != c["context"]
            or value["question"] != c["question"]
            or value["question_date"] != c["date"]
        ):
            raise ValueError("Runtime/source mismatch")
        case = dict(
            id=c["id"],
            runtime=value,
            gold=c["gold"],
            type=c["type"],
            abstention=c["abst"],
            runtime_sha256=record["sha256"],
        )
        runtime(case)
        package["cases"].append(case)
    package["request_bounds_nusd"] = {
        c["id"]: {s: reservation(request(package, c, s, upper_bound=True)) for s in STAGES}
        for c in package["cases"]
    }
    package["maximum_reservation_nusd"] = sum(
        sum(v.values()) for v in package["request_bounds_nusd"].values()
    )
    package["maximum_attempts"] = len(package["cases"]) * len(STAGES)
    return package


def validate(package):
    if package["code_sha256"] != code_hashes():
        raise ValueError("Changed experiment implementation")
    if (
        package["answer_settings"] != ANSWER_SETTINGS
        or package["planner_settings"] != SETTINGS
        or package["judge_settings"] != judge.SETTINGS
        or package["judge_rates"] != judge.RATES
    ):
        raise ValueError("Changed models or rates")
    if not package["cases"] or len({c["id"] for c in package["cases"]}) != len(package["cases"]):
        raise ValueError("Invalid case IDs")
    if package["maximum_attempts"] != len(package["cases"]) * len(STAGES) or package[
        "maximum_reservation_nusd"
    ] != sum(sum(v.values()) for v in package["request_bounds_nusd"].values()):
        raise ValueError("Invalid package totals")
    for case in package["cases"]:
        runtime(case)
        for stage in stages(case):
            req = request(package, case, stage, upper_bound=True)
            if (
                input_bound(req) >= 272000
                or reservation(req) != package["request_bounds_nusd"][case["id"]][stage]
            ):
                raise ValueError("Invalid stage bound")
    return sha(canonical(package))


def verify(package, state, *, complete=False):
    identity = validate(package)
    if state["binding"]["package_sha256"] != identity:
        raise ValueError("Changed package binding")
    jobs = state["jobs"]
    expected = set()
    receipts = set()
    rows = []
    reserved = 0
    usage = 0
    unreconciled = []
    usage_by_stage = {stage: 0 for stage in STAGES}
    for case in package["cases"]:
        outputs = {}
        grades = {}
        disposition = None
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
            if job["status"] != "complete" and "usage_upper_nusd" not in job:
                unreconciled.append(key)
            elif response:
                if response["id"] in receipts:
                    raise ValueError("Duplicate receipt")
                receipts.add(response["id"])
                cost = receipt_cost(req, response)
                if job.get("usage_upper_nusd") != cost or cost > bound:
                    raise ValueError("Invalid receipt cost")
                usage += cost
                usage_by_stage[stage] += cost
            if job["status"] == "complete":
                if not response:
                    raise ValueError("Missing response")
                accept(case, stage, response)
                outputs[stage] = response["text"]
                if stage.startswith("judge_"):
                    grades[stage.removeprefix("judge_")] = judge.verdict(response["text"])
                if stage == "plan":
                    _, disposition = apply_guarded(runtime(case), response["text"])
            elif complete:
                raise ValueError("Incomplete run")
        rows.append(dict(id=case["id"], grades=grades, focus=disposition))
    if (
        set(jobs) - expected
        or reserved != state["reserved_nusd"]
        or reserved > state["binding"]["budget_nusd"]
    ):
        raise ValueError("Invalid ledger")
    if complete and len(jobs) != package["maximum_attempts"]:
        raise ValueError("Missing attempts")
    paired = [r for r in rows if set(r["grades"]) == set(ARMS)]
    comparisons = {
        arm: dict(
            gains=sum(r["grades"]["focus"] and not r["grades"][arm] for r in paired),
            losses=sum(r["grades"][arm] and not r["grades"]["focus"] for r in paired),
        )
        for arm in ("baseline", "revision")
    }
    applied = sum(r["focus"] is not None and r["focus"]["status"] == "APPLIED" for r in rows)
    gate = package["gate"]
    passed = (
        len(paired) == len(package["cases"])
        and applied >= gate["minimum_applied_focus"]
        and all(
            v["gains"] - v["losses"] >= gate["minimum_net_gain_vs_each_control"]
            and v["losses"] <= gate["maximum_losses_vs_each_control"]
            for v in comparisons.values()
        )
    )
    return dict(
        status="PASS_EXPLORATORY_SCREEN" if passed else "FAIL_OR_INCOMPLETE",
        attempted=len(jobs),
        completed=sum(j["status"] == "complete" for j in jobs.values()),
        complete_pairs=len(paired),
        correct={arm: sum(r["grades"][arm] for r in paired) for arm in ARMS},
        comparisons=comparisons,
        applied_focus=applied,
        rows=rows,
        reserved_nusd=reserved,
        usage_upper_nusd=usage,
        unreconciled_usage_attempts=unreconciled,
        usage_upper_nusd_by_stage=usage_by_stage,
        full500_accuracy="NOT_MEASURED",
        mode=state["binding"]["mode"],
    )


def receipt_cost(req, result):
    import re

    model = result.get("model", "")
    if model != req["model"] and not re.fullmatch(
        re.escape(req["model"]) + r"-\d{4}-\d{2}-\d{2}", model
    ):
        raise ValueError("Unexpected provider model")
    if not result.get("id") or not result.get("request_id"):
        raise ValueError("Missing provider identity")
    use = result.get("usage") or {}
    inp, out, total = (use.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
    if (
        not all(type(x) is int and x >= 0 for x in (inp, out, total))
        or inp + out != total
        or inp > input_bound(req)
        or out > req["max_completion_tokens"]
    ):
        raise ValueError("Invalid provider usage")
    rates = (
        judge.RATES
        if req["model"] == judge.SETTINGS["model"]
        else {"cache_write": 250, "output": 1200}
    )
    return inp * rates["cache_write"] + out * rates["output"]


def accept(case, stage, response):
    if (
        response.get("finish_reason") != "stop"
        or not response.get("text")
        or len(response["text"].encode()) > MAX_BYTES
    ):
        raise ValueError("Incomplete or oversized output")
    if stage == "plan":
        apply_guarded(runtime(case), response["text"])
    if stage.startswith("judge_"):
        judge.verdict(response["text"])


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
