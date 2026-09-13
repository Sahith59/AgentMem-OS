#!/usr/bin/env python3
"""Paired-context variant of the V2 paid runner; default is offline only."""
import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path

from . import runner as base


_base_validate = base.validate
_base_request = base.request


def validate(package):
    identity = _base_validate(package)
    from .verification import verify_contract
    verify_contract(package)
    if package.get("experiment_kind") != "paired-context-retrieval-v1":
        raise ValueError("Expected paired-context retrieval experiment")
    current_contract = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if package.get("packet_runner_sha256") != current_contract:
        raise ValueError("Packet runner contract hash mismatch")
    for case in package["cases"]:
        if set(case.get("arm_contexts", {})) != {"baseline", "candidate"}:
            raise ValueError("Every case needs both arm contexts")
        if set(case.get("arm_context_sha256", {})) != {"baseline", "candidate"}:
            raise ValueError("Every arm context needs a hash")
        for arm in ("baseline", "candidate"):
            if base.digest(case["arm_contexts"][arm]) != case[
                    "arm_context_sha256"][arm]:
                raise ValueError("Changed arm context")
        if case["context"] != case["arm_contexts"]["baseline"]:
            raise ValueError("Compatibility context must equal baseline context")
    if package["prompts"]["baseline"] != package["prompts"]["candidate"]:
        raise ValueError("Retrieval screen requires one identical answer prompt")
    return identity


def request(package, case, arm, stage, answer=None):
    if stage == "generate":
        content = package["prompts"][arm].format(
            context=case["arm_contexts"][arm], question=case["question"],
            today_line=base.date_line(case))
        return dict(package["settings"][stage],
                    messages=[{"role": "user", "content": content}])
    return _base_request(package, case, arm, stage, answer)


def preflight(package):
    original_validate, original_request = base.validate, base.request
    base.validate, base.request = validate, request
    try:
        return base.preflight(package)
    finally:
        base.validate, base.request = original_validate, original_request


def run(*args, **kwargs):
    original_validate, original_request = base.validate, base.request
    base.validate, base.request = validate, request
    try:
        return base.run(*args, **kwargs)
    finally:
        base.validate, base.request = original_validate, original_request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--execute-paid", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--budget-usd")
    parser.add_argument("--approval-record", type=Path)
    args = parser.parse_args()
    package = json.loads(args.package.read_text())
    receipt = preflight(package)
    if not args.execute_paid:
        print(json.dumps(receipt, indent=2))
        return
    if not args.output or not args.budget_usd or not args.approval_record:
        parser.error("Paid execution needs output, budget and a real founder approval record")
    budget = int(Decimal(args.budget_usd) * 1_000_000_000)
    approval = json.loads(args.approval_record.read_text())
    if (approval.get("status") != "FOUNDER_APPROVED"
            or approval.get("package_sha256") != receipt["package_sha256"]
            or approval.get("budget_nusd") != budget
            or Path(approval.get("run_directory", "")).resolve()
            != args.output.resolve()
            or not approval.get("founder_message")):
        parser.error("Approval must bind exact package, budget, output and founder message")
    authorization = base.digest(base.canonical(approval))
    # Fail before constructing a provider on unapproved, corrupt or mismatched work.
    if budget != package["proposed_budget_nusd"]:
        parser.error("Budget differs from frozen proposed cap")
    result = run(package, args.output, base.OpenAIProvider(), budget,
                 authorization, "paid")
    from .verification import verify_paid
    state = json.loads((args.output / "checkpoint.json").read_text())
    verified = verify_paid(package, state, approval, args.output, require_complete=True)
    base.atomic(args.output / "post-run-verification.json", verified)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
