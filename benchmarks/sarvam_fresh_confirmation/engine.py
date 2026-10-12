"""Reuse the frozen ordinary actor once; fork only its final reporting call."""

import copy

from benchmarks import sarvam_ordinary_v2 as ordinary
from benchmarks.sarvam_native_controls.transcript_finalizer import classify
from benchmarks.sarvam_native_study.transport import save
from benchmarks.sarvam_receipt_reporting import project
from benchmarks.sarvam_reporter_wording import request_for


def episode(root, spec, case, dispatch):
    reports, calls, bodies = {}, [], {}

    def paired_dispatch(job, request):
        if not job.endswith("-final"):
            calls.append(job)
            return dispatch(job, request)
        for variant in spec["report_order"]:
            real_job = job + "-" + variant
            calls.append(real_job)
            try:
                bodies[variant] = dispatch(real_job, request_for(request, variant))
                reports[variant] = classify(bodies[variant])
            except Exception as exc:
                reports[variant] = {
                    "terminal": "infrastructure_error",
                    "error_class": type(exc).__name__,
                }
        # The unchanged legacy result is the control result. Both reporters have
        # already run, including when one fails. No extra native action or retry.
        if "control" not in bodies:
            raise RuntimeError("Control reporter failed; paired outcome preserved")
        return bodies["control"]

    original = ordinary.episode(root, spec, case, paired_dispatch)
    for variant in ("control", "candidate"):
        report = reports.setdefault(variant, {"terminal": "not_reached"})
        report["application"] = None
        if report.get("parsed") is not None:
            try:
                report["application"] = project(
                    intent=case["intent"],
                    requested_change=case["requested_change"],
                    prior_receipts=original["seed_receipts"],
                    tool_trace=original["tool_trace"],
                    raw_content=report["content"],
                    completed=True,
                )
                app = report["application"]["application_output"]
                assert all(app[k] == report["parsed"][k] for k in ("answer", "needs_clarification"))
            except (ValueError, KeyError, TypeError) as exc:
                report["application_error"] = type(exc).__name__
    result = {"original": original, "reports": reports, "actual_calls": calls}
    save(root / "episodes" / spec["id"] / "paired.json", result)
    return result


def report_view(result, variant):
    view = copy.deepcopy(result["original"])
    report = result["reports"][variant]
    view.update(final=report, terminal=report["terminal"], application=report["application"])
    return view
