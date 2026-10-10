"""python -m benchmarks.sarvam_discovery prepare|check|run|score --help"""

import argparse
from pathlib import Path

from benchmarks.english_screen.runner import atomic

from .contract import canonical, loads, runtime_cases
from .runner import build_package, preflight, run
from .score import evaluate, gold_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Offline; refuses an existing package directory")
    prepare.add_argument("--output-dir", type=Path, required=True)
    for action in ("check", "run", "score"):
        p = commands.add_parser(action)
        p.add_argument("--package", type=Path, required=True)
        if action in {"run", "score"}:
            p.add_argument("--run-dir", type=Path, required=True)
        if action == "run":
            p.add_argument("--approval", type=Path, required=True)
        if action == "score":
            p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        gold_labels(runtime_cases())  # Offline label audit; never part of inference.
        package = build_package()
        plan = preflight(package)
        directory = args.output_dir.resolve()
        directory.mkdir(parents=True, exist_ok=False)
        atomic(directory / "package.json", package)
        atomic(directory / "preflight.json", plan)
        approval = {
            "approved": False,
            "mode": "live",
            "package_sha256": plan["package_sha256"],
            "maximum_attempts": plan["calls_no_retries"],
            "budget_ninr": plan["maximum_reservation_ninr"],
            "output_directory": str(directory / "live-run"),
            "accepted_review_level": package["review_level"],
            "authorization_text": "",
        }
        atomic(directory / "approval-template.json", approval)
        print(canonical(plan))
        return
    package = loads(args.package.read_text())
    if args.command == "check":
        print(canonical(preflight(package)))
    elif args.command == "run":
        print(canonical(run(package, args.run_dir, loads(args.approval.read_text()))))
    elif args.command == "score":
        result = evaluate(package, args.run_dir)
        # Exclusive creation: never silently replace an earlier scoring record.
        with args.output.open("x") as stream:
            stream.write(canonical(result) + "\n")
        print(canonical({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
