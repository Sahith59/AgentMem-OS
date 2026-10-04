"""Default preflight is offline; run requires a separately approved exact package."""
import argparse
import json
from pathlib import Path
from .contract import preflight
from .runner import run
from .verify import verify

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('action', choices=['preflight', 'run', 'verify'])
ap.add_argument('package', type=Path)
ap.add_argument('--directory', type=Path)
ap.add_argument('--approval', type=Path)
args = ap.parse_args()
package = json.loads(args.package.read_text())
if args.action == 'preflight':
    result = preflight(package)
elif args.action == 'verify':
    if not args.directory:
        ap.error('--directory required')
    result = verify(package, json.loads((args.directory / 'checkpoint.json').read_text()), complete=True)
else:
    if not args.directory or not args.approval:
        ap.error('--directory and --approval required')
    result = run(package, args.directory, json.loads(args.approval.read_text()))
print(json.dumps(result, indent=2))
