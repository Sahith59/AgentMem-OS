# Fresh-workflow research code

The original `cases.py` bank is **v1 historical evidence**. Its
`missing_replacement` questions ambiguously ask for the historical last request
when the labels mean the currently wanted replacement. The live v1 comparison
was stopped after 32 episodes; it must never be resumed or presented as a passed
confirmation. Frozen code remains unchanged so its requests and failures replay.

Use `python -m benchmarks.sarvam_fresh_cases_v2 NEW_DIRECTORY` to create the
corrected offline bank. It retains all 120 renderings and changes only those 12
questions and their runtime hash bindings. It does not call a model or create a
paid-run manifest. This is a development set, not an independent holdout.

`engine.py` executes the unchanged ordinary native actor once and forks only the
final reporting call. `study.py` freezes/rechecks source and inputs, enforces the
budget and executes a once-only manifest. `audit.py` replays every request and
backend effect offline. A valid live package needs separate reviewed labels,
protocol, budget, pacing and manifest; neither generated files nor green tests
establish a model result. See the current public Sarvam research index.
