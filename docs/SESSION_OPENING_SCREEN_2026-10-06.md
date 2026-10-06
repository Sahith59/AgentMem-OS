# Session-opening retrieval: one approved diagnostic screen

October6,2026 America/New_York. Prepared offline, not yet executed.

Problem: the six-question partial comparison had a shared kitchen-item miss. The original coffee-maker donation/upgrade turn was absent from the frozen context. Focusing already-delivered turns could not recover it.

Candidate: TF-IDF (unigrams/bigrams, English stopwords, sublinear TF) ranks original sessions by their best matching user turn. For at most eight positively matching sessions, append the missing first user turn, whole and attributed, without clipping. Maximum4,000 added characters and40,000 total; preserve the original packet byte-for-byte. This is a limited context-expansion hypothesis, not exhaustive retrieval, semantic completeness, or a production default. No extra model, extraction change, generated fact, session label or answer enters retrieval. Ranking may admit distractors. The proposed policy was developed after examining the kitchen failure; these are exposed development cases.

Source controls: only observed date/role/text are projected. Filter out sessions later than question time BEFORE ranking. Package validation checks the original dataset and prior-package hashes, exact question population/order/prompt/baseline/reference metadata, exact projected sources and cutoff, sklearn1.8.0, implementation hashes, reproduced expansion and request bounds. Initial draft lacked the cutoff; independent review caught a real future-turn addition, repaired before any execution. Original baseline packets are preserved and are not newly certified free of future information.

Offline result:30/32 packets expanded, two unchanged; kitchen donation/upgrade recovered. This is evidence delivery, not an answer gain. An initial alternative requiring source anchors already in the packet did not reach the missing kitchen session and was not adopted. No reference-directed lookup occurs at runtime.

Experiment: reuse all32 previously SHA-selected question IDs with no outcome resampling. Regenerate baseline and candidate answers under identical gpt-5.6-luna settings and prompt; judge both with fixed gpt-5.6-terra. Two calls per arm/question (one answer, one judge),128requests maximum, no retries. There is no planner call. Both arms have the same40k context ceiling/4200output cap, but candidate actual input tokens may be larger; report realized tokens and do not call this exact compute equality. This is a new experiment, not continuation or replacement of the stopped three-arm screen.

Gate: complete32pairs, at least3netgains and at most1loss, followed by all changed-answer source/grade review. Passing alone does not establish90%, generalization, significance, research novelty or authorization for a full benchmark/repeat.

Reservation $4.349651; cap $4.35. User explicitly approved ONE paid run this turn, and requested notification for subsequent turns. Approval is bound to package207c5a4f3697be50534c3b6ac0a593c1d313334228b799bd631bbe43c96df215,128requests,paid-screen-001. No retry, second run or broader benchmark is authorized. Credits are not additional authorization.

Validation:158focused tests passed and Ruff passed. Tests cover source recovery, strict budgets, no clipping, provenance/cutoff rejection, gold isolation, saved receipts, tampering, no-repeat completed execution and no retry on errors. Source-opening recovery is offline evidence only. Code commit81abd31. Independent final review and GitHub checks are pending.

Runbook: use pinned Python3.13 and scikit-learn1.8.0; `uv run --no-project --python 3.13 --with scikit-learn==1.8.0 --with openai python ../codex-memory-2026-09-08/plans/2026-10-06-session-opening/execute_approved.py` from inner repo only after checks. Launcher reads credentials locally without printing them. On any error, preserve checkpoint and stop. Verify summary, unique response IDs, exact approval, source receipts and cost ledger; review every gain/loss and the known kitchen case. Record complete or partial outcome before any further recommendation.
