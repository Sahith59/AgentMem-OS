# Evidence bundle

The all-500 audit is negative for readiness. It contains no new model answers. See [the report](../../HYBRID_SOURCE_AUDIT_2026-10-06.md).

`audit.json` is the original result. `audit-v2.json` repeats the audit after adding a nonfinite-similarity guard. Their packets and per-case outcomes are identical. Frozen protocol, code/input/model hashes, numerical replay, independent verification and every question's measured additions/losses are retained. `manifest.json` binds these files; its `external/` entries identify packet files preserved in the local memory directory, not files included in this Git bundle.

`reproduction/` contains exact archived scripts. They are evidence of the executed procedure, not a portable benchmark launcher. **Do not run them from this Git directory.** Their relative paths expect a workspace layout like:

```
workspace/
  AgentMem-OS/                         # repository with audited implementation
  codex-memory-2026-09-08/
    runs/english-improvement-2026-09-11/ # historical inputs in preparation.json
    plans/<fresh-reproduction>/         # copied scripts, protocol, model files
```

Reproduction requires the exact historical inputs listed in `preparation.json`, the historical checkpoint, the pinned E5 assets from `model-files.json`, and the versions in `embeddings-complete.json`. These large inputs, model weights, vectors and raw packet text are not included here. Input hashes containing absolute paths reflect the original machine; relocating requires an explicitly versioned manifest and scripts rather than silently editing archived evidence. A regenerated embedding cache may differ by hardware and must be disclosed.

In a fresh, reviewed workspace, preparation separates runtime data from labels; local embedding reads runtime only; the audit builds all packets before reading evaluator labels. Output creation is exclusive. The v2 script expects v1 packets for byte-parity checks and the v2 frozen core implementation. Never rerun into the original evidence directory or overwrite outputs. Preserve the frozen scripts, original result, model identity, source hashes and evaluation thresholds.

The scripts make no paid LLM calls. Local embedding loads local-only model files. Acquiring missing public model assets is a separate step and must not send benchmark text. The numerical diagnostic and retrieval parity scripts are also offline. No script here is approval for a paid answer comparison.
