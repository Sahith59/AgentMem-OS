# Ordinary Sarvam integration and source challenge evidence

Read [REPORT.md](REPORT.md). Twelve live setup episodes yield 12/12 correct application reports and stored states, 2/2 useful approved writes, and 11/12 strict task successes. Raw reports are correct in 8/12; four status fields are corrected from trusted intent/receipts. One unnecessary rejected call remains a strict failure. Dates and clarification fields are never repaired. This is a small integration result, not a language benchmark or memory improvement.

The source challenge bank contains 24 related scenario instances in English, Hindi and Telugu (72 renderings). The 288-episode, two-model/two-enforcement-arm comparison is frozen but unstarted. A prospective attribution supplement separates desired-task failure from factual reporting, including truthful reports of unsafe writes. No comparison result or novelty claim is available.

`evidence.tar.gz` includes frozen source, protocol, original cases/separate gold labels, review, live requests/receipts, real SQLite databases, usage ledger, setup verification, and the prepared comparison including the supplemental evaluation contract. `artifact-manifest.json` records exact bytes/hashes. Founder-file inventory and credentials are excluded. Independent native-language review is absent; translations and known families are correlated.

From a clean checkout, extract into an empty directory and use a new output path:

```sh
mkdir /tmp/sarvam-source-evidence
tar -xzf benchmarks/audits/sarvam-ordinary-source-2026-10-11/evidence.tar.gz -C /tmp/sarvam-source-evidence
python -m benchmarks.sarvam_ordinary_audit /tmp/sarvam-source-evidence setup --output /tmp/sarvam-source-recheck.json
python -m benchmarks.sarvam_source_attribution check /tmp/sarvam-source-evidence
```

The first output must match archived `setup/verification.json` exactly. It reconstructs all 30 requests, actual backend operations, raw/application reports and charges with networking disabled. The second verifies the supplementary contract without credentials or calls. Verify every manifest hash. Do not restart setup or execute live inference from the public evidence copy; authorized future execution uses the original phase root and its carried meter, as documented in `NEXT_PHASE.md`.
