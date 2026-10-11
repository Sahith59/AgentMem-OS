# Architecture findings and next acceptance criteria

Status: evidence-backed diagnosis and proposed next design. The only new live-tested admission feature is the literal-date witness guard. No production memory migration or event compiler is implemented.

| Observed trace | Architectural issue | Next check |
|---|---|---|
| run004 r1-D07-hi: conditional callback becomes repair approval; wrong yes answer | A precondition is represented as a loosely typed fact and can be read as an event | Separate conditional rule from observed event; an explicit approval event is needed before authorization |
| Runs003–005: observed_at date becomes valid_from | Observation, effective date and scheduled action date are conflated | Preserve distinct time roles and their source spans; represent unknown times explicitly |
| Rejected backend request fails status enum or becomes withdrawn | Lifecycle, modality and backend outcome share one enum | Separate lifecycle, assertion modality and outcome; preserve rejected versus withdrawn |
| run003 r2-D09-en: two current delivery dates | No explicit versioned state transition links extracted facts | Match customer/order and version; late old receipt cannot overwrite current state |
| run002 r1-D01-hi: pickup_counter_id copied into required answer field | Internal memory attribute names compete with response contract | Validate answer shape prospectively; do not repair old answers or rename labels after scoring |
| Date-bearing wrong-role fact passes v6 | Literal occurrence is weaker than semantic support | Counterexample must remain failing until modality/role checks are demonstrated |

Proposed data model: immutable source record; candidate source-bound event with modality and time roles; admission decision with evidence; derived state with traceable transition. Generated candidates are proposals, not authoritative backend updates. A backend-confirmed transition needs an actual receipt; an ordinary user assertion can be stored as an assertion without inventing business confirmation.

Acceptance criteria before a paid package:

- The observed false approval produces no approval event or authorized action.
- A minimally changed, explicitly approved positive case does produce the correct result; do not solve safety by rejecting everything.
- Source language, names and dates can change without fixture IDs or gold values appearing in the implementation.
- Rejected, withdrawn, pending and accepted have distinct outcomes.
- Same-customer/order version updates and delayed replays behave correctly after a restart.
- Full originals and unsupported/ambiguous candidates are retained for inspection; uncertainty does not become a fabricated fact.
- Compare plain transactional SQLite and prompt-only controls with identical model and compute allocation. Document ordinary-control success as a valid result.
- Use fresh families for confirmation, group translations and repeats, and report source-state correctness separately from answer correctness.

Potential contribution: a reproducible multilingual set of condition/status/time counterexamples and a tested source-to-state contract. Whether this is novel or valuable beyond established event/state machinery remains open. Do not equate an engineering fix with a publishable research contribution.
