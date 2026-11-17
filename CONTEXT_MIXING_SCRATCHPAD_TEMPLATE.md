# Context mixing and sequence-continuity research scratchpad

**Companion PRD:** `PRD_CONTEXT_MIXING_SEQUENCE_INFORMATION.md`  
**Run root:** `results/context_mixing_sequence_continuity/v1/`  
**Rule:** append entries; do not rewrite past RED/GREEN evidence or delete negative outcomes.

## 1. Scientific thread

**Core question:** Why does the physical mixture of familiar A/B contexts disrupt prediction of the exact same sequence?

**Candidate explanations:** (H1) nonlinear cue mixing and recurrent propagation; (H2) sequence-progress representation/usage; (H3) missing learned robustness to within-sequence context changes.

**Primary tests:** exact `e=j+r` identity; history-dependent excess Mixed CE; post-split S−I continued-training effect on held-out Mixed CE.

**Do not claim:** a particular human representation, missing sequence information from a failed decoder, a formal fraction of CE mediated by Jensen mismatch, or causal mediation from concurrent changes in a diagnostic variable.

## 2. Frozen source audit (complete before analysis)

| Item | Actual path/identifier | SHA256 | Verified? | Notes |
|---|---|---|---|---|
| Canonical High checkpoints 0–19 | | | | |
| Executed recurrence + output | | | | |
| Original training objective | | | | |
| Original optimizer state availability | | | | |
| A/B/Mixed/Novel cue registry | | | | |
| Sequence inputs/targets and reset rule | | | | |
| Frozen CE and E1 references | | | | |

**Objective contract:** write exact source total loss, whether execution head trains, optimizer setup, how new arms restart weights and moments. Block D3 if this cannot be made identical across arms.

## 3. Prospective commitments (fill before revealing probe results)

- Primary cohort: canonical High seeds 0–19, final original checkpoint 170.
- D1 primary: seven-step exact decomposition and first-position correction.
- D2 masks: history-dependent `{1,3,4,7}` and token-diagnostic `{2,5,6}`.
- D2 decoder: fixed ridge λ1.0 (A/B only), familiarity transfer gate.
- D3 primary: 160 continuation updates; B 10-sequence blocks (80 A/80 B), I alternating full A/B (80/80), S 80 A→B and 80 B→A switched between scored positions 3/4.
- D3 primary contrast: seed-paired Mixed CE (S−I) after fixed final update.
- Interpretive check: cohort familiar-mean difference <=0.20 nat, A/B separation S at least 80% of I raw separation; descriptive guardrails, not post-hoc inclusion filters.
- Inference: 10,000 seed-cluster bootstrap draws, seed 20260925.
- Protected probes: Mixed, NoContext, Novel30 A/B, Novel60 A/B; never in training.

## 4. Stage completion board

| Stage | RED tests/log | GREEN tests/log | REFACTOR regression | Frozen manifest | Scientific status |
|---|---|---|---|---|---|
| D0 source | | | | | not started |
| D1 mixing | | | | | not started |
| D2 information | | | | | not started |
| D3 continuation | | | | | not started |
| D4 report | | | | | not started |

## 5. RED → GREEN → REFACTOR entries (append one per unit)

### YYYY-MM-DD HH:MM — stage.test_name

- **Purpose and scientific invariant:**
- **RED test authored:**
- **Command and exit code:**
- **Observed failure:**
- **GREEN minimal implementation:**
- **Command and passing evidence:**
- **REFACTOR / cleanup:**
- **Frozen source regression check:**
- **Files changed:**
- **Output and hash:**
- **Scientific caveat or decision:**

## 6. Runs and provenance ledger (append)

| Time | Stage | Seed/cohort | Arm/cue | Config hash | Source/checkpoint SHA | Outcome status | Output path | Failure notes |
|---|---|---|---|---|---|---|---|---|

## 7. Results ledger (fill even for negative outcomes)

### D1

- Maximum exact identity residual `e−j−r`:
- First-step propagation term:
- Natural Mixed excess CE by transition:
- Task-relevant Jensen/propagation components:
- First-step correction remaining-CE effect and uncertainty:
- Biggest counterexample to initial nonlinear-mismatch story:

### D2

- History-dependent vs diagnostic position excess CE difference:
- Per-position confounds and interpretation:
- Familiar cross-context decoder transfer sanity:
- Natural Mixed position-decoder result or `nonidentifiable`:
- How this constrains H2 and what it *doesn't* prove:

### D3

- Training objective verified; optimizer moments status:
- Per-arm training updates and A/B position exposure balance:
- Protected cue firewall passed?
- Mixed S−I paired effect and seed uncertainty:
- Familiar A/B competence separately by arm:
- A/B raw hidden and task-space separation by arm:
- Novel/NoContext transfer:
- History-dependent Mixed error before/after:
- Favorable case and adverse case:
- Interpretation status (`convergent_support`, `familiar_competence_confounded`, `repair_via_merging_possible`, `no_detectable_switch_specific_benefit`, etc.):

## 8. Decisions and blocked branches

For each issue, record: evidence, decision, rejected alternative, scope impact, whether new version is required. **Do not expand search/training budget, switch locations, decoder families or conditions because the first results are unfavorable.**

## 9. Researcher teach-back (refresh at every stage)

- What was changed?
- What was held constant?
- Exact algorithm and loss used?
- Main quantitative evidence?
- Which observation supports causality within the RNN?
- Strongest alternative explanation?
- Which outcomes would refute or narrow the proposed explanation?
- How does this help explain same-sequence failure under Mixed?

## 10. Completion checklist

- [ ] D0 source/checkpoint fidelity and training objective fully documented
- [ ] RED + GREEN evidence for every stage; refactor replay checks
- [ ] D1 exact recurrence identity and intervention correct
- [ ] D2 no probe decoder leakage; familiar sanity reported
- [ ] D3 60 matched new checkpoints or explicit blocked status
- [ ] No protected cue in training; unchanged original checkpoint hashes
- [ ] Seed-cluster inference and all adverse seeds retained
- [ ] Full-sequence and per-position outcomes, not just mean rescue
- [ ] Three figures, claim ledger, researcher handoff, final plain-language report
- [ ] All outputs sealed with reproducible commands and hashes
