# Context Cue Access — append-only coding-agent scratchpad

**PRD:** `PRD_CONTEXT_CUE_ACCESS_CONTROL.md`  
**Run root:** `results/context_cue_access/v1/`  
**Start status:** not started

## A. Scientific thread

**Core question:** Can a bounded context cue or brief cue history move an already-trained recurrent network into a viable continuation?

**Main hypotheses:** wrong single cue / history-dependent access / limited bounded input access.

**Meaning of viable:** record imported P3 implementation and threshold precisely. No new definition by assumption.

**Disconfirming or inconclusive outcomes:** record before running any probe search.

## B. Source and indexing audit

| Item | Actual path/key | SHA256 | Verification |
|---|---|---|---|
| Canonical High seeds 0–19 | | | |
| Network source implementation | | | |
| Frozen cue registry | | | |
| P3 viability function | | | |
| P4 native replay references | | | |
| Sequence token and label indexing | | | |

**Exact time convention:**

```
h_(t-1) --[x_t, c_t]--> h_t --[output head]--> p_t(y_t)
S1: change only c_k; future c_(k+1:7) fixed to target
S2: same h_(k-3); change c_(k-2), c_(k-1); c_k and future fixed to target
Score positions: k+1 through 7 only
```

## C. Frozen protocol decisions (write before running outcome analyses)

- Cue box: [-1.5,1.5]² with inactive third channel zero.
- S1 grid: 13×13, exact anchors separately.
- Cue oracle: 6 starts × 80 updates, Adam 0.04, only cue values trainable.
- S2 primary: R vs L, both 2 free input coordinates, k=3; k=5 secondary.
- Oracle P3 viability threshold imported verbatim: **TBD from code**.
- Primary inference: seed clusters, 20 High seeds; six registered probes.

## D. Stage gates

| Stage | RED | GREEN | REFACTOR | Evidence sealed | Status |
|---|---|---|---|---|---|
| S0 source | | | | | not started |
| S1 cue map | | | | | not started |
| S2 cue history | | | | | not started |
| S3 control sensitivity | | | | | not started |
| S4 reporting | | | | | not started |

## E. RED → GREEN → REFACTOR log (append entries; do not edit history)

### Entry YYYY-MM-DD HH:MM — stage / test ID

- **Goal:**
- **RED test and command:**
- **Expected failure:**
- **Observed failure (include assertion/stack location):**
- **GREEN code change:**
- **GREEN run and result:**
- **REFACTOR change:**
- **Golden replay / regression check:**
- **Files changed:**
- **Evidence path / hash:**

## F. Run ledger (append entries)

| Timestamp | Stage | Cohort/seeds | Cue family | Config hash | Status | Output directory | Reason if invalid |
|---|---|---|---|---|---|---|---|

## G. Findings and uncertainties (append, even for negative findings)

### Stage S1

- Native viability and failure patterns:
- Best finite-grid cue vs native:
- Bounded oracle vs finite grid:
- Familiar-line / familiar-square / signed-box breakdown:
- Eligible oracle-positive / native-failing subgroup denominator:
- Claim supported:
- Claim **not** supported:

### Stage S2

- Same-prestate and same-final-cue verification:
- Paired R vs L at k=3:
- E vs L, I vs R, k=5 secondary:
- Did any history arm cross viability while late-only failed?
- Optimization-budget/parameter-count caveats:
- Claim supported / not supported:

### Stage S3

- Singular values / effective ranks (actual, not upper bounds):
- Tied repeated derivative equals early + late:
- Finite-difference errors:
- Alignment with future-CE gradient and source rollouts:
- Interpretation and limits:

## H. Decision log (append)

- **Issue:**
- **Evidence:**
- **Chosen action:**
- **Alternative rejected and why:**
- **Does this amend the frozen design?** If yes, stop, label deviation, require new version; do not silently change v1.

## I. Researcher teach-back (update after each stage)

- In plain language, what did we vary?
- What exactly was held constant?
- What did the strongest result show?
- Why is it causal within this network, or why not?
- What is the most plausible alternative explanation?
- What would a skeptical lab member ask?
- What remains unresolved?

## J. Completion

- [ ] Complete test_evidence log with RED and GREEN outputs.
- [ ] All 20 seeds included with missing/failed statuses intact.
- [ ] Numeric tables and figures regenerated from sealed manifests.
- [ ] No model weights changed; before/after hashes agree.
- [ ] Final report claims directly mapped to tables.
- [ ] No future-informed oracle result described as ordinary retrieval.
- [ ] Speaker-ready one-paragraph summary and negative-case example.
