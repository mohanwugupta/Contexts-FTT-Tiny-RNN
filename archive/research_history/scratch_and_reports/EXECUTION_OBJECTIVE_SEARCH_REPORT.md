# Execution objective search report

## Outcome

No execution objective passed every five-seed behavioral gate under the frozen
8-unit Elman SRN, High initialization scale 2, learning rate `.005`, and 170
A10/B10 training sequences. Therefore no objective was selected and the
20-seed confirmatory paper pipeline was not started.

This is a scientifically informative null search. Sequence learning was strong
and context structure remained present, but execution grounding was not stable
across random initializations. The acceptance gates were not relaxed to create
a winner.

## Candidate equations and normative requirement

The prespecified A--D candidates use

```text
L_exec(v | u) = c_time*(1-v) + c_risk*u*v^gamma
v*(u) = clip((c_time/(gamma*c_risk*u))^(1/(gamma-1)), 0, 1)
```

where `u` is detached from sequence logits and is one minus correct-token
confidence (A/B), normalized predictive entropy (C), or one minus the top-two
probability margin (D). Candidate A fixes `gamma=2`; B tests `gamma=3,4`; C and
D use the quadratic risk term. The initial grid crosses `c_time={.05,.10}` and
`c_risk={.50,1.0}` at `lambda_exec=.20`.

All A--D families failed. As explicitly permitted only after that result, a
bounded Candidate-E diagnostic retained Candidate A's analytical `v*` but used

```text
L_exec = BCE(v, stopgrad(v*))
```

This avoids the direct cost's extra sigmoid-saturation factor while still using
only the network's own uncertainty. Candidate E swept `c_time/c_risk=.05--.75`;
`.75` was the largest mapping that retained an intermediate central optimum and
passed the fixed analytical saturation gate.

## Calibration and five-seed screening

| Family | Candidates | Calibration pass | Gate A | Gate B | Gate C | Gate D | Gate E | All gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A: quadratic confidence risk | 4 | 4 | 0 | 0 | 0 | 4 | 4 | 0 |
| B: steeper confidence risk | 8 | 8 | 0 | 0 | 0 | 8 | 8 | 0 |
| C: predictive entropy | 4 | 1 | 0 | 0 | 0 | 1 | 1 | 0 |
| D: probability margin | 4 | 4 | 0 | 0 | 0 | 4 | 4 | 0 |
| E: risk-optimal policy BCE | 10 | 10 | 5 | 0 | 2 | 10 | 10 | 0 |

For Candidate E, ratios `.50`, `.60`, `.65`, `.70`, and `.75` passed the
learning-direction gate; `.70` and `.75` also passed objective tracking. Every
ratio failed Gate B because the confidence--RT direction was correct in only
3/5 seeds. Two initializations retained arbitrary context-specific speed
offsets despite learning the sequence.

Candidate-level values and every gate flag are in:

- `outputs/execution_objective_search_screen/objective_candidate_summary.csv`
- `outputs/execution_objective_search_post_failure/objective_candidate_summary.csv`

The corresponding directories also contain the full calibration curves,
per-sequence results, token-level audits, configuration JSON, and seven figures
in both PNG and PDF formats.

## Exact decision

- Selected objective: **none**
- Selected hyperparameters: **none**
- Human target RT rows used: **0**
- Human Mixed/Novel probe rows used: **0**
- Human transition profiles used: **0**
- Trial progress supplied as input: **no**
- External competence state supplied: **no**
- Confirmatory 20-seed rerun started: **no**

The next scientifically defensible step would require changing a frozen
assumption (for example, execution-head initialization or available learning
exposure), so it is outside this objective-only PRD.
