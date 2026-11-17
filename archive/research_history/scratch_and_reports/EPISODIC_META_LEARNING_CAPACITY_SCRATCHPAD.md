# Episodic Meta-Learning Capacity v2 — Scratchpad

## Implementation status

- Added an isolated `episodic_meta_learning_capacity` package and runner.
- Existing eight-unit and frozen paper pipelines were not changed.
- Implemented the three required conditions at H8/H32/H64/H128.
- Implemented first-order MAML, V1--V5, capacity gates, conditional human
  transfer, temporal probes, statistics, figures, and conditional M1--M3.

## Decisions

1. **Exposure matching.** Every episode contains six support sequences, four
   unseen queries, and one copy query: 11 unique external sequences. Every
   condition makes one persistent Adam update per episode. Meta conditions also
   perform three virtual inner updates; ordinary training sees the same 11
   examples in one conventional batch update.
2. **FOMAML implementation.** Inner gradients use `create_graph=False` and are
   detached before each functional parameter update. The adapted tensors retain
   an identity path to the original parameters, allowing the query gradient to
   reach the meta-parameters without second-order derivatives.
3. **Prediction-only pretraining.** Only the variable-width Elman SRN exists in
   synthetic training. No execution policy is constructed and no speed outcome
   is part of support, query, gate, or selection calculations.
4. **Initialization.** The prior repository rule—independent normal weights
   with configured SD 2—is preserved for every width rather than silently
   introducing fan-in normalization. Realized weight SD, recurrent spectral
   radius, parameter count, and first hidden activation norm are exported.
5. **Novel families.** Six z values are sampled anew and sorted in every
   episode, with minimum spacing .15. Support includes a rank-lower-third and
   rank-upper-third context. Family hashes make novelty auditable.
6. **Paired shuffled control.** All conditions regenerate exactly the same
   seed-indexed source family. P1 changes only a cyclic task-context
   derangement; visible contexts, task marginals, splits, sequences, and update
   counts are retained.
7. **No latent input.** z generates the visible cue, empirical SVG-manifold
   layout, and task distribution, but the learner receives only the standard
   42-dimensional visible feature rows plus the four-dimensional token input.
8. **Validation.** V1--V5 use functional adapted parameters and frozen query
   scoring. Parent hashes are checked before and after every battery.
9. **Capacity selection.** All four widths run in the pilot. Confirmation is
   restricted to passing widths. Human transfer uses the smallest confirmed
   width, one larger confirmed width when available, both matched controls at
   the smallest width, and a same-width P0 baseline.
10. **Human transition.** SRN weights carry forward; optimizer and hidden state
    reset; the monotonic execution readout is created fresh at raw slope 0 and
    intercept 0. Frozen probes occur at 20/50/90/170 sequences.
11. **Mechanisms.** If the smallest selected P2_meta model passes familiar
    gates and improves early Mixed transfer, synthetic-prior retention is
    remeasured at every human checkpoint, input/recurrent/output subsystems are
    separately reinitialized, and hidden geometry is exported.

## Verification log

- AST parsing passed for all new Python modules.
- Tiny H8 episode: nonzero outer gradients for all three conditions.
- Registered engineering smoke: H8 and H64 × all conditions × 100 episodes
  completed. Exact paired hashes/exposure and immutable V1--V5 parents passed.
- Smoke result was not interpreted; both cells failed at least one hard gate
  and the human firewall remained closed.
- Forced integration branch completed five human groups and generated all
  downstream human tables/figures without synthetic readout carryover.
- Dedicated PRD tests: 9 passed.
- Full repository regression suite: 262 passed in 63.71 seconds.

## Commands

```bash
python scripts/run_episodic_meta_learning_capacity.py --smoke \
  --output /tmp/episodic-meta-capacity-smoke
```

```bash
python scripts/run_episodic_meta_learning_capacity.py \
  --config configs/episodic_meta_config.json \
  --capacity-grid configs/meta_capacity_grid.json \
  --generator-config configs/synthetic_family_generator_config.json \
  --output outputs/episodic_meta_learning_capacity_v2
```
