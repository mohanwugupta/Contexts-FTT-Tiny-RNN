# R3 — selective natural-gap repair and fixed reverse controls

{
  "selective_mechanism_gate": false,
  "softmax_violations": 0,
  "new_patch_rollouts": 33,
  "planned_forward_cells": 560,
  "planned_reverse_cells": 320,
  "planned_familiar_cells": 1280,
  "training_updates": 0,
  "optimization_steps": 0,
  "primary": "Mixed k3",
  "all20_retained": true
}

| group | context | sign | k | contrast | mean | ci_low | ci_high | n_valid | n_seeds | resampling_unit | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| forward | Mixed | 1 | 3 | U_minus_W | 5.5671e-06 | 5.5671e-06 | 5.5671e-06 | 1 | 20 | seed | conditional_estimable_seeds |
| forward | Mixed | 1 | 3 | U_minus_natural | 3.8284e-06 | 3.8284e-06 | 3.8284e-06 | 1 | 20 | seed | conditional_estimable_seeds |
| forward | Mixed | 1 | 3 | U_minus_random | 1.0772e-05 | 1.0772e-05 | 1.0772e-05 | 1 | 20 | seed | conditional_estimable_seeds |
| forward | Mixed | 1 | 3 | reference_minus_natural | 0.26231 | -0.22209 | 0.81611 | 20 | 20 | seed | complete_seed_support |
| forward | Mixed | 1 | 3 | silent_minus_natural | 5.7159e-08 | 5.7159e-08 | 5.7159e-08 | 1 | 20 | seed | conditional_estimable_seeds |
| forward | Mixed | 1 | 3 | visible_minus_natural | not established | not established | not established | 0 | 20 | seed | not_estimable |

Directions and amplitudes use only geometry. U=projection of actual Mixed→fixed reference gap into the two familiar-derived sensitive kernel directions. W is the complementary gap rescaled to the same norm (hash fallback only if zero); random is a hash-seeded kernel vector with the same norm. Shared a=.95×largest joint feasible scale≤1; no clipping, outcome-dependent sign, reference choice, rank or scale search. Projection arms use the same a where feasible. If no joint amplitude exists, the arms remain explicit missing outcomes, never zero CE. Exact cached natural/full-reference outcomes are shown as observed references; boundary references are not claimed strict-interior patch placements.

All actual prediction scores use source float32 recurrence/head; exact double kernel identity and float32 head roundoff audits are separate. Every silent intervention checks allfour probabilities before future scoring. Full-sequence CE uses the actual natural pre-k prefix, current score and unassisted future; it is not the primary endpoint. Reverse tests only originally viable references, fixed negative sign and same forward a; unmatched reverse feasibility is explicit. Familiar controls retain both signs and their own A/B future cue.

All20 seed rows, both reference sides, failed feasibility, adverse effects and rank/zero statuses are exported. Continuous intervals condition on estimable seeds when coverage is incomplete; this does not support a full-cohort mechanism claim. Primary selective gate needs both U−W and U−random upper95% seed limits<0, full20 estimable seeds and no current-softmax violation. Reverse evidence is required separately for bidirectionality. No useful full-state bound or raw component norm alone identifies a selective cause.
