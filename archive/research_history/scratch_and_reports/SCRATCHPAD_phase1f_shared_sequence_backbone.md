# Phase 1F implementation scratchpad

## Frozen decisions

- Only the 20 canonical High (`scale=2.0`) and 20 canonical Low (`scale=.02`) eight-unit checkpoints
  validated by the immutable training-timeline audit are admitted.
- Centered four-logit task space is primary; eight-dimensional hidden space is secondary.
- Raw loss is mean transition cross-entropy, matching the canonical repository definition.
- Mixed, Novel, and NoContextProbe remain separate throughout.
- All six incorrect transition positions are evaluated and averaged for wrong-position controls.
- Orthogonal controls are deterministic: project coordinate basis vectors away from the correct displacement,
  choose the largest surviving vector, and norm-match it.
- Free rollout enters at transition 1, then applies the frozen recurrent dynamics with no further patch.
- Stage 2 is disabled and no model parameters are optimized.

## Decision rule fixed before the full run

- A: Novel/NoContext task shared and axis rescue are positive; correct position and shared direction beat
  controls; hidden transition and free-rollout shared rescue are positive.
- B: the complete task-space pattern holds but transition-wise hidden rescue does not.
- C: the complete task-space pattern and transition-wise hidden rescue hold, but free rollout does not.
- D: the complete preregistered task-space rescue/specificity pattern does not hold.

This classification separates descriptive geometry, task-state sufficiency, hidden-state sufficiency,
sequence specificity, and learnability.
