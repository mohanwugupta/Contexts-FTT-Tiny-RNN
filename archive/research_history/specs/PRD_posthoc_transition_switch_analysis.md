# Short PRD: Posthoc Transition × Time-Since-Switch RT Analysis

## Goal

Determine whether the strong context-yoked RT splitting seen during training is expressed primarily **immediately after A/B switches** and then washes out once the network settles into the current context.

This is a **posthoc analysis only**. Do not retrain or modify any model.

Use the frozen 20-seed `stage2_layout` High- and Low-monotonic runs from the ordinal-layout ecology.

## Core hypothesis

For each ordinal transition position \(k\), compare RT as a function of time since an A/B context switch.

Prediction:

\[
RT^{switch}_{c,k} - RT^{settled}_{c,k}
\]

should be larger in the High-scale model than the Low-scale model, and the A/B transition profiles may be more differentiated immediately after switching than at the frozen/settled endpoint.

## Required analyses

For every seed, context, block, and ordinal transition position \(k=1,\ldots,7\), extract:

```text
RT
context
block
sequence-within-block
ordinal transition position
seed
condition
```

Define:

```text
switch = sequence 1 in each new A/B block
early = sequences 1-2
settled = sequences 8-10
```

Also retain sequence-within-block 1–10 so the full recovery trajectory can be plotted.

Compute separately for A and B:

```text
RT_switch(k)
RT_early(k)
RT_settled(k)
switch_cost(k) = RT_switch(k) - RT_settled(k)
```

Then compare:

```text
High monotonic
Low monotonic
```

## Primary figures

### Figure 1 — Transition profiles by retrieval state

For High scale, plot the 7-position RT profile for:

```text
A switch
A settled
B switch
B settled
```

Use seed means + 95% CI.

### Figure 2 — Switch cost by transition position

Plot:

\[
RT_{switch,k}-RT_{settled,k}
\]

for High vs Low scale across the seven positions.

### Figure 3 — Recovery across the block

For each ordinal transition position, plot RT across:

```text
sequence-within-block = 1 ... 10
```

Faceted or summarized cleanly for High vs Low.

### Optional Figure 4 — Frozen probe comparison

If the frozen A/B probe outputs are already available, overlay:

```text
A frozen
B frozen
```

against the settled A/B training profiles.

Do not rerun probes if saved outputs are sufficient.

## Primary statistical summaries

Report:

1. mean High-scale switch cost across positions;
2. mean Low-scale switch cost across positions;
3. paired High–Low difference across seeds;
4. whether A/B profile separation is larger at switch than at settled state.

Define A/B profile separation at state \(s\) as:

\[
D_s =
\sqrt{
\sum_{k=1}^{7}
(RT_{A,k,s}-RT_{B,k,s})^2
}.
\]

Compare:

```text
D_switch
D_settled
D_frozen   # if available
```

at the seed level.

## Interpretation

The key outcome is whether:

```text
representational splitting
-> strong switch/retrieval-specific RT expression
-> weaker settled item-level A/B RT difference
```

If `D_switch > D_settled` reliably in High scale, this explains why the training trajectory is strongly Cross/Split while frozen item-level A/B RT profiles are relatively close.

If not, report that the Figure 2 vs Figure 5 discrepancy is not explained by time-since-switch dynamics.

## Hard constraints

```text
NO retraining
NO hyperparameter changes
NO new RT objective
NO Mixed/Novel tuning
NO model selection
```

If transition-level training outputs are already saved, analyze them directly.

If they are not saved, re-evaluate the frozen checkpoints deterministically with:

```text
0 optimizer steps
0 parameter updates
```

and verify parameter hashes before/after.

## Outputs

```text
posthoc_switch_transition_profiles.csv
posthoc_switch_cost_by_transition.csv
posthoc_AB_profile_separation.csv
figure_posthoc_transition_profiles.png
figure_posthoc_switch_costs.png
figure_posthoc_block_recovery.png
POSTHOC_SWITCH_TRANSITION_REPORT.md
```
