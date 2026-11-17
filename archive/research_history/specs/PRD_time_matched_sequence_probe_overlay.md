# PRD: Sequence-Level Probes on the A/B Learning Curve

## Goal

Plot sequence-level Mixed and Novel probe RT at the points in learning where those probes are evaluated, alongside the surrounding A/B training sequences.

The primary question is:

> At the same stage of learning, does probe execution fall between execution in A and B?

This handoff takes priority over adding a retrieval mechanism or extending item-level model comparisons. Complete the temporal comparison before making another architectural recommendation.

---

## 1. Reason for the correction

Human probe RTs cannot be interpreted against endpoint profiles averaged over earlier learning. That compares different skill levels and can produce an apparent probe benefit or distort a mixture estimate.

The SRN audit must establish whether the same issue exists there. The current report describes its SRN inputs as frozen, zero-update probes, and earlier notes describe frozen A/B endpoints. Those endpoints may already be contemporaneous. Do not assume the reported SRN discrepancy disappears when timing is corrected.

The sequence-level question remains worth testing even if endpoint timing was valid: intermediate execution averaged across a sequence and intermediate RT at every ordinal transition are different criteria.

---

## 2. Preserve the reference experiment

Use the implementation that generated the supplied two-panel figure, “Raw context-yoked RT learning.” Resolve its exact input file, configuration, run identifiers, and aggregation code.

The supplied project record describes an 8-unit tanh Elman SRN, High scale 2.0, Low scale 0.02, 170 external training sequences in alternating A10/B10 blocks, and the existing monotonic execution readout. Verify these against the generating run rather than reconstructing parameters from the PNG.

Preserve the architecture, sequence, context encodings, training curriculum, optimizer, readout, seeds, recurrent-state reset policy, and normal learning updates.

Do not fit a new human target or change the environment to improve probe placement.

---

## 3. Stage 0 — recover the measurement timeline

Locate:

```text
sequence-level A/B training RTs for the reference figure
the reference training configuration and seed list
saved checkpoints and their training-sequence indices
existing Mixed and Novel probe outputs
source checkpoints for every A/B and probe evaluation
the task's existing probe schedule, if implemented
```

Known starting points from the prior diagnostics configuration:

```text
outputs/ordinal_layout_ecology_final/stage3_mixed_probe.csv
outputs/ordinal_layout_ecology_final/stage5_novel_probe.csv
```

For every existing probe comparison, establish:

```text
seed and condition
number of completed training sequences
checkpoint identity, including execution-readout parameters
whether A/B are online training observations or frozen evaluations
whether any A/B averaging spans different checkpoints
incoming recurrent-state policy
number of probe repetitions and any settling period
whether weights changed during evaluation
```

Also document whether online training RT is recorded before or after the corresponding learning update. A checkpoint comparison must use a clearly specified evaluation time.

Output a short audit stating whether the existing SRN result had a timing mismatch, already used a common checkpoint, or cannot be verified.

---

## 4. Primary outcome: execution within a sequence

Use the same sequence-level RT aggregation as the reference learning figure for both its training lines and probe markers.

If that outcome is the mean across the seven scored within-sequence transitions, use:

\[
R_{s,c}(t,r)=\frac{1}{7}\sum_{k=1}^{7}RT_{s,c,k}(t,r),
\]

where s is seed, c is context, t is completed training sequences, and r is evaluation repetition.

Keep initiation RT and the wraparound transition excluded if the reference figure excludes them. If the reference figure uses another aggregation, retain it consistently and document the difference from this formula.

Average transitions within a sequence first. Then summarize seeds at each measurement time. Never average A/B over the entire acquisition period to construct probe endpoints.

Keep individual probe repetitions visible. Use the first evaluated sequence as the immediate probe outcome; show later frozen repetitions separately if they already belong to the evaluation protocol. Do not selectively discard early probe sequences to improve the result.

---

## 5. Stage 1 — plot the existing observations in temporal order

Reproduce the supplied learning figure with High and Low panels, original A/B colors, and block shading.

Add Mixed probe markers in green and Novel markers in magenta at their actual evaluation positions. Produce separate Mixed and Novel versions of the figure to keep the comparisons readable.

The x-axis must distinguish the count of training sequences from evaluation events. A frozen probe after training sequence t can be shown in a clearly marked interval between t and t+1. It does not count as an additional learning update.

Requirements:

- Preserve the original A/B training curves.
- Add the actual sequence-mean probe RT without centering or offset correction.
- Show nearby observed A/B sequences at full resolution in a zoom panel.
- Show uncertainty across seeds and a supplementary panel for each seed.
- Keep observed training values distinguishable from frozen evaluation markers.
- If probes exist only at the final checkpoint, plot them only after the final training sequence. Never repeat a final-checkpoint probe at earlier boundaries.
- If actual interleaved evaluation changed the recurrent trajectory, retain that chronology and identify it explicitly.

Adjacent training A/B values provide the requested visual comparison, but they are measured at different times and may include different switch costs. Stage 2 supplies a stricter same-time reference.

---

## 6. Stage 2 — evaluate A, B, and probes from the same snapshot

At each eligible evaluation time, clone the same model snapshot and incoming recurrent state into independent branches:

```text
A branch
B branch
Mixed branch
Novel branch, when this probe is part of the existing protocol
```

Within each seed and time, these branches must have identical learned parameters, readout parameters, sequence inputs, reset policy, and evaluation duration. Only the context manipulation changes.

If the reference implementation resets recurrent state between sequences, use that reset in every branch. If it carries state forward, start every branch from the same cloned pre-probe state. Preserve A-preceded and B-preceded evaluation events separately.

Evaluation branches make no optimizer steps. Allow recurrent state to evolve normally within a branch, then discard that branch's state. Probe evaluation must not change subsequent acquisition through weights, recurrent state, optimizer buffers, random-number state, or mutable model buffers.

Do not run A, then B, then Mixed successively through one carried hidden state and call their endpoints matched.

### Evaluation-time policy

1. Reuse actual interleaved probe events or the recorded task-matched schedule when available.
2. If the baseline has no interleaved schedule, use every existing A/B block boundary after exposure to both contexts, plus the final checkpoint, as a fixed diagnostic schedule.
3. Label newly evaluated branches “checkpoint diagnostic probes.” They were not experienced during the original acquisition trajectory.

Use saved checkpoints first. If intermediate checkpoints are absent, replay the exact frozen training specification to collect them. This replay may execute the original learning updates; evaluation branches remain frozen. Do not redesign or tune the model. Confirm replay agreement with the existing training curve and final checkpoint within documented numerical tolerance before using the new probes.

If the run cannot be reproduced or the necessary state cannot be recovered, report the missing evidence and limit the figure to verifiable existing observations.

---

## 7. Required visual comparison

For each time and seed, plot three or four same-snapshot sequence means:

\[
R_{s,A}(t),\quad R_{s,B}(t),\quad R_{s,M}(t),\quad R_{s,N}(t).
\]

On the full learning curve, overlay distinct frozen A/B reference markers alongside the probe marker at the evaluation event. Connect only the paired frozen A/B endpoints with a thin vertical segment showing their range. Use a slight horizontal dodge solely for visibility and explain it in the caption.

Provide a close-up around representative probe events chosen by schedule (first eligible, middle, final), not by whether probes appear intermediate.

The reader should be able to see both:

```text
where the probe occurs among actual training sequences
where it falls relative to A and B evaluated from that same snapshot
```

Keep uncertainty bands separate from the interval between A and B. The endpoint interval is not a confidence interval.

---

## 8. Minimal quantitative summary

For each seed, evaluation time, and probe, retain the raw paired means and classify the probe as:

```text
between the same-snapshot A/B means, including endpoint ties
slower than both
faster than both
endpoints too close for a meaningful interpolation classification
```

Use a fixed, documented endpoint-gap tolerance established before inspecting probe results. Export the continuous A/B gap so conclusions do not depend entirely on the threshold. Low-scale near-lumping must not be treated as evidence of failed interpolation merely because its endpoint interval is tiny.

An optional sequence-level coordinate is:

\[
q_s(t)=\frac{R_{s,P}(t)-R_{s,A}(t)}{R_{s,B}(t)-R_{s,A}(t)}.
\]

This coordinate is descriptive and undefined for practically indistinguishable endpoints. It is not a posterior responsibility estimate. Keep original A/B identities; no transition-level relabeling is needed.

Report counts and proportions by checkpoint and incoming context, with seed-level bootstrap intervals where useful. Retain all times from each resampled seed. Distinguish a grand-mean probe lying between grand-mean endpoints from the fraction of individual seeds showing that pattern.

---

## 9. Interpretation

| Result | Supported conclusion |
| --- | --- |
| Probe sequence RT lies between same-snapshot A/B means within seeds | Intermediate sequence-level execution is present under the tested timing and evaluation policy. Item-level mixture failure does not negate that result. |
| Probe is slower than both at matched checkpoints | An excessive sequence-level probe RT remains. Averaging earlier training cannot explain this particular comparison. |
| Probe falls between nearby training observations but outside frozen A/B endpoints | The visual training comparison depends on learning stage or retrieval history; report both comparisons. |
| Probe changes position with checkpoint or incoming context | Probe expression is state-dependent; summarize where and after which context this occurs. |
| Only the grand mean is intermediate | Population averaging is insufficient to establish within-network intermediate execution. |
| Existing endpoints were already matched | Correct the interpretation of the old result accordingly; timing was not its confound. |

Sequence-level intermediacy is a behavioral result. It does not alone establish item-wise interpolation, continuous latent recombination, or a particular gating mechanism.

---

## 10. Verification and deliverables

Meaningful checks:

1. Training and probe aggregation use exactly the same scored transitions.
2. Every frozen A/B/probe comparison shares a checkpoint and incoming-state policy.
3. Evaluation leaves the continuation of acquisition unchanged.
4. Final-checkpoint results are never assigned earlier timestamps.
5. Classification is correct for synthetic between, above, below, and near-equal endpoint examples.
6. Any replay reproduces the reference run before new diagnostic probes are interpreted.

Deliver:

```text
time_matched_probe_config.json
probe_timing_audit.csv
sequence_learning_source.csv
same_snapshot_probe_source.csv
sequence_probe_classification.csv

learning_curve_with_mixed_probes.png / .pdf
learning_curve_with_novel_probes.png / .pdf, when available
probe_event_closeups.png / .pdf
probe_overlays_by_seed.pdf

TIME_MATCHED_SEQUENCE_PROBE_REPORT.md
```

The report should begin with the actual overlay and answer the timing question, the sequence-level intermediacy question, and whether the previous SRN failure survives this comparison. Record source paths and provide one reproducible command. Keep further architectural proposals outside this diagnostic task.
