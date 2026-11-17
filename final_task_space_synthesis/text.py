from __future__ import annotations

from pathlib import Path


def _row(frame, value):
    return frame[frame.condition.eq(value)].iloc[0]


def conclusions(formal, stats, seed_summary):
    dose = stats[stats.analysis.eq("P1 task-path dose response")].iloc[0]
    specificity = stats[stats.analysis.str.startswith("Exact-matched")].iloc[0]
    feasible = float(seed_summary.passes_full_feasible_gate.mean())
    return {
        "claim_A_behavioral_usefulness": bool(dose.estimate > 0 and dose.CI_low > 0),
        "claim_B_subset_sufficiency": bool(feasible > 0),
        "claim_C_directional_specificity": bool(specificity.estimate > 0 and specificity.CI_low > 0),
        "feasible_seed_fraction": feasible,
    }


def write_paper_text(output, formal, stats, seed_summary, heterogeneity):
    directory = Path(output) / "text"; directory.mkdir(parents=True, exist_ok=True)
    high = _row(formal, "Baseline High"); low = _row(formal, "Baseline Low")
    correct = _row(formal, "Best stable correct-path aggregate")
    wrong = _row(formal, "Matched wrong-path aggregate")
    dose = stats[stats.analysis.eq("P1 task-path dose response")].iloc[0]
    spec = stats[stats.analysis.str.startswith("Exact-matched")].iloc[0]
    claims = conclusions(formal, stats, seed_summary)
    feasible_n = int(seed_summary.passes_full_feasible_gate.sum()); n = len(seed_summary)
    selected_rescue = seed_summary.per_seed_rescue_fraction.mean()

    paragraph = f"""High-scale recurrent networks learned strongly differentiated familiar-context
representations but transferred poorly to the held-out ambiguous context (baseline Mixed loss
{high.Mixed_sequence_loss:.2f}). We formalized the behaviorally relevant representation as centered
output logits—the projection of hidden activity that can affect the next-response distribution—and
measured deviation from the endpoint-linear A-to-B path. Across the preregistered correct-path sweep,
lower normalized task-path error predicted lower Mixed loss (cluster-robust path coefficient
{dose.estimate:.2f}, seed-bootstrap 95% CI [{dose.CI_low:.2f}, {dose.CI_high:.2f}]) while familiar
separation remained close to baseline. A strictly behavior-blind, seed-specific stable-cell rule found
full endpoint/path/rescue solutions in {feasible_n}/{n} seeds; thus sufficiency was demonstrated for a
subset rather than universally. Direction mattered: exact seed-by-strength wrong-path controls had
Mixed loss {spec.estimate:.2f} higher on average than correct-path models (95% CI
[{spec.CI_low:.2f}, {spec.CI_high:.2f}]). Together with the earlier acute output-path patch, these results
show that context splitting and ambiguous-context generalization are compatible when the task-relevant
projection follows a compositional path. They do not establish that this path is necessary, universal,
or literally instantiated in human neural representations."""

    results = f"""## Full Results subsection

### Baseline computational tension

The frozen High-scale SRNs expressed the intended familiar-context split but generalized poorly to the
held-out Mixed cue. Their mean A/B hidden-state separation was {high.AB_separation:.2f}, whereas mean
Mixed sequence loss was {high.Mixed_sequence_loss:.2f}. The Low-scale comparison transferred much more
successfully (Mixed loss {low.Mixed_sequence_loss:.2f}) but did not provide the same representational
regime. This established the central computational tension: strong endpoint differentiation was present,
yet interpolation between those endpoints was not behaviorally useful.

### Formal task-space definition

We defined task space as the centered output-logit representation, which removes the softmax-invariant
common-logit component. Compositional interpolation requires only this task-relevant projection to follow
the endpoint-linear A-to-B path; it does not require the full hidden state, including output-null activity,
to be linear. The normalized midpoint residual therefore compares each fine-tuned network's centered-logit
error with its own frozen High baseline. This normalization equals one for every source High seed by
construction and approaches zero as the formal task path is reached.

We retained separate output-row-space and output-null residuals rather than treating whole-hidden-state
linearity as the mechanism. The row-space projector and its complement were audited for idempotence and
orthogonality on a frozen checkpoint. This matters because a hidden trajectory may remain curved in directions
that the readout cannot see. Accordingly, a small task-space error is a statement about behaviorally available
geometry, not about globally linear or low-dimensional recurrent dynamics.

### Acute patching diagnosis

The earlier causal intervention supplied the first sufficiency clue. Replacing only the output-relevant
component of the baseline Mixed state with the endpoint-linear component immediately reduced Mixed loss,
while replacing the output-null component did not. The seed-local baseline patch loss defines the available
rescue denominator used here. All rescue fractions were calculated within seed before aggregation; the Low
control was never assigned a High-baseline denominator.

### Learned path intervention

Phase 1C then trained recurrent parameters toward the formal path while freezing the output map and
anchoring familiar endpoints. The final synthesis corrected an ambiguity in the original labels: the
historical report's strongest seed-cell and the aggregate plotting representative were different objects.
We replaced both for inference with preregistered-variable-only rules. The strict aggregate stable cell was
the cell identified in the formal summary, with normalized task residual
{correct.normalized_task_path_residual:.2f}, A/B separation ratio {correct.AB_separation_ratio:.2f}, and
endpoint drift {correct.endpoint_hidden_drift:.2f}. Separately, one stable cell was selected within each seed
using only endpoint drift, familiar separation/loss, and minimum path residual. These selected cells averaged
a rescue fraction of {selected_rescue:.2f}; {feasible_n}/{n} simultaneously crossed every endpoint, path,
rescue, and remaining-gap threshold.

The selection firewall was applied structurally: the table passed to the selector omitted Mixed loss,
midpoint probability, sequence accuracy, execution proxy, rescue, and patch-gap columns. Eligibility used the
frozen endpoint-drift, familiar-loss, and A/B-separation gates; among eligible cells, only normalized task-path
residual determined rank, followed by preregistered deterministic tie-breaks. Thus the held-out outcome was
used to evaluate, not choose, the reported seed-specific interventions.

### Dose-response

Across all correct centered-logit cells, Mixed loss varied continuously with task-path error. The path
coefficient was {dose.estimate:.2f} (cluster-robust SE {dose.SE:.2f}; seed-bootstrap 95% CI
[{dose.CI_low:.2f}, {dose.CI_high:.2f}]), with Pearson r={dose.pearson_r:.2f} and Spearman rho=
{dose.spearman_rho:.2f}. Because larger residual means farther from the target, the positive coefficient
means that movement toward the path accompanied better held-out transfer. Endpoint drift was included as a
descriptive covariate. The regression itself is not treated as causal proof; causal leverage comes from the
experimentally manipulated correct versus wrong targets.

### Wrong-path specificity

For every seed, alpha, and anchor strength, the correct-path model was paired with the orthogonal,
norm-matched wrong-path model trained for the same duration from the same checkpoint. Wrong-minus-correct
Mixed loss averaged {spec.estimate:.2f} (median {spec['median']:.2f}, seed-bootstrap 95% CI
[{spec.CI_low:.2f}, {spec.CI_high:.2f}]); the wrong model was worse in {100*spec.fraction_positive:.1f}% of
matched cells. The aggregate wrong-path representative had Mixed loss {wrong.Mixed_sequence_loss:.2f}.
Thus representational movement alone was insufficient: its direction relative to the formal task path
predicted whether transfer improved.

### Seed heterogeneity and scope

The ordered seed profiles were classified descriptively as a **{heterogeneity['classification']}**
(largest adjacent-gap fraction {heterogeneity.get('largest_gap_fraction', float('nan')):.2f}). Only
{100*claims['feasible_seed_fraction']:.1f}% of seeds met the full selected-cell gate, so the evidence supports
sufficiency in a subset of split SRNs, not universal retrofittability. The result likewise does not establish
necessity: other geometries could support transfer. Finally, the human data motivate the computational
problem but do not show that human brains encode these exact latent vectors.
"""

    discussion = f"""## Discussion paragraph

These analyses refine the usual contrast between context specificity and generalization. The two are not
opposite ends of a single representational scale. A recurrent network can preserve a large A/B separation
and stable familiar behavior while improving transfer, provided that the component of its state visible to
the output map changes compositionally between the familiar solutions. Activity in the output-null subspace
may remain nonlinear or highly context specific without directly determining behavior. This distinction
explains why reducing generic curvature was insufficient, whereas an acute output-path patch and learned
movement toward the formally specified task path were behaviorally useful. The wrong-path control further
shows that benefit was not a generic consequence of perturbing the representation: an orthogonal movement
of matched magnitude did not yield equivalent transfer. At the same time, the seed analysis limits the
claim. Only {feasible_n} of {n} High-scale seeds satisfied the complete endpoint, path, rescue, and remaining-gap
criteria under the behavior-blind seed-specific selection rule. The mechanism is therefore a demonstrated
sufficient solution within this SRN family, not a universal or necessary principle. Human ambiguous-context
behavior is consistent with the same computational constraint—that task-relevant action predictions must
combine sensibly—but these simulations do not identify the neural code used by people, nor do they establish
the developmental experience that produces it. The paper-level contribution is consequently mechanistic
and computational: strong context splitting can coexist with transfer when the task-relevant projection,
rather than the entire hidden state, preserves a compositional route between familiar solutions.
The sequence-level plots also verify that this summary is not an artifact of isolated transitions.
"""
    (directory / "FINAL_MECHANISTIC_RESULTS_SUMMARY.md").write_text(
        "# Final Mechanistic Results Summary\n\n## One-paragraph result summary\n\n" + paragraph + "\n\n" + results + "\n\n" + discussion)
    return claims


def write_plain_language(output, formal, stats, seed_summary):
    directory = Path(output) / "text"; high = _row(formal, "Baseline High")
    spec = stats[stats.analysis.str.startswith("Exact-matched")].iloc[0]
    text = f"""# Plain-Language Task-Space Mechanism

## What is the network doing?

The network learns two clearly different internal solutions for the same motor sequence: one for context A
and one for context B. Those distinct solutions are useful because context matters. The problem appears when
the cue is halfway between A and B. A High-scale network does not automatically produce a sensible halfway
prediction; its baseline Mixed loss was {high.Mixed_sequence_loss:.2f}.

## What property fixes it without erasing the A/B split?

Only part of the hidden state can influence the next action. When that task-relevant part travels along a
sensible route between the familiar A and B solutions, the network handles the new Mixed cue better. The rest
of the hidden state can remain strongly split and nonlinear. This is why improved transfer does not require
collapsing A and B into the same representation.

## What does the wrong-path control establish?

We pushed matched networks by comparable amounts in a deliberately wrong direction. Those movements did not
provide the same benefit: wrong-path Mixed loss exceeded correct-path loss by {spec.estimate:.2f} on average.
The result therefore concerns the direction and organization of representational change, not merely its size.

## Why is this relevant to humans, and what are we not claiming?

People also need to retain context-specific skills while responding sensibly when context is ambiguous. The
model supplies a computational explanation for how both demands can coexist. It does not show that humans
literally encode the same vectors, that this is the only possible solution, or that every trained network can
be repaired this way.

## Minimal formal statement

""" + r"""Let centered output logits define task state:

\[
z_k(c)=C\,[W_{out}h_k(c)+b],\qquad C=I-\frac{1}{m}\mathbf 1\mathbf 1^\top.
\]

The compositional target between A and B is:

\[
z_k^*(\lambda)=(1-\lambda)z_k(A)+\lambda z_k(B).
\]

Task-space error is:

\[
E_{task}(\lambda)=\frac{1}{7}\sum_{k=1}^7\|z_k(c_\lambda)-z_k^*(\lambda)\|_2^2.
\]

The empirical result is that reducing this error can improve held-out Mixed behavior while familiar A/B
separation stays approximately fixed.
"""
    (directory / "FINAL_PLAIN_LANGUAGE_MECHANISM.md").write_text(text)


def write_captions(output):
    directory = Path(output) / "text"
    captions = """# Final Figure Captions

**FINAL_FIG01 — Mechanism triangle.** Each point is one seed × Phase 1C hyperparameter cell. Panel A compares correct-path and norm-matched orthogonal wrong-path sweeps; Panels B–C show that lower centered-logit task-path error need not collapse A/B separation or move familiar endpoints. The Mixed midpoint was held out from training and selection.

**FINAL_FIG02 — Correct-versus-wrong specificity.** Pairs match seed, alpha, anchor strength, training duration, and checkpoint rule exactly. The identity line denotes equal Mixed loss. The histogram shows wrong-minus-correct sequence loss; positive values favor the formal path.

**FINAL_FIG03 — Seed-level sufficiency.** One row of aligned values per seed, ordered by the task-path residual of its behavior-blind stable-cell selection. Dashed lines are preregistered gates. A cell was selected from endpoint drift, familiar loss/separation, and path residual only; Mixed outcomes were not accessed.

**FINAL_FIG04 — Causal chain.** Seed-mean empirical summaries connect the frozen High baseline, acute output-only midpoint patch, and learned correct-path intervention. The patch is a counterfactual state replacement, whereas the learned condition changes recurrent parameters with the output head frozen. This schematic does not imply that humans use identical latent vectors.

**FINAL_TS01 — Familiar sequence loss.** Lines are across-seed means and bands are 2.5th–97.5th seed quantiles for complete seven-transition sequences. A/B shading follows the registered schedule. Representatives were chosen without Mixed behavior.

**FINAL_TS02 — Familiar correct-token probability.** Same sequence-level aggregation, bands, schedule, and selection rule as FINAL_TS01.

**FINAL_TS03 — Familiar execution proxy.** Frozen-policy execution/RT proxy for complete familiar sequences; lines and bands summarize seeds, not transitions.

**FINAL_TS04 — Full probe-schedule loss.** Complete familiar schedule followed by held-out Mixed, NoContextProbe, and Novel sequences. The latter two are OOD under the current encoding. Probes are descriptive and did not select cells.

**FINAL_TS05 — Full probe-schedule probability.** Mean correct-token probability per complete sequence, with across-seed mean and 95% seed-quantile band.

**FINAL_TS06 — Full probe-schedule execution proxy.** Frozen-policy proxy at the complete-sequence level; the probe boundary is marked explicitly.

**FINAL_TS07 — Midpoint during fine-tuning.** Held-out Mixed loss, centered-logit path residual, endpoint drift, and remaining output-patch benefit at fixed 0%–100% checkpoints. These evaluations never affected optimization, early stopping, or selection.

**FINAL_TS08 — Dense cue-sweep loss.** Complete-sequence loss from lambda 0 to 1. The dashed midpoint was excluded from every fine-tuning loss.

**FINAL_TS09 — Dense cue-sweep task-path residual.** Centered-logit compositional error for the same readable representatives. Lines are seed means and bands are seed quantiles; all values aggregate seven transitions.
"""
    (directory / "FINAL_FIGURE_CAPTIONS.md").write_text(captions)
