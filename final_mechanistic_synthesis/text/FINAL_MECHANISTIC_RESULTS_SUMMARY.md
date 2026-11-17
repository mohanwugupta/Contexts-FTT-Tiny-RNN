# Final Mechanistic Results Summary

## One-paragraph result summary

High-scale recurrent networks learned strongly differentiated familiar-context
representations but transferred poorly to the held-out ambiguous context (baseline Mixed loss
2.25). We formalized the behaviorally relevant representation as centered
output logits—the projection of hidden activity that can affect the next-response distribution—and
measured deviation from the endpoint-linear A-to-B path. Across the preregistered correct-path sweep,
lower normalized task-path error predicted lower Mixed loss (cluster-robust path coefficient
1.03, seed-bootstrap 95% CI [0.37, 2.20]) while familiar
separation remained close to baseline. A strictly behavior-blind, seed-specific stable-cell rule found
full endpoint/path/rescue solutions in 4/20 seeds; thus sufficiency was demonstrated for a
subset rather than universally. Direction mattered: exact seed-by-strength wrong-path controls had
Mixed loss 0.33 higher on average than correct-path models (95% CI
[0.09, 0.57]). Together with the earlier acute output-path patch, these results
show that context splitting and ambiguous-context generalization are compatible when the task-relevant
projection follows a compositional path. They do not establish that this path is necessary, universal,
or literally instantiated in human neural representations.

## Full Results subsection

### Baseline computational tension

The frozen High-scale SRNs expressed the intended familiar-context split but generalized poorly to the
held-out Mixed cue. Their mean A/B hidden-state separation was 2.93, whereas mean
Mixed sequence loss was 2.25. The Low-scale comparison transferred much more
successfully (Mixed loss 0.05) but did not provide the same representational
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
0.55, A/B separation ratio 1.00, and
endpoint drift 0.09. Separately, one stable cell was selected within each seed
using only endpoint drift, familiar separation/loss, and minimum path residual. These selected cells averaged
a rescue fraction of 0.67; 4/20 simultaneously crossed every endpoint, path,
rescue, and remaining-gap threshold.

The selection firewall was applied structurally: the table passed to the selector omitted Mixed loss,
midpoint probability, sequence accuracy, execution proxy, rescue, and patch-gap columns. Eligibility used the
frozen endpoint-drift, familiar-loss, and A/B-separation gates; among eligible cells, only normalized task-path
residual determined rank, followed by preregistered deterministic tie-breaks. Thus the held-out outcome was
used to evaluate, not choose, the reported seed-specific interventions.

### Dose-response

Across all correct centered-logit cells, Mixed loss varied continuously with task-path error. The path
coefficient was 1.03 (cluster-robust SE 0.29; seed-bootstrap 95% CI
[0.37, 2.20]), with Pearson r=0.40 and Spearman rho=
0.40. Because larger residual means farther from the target, the positive coefficient
means that movement toward the path accompanied better held-out transfer. Endpoint drift was included as a
descriptive covariate. The regression itself is not treated as causal proof; causal leverage comes from the
experimentally manipulated correct versus wrong targets.

### Wrong-path specificity

For every seed, alpha, and anchor strength, the correct-path model was paired with the orthogonal,
norm-matched wrong-path model trained for the same duration from the same checkpoint. Wrong-minus-correct
Mixed loss averaged 0.33 (median 0.09, seed-bootstrap 95% CI
[0.09, 0.57]); the wrong model was worse in 61.3% of
matched cells. The aggregate wrong-path representative had Mixed loss 2.15.
Thus representational movement alone was insufficient: its direction relative to the formal task path
predicted whether transfer improved.

### Seed heterogeneity and scope

The ordered seed profiles were classified descriptively as a **continuous frontier**
(largest adjacent-gap fraction 0.20). Only
20.0% of seeds met the full selected-cell gate, so the evidence supports
sufficiency in a subset of split SRNs, not universal retrofittability. The result likewise does not establish
necessity: other geometries could support transfer. Finally, the human data motivate the computational
problem but do not show that human brains encode these exact latent vectors.


## Discussion paragraph

These analyses refine the usual contrast between context specificity and generalization. The two are not
opposite ends of a single representational scale. A recurrent network can preserve a large A/B separation
and stable familiar behavior while improving transfer, provided that the component of its state visible to
the output map changes compositionally between the familiar solutions. Activity in the output-null subspace
may remain nonlinear or highly context specific without directly determining behavior. This distinction
explains why reducing generic curvature was insufficient, whereas an acute output-path patch and learned
movement toward the formally specified task path were behaviorally useful. The wrong-path control further
shows that benefit was not a generic consequence of perturbing the representation: an orthogonal movement
of matched magnitude did not yield equivalent transfer. At the same time, the seed analysis limits the
claim. Only 4 of 20 High-scale seeds satisfied the complete endpoint, path, rescue, and remaining-gap
criteria under the behavior-blind seed-specific selection rule. The mechanism is therefore a demonstrated
sufficient solution within this SRN family, not a universal or necessary principle. Human ambiguous-context
behavior is consistent with the same computational constraint—that task-relevant action predictions must
combine sensibly—but these simulations do not identify the neural code used by people, nor do they establish
the developmental experience that produces it. The paper-level contribution is consequently mechanistic
and computational: strong context splitting can coexist with transfer when the task-relevant projection,
rather than the entire hidden state, preserves a compositional route between familiar solutions.
The sequence-level plots also verify that this summary is not an artifact of isolated transitions.
