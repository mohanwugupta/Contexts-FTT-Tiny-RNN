# Analysis overview

## Scientific order

### 1. SRN mechanism

`ftt_srn_execution_v2` is the registered base experiment. It fixes the sequence
to `(0, 1, 2, 0, 3, 1, 2, 3)`, uses an 8-unit tanh Elman SRN, and trains on 17
alternating A10/B10 blocks. The registered conditions are High scale (`.5`)
with execution control, Low scale (`.02`) with execution control, and High scale
with sequence prediction only. Mixed and Novel are frozen probes and cause zero
updates. The Novel input column stays exactly zero during training.

`ftt_srn_causal_representation` loads the saved checkpoints without training and
tests full A/B directions, subspace steering, ablation, Mixed/Novel patching,
random orthogonal controls, sequence controls, and off-manifold diagnostics. It
is retained because the strict necessity result is an important negative bound.

`ftt_srn_context_sequence_interaction` is the primary current-state causal
analysis. It decomposes matched natural A/B states into shared and
sequence-position-specific components, compares matched against every
mismatched token component, and checks random/sequence controls and sequence
preservation. Natural A/B execution models are fit leave-one-seed-out; Mixed and
Novel are generalization targets only.

### 2. Human behavior

`human_configured_state` inherits the validated retrieval exclusions, preserves
counterbalance and original sequence repetition, excludes the first sequence,
and maps both task encodings onto response transitions `1→2` through `7→8`.
Pure A/B profiles are frozen before any probe model is fit. The key participant
profile is

```text
delta_i(k) = logRT_A,i,k - logRT_B,i,k.
```

Mixed and Novel are modeled separately with participant-held-out M0 scalar
context, M1 scalar mixture, M2 token-specific, M3 low-rank, and M4 frozen-SRN
accounts. Recovery, Context × Position, continuous/categorical retrieval,
temporal probes, and Mixed/Novel comparisons are exported by the same pipeline.

`human_low_rank_interpretation` performs explicit rank selection over 1, 2, 3,
and full rank 7, participant bootstrap/fold alignment, transition-feature tests,
Mixed/Novel projection, independent cross-dataset replication, semantic
recovery, and the frozen human–SRN abstraction comparison. Rank 2 is the frozen
selected solution. One component receives limited late-learning support; the
beyond-difficulty and human–SRN abstraction-match gates fail.

## Canonical versus compatibility code

The five versioned packages above are the scientific implementation used by the
paper runners. `retrieval_modeling.data` and `probe_modeling.data` remain in the
repository only because the final human loader deliberately reuses their
validated exclusion and normalization contract. Their broader exploratory model
ladders are not part of the paper pipeline. Moving or rewriting that dependency
would risk changing participant exclusions and was therefore left untouched.

Historical GRU, partial-observability, Bayesian/gate, and early task-SRN work is
under `archive/exploratory/`. Superseded and verification-only results are under
`archive/old_outputs/`. They preserve the path by which the final hypotheses
were reached but are never called by `scripts/run_all_final.py`.

## Frozen inferential boundaries

- Low-scale networks do not show the High-scale representational split.
- Original task-trained SRNs did not naturally provide the final human averaging
  account; that exploration is archived, not erased.
- Strict sequence-preserving necessity and the v2.3 interaction advantage did
  not satisfy every registered criterion.
- The human rank-2 subspace replicates across datasets, but semantic
  interpretation remains limited.
- Exact human and frozen-SRN transition geometry does not match. No human result
  tunes, selects, or updates an SRN.

