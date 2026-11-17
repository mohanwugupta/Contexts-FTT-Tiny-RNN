# Similarity-weighted retrieval analysis (PRD3)

PRD3 is implemented as a separate package, `retrieval_modeling/`, and a
separate entry point, `scripts/run_retrieval_analysis.py`. It does not modify
or replace the PRD2 fitting pipeline.

The analysis treats the PRD2 A/B state estimates as frozen experts and asks how
task-derived context features assign retrieval responsibility to them. Models
are fit jointly across Combined and Novel participants and evaluated using
individual-sequence likelihoods, grouped participant splits, and bidirectional
zero-shot transfer.

## Context feature contract

The primary geometry is fixed before observing probe RT:

```text
A          [1, 0, 0]
B          [0, 1, 0]
Both       [1, 1, 0]
Novel      [0, 0, 1]
No Context [0, 0, 0]  -> canonical name: Default
```

The populated third-context export calls its novel condition `Space Training`
and `test spaceA`. The adapter maps those labels to `Novel` and preserves No
Context as the distinct target `Default`. It also separates trained target
cohorts from CB1/CB6 test-only controls. See
[`THREE_CONTEXT_DATA_DICTIONARY.md`](THREE_CONTEXT_DATA_DICTIONARY.md) for the
literal label and counterbalance contract.

## Implemented comparisons

- Equal and free condition-specific mixture baselines.
- Continuous similarity-weighted retrieval.
- Similarity plus a dynamically maintained history prior.
- Absolute-evidence modulation of residual variance.
- Categorical stochastic A/B retrieval using a Gaussian-mixture likelihood.
- Default/unknown-state extension and third-state falsification model.
- Participant-level A/B-separation prediction.
- Combined→Novel and Novel→Combined transfer without target-probe fitting.
- One shared logistic, 1D recurrent, and 2D recurrent gate, followed by
  distillation into `logit(qA) = beta_R R + beta_E E + lambda h + b`.
- Multi-regime recovery emphasizing continuous versus categorical retrieval.

## Running the full analysis

```bash
cd "/Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/RI-EVC/Contexts-FTT-Tiny-RNN"

MPLCONFIGDIR=/private/tmp/matplotlib-cache \
XDG_CACHE_HOME=/private/tmp/xdg-cache \
caffeinate -i python -u scripts/run_retrieval_analysis.py \
  --combined data/mixedBreak_clean.csv \
  --novel data/threeContextsBreak_clean.csv \
  --output outputs/retrieval_analysis
```

`--allow-partial` is available for development, but a partial run explicitly
marks the joint claim and cross-dataset transfer unavailable.

The run writes two additional audit artifacts,
`retrieval_label_dictionary.csv` and `retrieval_cohort_summary.csv`, alongside
the PRD3-required outputs. The current preprocessing contract yields 218
mixed-file participants and 92 three-context participants after conservative
exclusion of reused participant IDs containing two full sessions.

## Interpretation safeguards

- Context features and A/B prototypes never use probe RT.
- A/B trajectories and participant intercepts use A/B observations only.
- Shared retrieval parameters are not fit independently by dataset.
- Target probe behavior never enters cross-dataset transfer fitting.
- Continuous-versus-categorical claims use individual-sequence likelihoods.
- The original task-trained SRN is left unchanged as a negative control.
