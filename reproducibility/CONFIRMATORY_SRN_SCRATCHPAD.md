# Confirmatory SRN pipeline scratchpad

## Frozen decisions

1. This is a new `confirmatory_srn` package and output root. It does not modify the
   registered v2/v3 analyses, objective search, readout diagnostic, or failed recurrent-
   commitment experiment.
2. The primary execution mechanism is the two-parameter monotonic policy from the
   successful readout diagnostic: `sigmoid(softplus(alpha)*logit(stopgrad(p))+b)`.
   It is trained under the frozen normative candidate A (`c_time=.5`, `c_risk=1`,
   `gamma=2`, `lambda=.2`) with no human or RT target.
3. All primary SRNs contain only the recurrent layer and sequence head. There is no
   hidden-state execution head. High sequence-only has no policy, RT loss, or RT output.
4. High monotonic and High sequence-only use identical initial recurrent/sequence
   weights for each seed, and the pipeline asserts that paired initialization. Confidence
   is detached before the timing policy, so the execution objective sends no direct
   gradient into the SRN. The registered trainer's global gradient clipping is retained;
   because the policy gradient contributes to that shared clipping norm, final weights
   are not required to be bitwise identical across these two conditions.
5. Canonical seeds are the repository's existing `0--19` list.
6. The PRD labels the physical sequence `2-1-4-2-3-1-4-3`, but the registered configs,
   Methods, tests, and all current working outputs freeze `(0,1,2,0,3,1,2,3)`
   (`1-2-3-1-4-2-3-4` if simply one-indexed). The latter is retained because the PRD
   also says not to alter the existing working run. This discrepancy is surfaced in the
   final config/report rather than hidden.
7. The confirmatory Mixed vector is exactly `[.5,.5,0]`, local to this package. It does
   not change the older registered package's historical `1/sqrt(2)` encoding.
8. Responsibility q is never clipped. The predeclared usable-denominator threshold is
   `|A_RT-B_RT| >= .01` normalized RT.
9. Low-rank rank selection uses only A/B RT-difference profiles. Each CV prediction
   leaves out both a seed and transition entry. Probe evaluation then leaves out one
   probe transition at a time; probe rows never select rank.
10. Matched causal patches are compared with orthogonal-random, wrong-token A/B, and
    unrestricted magnitude-matched random controls. Every row reports RT, confidence,
    CE, and token preservation.
11. Strong causal success inherits the repository's established `.90` sequence-
    preservation criterion and applies it to both prediction identity and correct-target
    accuracy. Matched endpoint improvement and direction alignment must exceed every
    control, and matched absolute CE disruption must be smaller than every control.

## Progress

- [x] Parsed the full PRD and mapped frozen repository contracts.
- [x] Implemented architecture, training, probes, diagnostics, state-space, responsibility,
  low-rank, causal controls, standardized bridge, reports, figures, and CLIs.
- [x] T1--T9 tests pass.
- [x] Smoke pipeline and figures pass.
- [x] Full 60-network confirmatory run completes.
- [x] All required output schemas and checkpoints validate.
- [x] Repository regression suite passes (125/125).
- [x] The 112 registered final CSVs remain bitwise identical.

## Completed-run audit

- The full run contains 10,200 sequence rows, 71,400 transition rows, 2,100
  frozen-probe rows, and 60 checkpoints.
- Two independent full runs produced identical bytes for all 24 analysis CSVs
  and exactly equal tensors for all 60 checkpoints. Raw `.pt` archive hashes are
  not used because PyTorch serialization metadata is not byte-stable.
- High vs Low mean A/B hidden separation was `2.929756` vs `0.066354`.
- Mean-trajectory state-space winners were Cross for High and Lump for Low;
  seed winners were High = 5% Lump / 65% Cross / 30% Split and Low = 100% Lump.
- A/B-only rank selection chose full rank for High and rank 2 for Low, but neither
  selected model beat the global-scalar probe baseline; Figure 6 is therefore
  correctly omitted.
- Matched patches outperformed controls in endpoint proximity and direction, but
  High-scale prediction-identity preservation (`0.776786`) and correct-target
  accuracy (`0.414286`) failed the frozen `.90` criterion. The report therefore
  selects interpretation B and does not make the strong causal claim.
