# Corrected Mixed / Novel retrieval scratchpad

## Locked decisions

- Retrieval reads `outputs/human_state_space_adjudication` immutably and verifies the familiar selection sidecar hash before doing any work.
- The PRD's familiar candidate set is strictly Lump/Cross/Split. The historical pre-probe BIC sum selects Split; Cross is retained under the prespecified ΔBIC ≤ 20 sensitivity rule. M3/M4/M5 cannot be selected using probe behavior.
- Combined participants receive participant-specific frozen familiar parameters. Novel participants not present in the corrected Combined fit receive counterbalance-specific templates made by averaging fully decoded familiar predictions at the matched checkpoint. Parameters and decoder coefficients are never averaged separately, because that breaks their fitted covariance and produced invalid trajectories in the first smoke test.
- Familiar A/B sequences advance frozen familiar states. Mixed, Novel, NoContextTraining, and NoContextProbe never update them.
- `No Context Training` is explicitly labeled `NoContextTraining` and excluded from probe analysis. Only `No Context Test` is labeled `NoContextProbe`.
- The first analyzed probe sequence follows the validated retrieval contract: raw block sequence 1 is excluded, so exposure index 1 is raw sequence 2.
- R0 and R1 have the same q and sigma parameter count, rows, anchors, and Gaussian residual scale. R0 uses a mixture likelihood; R1 uses an interpolated mean.
- R0 makes one categorical A/B draw for an entire seven-transition participant/probe profile. It does not independently switch endpoints at each transition. R1 is evaluated on the same complete profiles.
- R3's rank-2 basis is generated only from decoded A/B profiles on familiar A/B rows. Probe-time rows and probe outcomes are both rejected by the basis function.
- Fast/Slow canonicalization compares whole-profile means and swaps the entire profile only.
- Participant bootstraps resample complete participant profiles. Familiar-anchor sensitivity resamples familiar anchor donors without using probe outcomes.
- Extreme frozen anchors are audited and reported, never clipped or repaired using probe outcomes. This matters for Cross, whose weakly identified state/loading combinations extrapolate to extreme values after the fitted familiar window; the primary Split anchors remain within the 5,000-ms audit threshold.
- The generated `probe_familiar_anchors.csv` is hashed immediately after creation and again after all retrieval fitting; the run aborts if it changes.

## Progress

- [x] Read the complete PRD and existing familiar/retrieval contracts.
- [x] Implement immutable input validation, timeline mapping, and probe-time anchor simulation.
- [x] Implement R0–R3, participant CV, geometry, bootstrap, familiar-model and anchor-uncertainty sensitivity.
- [x] Add required tables, ten figures, report, CLI, and separate Rmd wrapper.
- [x] Run tests and real-data smoke. The decoded-prediction template correction brought Novel Split anchors into the expected range; 24 unit/regression tests passed, the complete smoke pipeline produced every required table, both anchor hash checks passed, and all ten figures rendered.
