# Researcher handoff

We localized effects of state history and context on future performance, but did not identify a unique disruption whose correction explains the natural Mixed deficit.

## Actual algorithm and fixed source

Source `src/context_sequence/evaluation/rollout.py:Model.step`: float32 seven-input affine map (four token channels+three raw cue channels), recurrent8×8 map, tanh, four-logit linear next-token head. h_t=tanh(W_h h_(t−1)+W_x x_t+W_c c_t+b), z_t=W_o h_t+b_o. Original source biases are folded float32 as in all predecessors. The separate execution policy is neither substituted for this head nor updated. Teacher forcing uses exactly seven transitions, starts at zero, and never generates its own input tokens.

`experiments/context_failure_selective_repair/audit.py`: verifies source identities/cue definitions and replays76 fidelity cases. Original20 canonical High_monotonic seeds,170 updates, scale2; actual checkpoint paths and SHA256 in r0_checkpoints. `factorial.py`: consumes1440 saved P4 rows/360cells; no matched branch rerun. Both A/B-history references get one Mixed update and then the same Mixed future; zero best-reference selection. Current k and future k+1…7 scored separately via unchanged imported future_summary.

`geometry.py`: full-right SVD O=(I−11ᵀ/4)W_o; V rank3, N nullity5. If δ=Nα, Oδ=0, so softmax(z+W_oδ)=softmax(z), allfour classes. This does not force O diag(1−h_next²)W_hδ=0. Familiar-only G_A/G_B derivatives describe allfuture centered logits, not CE or labels. Stack, SVD G_fam N; fixed U2,W3. Identifiability uses absolute1e-8,relative1e-4σ1 and σ2−σ3>same bound. Exact double chain/autodiff, source finite differences eps1e-3/1e-4, one-sided at boundary. No fitting.

`repair.py`: d=r−h_M; natural h_M, full observed r, h+aVVᵀd, h+aNNᵀd, h+aUUᵀd, and h+a q_W/q_random. Controls have identical exact displacement norm, hash RNG identity and common a=.95×largest jointly feasible scale≤1. Interior1e-6, tolerance1e-10, zero norm1e-12. Infeasibility stays missing; no clipping. Current nullspace identity checks precede behavior; source float32 head has the imported P2 roundoff bound. Reverse r−aUUᵀd and matched controls use the fixed forward a and originally viable references, never a chosen harmful sign. Familiar controls use both signs and own A/B future cues.

`reporting.py`: independently recomputes summaries, paired within-reference differences, seed means and bootstrap10,000 draws/20260925. All20 seed rows retained; absent references are NA, never0. Complete20 support and favorable95% upper limits for U−W/U−random plus zero current violations are required for a selective claim. Reverse necessity is separate. Conditional one-seed degenerate bootstrap limits do not estimate population uncertainty.

## Results and limitations to teach back

Natural mean CE A=0.917284, B=1.455810, Mixed=2.248301. k3 inherited familiar histories give mean future difference+.262313[-.222091,.816107], not a clear improvement. Current Mixed-vs-own cue effects on future CE are+.749261 in A history and+.645418 in B history under the same subsequent Mixed cues. Seven of40 fixed references across5/20 seeds meet the inherited criterion; current correctness often deteriorates later.

U geometry identifiable40/40 does not imply a usable intervention. Strict joint geometry admits only1/40 primary cells; its U future CE change is+3.83e-6. Secondary k5 admits6/40 cells. No reverse matched cell is feasible. All complete current-softmax audits pass, but float32 roundoff and very small amplitudes limit interpretation. Local derivative FD discrepancies at saturated source states are recorded, not hidden. There is no selective-mechanism or necessary-computation result, and no new training is justified.

## Ninety-second teach-back

We asked whether Mixed breaks a specific recurring computation, rather than whether another full hidden state can work. First we reused the complete historical3×3 experiment: A, B or Mixed history followed by one A, B or Mixed cue, then identical Mixed inputs. Familiar history alone did not reliably repair the future; changing one current cue could affect later predictions. Next we divided the natural reference gap into directions visible to the current output and directions that leave allfour current probabilities unchanged. Familiar continuations, without correct-label losses, fixed two silent directions with large future-logit sensitivity. We tried that natural-gap component against equally large silent controls. Most natural states were too close to tanh boundaries for all the fixed control rays to stay in the required interior. Only one primary reference cell was admissible, with a tiny adverse effect; none of the reverse comparisons was admissible. So we localized state-and-cue effects but did not identify the selective mechanism. The right follow-up is to understand this intervention geometry under a separately fixed protocol, not train or search for a favorable rescue.

## Two adverse cases

1. Seed11/B-history reference at k3: future CE10.118783 versus natural Mixed2.777713. A familiar history can severely worsen the common Mixed future; familiar inheritance is not guaranteed competence. This adverse example is illustrative, never used for primary selection.
2. Seed12/A-history reference at k3: the only feasible matched U cell; a=.0001138595, exact displacement norm≈8.04e-5. U raises future CE≈3.83e-6 and is worse than both controls. The full reference itself has CE2.257857 versus natural Mixed.710387 and is not viable, while the tiny U state inherits native viability. A viability label alone therefore does not establish a successful repair. No population inference from one estimable seed.

## Validation

34 new tests and57 independent read-only predecessor tests passed. Phase1K: collection/import remained blocked in bounded retries; exact status {"status": "blocked_import_timeout45", "scientific_updates": 0, "source_unchanged": true}. No Phase1K pass is claimed. Source and predecessor data were not modified.

## Reproduce and check

Final authoritative output is `results/context_failure_selective_repair/v1_1`. The initial v1 R2 is explicitly superseded; no predecessor bytes were overwritten. The initial v1 R2 run was superseded by v1_1 after a precision error in a synthetic test and a shell failure to stop on RED. The original failing logs and test snapshot remain in v1; only the test’s weight averaging was corrected from float32 to float64, with tolerance unchanged. The scientific algorithms and thresholds did not change. R0/R1 outputs were copied byte-for-byte and hash-verified; R2 was rerun only after GREEN. The accepted scratchpad uses PRD sections because the named template was unavailable; the user explicitly accepted this.

Read the five stage reports, append-only scratchpad, claim ledger and source tables. Run `python -m experiments.context_failure_selective_repair.verify --output results/context_failure_selective_repair/v1_1` with PYTHONPATH=.:src; regenerate with the command in the final report. New stage execution (to a distinct initialized directory) is `python -m experiments.context_failure_selective_repair.core --stage r0 ...`, then r1,r2,r3,r4,r5 in order. Never rerun already matched factorial branches. No R4 weights were updated. Preserve the existing failed/infeasible evidence if a separately preregistered protocol is proposed.
