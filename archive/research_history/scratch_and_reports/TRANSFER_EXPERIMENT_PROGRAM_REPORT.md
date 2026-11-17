# Human/SRN Transfer Experiment Program Report

Analysis version `transfer-experiment-program-1.0.0`; configuration `fb990ca2d96eb14714a86f805b069c7adea221b8f1a0fa548ba49914f4ec88fe`. This is the preregistered 20-seed execution.

## Outcome

All five experiment implementations plus H0 are runnable. H0 uses existing empirical keypress data. E1, E2, E3, and E5 use real frozen SRN code and prespecified seeds. E4 and the human components of E2/E3 are engineering-complete at the code/static-test level but remain **awaiting human data**; interactive browser screenshot QA is separately marked pending, and synthetic power rows are never human evidence.

The reference result remains partial transfer, not no transfer. Every computational interpretation must jointly inspect absolute learning, relative probe deficit, and A/B differentiation.

## Evidence status

| work_package   | question                                                  | implementation_status               | evidence_status                                | prerequisites                           | outputs                                                                                                                                           | failed_checks                                                                                                      | scientific_interpretation_limit                                                                                                                 |
|:---------------|:----------------------------------------------------------|:------------------------------------|:-----------------------------------------------|:----------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------|
| Stage 0        | Does the frozen baseline reproduce?                       | validated                           | completed                                      | reference outputs and model source      | /Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/RI-EVC/Contexts-FTT-Tiny-RNN/outputs/transfer_experiment_program/audit                | nan                                                                                                                | Reference reproduction only; not a mechanism test.                                                                                              |
| H0             | Are human/network probes comparable in time and exposure? | validated                           | completed                                      | existing event-level Mixed/Novel files  | /Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/RI-EVC/Contexts-FTT-Tiny-RNN/outputs/transfer_experiment_program/h0_human_audit       | elapsed/rest timestamps unavailable in legacy data                                                                 | Local A/B values are model-estimated counterfactuals, not same-snapshot observations.                                                           |
| E1             | Does the deficit persist at matched acquisition?          | validated                           | completed                                      | additional-seed A/B training            | /Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/RI-EVC/Contexts-FTT-Tiny-RNN/outputs/transfer_experiment_program/e1_acquisition       | nan                                                                                                                | Duration does not isolate all initialization effects; unsupported strict matches remain inconclusive.                                           |
| E2             | Do recurrent history or probe updates alter transfer?     | validated                           | completed_computational_awaiting_human_data    | cached additional-seed checkpoints      | /Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/RI-EVC/Contexts-FTT-Tiny-RNN/outputs/transfer_experiment_program/e2_probe_experience  | No new human task observations supplied.                                                                           | Carry/update effects do not establish Bayesian inference or a matching human optimizer.                                                         |
| E3             | Does cue, layout, or their conjunction drive the deficit? | validated                           | completed_computational_awaiting_human_data    | cached original-cohort checkpoints      | /Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/RI-EVC/Contexts-FTT-Tiny-RNN/outputs/transfer_experiment_program/e3_cue_decomposition | Human graded layout intentionally uninstantiated.                                                                  | Factorial sensitivity localizes inputs but is not a unique configural mechanism.                                                                |
| E4             | Does action visibility alter human probe disadvantage?    | validated_static_browser_QA_pending | awaiting_human_data                            | new randomized participant observations | /Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/RI-EVC/Contexts-FTT-Tiny-RNN/outputs/transfer_experiment_program/e4_human_task        | No new browser-task observations supplied. In-app browser screenshot QA unavailable in this execution environment. | Task/power/model-information evidence is not a human outcome; no new execution architecture was tested.                                         |
| E5             | Does cross-layout prior structure improve transfer?       | validated                           | completed_with_partial_no_common_support       | additional prespecified seeds           | /Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/RI-EVC/Contexts-FTT-Tiny-RNN/outputs/transfer_experiment_program/e5_prior_experience  | 26 arm/seed comparisons lacked A/B support through cap 680.                                                        | A dose-specific structural contrast cannot recreate or identify lifetime human experience; unsupported matched comparisons remain inconclusive. |
| Synthesis      | Which tested differences explain partial transfer?        | validated                           | completed_computational_human_tasks_await_data | completed independent stages            | /Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/RI-EVC/Contexts-FTT-Tiny-RNN/outputs/transfer_experiment_program                      | nan                                                                                                                | No single manipulation is a unique mechanistic explanation; new human E2-E4 evidence awaits collection.                                         |

## H0 — human measurement audit

**Question/intervention/controls.** Chronology and first exposure were reconstructed from the complete source stream before RT/correctness exclusions. Local A/B references were fit only to familiar events, with ridge strength selected by held-out whole A/B blocks; probe outcomes were unavailable to fitting and selection.

**Sample/status.** The audit covered 520,717 raw events (318 dataset-namespaced target participants before complete-case filtering). Existing empirical evidence is complete, but legacy elapsed/rest time is unrecoverable and local counterfactual A/B estimates are not simultaneous observations.

| dataset   | probe_context   | contrast                                         |   participant_count |   estimate_log_RT |   ci_lower |   ci_upper |   bootstrap_draws | bootstrap_unit    | estimability_status                            |
|:----------|:----------------|:-------------------------------------------------|--------------------:|------------------:|-----------:|-----------:|------------------:|:------------------|:-----------------------------------------------|
| Combined  | Mixed           | first_ever_minus_later_first_block_local_deficit |                   0 |               nan |        nan |        nan |              2000 | whole_participant | not_estimable_under_existing_cleaning_contract |
| Novel     | Novel           | first_ever_minus_later_first_block_local_deficit |                   0 |               nan |        nan |        nan |              2000 | whole_participant | not_estimable_under_existing_cleaning_contract |

**Result/decision.** The legacy cleaning contract removes original sequence 1 from every block. Consequently the registered first-versus-later inferential contrast is not estimable under that contract; first encounters remain in the audit/local-reference tables as explicitly ineligible descriptive rows and are not renamed. The human first-encounter mismatch therefore remains temporally unresolved; collect the new tasks without changing this exclusion post hoc.

## E1 — acquisition

**Question/intervention/controls.** The additional prespecified cohort continued the unmodified alternating A/B schedule to 340 and 680, with frozen zero-state A/B/Mixed/Novel probes every ten sequences. Matching used labeled A/B NLL, labeled A/B accuracy, and incoming phase only.

**Result.** Strict `.05`-nat matches selected: **33**. Unmatched anchors remain in the candidate table. Among valid matches, High-minus-Low relative deficits were:

|                |   ('High_minus_Low_NLL_deficit', 'mean') |   ('High_minus_Low_NLL_deficit', 'count') |   ('High_minus_Low_RT_deficit', 'mean') |   ('High_minus_Low_RT_deficit', 'count') |
|:---------------|-----------------------------------------:|------------------------------------------:|----------------------------------------:|-----------------------------------------:|
| (170, 'Mixed') |                                  1.22223 |                                         3 |                                0.333062 |                                        3 |
| (170, 'Novel') |                                  2.14618 |                                         3 |                                0.369576 |                                        3 |
| (340, 'Mixed') |                                  1.79702 |                                        12 |                                0.359046 |                                       12 |
| (340, 'Novel') |                                  1.74908 |                                        12 |                                0.394191 |                                       12 |
| (680, 'Mixed') |                                  1.56291 |                                        18 |                                0.335648 |                                       18 |
| (680, 'Novel') |                                  1.98964 |                                        18 |                                0.431677 |                                       18 |

Whole-seed trajectory uncertainty (2000 draws):

| condition      |   parent_checkpoint | probe_context   |        mean |   CI95_lower |   CI95_upper | bootstrap_unit   |   bootstrap_draws | experiment   | metric                 |
|:---------------|--------------------:|:----------------|------------:|-------------:|-------------:|:-----------------|------------------:|:-------------|:-----------------------|
| High_monotonic |                 170 | Mixed           | 1.36711     |  0.74306     |  2.25901     | seed             |              2000 | E1           | NLL_deficit_over_worse |
| High_monotonic |                 170 | Novel           | 2.07187     |  1.39361     |  2.78611     | seed             |              2000 | E1           | NLL_deficit_over_worse |
| High_monotonic |                 340 | Mixed           | 1.52301     |  0.867218    |  2.40089     | seed             |              2000 | E1           | NLL_deficit_over_worse |
| High_monotonic |                 340 | Novel           | 1.96589     |  1.37334     |  2.69235     | seed             |              2000 | E1           | NLL_deficit_over_worse |
| High_monotonic |                 680 | Mixed           | 1.65624     |  1.06933     |  2.43405     | seed             |              2000 | E1           | NLL_deficit_over_worse |
| High_monotonic |                 680 | Novel           | 2.10111     |  1.61126     |  2.65907     | seed             |              2000 | E1           | NLL_deficit_over_worse |
| Low_monotonic  |                 170 | Mixed           | 0.00684173  |  0.00564336  |  0.00798117  | seed             |              2000 | E1           | NLL_deficit_over_worse |
| Low_monotonic  |                 170 | Novel           | 0.00946499  |  0.00728702  |  0.0117338   | seed             |              2000 | E1           | NLL_deficit_over_worse |
| Low_monotonic  |                 340 | Mixed           | 0.00254142  |  0.00212229  |  0.00293497  | seed             |              2000 | E1           | NLL_deficit_over_worse |
| Low_monotonic  |                 340 | Novel           | 0.00404448  |  0.00291952  |  0.0053292   | seed             |              2000 | E1           | NLL_deficit_over_worse |
| Low_monotonic  |                 680 | Mixed           | 0.00099596  |  0.000806602 |  0.0011763   | seed             |              2000 | E1           | NLL_deficit_over_worse |
| Low_monotonic  |                 680 | Novel           | 0.00167568  |  0.00113813  |  0.00230904  | seed             |              2000 | E1           | NLL_deficit_over_worse |
| High_monotonic |                 170 | Mixed           | 0.24964     |  0.173342    |  0.334492    | seed             |              2000 | E1           | RT_deficit_over_worse  |
| High_monotonic |                 170 | Novel           | 0.278498    |  0.216697    |  0.33854     | seed             |              2000 | E1           | RT_deficit_over_worse  |
| High_monotonic |                 340 | Mixed           | 0.330089    |  0.243175    |  0.422351    | seed             |              2000 | E1           | RT_deficit_over_worse  |
| High_monotonic |                 340 | Novel           | 0.381305    |  0.286967    |  0.467122    | seed             |              2000 | E1           | RT_deficit_over_worse  |
| High_monotonic |                 680 | Mixed           | 0.353256    |  0.270173    |  0.440372    | seed             |              2000 | E1           | RT_deficit_over_worse  |
| High_monotonic |                 680 | Novel           | 0.417477    |  0.342244    |  0.485854    | seed             |              2000 | E1           | RT_deficit_over_worse  |
| Low_monotonic  |                 170 | Mixed           | 0.00259531  |  0.00212651  |  0.00304036  | seed             |              2000 | E1           | RT_deficit_over_worse  |
| Low_monotonic  |                 170 | Novel           | 0.00365155  |  0.00274129  |  0.00460751  | seed             |              2000 | E1           | RT_deficit_over_worse  |
| Low_monotonic  |                 340 | Mixed           | 0.000404584 |  0.000337071 |  0.000470127 | seed             |              2000 | E1           | RT_deficit_over_worse  |
| Low_monotonic  |                 340 | Novel           | 0.000725686 |  0.000486254 |  0.00101211  | seed             |              2000 | E1           | RT_deficit_over_worse  |
| Low_monotonic  |                 680 | Mixed           | 7.30874e-05 |  5.92467e-05 |  8.68642e-05 | seed             |              2000 | E1           | RT_deficit_over_worse  |
| Low_monotonic  |                 680 | Novel           | 0.000151924 |  8.89194e-05 |  0.000228873 | seed             |              2000 | E1           | RT_deficit_over_worse  |

**Remaining explanation/decision.** Valid matched High deficits remain larger in the available matches, so acquisition stage alone is insufficient there; unmatched seeds/anchors remain inconclusive. Duration is not a pure initialization manipulation, and the next decision is to interpret E2/E3/E5 without adding training or selecting favorable matches.

## E2 — state and probe learning

**Question/intervention/controls.** Eight probe repetitions crossed frozen/learn with reset/carry after frozen A/B primers at checkpoints 20/100/170. Parallel A/B-practice controls, probe-adapted same-parameter zero-state endpoints, no-primer references, and isolated A/B return descendants are distinct tables.

**Result.** First-to-late changes (negative means absolute improvement):

|                     |   NLL_change_first_to_late |   RT_change_first_to_late |
|:--------------------|---------------------------:|--------------------------:|
| ('frozen', 'carry') |               -0.0853362   |              -0.00839883  |
| ('frozen', 'reset') |                1.23744e-17 |               2.02023e-18 |
| ('learn', 'carry')  |               -0.27509     |              -0.0323121   |
| ('learn', 'reset')  |               -0.367357    |              -0.0423204   |

Whole-seed interaction uncertainty:

| condition      |   parent_checkpoint |         mean |   CI95_lower |   CI95_upper | bootstrap_unit   |   bootstrap_draws | experiment   | metric                                          | probe_family   | primer   |
|:---------------|--------------------:|-------------:|-------------:|-------------:|:-----------------|------------------:|:-------------|:------------------------------------------------|:---------------|:---------|
| High_monotonic |                  20 | -0.0388131   | -0.702794    |  0.669477    | seed             |              2000 | E2           | update_x_carry_NLL_deficit_change_first_to_late | Mixed          | A        |
| High_monotonic |                  20 |  0.12696     | -0.230571    |  0.658867    | seed             |              2000 | E2           | update_x_carry_NLL_deficit_change_first_to_late | Mixed          | B        |
| High_monotonic |                  20 | -0.0335882   | -0.596541    |  0.433843    | seed             |              2000 | E2           | update_x_carry_NLL_deficit_change_first_to_late | Novel          | A        |
| High_monotonic |                  20 |  0.417569    |  0.0798923   |  0.805698    | seed             |              2000 | E2           | update_x_carry_NLL_deficit_change_first_to_late | Novel          | B        |
| High_monotonic |                 100 |  0.258855    | -0.00828976  |  0.537698    | seed             |              2000 | E2           | update_x_carry_NLL_deficit_change_first_to_late | Mixed          | A        |
| High_monotonic |                 100 |  0.343719    |  0.0385024   |  0.667535    | seed             |              2000 | E2           | update_x_carry_NLL_deficit_change_first_to_late | Mixed          | B        |
| High_monotonic |                 100 |  0.327412    | -0.254826    |  0.894327    | seed             |              2000 | E2           | update_x_carry_NLL_deficit_change_first_to_late | Novel          | A        |
| High_monotonic |                 100 |  0.326423    | -0.268239    |  0.943029    | seed             |              2000 | E2           | update_x_carry_NLL_deficit_change_first_to_late | Novel          | B        |
| High_monotonic |                 170 |  0.420244    |  0.157163    |  0.685588    | seed             |              2000 | E2           | update_x_carry_NLL_deficit_change_first_to_late | Mixed          | A        |
| High_monotonic |                 170 |  0.426688    | -0.0120851   |  0.891415    | seed             |              2000 | E2           | update_x_carry_NLL_deficit_change_first_to_late | Mixed          | B        |
| High_monotonic |                 170 |  0.794443    |  0.374788    |  1.23284     | seed             |              2000 | E2           | update_x_carry_NLL_deficit_change_first_to_late | Novel          | A        |
| High_monotonic |                 170 |  0.873537    |  0.437125    |  1.30259     | seed             |              2000 | E2           | update_x_carry_NLL_deficit_change_first_to_late | Novel          | B        |
| Low_monotonic  |                  20 |  0.00183511  |  0.00119308  |  0.00243785  | seed             |              2000 | E2           | update_x_carry_NLL_deficit_change_first_to_late | Mixed          | A        |
| Low_monotonic  |                  20 |  0.00180787  |  0.00117912  |  0.00241832  | seed             |              2000 | E2           | update_x_carry_NLL_deficit_change_first_to_late | Mixed          | B        |
| Low_monotonic  |                  20 |  0.000472801 | -0.000650299 |  0.00172627  | seed             |              2000 | E2           | update_x_carry_NLL_deficit_change_first_to_late | Novel          | A        |
| Low_monotonic  |                  20 | -0.00055474  | -0.00177996  |  0.000776437 | seed             |              2000 | E2           | update_x_carry_NLL_deficit_change_first_to_late | Novel          | B        |
| Low_monotonic  |                 100 | -0.00178404  | -0.00533275  |  0.00251992  | seed             |              2000 | E2           | update_x_carry_NLL_deficit_change_first_to_late | Mixed          | A        |
| Low_monotonic  |                 100 | -0.00176404  | -0.00530236  |  0.0025242   | seed             |              2000 | E2           | update_x_carry_NLL_deficit_change_first_to_late | Mixed          | B        |
| Low_monotonic  |                 100 | -0.00654152  | -0.0123897   | -0.00122998  | seed             |              2000 | E2           | update_x_carry_NLL_deficit_change_first_to_late | Novel          | A        |
| Low_monotonic  |                 100 | -0.00655427  | -0.0124069   | -0.00123821  | seed             |              2000 | E2           | update_x_carry_NLL_deficit_change_first_to_late | Novel          | B        |
| Low_monotonic  |                 170 | -0.00119267  | -0.00268381  |  4.7876e-05  | seed             |              2000 | E2           | update_x_carry_NLL_deficit_change_first_to_late | Mixed          | A        |
| Low_monotonic  |                 170 | -0.00119561  | -0.0026863   |  4.28191e-05 | seed             |              2000 | E2           | update_x_carry_NLL_deficit_change_first_to_late | Mixed          | B        |
| Low_monotonic  |                 170 | -0.000930076 | -0.00374639  |  0.00136294  | seed             |              2000 | E2           | update_x_carry_NLL_deficit_change_first_to_late | Novel          | A        |
| Low_monotonic  |                 170 | -0.000926875 | -0.00373766  |  0.00136297  | seed             |              2000 | E2           | update_x_carry_NLL_deficit_change_first_to_late | Novel          | B        |
| High_monotonic |                  20 | -0.0107194   | -0.0531349   |  0.0329899   | seed             |              2000 | E2           | update_x_carry_RT_deficit_change_first_to_late  | Mixed          | A        |
| High_monotonic |                  20 |  0.0164445   | -0.0232407   |  0.0594748   | seed             |              2000 | E2           | update_x_carry_RT_deficit_change_first_to_late  | Mixed          | B        |
| High_monotonic |                  20 |  0.0140219   | -0.0159885   |  0.0472483   | seed             |              2000 | E2           | update_x_carry_RT_deficit_change_first_to_late  | Novel          | A        |
| High_monotonic |                  20 |  0.0307407   |  0.00471752  |  0.0575905   | seed             |              2000 | E2           | update_x_carry_RT_deficit_change_first_to_late  | Novel          | B        |
| High_monotonic |                 100 |  0.0313239   |  0.00138355  |  0.0619606   | seed             |              2000 | E2           | update_x_carry_RT_deficit_change_first_to_late  | Mixed          | A        |
| High_monotonic |                 100 |  0.0343987   |  0.0040403   |  0.0674005   | seed             |              2000 | E2           | update_x_carry_RT_deficit_change_first_to_late  | Mixed          | B        |
| High_monotonic |                 100 |  0.064724    |  0.00755704  |  0.119064    | seed             |              2000 | E2           | update_x_carry_RT_deficit_change_first_to_late  | Novel          | A        |
| High_monotonic |                 100 |  0.059105    |  0.00155723  |  0.111381    | seed             |              2000 | E2           | update_x_carry_RT_deficit_change_first_to_late  | Novel          | B        |
| High_monotonic |                 170 |  0.0307439   |  0.000710003 |  0.0673274   | seed             |              2000 | E2           | update_x_carry_RT_deficit_change_first_to_late  | Mixed          | A        |
| High_monotonic |                 170 |  0.0356224   | -0.0191711   |  0.0954844   | seed             |              2000 | E2           | update_x_carry_RT_deficit_change_first_to_late  | Mixed          | B        |
| High_monotonic |                 170 |  0.101207    |  0.0551292   |  0.155372    | seed             |              2000 | E2           | update_x_carry_RT_deficit_change_first_to_late  | Novel          | A        |
| High_monotonic |                 170 |  0.111178    |  0.0510322   |  0.173827    | seed             |              2000 | E2           | update_x_carry_RT_deficit_change_first_to_late  | Novel          | B        |
| Low_monotonic  |                  20 |  0.000498033 |  0.000327959 |  0.000660929 | seed             |              2000 | E2           | update_x_carry_RT_deficit_change_first_to_late  | Mixed          | A        |
| Low_monotonic  |                  20 |  0.000486479 |  0.000314554 |  0.000651194 | seed             |              2000 | E2           | update_x_carry_RT_deficit_change_first_to_late  | Mixed          | B        |
| Low_monotonic  |                  20 |  3.69868e-06 | -0.000281818 |  0.000300224 | seed             |              2000 | E2           | update_x_carry_RT_deficit_change_first_to_late  | Novel          | A        |
| Low_monotonic  |                  20 | -0.000241292 | -0.000545969 |  7.37684e-05 | seed             |              2000 | E2           | update_x_carry_RT_deficit_change_first_to_late  | Novel          | B        |
| Low_monotonic  |                 100 | -0.00100713  | -0.00309178  |  0.00140014  | seed             |              2000 | E2           | update_x_carry_RT_deficit_change_first_to_late  | Mixed          | A        |
| Low_monotonic  |                 100 | -0.000994482 | -0.00306047  |  0.00140058  | seed             |              2000 | E2           | update_x_carry_RT_deficit_change_first_to_late  | Mixed          | B        |
| Low_monotonic  |                 100 | -0.00401033  | -0.00759876  | -0.000831556 | seed             |              2000 | E2           | update_x_carry_RT_deficit_change_first_to_late  | Novel          | A        |
| Low_monotonic  |                 100 | -0.00401814  | -0.00761257  | -0.000834438 | seed             |              2000 | E2           | update_x_carry_RT_deficit_change_first_to_late  | Novel          | B        |
| Low_monotonic  |                 170 | -0.000539264 | -0.00120319  | -6.77086e-06 | seed             |              2000 | E2           | update_x_carry_RT_deficit_change_first_to_late  | Mixed          | A        |
| Low_monotonic  |                 170 | -0.000540171 | -0.00120383  | -9.59175e-06 | seed             |              2000 | E2           | update_x_carry_RT_deficit_change_first_to_late  | Mixed          | B        |
| Low_monotonic  |                 170 | -0.000375063 | -0.00158482  |  0.000587341 | seed             |              2000 | E2           | update_x_carry_RT_deficit_change_first_to_late  | Novel          | A        |
| Low_monotonic  |                 170 | -0.000373803 | -0.00158162  |  0.00058734  | seed             |              2000 | E2           | update_x_carry_RT_deficit_change_first_to_late  | Novel          | B        |

High-scale checkpoint-170 mean update × carry interactions on relative deficit:

| probe_family   |   update_x_carry_NLL_deficit_change_first_to_late |   update_x_carry_RT_deficit_change_first_to_late |
|:---------------|--------------------------------------------------:|-------------------------------------------------:|
| Mixed          |                                          0.423038 |                                        0.0332456 |
| Novel          |                                          0.831643 |                                        0.10673   |

Frozen/reset equality, branch isolation, zero-state adapted endpoints, parallel familiar practice, and separate A/B return descendants are preserved in the branch audit and transition tables. Absolute probe performance can improve while relative deficit worsens; carry and learning are interventions on the SRN, not literal human manipulations. Human history/practice evidence remains awaiting data, so no shared internal process is claimed.

## E3 — cue/layout composition

**Question/intervention/controls.** Original-cohort frozen parents received cue-only, layout-only, combined, and unchanged familiar-anchor inputs at constant physical sequence, targets, dimensionality, parameters, and zero state. Cell 00 and cell 11 reproduce repository encoders exactly.

**Result.** Mean prespecified factorial interactions across scales/checkpoints:

| probe_family   |   interaction_NLL |   interaction_RT |
|:---------------|------------------:|-----------------:|
| Mixed          |         -0.733837 |       -0.0846486 |
| Novel          |         -0.876199 |       -0.0946921 |

Whole-seed factorial uncertainty:

| condition      |   parent_checkpoint |         mean |   CI95_lower |   CI95_upper | bootstrap_unit   |   bootstrap_draws | experiment   | metric          | probe_family   | anchor   |
|:---------------|--------------------:|-------------:|-------------:|-------------:|:-----------------|------------------:|:-------------|:----------------|:---------------|:---------|
| High_monotonic |                  20 | -1.16091     | -1.90306     | -0.433472    | seed             |              2000 | E3           | interaction_NLL | Mixed          | A        |
| High_monotonic |                  20 | -1.51445     | -2.71669     | -0.433776    | seed             |              2000 | E3           | interaction_NLL | Mixed          | B        |
| High_monotonic |                  20 | -0.684105    | -1.30655     | -0.0358194   | seed             |              2000 | E3           | interaction_NLL | Novel          | A        |
| High_monotonic |                  20 | -1.15661     | -2.32956     | -0.136537    | seed             |              2000 | E3           | interaction_NLL | Novel          | B        |
| High_monotonic |                 100 | -1.19357     | -1.97516     | -0.34722     | seed             |              2000 | E3           | interaction_NLL | Mixed          | A        |
| High_monotonic |                 100 | -1.80106     | -2.88752     | -0.723206    | seed             |              2000 | E3           | interaction_NLL | Mixed          | B        |
| High_monotonic |                 100 | -1.08855     | -1.88897     | -0.357049    | seed             |              2000 | E3           | interaction_NLL | Novel          | A        |
| High_monotonic |                 100 | -2.82674     | -4.11765     | -1.6334      | seed             |              2000 | E3           | interaction_NLL | Novel          | B        |
| High_monotonic |                 170 | -1.57423     | -2.23521     | -0.907253    | seed             |              2000 | E3           | interaction_NLL | Mixed          | A        |
| High_monotonic |                 170 | -1.64962     | -2.419       | -0.919151    | seed             |              2000 | E3           | interaction_NLL | Mixed          | B        |
| High_monotonic |                 170 | -1.79172     | -2.76441     | -0.816301    | seed             |              2000 | E3           | interaction_NLL | Novel          | A        |
| High_monotonic |                 170 | -3.01848     | -4.23476     | -1.9661      | seed             |              2000 | E3           | interaction_NLL | Novel          | B        |
| Low_monotonic  |                  20 |  0.000434294 |  0.000328671 |  0.000543414 | seed             |              2000 | E3           | interaction_NLL | Mixed          | A        |
| Low_monotonic  |                  20 |  0.00462438  |  0.00418915  |  0.00510976  | seed             |              2000 | E3           | interaction_NLL | Mixed          | B        |
| Low_monotonic  |                  20 |  0.00691075  |  0.00635389  |  0.00744506  | seed             |              2000 | E3           | interaction_NLL | Novel          | A        |
| Low_monotonic  |                  20 |  0.00120998  |  0.00102637  |  0.00138851  | seed             |              2000 | E3           | interaction_NLL | Novel          | B        |
| Low_monotonic  |                 100 |  0.00587336  |  0.00501123  |  0.00684352  | seed             |              2000 | E3           | interaction_NLL | Mixed          | A        |
| Low_monotonic  |                 100 |  0.00527761  |  0.00446346  |  0.00615871  | seed             |              2000 | E3           | interaction_NLL | Mixed          | B        |
| Low_monotonic  |                 100 |  0.00572833  |  0.00473045  |  0.00677594  | seed             |              2000 | E3           | interaction_NLL | Novel          | A        |
| Low_monotonic  |                 100 |  0.00648375  |  0.00525413  |  0.00784386  | seed             |              2000 | E3           | interaction_NLL | Novel          | B        |
| Low_monotonic  |                 170 |  0.0030121   |  0.00218271  |  0.00406691  | seed             |              2000 | E3           | interaction_NLL | Mixed          | A        |
| Low_monotonic  |                 170 |  0.00289849  |  0.00214639  |  0.00382184  | seed             |              2000 | E3           | interaction_NLL | Mixed          | B        |
| Low_monotonic  |                 170 |  0.00316314  |  0.00233277  |  0.00413697  | seed             |              2000 | E3           | interaction_NLL | Novel          | A        |
| Low_monotonic  |                 170 |  0.00332863  |  0.0023587   |  0.00452247  | seed             |              2000 | E3           | interaction_NLL | Novel          | B        |
| High_monotonic |                  20 | -0.101004    | -0.168688    | -0.0421774   | seed             |              2000 | E3           | interaction_RT  | Mixed          | A        |
| High_monotonic |                  20 | -0.0815461   | -0.153333    | -0.0136184   | seed             |              2000 | E3           | interaction_RT  | Mixed          | B        |
| High_monotonic |                  20 | -0.0456531   | -0.13806     |  0.0515901   | seed             |              2000 | E3           | interaction_RT  | Novel          | A        |
| High_monotonic |                  20 | -0.0444077   | -0.123582    |  0.0361244   | seed             |              2000 | E3           | interaction_RT  | Novel          | B        |
| High_monotonic |                 100 | -0.191935    | -0.306737    | -0.080022    | seed             |              2000 | E3           | interaction_RT  | Mixed          | A        |
| High_monotonic |                 100 | -0.165894    | -0.262634    | -0.0782613   | seed             |              2000 | E3           | interaction_RT  | Mixed          | B        |
| High_monotonic |                 100 | -0.135363    | -0.201275    | -0.0663284   | seed             |              2000 | E3           | interaction_RT  | Novel          | A        |
| High_monotonic |                 100 | -0.26075     | -0.378828    | -0.147873    | seed             |              2000 | E3           | interaction_RT  | Novel          | B        |
| High_monotonic |                 170 | -0.286947    | -0.374491    | -0.193747    | seed             |              2000 | E3           | interaction_RT  | Mixed          | A        |
| High_monotonic |                 170 | -0.203052    | -0.306113    | -0.103509    | seed             |              2000 | E3           | interaction_RT  | Mixed          | B        |
| High_monotonic |                 170 | -0.250326    | -0.365485    | -0.140707    | seed             |              2000 | E3           | interaction_RT  | Novel          | A        |
| High_monotonic |                 170 | -0.412212    | -0.541609    | -0.289793    | seed             |              2000 | E3           | interaction_RT  | Novel          | B        |
| Low_monotonic  |                  20 |  9.29389e-05 |  7.13057e-05 |  0.000115508 | seed             |              2000 | E3           | interaction_RT  | Mixed          | A        |
| Low_monotonic  |                  20 |  0.000962131 |  0.000878288 |  0.00104747  | seed             |              2000 | E3           | interaction_RT  | Mixed          | B        |
| Low_monotonic  |                  20 |  0.00139567  |  0.00126999  |  0.00151431  | seed             |              2000 | E3           | interaction_RT  | Novel          | A        |
| Low_monotonic  |                  20 |  0.000240478 |  0.000200741 |  0.000278893 | seed             |              2000 | E3           | interaction_RT  | Novel          | B        |
| Low_monotonic  |                 100 |  0.00363983  |  0.00310337  |  0.00421611  | seed             |              2000 | E3           | interaction_RT  | Mixed          | A        |
| Low_monotonic  |                 100 |  0.00325114  |  0.00278028  |  0.00373673  | seed             |              2000 | E3           | interaction_RT  | Mixed          | B        |
| Low_monotonic  |                 100 |  0.00342592  |  0.00288187  |  0.00399056  | seed             |              2000 | E3           | interaction_RT  | Novel          | A        |
| Low_monotonic  |                 100 |  0.00389837  |  0.00321128  |  0.00467346  | seed             |              2000 | E3           | interaction_RT  | Novel          | B        |
| Low_monotonic  |                 170 |  0.00116946  |  0.000835478 |  0.00159287  | seed             |              2000 | E3           | interaction_RT  | Mixed          | A        |
| Low_monotonic  |                 170 |  0.00112889  |  0.000824733 |  0.00150096  | seed             |              2000 | E3           | interaction_RT  | Mixed          | B        |
| Low_monotonic  |                 170 |  0.00129856  |  0.000930209 |  0.00173405  | seed             |              2000 | E3           | interaction_RT  | Novel          | A        |
| Low_monotonic  |                 170 |  0.00136456  |  0.000943365 |  0.00190018  | seed             |              2000 | E3           | interaction_RT  | Novel          | B        |

High-scale checkpoint-170 decomposition:

| probe_family   |   interaction_NLL |   interaction_RT |   cue_effect_NLL |   layout_effect_NLL |
|:---------------|------------------:|-----------------:|-----------------:|--------------------:|
| Mixed          |          -1.60215 |        -0.244109 |          1.73826 |             1.64135 |
| Novel          |          -2.40568 |        -0.331021 |          2.30306 |             2.37334 |

The large negative interactions show non-additivity: cue-only and layout-only costs do not simply sum in the combined input. This localizes sensitivity to composition without proving configural learning. Layout interpolation is labeled an input-space diagnostic and is not instantiated as a psychologically valid human display; the human factorial remains awaiting data.

## E4 — action visibility

**Question/intervention/controls.** The local task crosses between-participant full-preview/one-target visibility with A/B/Mixed/Novel first tests and preceding A/B context. Layout and response counterbalances are layered over the 16 core cells. A/B test arms control the boundary/pause; acquisition and return schedules use shared timing logic.

**Status/decision.** The task implements deterministic assignment, exact event timestamps, focus/abort/resume handling, repeat-key suppression, and a future-label-free one-target DOM. The SRN already receives ordinal, cue, and full-layout information; feeding the scored target directly was not implemented. Power results are prospective simulations, not outcomes. Static/task-contract testing passed, but the in-app browser connection was unavailable in this execution environment, so interactive screenshot QA remains pending. Collecting participants and estimating the visibility interaction are the next empirical steps.

## E5 — prior structural experience

**Question/intervention/controls.** P1 and P2 have identical 640-sequence token streams and sequence/layout/cue marginals; P1 distributes each sequence across layouts while P2 uses a seed-counterbalanced bijection. P0 has no pretraining. Predictive weights carry into identical target acquisition while the execution readout and optimizer buffers reset.

**Result.** Checkpoint-170 arm means:

|                 |   NLL_deficit_over_worse |   RT_deficit_over_worse |
|:----------------|-------------------------:|------------------------:|
| ('P0', 'Mixed') |                 0.692525 |               0.12654   |
| ('P0', 'Novel') |                 1.03597  |               0.141087  |
| ('P1', 'Mixed') |                 0.54955  |               0.108373  |
| ('P1', 'Novel') |                 0.627593 |               0.144594  |
| ('P2', 'Mixed') |                 0.45436  |               0.0909164 |
| ('P2', 'Novel') |                 0.788268 |               0.146784  |

Whole-seed P1-minus-P2 uncertainty:

| condition      | probe_context   |         mean |   CI95_lower |   CI95_upper | bootstrap_unit   |   bootstrap_draws | experiment   | metric                             |
|:---------------|:----------------|-------------:|-------------:|-------------:|:-----------------|------------------:|:-------------|:-----------------------------------|
| High_monotonic | Mixed           |  0.184105    | -0.193453    |  0.593274    | seed             |              2000 | E5           | P1_minus_P2_NLL_deficit_over_worse |
| High_monotonic | Novel           | -0.319842    | -0.639603    |  0.0205125   | seed             |              2000 | E5           | P1_minus_P2_NLL_deficit_over_worse |
| Low_monotonic  | Mixed           | -4.07652e-05 | -0.00316292  |  0.00295825  | seed             |              2000 | E5           | P1_minus_P2_NLL_deficit_over_worse |
| Low_monotonic  | Novel           | -0.00023501  | -0.000956304 |  0.000484204 | seed             |              2000 | E5           | P1_minus_P2_NLL_deficit_over_worse |
| High_monotonic | Mixed           |  0.0335533   | -0.052086    |  0.127951    | seed             |              2000 | E5           | P1_minus_P2_RT_deficit_over_worse  |
| High_monotonic | Novel           | -0.00453406  | -0.067257    |  0.0572991   | seed             |              2000 | E5           | P1_minus_P2_RT_deficit_over_worse  |
| Low_monotonic  | Mixed           | -1.12524e-05 | -0.00188728  |  0.00176017  | seed             |              2000 | E5           | P1_minus_P2_RT_deficit_over_worse  |
| Low_monotonic  | Novel           | -0.0001141   | -0.000525334 |  0.000304097 | seed             |              2000 | E5           | P1_minus_P2_RT_deficit_over_worse  |

A/B-only acquisition support remained absent through 680 for **26** of 80 condition/seed/candidate-arm comparisons. The registered P1-minus-P2 High-scale effect does not show a uniform joint transfer advantage: Mixed NLL deficit is numerically higher for P1, Novel is numerically lower, and both intervals include zero.

The tested structural dose therefore does not solve the joint human target, and unsupported matched comparisons remain inconclusive. No new dose, seed, or architecture is added; any refinement belongs in a separately frozen follow-up.

## Reproduction

```bash
python scripts/run_transfer_experiment_program.py --config configs/transfer_experiment_program.yaml --stage all
python scripts/serve_transfer_human_task.py --port 8765
```

Use `--dry-run` for source/branch/runtime inspection and `--smoke` for a non-inferential two-seed validation. To analyze a real browser export, rerun E4 with `--human-events PATH`.

## Scientific boundary

Negative interventions are informative and do not select a replacement architecture. Human E2–E4 outcomes require new participants. The program does not tune against human probe RTs, add seeds after inspecting outcomes, or turn missing evidence into a null result.
