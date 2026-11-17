# Three-context data dictionary for PRD3

This dictionary records the literal labels observed in
`data/threeContextsBreak_clean.csv` and the analysis labels used by the
standalone `retrieval_modeling/` pipeline. It was derived from the populated
CSV rather than inferred from the counterbalance numbers.

## Raw-label mapping

| Raw `trialType` | Analysis `retrieval_context` | Role | Raw rows |
|---|---:|---|---:|
| `Maze Training` | `A` | established A skill | 57,288 |
| `Maze Test` | `A` | A return/test | 8,491 |
| `Melody Training` | `B` | established B skill | 57,150 |
| `test melodyA` | `B` | B return/test | 8,508 |
| `Space Training` | `Novel` | third-context exposure | 6,512 |
| `test spaceA` | `Novel` | third-context test | 5,918 |
| `No Context Training` | `Default` | no-cue exposure | 5,413 |
| `No Context Test` | `Default` | no-cue test | 2,640 |
| `warmup MazeA` | excluded | warmup | 2,020 |
| `warmup melodyA` | excluded | warmup | 1,899 |
| `test warmupA` | excluded | warmup | 1,770 |

`dataset=Novel` remains the internal PRD3 name for the three-context source
experiment. Within it, `retrieval_context=Novel` specifically means the
export's Space context. `Default` always means No Context; it is never pooled
with Space.

## Exposure cohorts and counterbalances

The cohort labels are determined from each participant's observed training
and test blocks, not assigned from the counterbalance number alone.

| Counterbalance | Exposure cohort | Context immediately before the first primary probe | Included participants |
|---:|---|---:|---:|
| 1 | `Novel_test_only_control` | B | 15 |
| 2 | `Novel_training` | A | 15 |
| 3 | `Novel_training` | B | 20 |
| 4 | `Default_training` | B | 16 |
| 5 | `Default_training` | A | 11 |
| 6 | `Novel_test_only_control` | A | 15 |

Thus CB1/CB6 are not missing-data participants: they are intentional Space
test-only controls. CB2/CB3 receive Space training. CB4/CB5 receive No Context
training. The paired counterbalances reverse whether A or B immediately
precedes the primary target block.

After the standard complete-case rules and removal of raw sequence 1, the
three-context analysis contains 92 participants, 17 blocks per participant,
and 14,076 sequence observations (9 per block). The cohort totals are:

- `Novel_training`: 35
- `Default_training`: 27
- `Novel_test_only_control`: 30

Three reused IDs are conservatively excluded because they contain two full
17-block sessions under one participant identifier: `MG578955`, `MG600555`,
and `MG783355`. Treating those sessions as independent people would risk
pseudoreplication.

## Parallel mixed-context labels

The mixed-context file has the same design structure:

- `Mixed Training` / `Mixed Test` -> `Both`
- `No Context Training` / `No Context Test` -> `Default`
- CB1/CB6 -> `Both_test_only_control`
- CB2/CB3 -> `Both_training`
- CB4/CB5 -> `Default_training`

After exclusions it contains 218 participants and 33,354 sequence
observations: 76 `Both_training`, 76 `Default_training`, and 66
`Both_test_only_control` participants.

## Output columns

The PRD3 clean data, prediction, responsibility, and evidence outputs retain:

- `raw_context`: literal source label;
- `retrieval_context`: A, B, Both, Novel, or Default;
- `probe_phase`: training or test according to the literal label;
- `probe_training_context`: trained probe family, or missing for controls;
- `probe_exposure_cohort`: explicit trained/test-only cohort;
- `is_probe_control`: test-only control flag;
- `probe_entry_context`: A or B immediately preceding the first primary probe;
- `counterbalance`: original counterbalance code.

The pipeline also writes `retrieval_label_dictionary.csv` and
`retrieval_cohort_summary.csv` so every analysis run records the mappings and
post-exclusion cohort counts used for that run.
