# Supervised native Cloud trial v1

This is the next executable training trial, not the full autonomous72-hour campaign. It reuses the accepted Cloud Docker image and executes one fixed registered hypothesis plus its matched control. It does not make research-agent calls. The owner launches it explicitly using START_IN_CLOUD.md.

## Fixed experiment

Open hypothesis: `research_search/history/OPEN_HYPOTHESES.json:prior-cue-coverage`. Does broader prior cue-code coverage change later functional A/B specificity and C transfer, when the prior has received its ordinary full training budget?

Control: unchanged `research_search.runners.candidate.DEFAULT` (H8 Linear, nominal scale5, recurrent norm.9, continuous uncentered cues, P2). Candidate changes only `prior.cue_rotation` from false to true. The registered native operator applies an independently seeded Haar orthogonal3D cue basis within each background episode. It does not rotate or train C/Mixed/NoContext during target practice. This is a full-prior retest of an unresolved registered hypothesis; it is not a new discovery or a claim that earlier insufficient-prior results refuted it.

- Seeds21000 and21001, all four existing development tasks; independent unit is initialization seed, not the eight task rows per arm.
-1920 ordinary Reptile episodes x20 native updates per prior, native epsilon schedule; no dose shortening.
-Three unrelated shared competence tasks (AB/AC/BC),20 updates each, same paths in both arms. Inherited useful criterion: mean CE<=.5 and>=6/7 correct in each practiced context.
-330 target updates, A10/B10 only; all331 frozen evaluation states and five physical probes. Candidate and control use identical task paths, schedules and registered initialization seeds.
-All original flags, floor measures, signed own/cross/net gains, directional and sustained costs retained. Additional same-network criterion: original early phenotype plus useful A, B, C, Mixed-average and NoContext at330. Original screen and late functional status remain unassessed.
-One candidate, four priors, twelve competence tests, sixteen target histories. No promotion to more seeds, canonical evaluation, optimization of probes, thresholds, searches or architecture extensions.

## Resource contract

Fresh trial, max6h wall from first numerical start and5h process CPU across resumes. Maximum159120 Adam attempts =4x1920x20+12x20+16x330. Reptile outer interpolations separately capped at7680; combined ceiling166800. These outer operations are not Adam steps. A global host lock spans numerical execution and export. Native uncertain attempts remain charged and cannot be replayed. Only a clean paused branch can resume explicitly; source/config/task/checkpoint hashes and full model/policy/Adam/RNG/task cursor must agree. Neither resume nor export resets the original start time or ceilings.

The old six-step laptop benchmark implies about3.76 CPU hours for the maximum Adam count before unmeasured overhead. Actual Cloud throughput is unknown. Budget exhaustion is incomplete/inconclusive. Never shorten prior learning, expand budgets after results, or interpret failed competence as a mechanism rejection.

Token allowance for automatic calls is zero. The previous117000 planning holds remain historical holds, with actual provider usage unknown. Executing deterministic Python makes no LLM calls; asking Codex Cloud chat to operate it still consumes that enclosing chat turn's usage, which this worker cannot measure. No new specialist, critic or coordinator model call is launched. The full autonomous campaign is still blocked on verified provider receipts, manual per-call approval and parent-agent isolation.

## Source audit and necessary corrections

| Native source | Reused contract |
| --- | --- |
| `research_search/runners/candidate.py` | exact model, registered cue rotation, target optimizer and330-update schedule |
| `research_search/runners/train.py` | ordinary CE+.2 detached-confidence execution objective, jointclip1, native Adam, Reptile, full-state checkpoints and guarded charge-before-attempt |
| `research_search/runners/evaluate.py` | five physical probes, static quarter-readout diagnostic, position logits/probabilities, inherited score and linear midpoint/CE controls |
| `research_search/evaluation/scoring.py`, screen `analysis.py`/`rules.py` | original functional/switch/floor measures and thresholds |
| `research_search/orchestration/store.py`, `queue.py` | durable reservations, step hashes, unreconciled/uncertain work, one worker |
| `research_search/runners/tasks.py` | unchanged native1920-episode background generator, original exclusion identities and RNG |

Original sources, evaluator and72h configuration are not edited. SOURCE_FILES.json pins151 required source/registry identities; assets/FILES.json pins the registered libraries. No historical checkpoints, human data, credentials or private confirmation vectors enter the new container. Task registration verifies both existing bank commitments locally; Cloud receives only development tasks and the excluded-background libraries.

Three versioned infrastructure corrections are explicit:

1. Native `train.py` has an unimported BudgetError in its clean CPU branch. A new subclass overrides only the pre-update resource guard, preserving all construction, optimization, save/restore and evaluation methods. It also checks owner stop before every update. This is not an edit to sealed source.
2. Original competence registration asks the general mixed-task generator to construct conflicting second paths that its eventual shared-only tests discard. For seed21000/full1920 exclusions this exhausted the existing100000-attempt limit before any model ran. The log is retained. New fixed shared-only registration uses the same native draw/allowed/orbit routines, original seed construction, three shared paths,20-update schedules, all exclusions and the unchanged100000 ceiling. It does not generate unused conflict branches. Competence paths are a new registered library and are not claimed identical to the unavailable original library. Neither generation failure nor its repair is a scientific result.
3. Native atomic files are0600. A separate container under the same writer identity reads the native directory through a read-only mount, writes a fresh ZIP/checksum manifest elsewhere and verifies every member. It does not chmod or alter native files. Named-container cleanup works even if docker create times out before returning an ID; create has the owner-approved60-second timeout.

Exclusion acceleration uses set distributivity: union forbidden full-path orbits, and separately forbidden seven-token window orbits; intersection with each candidate yields exactly `any(equivalent(candidate,p))`. It makes no RNG calls. Tests check equivalence on a fixed path grid and exact generator output/order against the native function. The initial slow attempt was interrupted; its log and partial development JSON remain in the scratchpad audit.

The accepted Linux image is a new runtime, not an arm64 bitwise-replay certificate. New results record its immutable image, code hashes, native checkpoint hashes, task identities, queue events and scientific uncertainty. Container restrictions do not certify isolation of the enclosing Cloud model agent.

## Outputs and recovery

`OUT/native/` is container-owned full state; do not fix its modes or manually change SQLite. Each run produces `OUT/launches/<unique-name>/` lifecycle logs and a fresh `OUT/exports/<unique-name>/bundle/results.zip` with FILES.json. The host verifies ZIP bytes directly. No host recursive hash over native0600 files is attempted.

The bundle includes all job outputs: full checkpoint states, practice/exposure logs, metrics/position predictions, current and static-readout tables, original signed summaries and linear controls. Aggregate products: coverage.csv, seed_task_results.csv, seed_intervals.csv, checkpoint_hashes.json, background_competence.json, RESULT.json, REPORT.md, plot_values.csv and trial_summary.png/svg. Missing, failed, paused and negative branches remain explicit. If generation, runtime or worker failure stops execution, keep logs and review; no automatic retry.

`stop` requests terminal stop at the next update boundary. `resume` only continues clean paused work within the same original ceilings; terminal, uncertain or failed branches require adjudication and do not silently restart. `export` performs table/state collection only under a read-only native mount, including after a failed worker. A hard termination can leave a running/uncertain ledger; report it rather than clearing it.

## Teach-back

Ordinary Reptile repeatedly lets a copy of the meta-initialization learn an unrelated task with native Adam, then interpolates the meta-weights toward that adapted copy. The candidate changes the context basis seen in those unrelated experiences; it adds no reward for splitting or transfer. Each final task starts from its arm's corresponding pretrained state, with fresh target execution policy/Adam, and practices only A/B. The fixed scorer measures what that ordinary learning produced. Competence controls tell us whether failure followed inadequate prior adaptation; they do not select the best checkpoint. Lower transfer CE alone is not the full desired phenotype.
