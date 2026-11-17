# Bridge result map

| Claim | Level | Script | Input | Output | Figure | Statistic | Boundary |
|---|---|---|---|---|---|---|---|
| A/B learning architecture | Scalar learning | `scripts/run_state_space_bridge.py` | Validated sequence RT / corrected normalized RT | `state_space_bic_comparison.csv` | SSB1/SSB2 | Delta BIC | Pure A/B only |
| Frozen A/B endpoint expression | Scalar retrieval | Bridge pipeline | Selected pure-A/B state model | `frozen_AB_counterfactuals.csv` | SSB3 | Endpoint separation | No probe update |
| Mixed, Novel, and No Context retrieval | Retrieval | Validated v3 adapter | Frozen A/B profiles | `retrieval_model_comparison.csv` | SSB4 | Held-out log likelihood | No learning-model feedback |
| Rank-2 vector state | Vector state | Validated v3.1 adapter | Pure A/B transition profiles | `human_low_rank_rank_selection.csv` | SSB5 | Held-out rank selection | Rank not hard-coded |
| SRN implementation | Representation/causal | Corrected v2.2/v2.3 | Frozen scale-2 checkpoints | `cross_system_bridge_summary.csv` | SSB6 | Registered result code | No human tuning |

The current corrected scale-2 smoke is stopped at the execution sanity gate, so downstream entries
describe implemented artifacts rather than completed corrected-run results.

