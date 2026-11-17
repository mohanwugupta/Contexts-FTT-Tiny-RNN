# Phase 1E Optimizer Audit

| Phase 1D quantity | Recomputed each attempted update? | Audit finding |
|---|---:|---|
| Endpoint constraint residual | Yes | Evaluated at current parameters. |
| Endpoint constraint Jacobian, J_C | Yes | Rebuilt before each proposal and therefore after accepted updates. |
| Task-path residual | Yes | Evaluated at current parameters. |
| Exact task-path Jacobian, J_P | No | Formed for the initial local-reachability audit, while training used the current scalar path gradient. |
| Tangent/null geometry | Yes for J_C | Rebuilt by SVD inside the equality-constrained proposal. |
| Damping | Partly | Absolute damping followed current J_C; relative damping was fixed. |
| Trust-region scale | No | The source-relative radius was fixed for the run. |
| Equality correction | Yes | Recomputed for each proposal. |
| Nonlinear restoration | No | Rejected proposals triggered rollback/learning-rate halving, not Jacobian-recomputed restoration. |

Answer: Phase 1D recomputed `J_C` after every accepted update, but did not form a fresh exact `J_P` on
every optimizer step. Thus the two Jacobians were not jointly relinearized after every accepted update.

Phase 1E consequently implements the full distinct manipulation: fresh scaled residuals and exact `J_C` and
`J_P`, normal-plus-tangent SQP steps, nonlinear feasibility restoration with a fresh `J_C` at each restoration
iteration, and adaptive trust-region acceptance using fixed 0.25/0.75 thresholds.
