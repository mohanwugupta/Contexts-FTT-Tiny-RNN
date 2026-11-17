# Scratchpad: SRN Mixed / Novel Representation Diagnostics

## Decisions

- Use the persisted `confirmatory_srn_final` High/Low checkpoints (20 seeds each).
  They are the frozen 8-unit Elman SRNs with the exact A/B/Mixed/Novel 3D context basis.
- Map `High_monotonic` to **High-Scale + High Context** and `Low_monotonic` to
  **Low-Scale + Low Context**. The monotonic execution policy is not loaded or analyzed;
  this PRD concerns hidden states and the next-token head.
- The historical 6-unit code defines the requested hybrid controls, but no trained hybrid
  checkpoints are present. Do not recreate or retrain them. Record their absence in the manifest.
- Analyze seven trained transitions, not an invented eighth cyclic transition: the eight-item
  training sequence supplies seven next-token targets.
- Sequence-decoder familiar generalization is cross-context (A→B and B→A). The production
  probe decoder is fit once on A+B and never on Mixed, Novel, or NoContextProbe.
- The registered SRN resets hidden state at every sequence boundary. Therefore the actual
  preceding-A/B conditions equal reset. `matched terminal` is explicitly counterfactual
  state carryover after ten repeated familiar sequences and cannot drive classification D.
- Causal patches intervene on hidden states immediately before the frozen output layer.
  They diagnose readout-relevant rescue and do not alter recurrent weights/checkpoints.

## Progress

- [x] Locate and inspect the exact frozen checkpoint payloads.
- [x] Implement frozen extraction and parameter-hash audit.
- [x] Implement analyses A–I in PRD order.
- [x] Implement exact nine-table, ten-figure, and report contracts.
- [x] Add T1–T10 executable checks and a smoke integration test.
- [x] Run tests and correct float32 decomposition tolerance/accumulation defects.
- [x] Run the full 40-network frozen diagnostic battery.
- [x] Audit report interpretation and separate registered reset history from counterfactual carryover.
