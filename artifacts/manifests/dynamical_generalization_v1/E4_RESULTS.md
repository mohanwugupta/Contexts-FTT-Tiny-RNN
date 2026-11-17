# E4 — Fixed sequence replication

Design-only; evidence_status=blocked_by_e3. No feasible frozen E3 familiar-only configuration; no replacement model or sequence-dependent calibration permitted.

| sequence_id | tokens | valid | replaced_equivalent_candidate |
| --- | --- | --- | --- |
| original | [0, 1, 2, 0, 3, 1, 2, 3] | True | False |
| S1 | [0, 1, 0, 1, 2, 3, 2, 3] | True | False |
| S2 | [0, 1, 0, 2, 1, 3, 2, 3] | True | False |
| S3 | [0, 1, 2, 0, 3, 2, 1, 3] | True | False |

The registered design retains 20 paired seeds for each sequence and all four factorial arms, with no per-sequence coefficient search. No replication model was trained; empty outcome tables contain no imputed measurements.

