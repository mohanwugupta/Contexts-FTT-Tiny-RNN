# E3 pilot feasibility — numerical stop rationale

The fixed pilot tested 48 candidate settings × five reserved seeds × four arms (960 records, with 400 distinct objective-equivalent training runs). All settings were retained. No protected factorial probe was evaluated.

| criterion | passing_candidates | total_candidates |
| --- | --- | --- |
| finite | 48 | 48 |
| competence_passed | 0 | 48 |
| learning_passed | 48 | 48 |
| separation_passed | 12 | 48 |
| closure_passed | 30 | 48 |
| gradient_passed | 23 | 48 |
| feasible | 0 | 48 |

No candidate passed the familiar-competence gate. That gate required all four arms to have seed-mean A CE ≤1.5 nats, B CE ≤1.5 nats, and familiar A/B mean CE ≤1.2 nats. All candidates nevertheless passed the distinct learning-over-initialization/nonconstant-output checks. Thus failure to meet the competence bound must not be described as absence of any learning.

Range of each candidate’s worst-arm familiar means (descriptive calibration diagnostics):

| across_candidates | A_ce | B_ce | familiar_mean_ce |
| --- | --- | --- | --- |
| min | 1.3877 | 1.429 | 1.5155 |
| max | 1.9771 | 2.4117 | 2.0767 |

Some candidates engaged separation or reduced neutral closure, but none met every required condition jointly. The result is insufficient_familiar_support; the causal factorial and its sequence replication remain unestimated. No threshold was relaxed, no coefficient grid was expanded, no seed was removed, and no substitute architecture was trained.

Source: tables/e3_pilot_ab_only.csv SHA256 456d50b14101fcc303a2a6826e84f2100701bdcdced6e52e1f968b5e6f6d9c16; tables/e3_pilot_candidate_audit.csv SHA256 1568ef8480289b9ad4d98f3cefc6b7c9a742e3884f28966e5e3bbd90c2040f7a.
