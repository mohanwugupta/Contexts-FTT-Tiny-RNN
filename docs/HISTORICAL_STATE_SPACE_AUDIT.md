# Historical state-space audit

## Canonical model chosen by the investigator

The bridge ports the **no-fatigue Lump/Cross/Split family in `state-space_SRN.Rmd`**. It preserves the notebook's state initialization, gated A/B updates, OLS decoder, bounds, and BIC parameter counts (3/5/4). This explicit instruction supersedes the PRD's earlier request to reconstruct a fatigue-bearing PNAS model.

The canonical Python engine accepts only A/B contexts. The R notebook's Mixed/NoContext branch is historical code and is not ported.

## Historical references

The V4-5 and V5-adjusted notebooks contain reusable examples of latent simulation, fatigue accumulation/recovery, OLS decoding, counterbalance grouping, RSS/RMSE, Gaussian likelihood/BIC, and the three learning topologies. They also contain duplicated and inconsistent ambiguous-context rules, so none of those readouts enter Stage 1.

## Rejected Mixed/Novel logic

| File | Matches | Final status |
|---|---:|---|
| `state-space_SRN.Rmd` | 20 | deprecated; listed row-by-row in `historical_state_space_audit.csv` |
| `state-space-trial-V4-5-generative-fits.Rmd` | 111 | deprecated; listed row-by-row in `historical_state_space_audit.csv` |
| `state-space-trial-V5-adjusted.Rmd` | 92 | deprecated; listed row-by-row in `historical_state_space_audit.csv` |

Rejected patterns include free `w_Mixed`, generic `w_mix`, strict 50/50 rules, coupling-derived retrieval, and shared Mixed/NoContext branches. Probe retrieval is a separate frozen-endpoint model family.

## Data-dependent warning

The historical hard-coded human BIC values are not treated as data. If the cleaned validated cohort changes the historical human selection, the bridge report flags the preprocessing/cohort difference instead of overwriting the earlier interpretation.

Repository audited: `/Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/RI-EVC/Contexts-FTT-Tiny-RNN`
