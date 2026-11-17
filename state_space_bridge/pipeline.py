"""End-to-end assembly of the state-space/retrieval/vector/SRN bridge."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .audit import audit_notebooks, write_historical_audit
from .config import BridgeConfig
from .contracts import assert_corrected_srn_source
from .data import (
    load_corrected_srn_traces,
    load_human_sequence_data,
    make_human_average_traces,
)
from .recovery import run_state_space_recovery
from .retrieval import assemble_retrieval_outputs
from .state_space import fit_model_family


REQUIRED_TABLES = (
    "historical_state_space_audit", "state_space_model_fits",
    "state_space_parameters", "state_space_decoders", "state_space_bic_comparison",
    "state_space_predictions", "state_space_latent_states", "state_space_recovery",
    "frozen_AB_counterfactuals", "retrieval_model_comparison",
    "retrieval_predictions", "retrieval_recovery", "human_vector_AB_profiles",
    "human_low_rank_rank_selection", "human_low_rank_basis",
    "human_low_rank_predictions", "cross_system_bridge_summary",
)


def _progress(enabled, message):
    if enabled:
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"[{stamp}] STATE-SPACE-BRIDGE | {message}", flush=True)


def _fit_traces(traces: pd.DataFrame):
    keys = [
        "agent_type", "dataset", "condition", "counterbalance_group",
        "seed_or_group", "fit_scope",
    ]
    metric_rows, parameter_rows, decoder_rows, prediction_rows, latent_rows = [], [], [], [], []
    lookup = {}
    for identity, trace in traces.groupby(keys, sort=False, dropna=False):
        trace = trace.sort_values("global_step", kind="stable").reset_index(drop=True)
        fits = fit_model_family(trace)
        minimum_bic = min(fit.bic for fit in fits.values())
        winner = min(fits, key=lambda model: fits[model].bic)
        identity_record = dict(zip(keys, identity))
        identity_record["counterbalance"] = identity_record.pop("counterbalance_group")
        lookup[identity] = fits
        for model, fit in fits.items():
            common = {
                **identity_record, "model": model, "BIC": fit.bic,
                "delta_BIC": fit.bic - minimum_bic, "RMSE": fit.rmse, "R2": fit.r2,
                "log_likelihood": fit.log_likelihood, "RSS": fit.rss,
                "n_observations": len(fit.observed), "winner": model == winner,
                "converged": fit.converged, "optimizer_message": fit.optimizer_message,
                "state_space_engine": "state_space_bridge.state_space",
                "canonical_R_source": "state-space_SRN.Rmd",
                "probe_rows_used": 0,
            }
            metric_rows.append(common)
            for name, value in zip(fit.parameter_names, fit.parameters):
                parameter_rows.append({**common, "parameter": name, "value": float(value)})
            for name, value in zip(fit.decoder_names, fit.decoder):
                decoder_rows.append({**common, "decoder_term": name, "value": float(value)})
            for index, source in trace.iterrows():
                row_common = {
                    **identity_record, "model": model,
                    "global_step": int(source.global_step),
                    "block_id": source.block_id,
                    "sequence_in_block": int(source.analysis_sequence_in_block),
                    "context": source.context, "observed": float(source.observed),
                    "observed_sem": float(source.get("observed_sem", np.nan)),
                    "participants": int(source.get("participants", 0)),
                    "BIC": fit.bic, "delta_BIC": fit.bic - minimum_bic,
                    "winner": model == winner,
                }
                prediction_rows.append({
                    **row_common, "predicted": float(fit.predicted[index]),
                    "residual": float(fit.observed[index] - fit.predicted[index]),
                })
                latent_rows.append({
                    **row_common, "x_readout": float(fit.x_readout[index]),
                    "x_A": float(fit.x_a[index]), "x_B": float(fit.x_b[index]),
                    "fatigue_A": np.nan, "fatigue_B": np.nan,
                    "fatigue_model": False,
                })
    fits = pd.DataFrame(metric_rows)
    return {
        "state_space_model_fits": fits,
        "state_space_bic_comparison": fits.copy(),
        "state_space_parameters": pd.DataFrame(parameter_rows),
        "state_space_decoders": pd.DataFrame(decoder_rows),
        "state_space_predictions": pd.DataFrame(prediction_rows),
        "state_space_latent_states": pd.DataFrame(latent_rows),
    }, lookup


def _human_endpoints(human_frames, state_tables, lookup, reliability_fraction):
    fits = state_tables["state_space_model_fits"]
    primary = fits[(fits.agent_type == "human") & (fits.fit_scope == "pre_probe_AB")]
    rows = []
    for dataset, source in human_frames.items():
        probes = source[~source.context.isin(["A", "B"])].copy()
        for cb, group in probes.groupby("counterbalance_group", sort=False):
            choices = primary[(primary.dataset == dataset) & (primary.counterbalance == cb)]
            if choices.empty:
                continue
            winner_row = choices.sort_values("BIC").iloc[0]
            identity = (
                "human", dataset, dataset, cb, "AVERAGE", "pre_probe_AB"
            )
            fit = lookup[identity][winner_row.model]
            a_state, b_state = float(fit.x_a[-1]), float(fit.x_b[-1])
            intercept, slope = map(float, fit.decoder)
            a_prediction = intercept + slope * a_state
            b_prediction = intercept + slope * b_state
            separation = abs(a_prediction - b_prediction)
            threshold = reliability_fraction * float(source.observed.median())
            for row in group.itertuples(index=False):
                raw = str(row.raw_context).replace(" ", "_")
                rows.append({
                    "agent_type": "human", "dataset": dataset,
                    "condition": dataset, "participant_or_seed": row.participant_id,
                    "counterbalance": cb, "probe_context": row.context,
                    "probe_epoch": f"{raw}_block_{row.block_id}",
                    "sequence": int(row.analysis_sequence_in_block),
                    "A_prediction": a_prediction, "B_prediction": b_prediction,
                    "A_state": a_state, "B_state": b_state,
                    "A_fatigue": np.nan, "B_fatigue": np.nan,
                    "endpoint_separation": separation,
                    "reliable_endpoint": separation >= threshold,
                    "reliability_threshold": threshold,
                    "probe_observed": float(row.observed),
                    "selected_state_space_model": winner_row.model,
                    "probe_outcome_used_for_endpoint": False,
                    "probe_driven_state_updates": 0,
                })
    return pd.DataFrame(rows)


def _srn_endpoints(source_dir):
    source = pd.read_csv(Path(source_dir) / "pure_context_counterfactuals.csv")
    source = source[
        source.model_condition.isin(["High", "Low"])
        & source.context.isin(["A", "B"])
    ]
    pivot = source.pivot_table(
        index=["seed", "model_condition", "checkpoint"], columns="context",
        values="normalized_rt", aggfunc="first",
    ).reset_index()
    rows = []
    for row in pivot.itertuples(index=False):
        rows.append({
            "agent_type": "SRN", "dataset": "FTT_SRN",
            "condition": "High_scale_2_execution" if row.model_condition == "High" else "Low_scale_execution",
            "participant_or_seed": row.seed, "counterbalance": "AFirst",
            "probe_context": "frozen_pure_counterfactual", "probe_epoch": row.checkpoint,
            "sequence": np.nan, "A_prediction": float(row.A), "B_prediction": float(row.B),
            "A_state": np.nan, "B_state": np.nan, "A_fatigue": np.nan, "B_fatigue": np.nan,
            "endpoint_separation": abs(float(row.A) - float(row.B)),
            "reliable_endpoint": abs(float(row.A) - float(row.B)) >= .02,
            "reliability_threshold": .02, "probe_observed": np.nan,
            "selected_state_space_model": "not_applicable_raw_frozen_SRN",
            "probe_outcome_used_for_endpoint": False, "probe_driven_state_updates": 0,
        })
    return pd.DataFrame(rows)


def _cross_system_summary(state, retrieval, dimensions, recovery, corrected_record,
                          causal_dir=None, interaction_dir=None):
    fits = state[state.fit_scope.isin(["pre_probe_AB", "all_AB_training"])].copy()
    aggregate = fits.groupby(
        ["agent_type", "condition", "model"], as_index=False
    ).BIC.sum()
    winners = aggregate.loc[aggregate.groupby(
        ["agent_type", "condition"]
    ).BIC.idxmin()].set_index(["agent_type", "condition"]).model.to_dict()
    human = winners.get(("human", "Combined"), "not_available")
    high = winners.get(("SRN", "High_scale_2_execution"), "not_available")
    low = winners.get(("SRN", "Low_scale_execution"), "not_available")
    retrieval_winners = retrieval.loc[
        retrieval.groupby("probe_context").heldout_log_likelihood.idxmax()
    ].set_index("probe_context").model.to_dict()
    selected = dimensions[
        dimensions.selected_primary_rank.fillna(False).astype(bool)
    ]["rank"].astype(int).iloc[0]
    causal_code, causal_text = "not_run", "Corrected causal analysis was not supplied."
    if causal_dir and (Path(causal_dir) / "causal_mechanism_summary.csv").exists():
        causal = pd.read_csv(Path(causal_dir) / "causal_mechanism_summary.csv")
        causal_code = str(causal.main_result_code.iloc[0])
        causal_text = str(causal.main_result_text.iloc[0])
    interaction_code = "not_run"
    if interaction_dir and (Path(interaction_dir) / "interaction_mechanism_summary.csv").exists():
        interaction = pd.read_csv(Path(interaction_dir) / "interaction_mechanism_summary.csv")
        interaction_code = str(interaction.main_result_code.iloc[0])
    human_split = human in {"SS_CROSS", "SS_SPLIT"}
    high_split = high in {"SS_CROSS", "SS_SPLIT"}
    low_lump = low == "SS_LUMP"
    probe_lowrank = any(value == "RET_LOWRANK" for value in retrieval_winners.values())
    causal_supported = causal_code not in {"not_run", "C", "D"}
    if human_split and high_split and low_lump and probe_lowrank and causal_supported:
        conclusion = "A. strong functional + representational convergence"
    elif human_split and high == "SS_LUMP" and causal_code != "not_run":
        conclusion = "B. functional/representational dissociation in the corrected SRN"
    elif human != high and probe_lowrank:
        conclusion = "C. retrieval convergence without matching learning dynamics"
    else:
        conclusion = "D. limited cross-system convergence"
    recovery_summary = recovery.drop_duplicates(["generating_model", "replicate"]).groupby(
        "generating_model"
    ).selected_correctly.mean()
    return pd.DataFrame([{
        "human_combined_state_space_winner": human,
        "corrected_high_state_space_winner": high,
        "low_state_space_winner": low,
        "mixed_retrieval_winner": retrieval_winners.get("Both", "not_available"),
        "novel_retrieval_winner": retrieval_winners.get("Novel", "not_available"),
        "human_low_rank_selected_rank": int(selected),
        "corrected_high_scale": corrected_record.get("high_scale"),
        "execution_sanity_gate": corrected_record.get("execution_sanity_gate"),
        "corrected_causal_result_code": causal_code,
        "corrected_causal_result_text": causal_text,
        "corrected_context_sequence_result_code": interaction_code,
        "state_space_recovery_minimum_accuracy": float(recovery_summary.min()),
        "state_space_recovery_all_models_pass_80pct": bool((recovery_summary >= .80).all()),
        "cross_system_conclusion": conclusion,
        "human_outcomes_used_to_modify_srn": 0,
        "probe_rows_used_for_state_space_learning": 0,
        "rank_two_hard_coded": False,
    }])


def _write_reports(repo, output, summary, tables):
    docs = repo / "docs"
    docs.mkdir(exist_ok=True)
    row = summary.iloc[0]
    (docs / "STATE_SPACE_BRIDGE_README.md").write_text(f"""# State-space bridge

## What is a “state” here?

A state is a compact, dynamically updated estimate of current task performance. The canonical
implementation is the no-fatigue model family in `state-space_SRN.Rmd`: Lump has one state;
Cross and Split have separately gated A/B states. A two-term OLS decoder maps state to observed
sequence RT (human) or normalized RT (SRN).

## Why is Mixed not part of the state-space learning model?

Lump/Cross/Split answer what was learned during pure A/B practice. Mixed, Novel, and No Context
are retrieval tests. They contribute zero rows to state dynamics, decoder fitting, BIC, and model
selection. Their outcomes can therefore be permuted or removed without changing Stage 1.

## Why is low rank a vector extension?

The scalar model gives one A and B value. The transition analysis replaces each value with a
seven-position vector and estimates a smaller basis spanning those vectors. Its selected rank is
read from held-out dimension selection, not fixed in bridge code. The frozen result is rank
**{int(row.human_low_rank_selected_rank)}**.

## Canonical-model decision

The original PRD requested historical fatigue machinery, but the investigator subsequently chose
`state-space_SRN.Rmd` as canonical. Its no-fatigue topology is therefore used exactly; fatigue
columns remain explicit missing values so this deviation cannot be overlooked.
""", encoding="utf-8")
    result_map = pd.DataFrame([
        ["A/B learning architecture", "scalar learning", "scripts/run_state_space_bridge.py", "validated sequence RT / corrected normalized RT", "state_space_bic_comparison.csv", "SSB1/SSB2", "delta BIC", "pure A/B only"],
        ["Probe retrieval", "retrieval", "validated v3 adapter", "frozen A/B profiles", "retrieval_model_comparison.csv", "SSB4", "held-out log likelihood", "no learning update"],
        ["Rank-2 vector state", "vector", "validated v3.1 adapter", "pure A/B transition profiles", "human_low_rank_rank_selection.csv", "SSB5", "held-out rank selection", "rank not hard-coded"],
        ["SRN implementation", "representation/causal", "corrected v2.2/v2.3", "scale-2 frozen checkpoints", "cross_system_bridge_summary.csv", "SSB6", "registered result code", "no human tuning"],
    ], columns=["claim", "analysis_level", "script", "input", "output", "figure", "statistic", "interpretive_boundary"])
    result_map.to_csv(output / "bridge_result_map.csv", index=False)
    (docs / "BRIDGE_RESULT_MAP.md").write_text(
        "# Bridge result map\n\n" + result_map.to_markdown(index=False) + "\n", encoding="utf-8"
    )
    report = f"""# State-space bridge report

## 1. Functional learning architecture

- Human Combined: **{row.human_combined_state_space_winner}**
- Corrected High scale 2 + execution: **{row.corrected_high_state_space_winner}**
- Low scale + execution: **{row.low_state_space_winner}**

These are sequence-trajectory BIC results from the exact no-fatigue models ported from
`state-space_SRN.Rmd`. Probe rows used in these fits: **0**.

## 2. Probe retrieval

- Mixed: **{row.mixed_retrieval_winner}**
- Novel: **{row.novel_retrieval_winner}**

The primary comparison is held-out log likelihood; RMSE and MAE are also exported.

## 3. Vector-state structure

The cleaned, held-out rank-selection regression selected rank **{row.human_low_rank_selected_rank}**.
Rank two is checked against the frozen data but is not an estimator constant.

## 4. SRN implementation

Corrected causal result: **{row.corrected_causal_result_code}** — {row.corrected_causal_result_text}

## 5. Cross-system conclusion

**{row.cross_system_conclusion}**

State-space recovery minimum class accuracy was **{row.state_space_recovery_minimum_accuracy:.1%}**.
If recovery or a corrected SRN gate fails, this report is descriptive rather than inferential.
"""
    (output / "BRIDGE_REPORT.md").write_text(report, encoding="utf-8")


def run_state_space_bridge(
    combined_path, novel_path, corrected_srn_dir, v3_dir, v31_dir, output_dir,
    *, causal_dir=None, interaction_dir=None, config: BridgeConfig | None = None,
):
    config = config or BridgeConfig(); config.validate()
    repo = Path(__file__).resolve().parents[1]
    output = Path(output_dir).expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    _progress(config.show_progress, "Validating corrected scale-2 SRN source and sanity gate")
    corrected_record = assert_corrected_srn_source(corrected_srn_dir)
    v3_record_path = Path(v3_dir).expanduser().resolve() / "human_run_config.json"
    if not v3_record_path.exists():
        raise FileNotFoundError(v3_record_path)
    v3_record = json.loads(v3_record_path.read_text(encoding="utf-8"))
    v3_source = Path(v3_record.get("srn_source_dir", "")).expanduser().resolve()
    if v3_source != Path(corrected_srn_dir).expanduser().resolve():
        raise ValueError(
            "Human v3 retrieval outputs were not generated from the supplied corrected scale-2 SRN."
        )
    _progress(config.show_progress, "Auditing canonical and historical R state-space code")
    audit = audit_notebooks(repo)
    audit.to_csv(output / "historical_state_space_audit.csv", index=False)
    write_historical_audit(repo, audit, repo / "docs/HISTORICAL_STATE_SPACE_AUDIT.md")

    _progress(config.show_progress, "Loading validated sequence RT and constructing pure-A/B traces")
    human_frames = {
        "Combined": load_human_sequence_data(combined_path, "Combined"),
        "Novel": load_human_sequence_data(novel_path, "Novel"),
    }
    human_traces = pd.concat(
        [make_human_average_traces(frame) for frame in human_frames.values()], ignore_index=True
    )
    srn_traces = load_corrected_srn_traces(corrected_srn_dir)
    traces = pd.concat([human_traces, srn_traces], ignore_index=True, sort=False)
    _progress(config.show_progress, f"Fitting one shared R-equivalent engine to {traces.groupby(['agent_type','condition','counterbalance_group','fit_scope']).ngroups} traces")
    state_tables, lookup = _fit_traces(traces)

    _progress(config.show_progress, "Running schedule-matched state-space recovery")
    schedule = srn_traces[srn_traces.condition.eq("High_scale_2_execution")].sort_values(
        "global_step"
    ).context.to_numpy()
    noise = max(float(srn_traces.groupby("global_step").observed.std().median()), .01)
    recovery = run_state_space_recovery(
        schedule, replicates=config.state_space_recovery_replicates,
        noise_sd=noise, random_seed=config.random_seed,
    )
    state_tables["state_space_recovery"] = recovery
    human_endpoints = _human_endpoints(
        human_frames, state_tables, lookup, config.endpoint_reliability_fraction
    )
    state_tables["frozen_AB_counterfactuals"] = pd.concat(
        [human_endpoints, _srn_endpoints(corrected_srn_dir)], ignore_index=True, sort=False
    )

    _progress(config.show_progress, "Assembling frozen held-out retrieval and low-rank outputs")
    retrieval_tables = assemble_retrieval_outputs(
        v3_dir, v31_dir, combined_path=combined_path, novel_path=novel_path,
        participant_limit=12 if config.profile == "smoke" else None,
        require_rank_two=config.profile == "final",
    )
    summary = _cross_system_summary(
        state_tables["state_space_model_fits"],
        retrieval_tables["retrieval_model_comparison"],
        retrieval_tables["human_low_rank_rank_selection"], recovery, corrected_record,
        causal_dir=causal_dir, interaction_dir=interaction_dir,
    )
    tables = {
        "historical_state_space_audit": audit,
        **state_tables, **retrieval_tables,
        "cross_system_bridge_summary": summary,
    }
    _progress(config.show_progress, "Writing bridge tables, reports, and configuration audit")
    for name in REQUIRED_TABLES:
        frame = tables[name]
        if frame.empty:
            raise AssertionError(f"Required bridge table {name} is empty.")
        frame.to_csv(output / f"{name}.csv", index=False)
    record = {
        "analysis_version": "state-space-bridge-1.0.0",
        "profile": config.profile,
        "canonical_state_space_source": "state-space_SRN.Rmd",
        "canonical_state_space_variant": "no-fatigue OLS Lump/Cross/Split",
        "corrected_srn_dir": str(Path(corrected_srn_dir).resolve()),
        "corrected_high_scale": 2.0,
        "execution_sanity_gate": "passed",
        "human_state_space_observable": "sequence mean correct-keypress RT (ms)",
        "srn_state_space_observable": "normalized_rt = 1 - execution_speed",
        "state_space_recovery_replicates_per_model": config.state_space_recovery_replicates,
        "probe_rows_used_for_state_space_fit": 0,
        "mixed_or_novel_state_updates": 0,
        "required_tables": [f"{name}.csv" for name in REQUIRED_TABLES],
        "frozen_previous_outputs_modified": False,
    }
    (output / "bridge_run_config.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_reports(repo, output, summary, tables)
    if config.make_plots:
        _progress(config.show_progress, "Rendering six registered bridge figures")
        from .plotting import make_bridge_figures
        make_bridge_figures(output / "figures", tables)
    _progress(config.show_progress, f"Complete: {output}")
    return tables
