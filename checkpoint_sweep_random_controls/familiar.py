"""Familiar-only checkpoint measurements, transfer indices, and references."""

from __future__ import annotations

from itertools import product
import numpy as np
import pandas as pd

from confirmatory_srn.state_space import fit_confirmatory_state_space
from human_configured_state.data.load_human import load_human_token_data
from human_context_control_transfer.engine import probe_run
from human_context_control_transfer.h8 import probe_h8


def _columns(frame, prefix):
    return sorted((c for c in frame if c.startswith(prefix)), key=lambda c: int(c.rsplit("_", 1)[1]))


def paired_signflip_null(a, b, quantile=.95):
    """Exact paired null for coherent A/B vector separation across transitions."""
    differences = np.asarray(a, float) - np.asarray(b, float)
    observed = float(np.linalg.norm(differences.mean(0)))
    distribution = []
    for signs in product((-1., 1.), repeat=len(differences)):
        distribution.append(float(np.linalg.norm((differences * np.asarray(signs)[:, None]).mean(0))))
    return observed, float(np.quantile(distribution, quantile)), float(np.linalg.norm(differences, axis=1).mean())


def familiar_probes(config, registry, runs):
    sequence = tuple(config.payload["human_training"]["token_sequence"]); parts = []
    for run in runs:
        for checkpoint in config.checkpoints:
            if run.architecture == "H8":
                frame = probe_h8(run, registry, checkpoint, f"FamiliarSweep{checkpoint}", ("A", "B"), sequence, role="familiar_selection")
            else:
                frame = probe_run(run, registry, checkpoint, f"FamiliarSweep{checkpoint}", ("A", "B"), sequence, role="familiar_selection")
            frame["analysis_type"] = "familiar_only_selection"; parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def summarize_familiar(config, probes):
    transition_rows, summaries, hidden_rows, controller_rows = [], [], [], []
    all_h = _columns(probes, "hidden_state_"); all_c = _columns(probes, "context_state_")
    keys = ["group", "architecture", "prior_condition", "seed", "checkpoint"]
    for key, one in probes.groupby(keys, sort=True):
        a = one[one.context == "A"].sort_values("transition_position"); b = one[one.context == "B"].sort_values("transition_position")
        hcols = [c for c in all_h if a[c].notna().all() and b[c].notna().all()]
        ccols = [c for c in all_c if a[c].notna().all() and b[c].notna().all()]
        observed, null, mean_distance = paired_signflip_null(a[hcols], b[hcols], float(config.payload["selection"]["null_quantile"]))
        hidden_rows.append(dict(zip(keys, key)) | {"coherent_AB_hidden_distance": observed, "paired_signflip_null_95": null,
            "mean_token_matched_AB_hidden_distance": mean_distance, "hidden_separation_exceeds_null": observed > null,
            "null_method": "exact_2^7_token_matched_sign_flip"})
        if ccols:
            c_observed, c_null, c_mean = paired_signflip_null(a[ccols], b[ccols], float(config.payload["selection"]["null_quantile"]))
            controller_rows.append(dict(zip(keys, key)) | {"coherent_AB_controller_distance": c_observed, "paired_signflip_null_95": c_null,
                "mean_token_matched_AB_controller_distance": c_mean, "controller_separation_exceeds_null": c_observed > c_null,
                "null_method": "exact_2^7_token_matched_sign_flip"})
        for context_frame in (a, b):
            for _, row in context_frame.iterrows():
                transition_rows.append(dict(zip(keys, key)) | {"row_type": "transition", "context": row.context,
                    "transition_position": int(row.transition_position), "correct_token_nll": row.correct_token_nll,
                    "token_correct": row.token_correct, "correct_token_probability": row.correct_token_probability,
                    "normalized_RT": row.normalized_RT})
        art = a.normalized_RT.to_numpy(float); brt = b.normalized_RT.to_numpy(float); difference = art - brt
        corr = float(np.corrcoef(one.correct_token_probability, one.normalized_RT)[0,1]) if one.correct_token_probability.std() > 1e-12 else np.nan
        summaries.append(dict(zip(keys, key)) | {"row_type": "seed_summary", "context": "A_B", "transition_position": np.nan,
            "familiar_NLL": one.correct_token_nll.mean(), "familiar_accuracy": one.token_correct.mean(),
            "familiar_confidence": one.correct_token_probability.mean(), "familiar_normalized_RT": one.normalized_RT.mean(),
            "A_mean_RT": art.mean(), "B_mean_RT": brt.mean(), "AB_RT_profile_RMSE": float(np.sqrt(np.mean(difference**2))),
            "AB_RT_mean_absolute_difference": float(np.mean(np.abs(difference))), "AB_RT_profile_L2": float(np.linalg.norm(difference)),
            "confidence_RT_correlation": corr, "coherent_AB_hidden_distance": observed,
            "hidden_separation_exceeds_null": observed > null})
    metrics = pd.concat([pd.DataFrame(transition_rows), pd.DataFrame(summaries)], ignore_index=True, sort=False)
    return metrics, pd.DataFrame(hidden_rows), pd.DataFrame(controller_rows)


def _settled(frame, indices, column):
    values = frame[frame.training_sequence.isin(indices)][column]
    return float(values.mean()) if len(values) else np.nan


def transfer_indices(config, runs):
    rows = []; epsilon = 1e-12
    metric_specs = (("RT", "mean_normalized_RT", 1.), ("NLL", "mean_prediction_NLL", 1.), ("confidence", "mean_correct_token_probability", -1.))
    for run in runs:
        trace = run.training.sort_values("training_sequence")
        for checkpoint in config.checkpoints:
            prefix = trace[trace.training_sequence <= checkpoint]; switches = prefix[prefix.is_context_switch]
            base = {"group":run.group,"architecture":run.architecture,"prior_condition":run.prior_condition,"seed":run.seed,"checkpoint":checkpoint}
            for label, column, direction in metric_specs:
                initial_costs = {}
                for context, first, settled in (("A",1,range(8,11)), ("B",11,range(18,21))):
                    first_values = trace.loc[trace.training_sequence.eq(first), column]
                    settled_value = _settled(trace, settled, column) if checkpoint >= max(settled) else np.nan
                    initial_costs[context] = direction * (float(first_values.iloc[0]) - settled_value) if len(first_values) and np.isfinite(settled_value) else np.nan
                costs, denominators, valid_costs, valid_denominators = [], [], [], []
                for _, switch in switches.iterrows():
                    index = int(switch.training_sequence); previous = trace[(trace.training_sequence < index) & (trace.training_sequence >= index-3)][column].mean()
                    cost = direction * (float(switch[column]) - float(previous)); denominator = initial_costs[str(switch.context)]
                    costs.append(cost);denominators.append(denominator)
                    if np.isfinite(denominator) and abs(denominator) > epsilon: valid_costs.append(cost);valid_denominators.append(denominator)
                cost = float(np.mean(costs)) if costs else np.nan; denominator = float(np.mean(valid_denominators)) if valid_denominators else np.nan
                transfer = float(1 - np.mean(valid_costs) / denominator) if valid_costs and np.isfinite(denominator) and abs(denominator)>epsilon else np.nan
                rows.append(base | {"metric":label,"switch_count":len(switches),"switch_cost":cost,"initial_learning_cost":denominator,
                                    "cross_context_transfer_index":transfer,"raw_switch_costs":";".join(map(str,costs)),
                                    "raw_initial_costs":";".join(map(str,denominators)),"transfer_direction":"1=complete_transfer_0=no_transfer"})
    return pd.DataFrame(rows)


def checkpoint_state_space(config, training, progress=None):
    frames = []
    minimum = int(config.payload["selection"]["minimum_checkpoint_for_state_space"])
    for checkpoint in config.checkpoints:
        if checkpoint < minimum:
            for (group, seed), _ in training.groupby(["group","seed"]):
                frames.append(pd.DataFrame([{"condition":group,"seed":seed,"checkpoint":checkpoint,"fit_scope":"seed","fit_status":"insufficient_observations"}]))
            continue
        prefix = training[training.training_sequence <= checkpoint].copy()
        try:
            fits, _, _ = fit_confirmatory_state_space(prefix)
            fits["checkpoint"] = checkpoint; fits["fit_status"] = "fit"; frames.append(fits)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            for (group, seed), _ in prefix.groupby(["group","seed"]):
                frames.append(pd.DataFrame([{"condition":group,"seed":seed,"checkpoint":checkpoint,"fit_scope":"seed","fit_status":"failed","failure_reason":str(error)}]))
        if progress: progress(f"state-space familiar fits checkpoint {checkpoint}")
    result = pd.concat(frames, ignore_index=True, sort=False).rename(columns={"condition":"group"})
    fit=result[result.fit_status.eq("fit")].copy()
    if len(fit):
        index=["group","fit_scope","seed","checkpoint"]
        pivot=fit.pivot_table(index=index,columns="model_label",values="BIC",aggfunc="first").reset_index()
        winners=fit[fit.best_model==True][index+["model_label"]].rename(columns={"model_label":"state_space_winner"})
        evidence=pivot.merge(winners,on=index,how="left")
        evidence["Cross_vs_Lump_evidence"]=evidence["Lump"]-evidence["Cross"]
        evidence["Split_vs_Lump_evidence"]=evidence["Lump"]-evidence["Split"]
        result=result.merge(evidence[index+["state_space_winner","Cross_vs_Lump_evidence","Split_vs_Lump_evidence"]],on=index,how="left")
    return result


def h8_and_human_reference(config, familiar_metrics, transfer):
    h8 = familiar_metrics.query("row_type == 'seed_summary' and group == 'H8'").copy(); h8["reference_type"] = "H8"
    combined = load_human_token_data(config.combined_data, "Combined", target_probe="Both")
    familiar = combined[combined.context.isin(["A","B"])].copy()
    profile = familiar.groupby(["context","transition_position"]).log_rt.mean().reset_index()
    a = profile[profile.context=="A"].sort_values("transition_position").log_rt.to_numpy(); b = profile[profile.context=="B"].sort_values("transition_position").log_rt.to_numpy()
    sequence=familiar.groupby(["participant_uid","block_order","sequence_repetition","context"],as_index=False).log_rt.mean()
    human_switch_costs=[];human_transfer=[]
    for _,person in sequence.groupby("participant_uid"):
        blocks=[]
        for block,one in person.groupby("block_order",sort=True):
            ordered=one.sort_values("sequence_repetition");blocks.append((block,str(ordered.context.iloc[0]),ordered))
        initial={}
        for context in ("A","B"):
            candidates=[one for _,label,one in blocks if label==context]
            if candidates:
                one=candidates[0];initial[context]=float(one.log_rt.iloc[0]-one.tail(3).log_rt.mean())
        for (_,previous_context,previous),(_,context,current) in zip(blocks[:-1],blocks[1:]):
            if context==previous_context:continue
            cost=float(current.log_rt.iloc[0]-previous.tail(3).log_rt.mean());human_switch_costs.append(cost)
            denominator=initial.get(context,np.nan)
            if np.isfinite(denominator) and abs(denominator)>1e-12:human_transfer.append(1-cost/denominator)
    human_row = {"group":"Human","architecture":"Human","prior_condition":"observed","seed":np.nan,"checkpoint":np.nan,
        "row_type":"reference_summary","reference_type":"human_familiar","AB_RT_profile_RMSE":float(np.sqrt(np.mean((a-b)**2))),
        "AB_RT_mean_absolute_difference":float(np.mean(np.abs(a-b))),"AB_RT_profile_L2":float(np.linalg.norm(a-b)),
        "switch_cost_RT":float(np.mean(human_switch_costs)) if human_switch_costs else np.nan,
        "cross_context_transfer_index_RT":float(np.mean(human_transfer)) if human_transfer else np.nan,
        "human_A_profile":";".join(map(str,a)),"human_B_profile":";".join(map(str,b))}
    return pd.concat([h8, pd.DataFrame([human_row])], ignore_index=True, sort=False)
