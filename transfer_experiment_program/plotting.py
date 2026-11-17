"""Standalone figures; every panel writes the exact plotted rows."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"A": "#286c8e", "B": "#ca6b35", "Mixed": "#7856a8", "Novel": "#4e8b57",
          "High_monotonic": "#a94747", "Low_monotonic": "#397ca8", "P0": "#777777",
          "P1": "#3c8b68", "P2": "#c47738"}


def _save(fig, directory: Path, stem: str, source: pd.DataFrame) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    source.to_csv(directory / f"{stem}_source.csv", index=False)
    plt.close(fig)


def figure_h0(human: pd.DataFrame, references: pd.DataFrame, srn: pd.DataFrame, directory: Path):
    participants = human.participant_uid.drop_duplicates().head(8)
    observed = human[human.participant_uid.isin(participants) & human.existing_cleaning_included].copy()
    local = references[references.participant_uid.isin(participants)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for uid, one in observed.groupby("participant_uid"):
        axes[0].plot(one.absolute_experimental_sequence_order, one.mean_execution_RT_ms,
                     color="#a9b4be", alpha=.5, lw=.8)
    for context in ("Mixed", "Novel"):
        one = observed[observed.context.eq(context)]
        axes[0].scatter(one.absolute_experimental_sequence_order, one.mean_execution_RT_ms,
                        s=12, alpha=.65, color=COLORS[context], label=context)
    axes[0].set(title="Human sequences in source order", xlabel="Experimental sequence", ylabel="Execution RT (ms)")
    axes[0].legend(frameon=False)
    if len(srn):
        mean = srn.groupby(["condition", "parent_checkpoint", "context"], as_index=False).mean(numeric_only=True)
        for (condition, context), one in mean.groupby(["condition", "context"]):
            axes[1].plot(one.parent_checkpoint, one.mean_normalized_RT,
                         color=COLORS.get(condition), ls="-" if context in {"A", "B"} else "--",
                         alpha=.8, label=f"{condition.replace('_monotonic','')} {context}")
    axes[1].set(title="Reference SRN same-checkpoint probes", xlabel="Completed A/B sequences", ylabel="Normalized RT")
    axes[1].legend(frameon=False, ncol=2, fontsize=7)
    fig.suptitle("Measurement audit: chronology and time-matched reference")
    _save(fig, directory, "figure_01_reference_human_measurement_audit",
          pd.concat([observed.assign(figure_panel="human"), local.assign(figure_panel="local_reference"),
                     srn.assign(figure_panel="SRN")], ignore_index=True, sort=False))


def figure_e1(sequences: pd.DataFrame, matches: pd.DataFrame, directory: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    means = sequences.groupby(["condition", "parent_checkpoint", "context"], as_index=False).mean(numeric_only=True)
    for (condition, context), one in means.groupby(["condition", "context"]):
        if context not in {"Mixed", "Novel"}: continue
        axes[0].plot(one.parent_checkpoint, one.mean_prediction_NLL, color=COLORS[condition],
                     ls="-" if context == "Mixed" else "--", label=f"{condition[:4]} {context}")
        axes[1].plot(one.parent_checkpoint, one.mean_normalized_RT, color=COLORS[condition],
                     ls="-" if context == "Mixed" else "--", label=f"{condition[:4]} {context}")
    axes[0].set(title="Prediction transfer", xlabel="Completed A/B sequences", ylabel="Mean NLL")
    axes[1].set(title="Expressed execution", xlabel="Completed A/B sequences", ylabel="Normalized RT")
    for ax in axes: ax.legend(frameon=False, fontsize=8)
    support = matches[matches.selected].copy() if len(matches) else matches
    fig.suptitle("E1 acquisition and transfer; strict matches remain explicit in source")
    _save(fig, directory, "figure_02_acquisition_versus_transfer",
          pd.concat([means.assign(figure_layer="trajectory"), support.assign(figure_layer="strict_match")],
                    ignore_index=True, sort=False))


def figure_differentiation(contrasts: pd.DataFrame, splitting: pd.DataFrame, directory: Path):
    merged = contrasts.merge(splitting, on=["experiment", "cohort", "condition", "seed", "parent_checkpoint"], how="left")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for probe in ("Mixed", "Novel"):
        one = merged[merged.probe_context.eq(probe)]
        axes[0].scatter(one.absolute_AB_RT_gap, one.RT_deficit_over_worse, s=12, alpha=.45,
                        color=COLORS[probe], label=probe)
        axes[1].scatter(one.scale_normalized_AB_hidden_distance, one.NLL_deficit_over_worse,
                        s=12, alpha=.45, color=COLORS[probe], label=probe)
    axes[0].axhline(0,color="#555",lw=.8); axes[1].axhline(0,color="#555",lw=.8)
    axes[0].set(xlabel="Absolute A/B RT gap", ylabel="Probe RT deficit over worse endpoint")
    axes[1].set(xlabel="Scale-normalized A/B hidden distance", ylabel="Probe NLL deficit over worse endpoint")
    for ax in axes: ax.legend(frameon=False)
    fig.suptitle("E1 transfer and differentiation are reported jointly")
    _save(fig, directory, "figure_03_transfer_and_differentiation", merged)


def figure_e2(outcomes: pd.DataFrame, directory: Path):
    summary = outcomes.groupby(["probe_family", "primer", "weight_policy", "state_policy"], as_index=False).mean(numeric_only=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    x = np.arange(len(summary))
    labels = summary.weight_policy.str[0].str.upper()+"/"+summary.state_policy.str[0].str.upper()+"/"+summary.primer
    axes[0].scatter(x, summary.RT_change_first_to_late, c=summary.probe_family.map(COLORS), s=38)
    axes[1].scatter(x, summary.NLL_change_first_to_late, c=summary.probe_family.map(COLORS), s=38)
    for ax, title, ylabel in zip(axes, ("RT", "Prediction NLL"),
                                ("Late minus first normalized RT", "Late minus first mean NLL")):
        ax.axhline(0,color="#555",lw=.8); ax.set_xticks(x,labels,rotation=60,ha="right",fontsize=7)
        ax.set(title=title,ylabel=ylabel)
    fig.suptitle("E2 probe experience: weight/state policy and primer")
    _save(fig, directory, "figure_04_probe_experience", summary)


def figure_e3(effects: pd.DataFrame, sequences: pd.DataFrame, directory: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    mean = effects.groupby(["condition", "probe_family", "anchor"], as_index=False).mean(numeric_only=True)
    for family in ("Mixed", "Novel"):
        one=mean[mean.probe_family.eq(family)]
        axes[0].scatter(one.interaction_NLL, one.interaction_RT, color=COLORS[family], label=family)
    grid=sequences[sequences.evaluation_role.eq("prespecified_generalization_grid")]
    grid=grid.groupby(["condition","gradient_channel","lambda"],as_index=False).mean(numeric_only=True)
    for (condition,channel),one in grid.groupby(["condition","gradient_channel"]):
        axes[1].plot(one["lambda"],one.mean_prediction_NLL,color=COLORS[condition],
                     ls="-" if channel=="cue" else "--",label=f"{condition[:4]} {channel}")
    axes[0].axhline(0,color="#555",lw=.7);axes[0].axvline(0,color="#555",lw=.7)
    axes[0].set(xlabel="Cue × layout interaction (NLL)",ylabel="Cue × layout interaction (RT)")
    axes[1].set(xlabel="A→B input-space λ",ylabel="Mean prediction NLL")
    for ax in axes:ax.legend(frameon=False,fontsize=8)
    fig.suptitle("E3 cue/layout decomposition and fixed generalization diagnostics")
    _save(fig,directory,"figure_05_cue_decomposition",
          pd.concat([mean.assign(figure_layer="factorial"),grid.assign(figure_layer="grid")],ignore_index=True,sort=False))


def figure_e4(info: pd.DataFrame, power: pd.DataFrame, screenshots: list[Path], directory: Path):
    fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    for index,visibility in enumerate(("full_preview","one_target")):
        path=next((p for p in screenshots if visibility in p.name and p.exists()),None)
        if path is not None:
            axes[index].imshow(plt.imread(path)); axes[index].axis("off")
        else:
            axes[index].text(.5,.55,"FULL 8-TARGET PREVIEW" if index==0 else "CURRENT TARGET ONLY",
                             ha="center",va="center",fontsize=14,weight="bold")
            axes[index].text(.5,.4,"Context cue and layout remain visible",ha="center",va="center")
            axes[index].set_xticks([]);axes[index].set_yticks([])
        axes[index].set_title(visibility.replace("_"," ").title())
    fig.suptitle("E4 action-visibility task (engineering evidence; not participant outcomes)")
    _save(fig,directory,"figure_06_action_visibility",info)
    p4=power[power.experiment.eq("e4") & power.standardized_interaction_effect.gt(0)]
    fig,ax=plt.subplots(figsize=(7,4.3))
    for effect,one in p4.groupby("standardized_interaction_effect"):
        ax.plot(one.total_N,one.rejection_probability,marker="o",label=f"{effect:g} residual SD")
    ax.set(xlabel="Total randomized N",ylabel="Simulated rejection probability",ylim=(0,1),
           title="E4 prospective power planning (synthetic)");ax.legend(frameon=False)
    _save(fig,directory,"figure_06_power_planning",p4)


def figure_e5(contrasts: pd.DataFrame, pairing: pd.DataFrame, directory: Path):
    fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    end=contrasts[contrasts.parent_checkpoint.eq(170)]
    mean=end.groupby(["arm","probe_context"],as_index=False).mean(numeric_only=True)
    for arm,one in mean.groupby("arm"):
        axes[0].plot(one.probe_context,one.NLL_deficit_over_worse,marker="o",color=COLORS[arm],label=arm)
        axes[1].plot(one.probe_context,one.RT_deficit_over_worse,marker="o",color=COLORS[arm],label=arm)
    axes[0].axhline(0,color="#555",lw=.8);axes[1].axhline(0,color="#555",lw=.8)
    axes[0].set(ylabel="NLL deficit over worse A/B",title="Prediction")
    axes[1].set(ylabel="RT deficit over worse A/B",title="Execution readout")
    for ax in axes:ax.legend(frameon=False)
    fig.suptitle("E5 prior structural experience at target checkpoint 170")
    _save(fig,directory,"figure_07_prior_experience",
          pd.concat([mean.assign(figure_layer="outcome"),pairing.assign(figure_layer="pairing")],ignore_index=True,sort=False))


def figure_synthesis(status: pd.DataFrame, directory: Path):
    fig,ax=plt.subplots(figsize=(12,4.8));ax.axis("off")
    columns=["work_package","question","implementation_status","evidence_status","scientific_interpretation_limit"]
    shown=status[columns].fillna("")
    table=ax.table(cellText=shown.values,colLabels=[x.replace("_"," ") for x in columns],loc="center",cellLoc="left")
    table.auto_set_font_size(False);table.set_fontsize(7);table.scale(1,1.6)
    ax.set_title("Transfer experiment program: evidence synthesis",pad=18)
    _save(fig,directory,"figure_08_evidence_synthesis",status)

