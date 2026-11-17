"""Validated familiar/probe timelines with NoContext naming kept explicit."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from human_configured_state.data.load_human import RAW_COLUMNS,build_token_rows
from retrieval_modeling.data import load_retrieval_data


def _probe_label(raw):
    value=str(raw).strip().lower()
    if "mixed" in value: return "Mixed"
    if "space" in value or "novel" in value or "third" in value: return "Novel"
    if "no context test" in value: return "NoContextProbe"
    if "no context training" in value: return "NoContextTraining"
    return "Familiar"


def load_dataset(path: str|Path,dataset: str):
    contract=load_retrieval_data(path,dataset).copy()
    contract["participant_uid"]=dataset+":"+contract.participant_id.astype(str)
    contract["probe_type"]=contract.raw_context.map(_probe_label)
    first=(contract[contract.retrieval_context.isin(["A","B"])]
           .sort_values(["participant_id","global_sequence"])
           .groupby("participant_id").retrieval_context.first())
    contract["counterbalance_order"]=contract.participant_id.map(first).map({"A":"AFirst","B":"BFirst"})
    contract["probe_phase"]=np.where(contract.raw_context.str.lower().str.contains("training"),"Training",
                              np.where(contract.raw_context.str.lower().str.contains("test"),"Test","Familiar"))
    contract["eligible_probe"]=contract.probe_type.isin(["Mixed","Novel","NoContextProbe"])
    contract["probe_exposure_index"]=pd.Series(pd.NA,index=contract.index,dtype="Int64")
    for (_,probe),indices in contract[contract.eligible_probe].groupby(["participant_uid","probe_type"],sort=False).groups.items():
        ordered=contract.loc[indices].sort_values(["block_id","original_sequence_in_block"],kind="stable")
        contract.loc[ordered.index,"probe_exposure_index"]=np.arange(1,len(ordered)+1)
    contract["familiar_state_checkpoint"]=(contract.retrieval_context.isin(["A","B"])
        .groupby(contract.participant_uid).cumsum()-contract.retrieval_context.isin(["A","B"]).astype(int))

    raw=pd.read_csv(path,usecols=list(RAW_COLUMNS))
    tokens=build_token_rows(raw,contract,dataset=dataset)
    token_keys=["participant_id","block_id","original_sequence_in_block"]
    extras=contract[token_keys+["participant_uid","probe_type","probe_phase","eligible_probe",
        "probe_exposure_index","counterbalance_order","familiar_state_checkpoint","global_sequence"]].drop_duplicates(token_keys)
    tokens=tokens.drop(columns=[c for c in ["participant_uid","probe_type","probe_phase"] if c in tokens])
    tokens=tokens.merge(extras,on=token_keys,how="inner",validate="many_to_one")
    tokens["observed_rt"]=tokens.mean_rt.astype(float)
    return contract.sort_values(["participant_uid","global_sequence"]),tokens.sort_values(
        ["participant_uid","block_order","original_sequence_in_block","transition_position"],kind="stable")


def load_all(combined_path,novel_path):
    combined_contract,combined_tokens=load_dataset(combined_path,"Combined")
    novel_contract,novel_tokens=load_dataset(novel_path,"Novel")
    return (pd.concat([combined_contract,novel_contract],ignore_index=True,sort=False),
            pd.concat([combined_tokens,novel_tokens],ignore_index=True,sort=False))


def probe_timeline(contract):
    columns=["dataset","participant_uid","participant_id","probe_type","probe_exposure_index",
        "probe_phase","block_id","original_sequence_in_block","counterbalance_order",
        "familiar_state_checkpoint","raw_context","eligible_probe"]
    return contract[contract.probe_type.ne("Familiar")][columns].rename(columns={
        "block_id":"block","original_sequence_in_block":"sequence","counterbalance_order":"counterbalance"
    }).sort_values(["dataset","participant_uid","block","sequence"])
