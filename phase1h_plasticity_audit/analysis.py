"""Shared/private decompositions, subspaces, update projections, and LOSO decoders."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HIGH, LOW, AuditConfig


HIDDEN_COLUMNS=tuple(f"hidden_state_{i}" for i in range(1,9))
TASK_COLUMNS=tuple(f"centered_logit_{i}" for i in range(1,5))


def assert_familiar_only(frame):
    values=set(frame.evaluation_context.unique())
    if values != {"A","B"}: raise AssertionError("FAIL: NONFAMILIAR_SUBSPACE_INPUT")
    return True


def endpoint_decomposition(a,b):
    shared=.5*(a+b); private=.5*(a-b)
    if not np.allclose(shared+private,a,atol=1e-10) or not np.allclose(shared-private,b,atol=1e-10):
        raise AssertionError("FAIL: ENDPOINT_DECOMPOSITION_IDENTITY")
    return shared,private


def _state_lookup(states):
    lookup={}
    for keys,group in states.groupby(["seed","scale_condition","training_sequence_index","evaluation_context"],sort=False):
        group=group.sort_values("transition")
        lookup[keys]={"hidden":group[list(HIDDEN_COLUMNS)].to_numpy(float),
                      "task":group[list(TASK_COLUMNS)].to_numpy(float),
                      "loss":float(group.raw_sequence_loss.iloc[0]),
                      "checkpoint_hash":group.checkpoint_hash.iloc[0],
                      "training_context":group.training_context.iloc[0],
                      "training_schedule_hash":group.training_schedule_hash.iloc[0],
                      "model_config_hash":group.model_config_hash.iloc[0]}
    return lookup


def shared_private_states(states,config):
    assert_familiar_only(states); lookup=_state_lookup(states); rows=[]
    for seed in config.seeds:
        for condition in config.conditions:
            previous=None
            for t in range(config.training_sequences+1):
                a=lookup[(seed,condition,t,"A")];b=lookup[(seed,condition,t,"B")]
                hs,hd=endpoint_decomposition(a["hidden"],b["hidden"])
                zs,zd=endpoint_decomposition(a["task"],b["task"])
                for k in range(config.transitions):
                    shared_change=private_change=task_shared_change=task_private_change=np.nan
                    if previous is not None:
                        shared_change=float(np.linalg.norm(hs[k]-previous[0][k]));private_change=float(np.linalg.norm(hd[k]-previous[1][k]))
                        task_shared_change=float(np.linalg.norm(zs[k]-previous[2][k]));task_private_change=float(np.linalg.norm(zd[k]-previous[3][k]))
                    rows.append({"seed":seed,"scale_condition":condition,"training_sequence_index":t,
                                 "training_context":a["training_context"],"transition":k+1,
                                 "checkpoint_hash":a["checkpoint_hash"],
                                 "training_schedule_hash":a["training_schedule_hash"],"model_config_hash":a["model_config_hash"],
                                 "shared_state_norm":float(np.linalg.norm(hs[k])),
                                 "private_difference_norm":float(np.linalg.norm(hd[k])),
                                 "AB_distance":float(np.linalg.norm(a["hidden"][k]-b["hidden"][k])),
                                 "shared_state_change":shared_change,"private_state_change":private_change,
                                 "shared_change_squared":shared_change**2 if t else np.nan,
                                 "private_change_squared":private_change**2 if t else np.nan,
                                 "private_shared_ratio":private_change**2/(shared_change**2+config.epsilon) if t else np.nan,
                                 "task_shared_norm":float(np.linalg.norm(zs[k])),"task_private_norm":float(np.linalg.norm(zd[k])),
                                 "task_AB_separation":float(np.linalg.norm(a["task"][k]-b["task"][k])),
                                 "task_shared_change":task_shared_change,"task_private_change":task_private_change,
                                 "task_shared_change_squared":task_shared_change**2 if t else np.nan,
                                 "task_private_change_squared":task_private_change**2 if t else np.nan,
                                 "task_private_shared_ratio":task_private_change**2/(task_shared_change**2+config.epsilon) if t else np.nan,
                                 **{f"shared_hidden_{j+1}":hs[k,j] for j in range(8)},
                                 **{f"private_hidden_{j+1}":hd[k,j] for j in range(8)},
                                 **{f"shared_task_{j+1}":zs[k,j] for j in range(4)},
                                 **{f"private_task_{j+1}":zd[k,j] for j in range(4)}})
                previous=(hs,hd,zs,zd)
    result=pd.DataFrame(rows)
    for (seed,condition,transition),group in result.groupby(["seed","scale_condition","transition"]):
        ordered=group.sort_values("training_sequence_index").training_sequence_index.to_numpy()
        if not np.array_equal(ordered,np.arange(config.training_sequences+1)):
            raise AssertionError("FAIL: TEMPORAL_UPDATE_NOT_ADJACENT")
    return result,lookup


def _bases(shared,difference,rank,epsilon=1e-12):
    _,_,v_shared=np.linalg.svd(shared,full_matrices=False);_,_,v_private=np.linalg.svd(difference,full_matrices=False)
    us=v_shared[:rank].T;uc=v_private[:rank].T
    ps=us@us.T
    residual=difference@(np.eye(shared.shape[1])-ps)
    _,singular,v_orth=np.linalg.svd(residual,full_matrices=False)
    effective=min(rank,int(np.sum(singular>epsilon)))
    uco=v_orth[:effective].T if effective else np.zeros((shared.shape[1],0))
    pc=uco@uco.T
    cosines=np.linalg.svd(us.T@uc,compute_uv=False);angles=np.degrees(np.arccos(np.clip(cosines,-1,1)))
    return us,uc,ps,pc,angles,effective


def subspaces_and_updates(states,config):
    assert_familiar_only(states); lookup=_state_lookup(states); subspace=[];updates=[]
    for seed in config.seeds:
        for condition in config.conditions:
            for t in range(config.training_sequences+1):
                a=lookup[(seed,condition,t,"A")];b=lookup[(seed,condition,t,"B")]
                shared=.5*(a["hidden"]+b["hidden"]);difference=a["hidden"]-b["hidden"]
                primary=None
                for rank in config.subspace_ranks:
                    us,uc,ps,pc,angles,effective=_bases(shared,difference,rank,config.epsilon)
                    shared_reconstruction=float(np.sum((shared@ps)**2)/(np.sum(shared**2)+config.epsilon))
                    private_reconstruction=float(np.sum((difference@(uc@uc.T))**2)/(np.sum(difference**2)+config.epsilon))
                    subspace.append({"seed":seed,"scale_condition":condition,"training_sequence_index":t,
                                     "training_context":a["training_context"],"transition":0,
                                     "checkpoint_hash":a["checkpoint_hash"],"training_schedule_hash":a["training_schedule_hash"],
                                     "model_config_hash":a["model_config_hash"],"rank":rank,
                                     "orthogonal_private_effective_rank":effective,
                                     "mean_principal_angle_degrees":float(np.mean(angles)),
                                     "minimum_principal_angle_degrees":float(np.min(angles)),
                                     "shared_state_energy_captured":shared_reconstruction,
                                     "private_state_energy_captured":private_reconstruction})
                    if rank==config.primary_subspace_rank: primary=(ps,pc)
                if t==config.training_sequences: continue
                ps,pc=primary
                next_a=lookup[(seed,condition,t+1,"A")];next_b=lookup[(seed,condition,t+1,"B")]
                actual_context=next_a["training_context"]
                for evaluated,current,nxt in (("A",a,next_a),("B",b,next_b)):
                    delta=nxt["hidden"]-current["hidden"]
                    for k in range(config.transitions):
                        total=float(delta[k]@delta[k]);shared_energy=float((ps@delta[k])@(ps@delta[k]))
                        private_energy=float((pc@delta[k])@(pc@delta[k]))
                        updates.append({"seed":seed,"scale_condition":condition,"training_sequence_index":t+1,
                                        "training_context":actual_context,"transition":k+1,"evaluation_context":evaluated,
                                        "checkpoint_hash":nxt["checkpoint_hash"],"training_schedule_hash":a["training_schedule_hash"],
                                        "model_config_hash":a["model_config_hash"],"subspace_rank":config.primary_subspace_rank,
                                        "total_update_energy":total,"shared_update_energy":shared_energy,
                                        "private_update_energy":private_energy,
                                        "shared_update_energy_fraction":shared_energy/(total+config.epsilon),
                                        "private_update_energy_fraction":private_energy/(total+config.epsilon),
                                        "residual_update_energy_fraction":max(0.,1-(shared_energy+private_energy)/(total+config.epsilon)),
                                        "state_change_norm":float(np.linalg.norm(delta[k])),
                                        "is_within_training_context":evaluated==actual_context})
    return pd.DataFrame(subspace),pd.DataFrame(updates)


def _ridge_predict(train_x,train_y,test_x,n_classes,ridge):
    x=np.column_stack([np.ones(len(train_x)),train_x]);xt=np.column_stack([np.ones(len(test_x)),test_x])
    y=np.eye(n_classes)[np.asarray(train_y,int)]
    penalty=np.eye(x.shape[1])*ridge;penalty[0,0]=0
    weights=np.linalg.solve(x.T@x+penalty,x.T@y)
    return (xt@weights).argmax(axis=1)


def decoder_results(shared_private,config):
    rows=[]
    shared_cols=[f"shared_hidden_{i}" for i in range(1,9)];private_cols=[f"private_hidden_{i}" for i in range(1,9)]
    selected=shared_private[shared_private.training_sequence_index.isin(config.diagnostic_checkpoints)]
    for condition in config.conditions:
        for t in config.diagnostic_checkpoints:
            frame=selected[(selected.scale_condition==condition)&(selected.training_sequence_index==t)]
            for held_out in config.seeds:
                train=frame[frame.seed!=held_out];test=frame[frame.seed==held_out]
                if train.empty or test.empty: raise AssertionError("Leave-seed-out decoding requires at least two seeds.")
                target_train=train.transition.to_numpy()-1;target_test=test.transition.to_numpy()-1
                pred_s=_ridge_predict(train[shared_cols].to_numpy(),target_train,test[shared_cols].to_numpy(),7,config.ridge)
                pred_p=_ridge_predict(train[private_cols].to_numpy(),target_train,test[private_cols].to_numpy(),7,config.ridge)
                # Private context contribution is +d for A and -d for B.
                train_private=train[private_cols].to_numpy();test_private=test[private_cols].to_numpy()
                context_train_x=np.vstack([train_private,-train_private]);context_train_y=np.r_[np.ones(len(train),int),np.zeros(len(train),int)]
                context_test_x=np.vstack([test_private,-test_private]);context_test_y=np.r_[np.ones(len(test),int),np.zeros(len(test),int)]
                pred_c=_ridge_predict(context_train_x,context_train_y,context_test_x,2,config.ridge)
                source=test.iloc[0]
                rows.append({"seed":held_out,"scale_condition":condition,"training_sequence_index":t,
                             "training_context":source.training_context,"transition":0,
                             "checkpoint_hash":source.checkpoint_hash,"training_schedule_hash":source.training_schedule_hash,
                             "model_config_hash":source.model_config_hash,
                             "sequence_position_accuracy_shared":float(np.mean(pred_s==target_test)),
                             "sequence_position_accuracy_private":float(np.mean(pred_p==target_test)),
                             "context_accuracy_private":float(np.mean(pred_c==context_test_y)),
                             "cross_validation":"leave_one_seed_out","held_out_seed":held_out})
    return pd.DataFrame(rows)


def attach_actual_state_changes(functional,updates):
    actual=functional[functional.analysis_type.eq("actual_training_update")].copy()
    means=updates.groupby(["seed","scale_condition","training_sequence_index","is_within_training_context"],as_index=False).state_change_norm.mean()
    within=means[means.is_within_training_context][["seed","scale_condition","training_sequence_index","state_change_norm"]].rename(columns={"state_change_norm":"within_context_state_change"})
    cross=means[~means.is_within_training_context][["seed","scale_condition","training_sequence_index","state_change_norm"]].rename(columns={"state_change_norm":"cross_context_state_change"})
    actual=actual.drop(columns=["within_context_state_change","cross_context_state_change"]).merge(within,on=["seed","scale_condition","training_sequence_index"]).merge(cross,on=["seed","scale_condition","training_sequence_index"])
    other=functional[~functional.analysis_type.eq("actual_training_update")]
    return pd.concat([actual,other],ignore_index=True,sort=False)


def seed_summary(shared_private,updates,gradients,functional,decoders,config):
    rows=[]
    actual=functional[functional.analysis_type.eq("actual_training_update")]
    for seed in config.seeds:
        for condition in config.conditions:
            sp=shared_private[(shared_private.seed==seed)&(shared_private.scale_condition==condition)]
            change=sp[sp.training_sequence_index>0];up=updates[(updates.seed==seed)&(updates.scale_condition==condition)]
            gr=gradients[(gradients.seed==seed)&(gradients.scale_condition==condition)]
            ft=actual[(actual.seed==seed)&(actual.scale_condition==condition)]
            final=sp[sp.training_sequence_index==config.training_sequences]
            decoder=decoders[(decoders.seed==seed)&(decoders.scale_condition==condition)&
                             (decoders.training_sequence_index==config.training_sequences)].iloc[0]
            a_to_b=ft[ft.training_context.eq("A")].cross_context_loss_change
            b_to_a=ft[ft.training_context.eq("B")].cross_context_loss_change
            # Per-seed associations preserve seed as the inferential unit.
            checkpoint_transfer=ft[ft.training_sequence_index.isin(config.diagnostic_checkpoints)].set_index("training_sequence_index").cross_context_loss_change
            checkpoint_gradient=gr.set_index("training_sequence_index").cosine_gradient_alignment
            aligned=pd.concat([checkpoint_transfer,checkpoint_gradient],axis=1).dropna()
            checkpoint_shared=up.groupby("training_sequence_index").shared_update_energy_fraction.mean()
            aligned_shared=pd.concat([checkpoint_transfer,checkpoint_shared],axis=1).dropna()
            rows.append({"seed":seed,"scale_condition":condition,
                         "mean_shared_change":change.shared_state_change.mean(),"mean_private_change":change.private_state_change.mean(),
                         "median_private_shared_ratio":change.private_shared_ratio.median(),
                         "mean_task_shared_change":change.task_shared_change.mean(),"mean_task_private_change":change.task_private_change.mean(),
                         "shared_update_energy_fraction":up.shared_update_energy_fraction.mean(),
                         "private_update_energy_fraction":up.private_update_energy_fraction.mean(),
                         "mean_gradient_alignment":gr.cosine_gradient_alignment.mean(),
                         "mean_A_to_B_transfer":a_to_b.mean(),"mean_B_to_A_transfer":b_to_a.mean(),
                         "mean_within_context_transfer":ft.within_context_loss_change.mean(),
                         "mean_cross_context_transfer":ft.cross_context_loss_change.mean(),
                         "AB_hidden_separation":final.AB_distance.mean(),"AB_task_separation":final.task_AB_separation.mean(),
                         "private_structure_growth":final.private_difference_norm.mean()-sp[sp.training_sequence_index==0].private_difference_norm.mean(),
                         "shared_sequence_decoder_accuracy":decoder.sequence_position_accuracy_shared,
                         "private_sequence_decoder_accuracy":decoder.sequence_position_accuracy_private,
                         "private_context_decoder_accuracy":decoder.context_accuracy_private,
                         "transfer_gradient_alignment_correlation":aligned.corr().iloc[0,1] if len(aligned)>2 else np.nan,
                         "transfer_shared_energy_correlation":aligned_shared.corr().iloc[0,1] if len(aligned_shared)>2 else np.nan})
    return pd.DataFrame(rows)


def statistical_summary(seed_summary,config):
    rng=np.random.default_rng(config.bootstrap_seed);metrics=[c for c in seed_summary.columns if c not in ("seed","scale_condition")]
    rows=[]
    for metric in metrics:
        for condition in config.conditions:
            values=seed_summary[seed_summary.scale_condition.eq(condition)][metric].dropna().to_numpy()
            boot=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(config.bootstrap_samples)])
            rows.append({"metric":metric,"comparison":condition,"n_seeds":len(values),"mean":values.mean(),
                         "median":np.median(values),"ci_low":np.quantile(boot,.025),"ci_high":np.quantile(boot,.975),
                         "resampling_unit":"seed"})
        high=seed_summary[seed_summary.scale_condition.eq(HIGH)].sort_values("seed")
        low=seed_summary[seed_summary.scale_condition.eq(LOW)].sort_values("seed")
        if not np.array_equal(high.seed.to_numpy(),low.seed.to_numpy()): raise AssertionError("FAIL: SEED_LEVEL_INFERENCE")
        differences=high[metric].to_numpy()-low[metric].to_numpy();differences=differences[np.isfinite(differences)]
        boot=np.array([rng.choice(differences,len(differences),replace=True).mean() for _ in range(config.bootstrap_samples)])
        rows.append({"metric":metric,"comparison":"High_minus_Low_paired","n_seeds":len(differences),
                     "mean":differences.mean(),"median":np.median(differences),"ci_low":np.quantile(boot,.025),
                     "ci_high":np.quantile(boot,.975),"resampling_unit":"paired_seed"})
    return pd.DataFrame(rows)
