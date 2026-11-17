"""Prospective norm/direction comparisons and familiar-only factorial gates."""
import numpy as np
import pandas as pd
from context_sequence.evaluation.cue_grid import cue_geometry


def grid_controls(seed,registry):
    pairs=[];grid=[x for x in registry['cues'] if x['kind']=='grid']
    for a in registry['alpha']:
        for beta in [.366025403784,1.]:
            p=next(c for c in grid if c['cue_alpha']==a and c['cue_beta']==beta)
            n=next(c for c in grid if c['cue_alpha']==a and c['cue_beta']==-beta)
            pairs.append(('same_alpha_norm_matched_beta_sign',p['name'],n['name']))
    for i,a in enumerate(grid):
        for b in grid[i+1:]:
            if abs(a['cue_norm']-b['cue_norm'])<1e-10 and abs(a['cue_alpha']-b['cue_alpha'])>.1:pairs.append(('same_norm_different_direction',a['name'],b['name']))
    pairs.extend([('anchor_Mixed_minus_NoContext','Mixed','NoContext'),('exact_Novel_A_beta_sign','Novel30_A','Novel60_A'),('exact_Novel_B_beta_sign','Novel30_B','Novel60_B')])
    for scale in [.5,1.,1.5]:pairs.append(('same_norm_ray_direction',f'ray_A_{scale:g}',f'ray_B_{scale:g}'))
    rows=[]
    for key,f in seed.groupby(['cohort','condition','seed']):
        f=f.set_index('probe')
        for kind,left,right in pairs:
            a=f.loc[left];b=f.loc[right];name=left+'__minus__'+right
            for metric in ['raw_ce','closure_task_norm','error_task_norm','task_residual']:
                rows.append({'sequence_id':'original','cohort':key[0],'condition':key[1],'seed':key[2],'probe':kind,'contrast':name,'comparison_kind':kind,'left':left,'right':right,'metric':metric,'effect':a[metric]-b[metric],
                    'norm_difference':a.cue_norm-b.cue_norm,'alpha_difference':a.cue_alpha-b.cue_alpha,'beta_difference':a.cue_beta-b.cue_beta,
                    'beta_sensitivity':(a[metric]-b[metric])/(a.cue_beta-b.cue_beta) if abs(a.cue_beta-b.cue_beta)>1e-12 else np.nan})
    return pd.DataFrame(rows)


def projection_invariants(transitions,registry):
    rows=[];grids=[x for x in registry['cues'] if x['kind']=='grid']
    for key,f in transitions.groupby(['cohort','condition','seed']):
        index={p:g.sort_values('transition') for p,g in f.groupby('probe')}
        pairs=[]
        for alpha in registry['alpha']:
            selected=[c['name'] for c in grids if c['cue_alpha']==alpha]
            pairs.extend((selected[0],s) for s in selected[1:])
        pairs.extend([('Mixed','NoContext'),('Novel30_A','Novel60_A'),('Novel30_B','Novel60_B')])
        for left,right in pairs:
            a=index[left];b=index[right];expected=key[1] in ['C1_contrast_only','C3_full']
            h=np.max(np.abs(a[[f'hidden_{i}' for i in range(8)]].values-b[[f'hidden_{i}' for i in range(8)]].values))
            z=np.max(np.abs(a[[f'logit_{i}' for i in range(4)]].values-b[[f'logit_{i}' for i in range(4)]].values))
            ce=np.max(np.abs(a.raw_ce.values-b.raw_ce.values))
            drive=np.max(np.abs(a[[f'context_drive_{i}' for i in range(8)]].values-b[[f'context_drive_{i}' for i in range(8)]].values))
            rows.append({'cohort':key[0],'condition':key[1],'seed':key[2],'left':left,'right':right,'structural_identity_expected':expected,'max_hidden_difference':h,'max_logit_difference':z,'max_ce_difference':ce,'max_drive_difference':drive,'passed':not expected or (h<=2e-4 and z<=2e-3 and ce<=2e-3 and drive<=2e-6)})
    return pd.DataFrame(rows)


def familiar_feasibility(familiar,settings):
    rows=[]
    for sid,seq in familiar.groupby('sequence_id'):
        for close in [0,1]:
            weak=seq[seq.condition.eq(f'F0{close}')].set_index('seed').sort_index();strong=seq[seq.condition.eq(f'F1{close}')].set_index('seed').sort_index()
            if not weak.index.equals(strong.index):raise ValueError('Familiar seed pairing mismatch')
            gap=strong.hidden_separation.mean()-weak.hidden_separation.mean();ratio=strong.hidden_separation.mean()/(weak.hidden_separation.mean()+1e-12)
            da=strong.A_ce-weak.A_ce;db=strong.B_ce-weak.B_ce
            rows.append({'sequence_id':sid,'closure_arm':close,'n_seeds':len(weak),'A_ce_strong_minus_weak':da.mean(),'B_ce_strong_minus_weak':db.mean(),'A_gap_sd':da.std(),'B_gap_sd':db.std(),
                'competence_comparable':abs(da.mean())<=.30 and abs(db.mean())<=.30,'hidden_separation_gap':gap,'hidden_separation_ratio':ratio,
                'separation_target_engaged':gap>=.25 and ratio>=1.25,'nonoverlapping_hidden_regimes':strong.hidden_separation.min()>weak.hidden_separation.max(),
                'strong_margin_attainment':strong.margin_attainment.mean(),'weak_margin_attainment':weak.margin_attainment.mean()})
    return pd.DataFrame(rows)
