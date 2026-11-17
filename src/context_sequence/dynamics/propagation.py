"""Nested grouped ridge prediction. All rows of a seed remain in one fold."""
import json
import numpy as np
import pandas as pd

STATIC=['familiar_mean_ce','cue_alpha','cue_norm','task_separation']
DYNAMIC=STATIC+['closure_task_norm','finite_horizon_amplification']


def _predict(train,test,features,penalty):
    x=train[features].to_numpy(float);v=test[features].to_numpy(float)
    mu=x.mean(0);sd=x.std(0);sd[sd<1e-12]=1
    x=np.c_[np.ones(len(x)),(x-mu)/sd];v=np.c_[np.ones(len(v)),(v-mu)/sd]
    ridge=np.eye(x.shape[1])*penalty;ridge[0,0]=0
    coef=np.linalg.solve(x.T@x+ridge,x.T@train.raw_ce.to_numpy())
    return v@coef


def grouped_predictions(frame):
    rows=[]
    for cohort,f in frame.groupby('cohort',sort=True):
        for fold in range(5):
            train=f[f.seed%5!=fold];test=f[f.seed%5==fold]
            if len(test)==0 or train.seed.nunique()<4:raise ValueError('Grouped CV needs complete held-out seed support')
            for label,features in [('static',STATIC),('dynamics_augmented',DYNAMIC)]:
                scores=[]
                for penalty in [.01,.1,1.,10.]:
                    errors=[]
                    for inner in range(4):
                        tr=train[train.seed%4!=inner];va=train[train.seed%4==inner]
                        if len(va) and len(tr):
                            e=va[['seed']].copy();e['mse']=(va.raw_ce.to_numpy()-_predict(tr,va,features,penalty))**2
                            errors.extend(e.groupby('seed').mse.mean().tolist())
                    scores.append((float(np.mean(errors)),penalty))
                penalty=min(scores)[1];prediction=_predict(train,test,features,penalty)
                for (_,r),p in zip(test.iterrows(),prediction):
                    rows.append({**r.to_dict(),'model':label,'prediction':p,'squared_error':(p-r.raw_ce)**2,
                        'outer_fold':fold,'ridge_penalty':penalty,'inner_cv_mse':min(scores)[0],
                        'training_seeds':json.dumps(sorted(int(x) for x in train.seed.unique())),
                        'test_seeds':json.dumps(sorted(int(x) for x in test.seed.unique()))})
    return pd.DataFrame(rows)
