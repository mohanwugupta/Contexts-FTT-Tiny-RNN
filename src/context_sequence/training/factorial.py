"""Faithful SRN recreation and strictly familiar-only prospective training."""
import hashlib
import itertools
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from context_sequence.evaluation.rollout import Model
from context_sequence.task.sequence import SEQUENCE
from .separation_regularizers import weak_loss,strong_loss,closure_loss

ARMS=('F00','F01','F10','F11')


class CanonicalSRN(nn.Module):
    def __init__(self):
        super().__init__();self.recurrent=nn.RNN(7,8,nonlinearity='tanh',batch_first=True);self.sequence_head=nn.Linear(8,4)
        for module in (self.recurrent,self.sequence_head):
            for name,p in module.named_parameters():
                if 'weight' in name:nn.init.normal_(p,mean=0.,std=2.)
                else:nn.init.zeros_(p)
        self.enforce_novel_neutral()
        self.recurrent.weight_ih_l0.register_hook(self._zero_third)

    @staticmethod
    def _zero_third(g):
        g=g.clone();g[:,6]=0;return g

    def enforce_novel_neutral(self):
        with torch.no_grad():self.recurrent.weight_ih_l0[:,6].zero_()

    def forward(self,tokens,cues):
        if tokens.ndim==2:tokens=tokens.unsqueeze(0)
        if cues.ndim==1:cues=cues.unsqueeze(0)
        tokens=tokens.expand(len(cues),-1,-1);c=cues[:,None,:].expand(-1,tokens.shape[1],-1)
        h,_=self.recurrent(torch.cat([tokens,c],-1))
        return self.sequence_head(h),h

    def trajectory(self,sequence,cue):
        tokens=F.one_hot(torch.tensor(sequence[:-1]),4).float()
        z,h=self(tokens,torch.as_tensor(cue,dtype=torch.float32))
        return z[0],h[0]


def initialize(seed):
    torch.use_deterministic_algorithms(True);torch.manual_seed(int(seed));return CanonicalSRN()


def training_schedule():
    return ['A' if (i//10)%2==0 else 'B' for i in range(170)]


def state_hash(model):
    return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for k,v in sorted(model.state_dict().items()))).hexdigest()


def numpy_model(model):
    s=model.state_dict()
    def n(k):return s[k].detach().cpu().numpy().copy()
    return Model(n('recurrent.weight_ih_l0'),n('recurrent.weight_hh_l0'),n('recurrent.bias_ih_l0')+n('recurrent.bias_hh_l0'),n('sequence_head.weight'),n('sequence_head.bias'))


def objective(model,sequence,label,condition,settings):
    if label not in ('A','B'):raise ValueError('Protected probe cannot enter training objective: '+str(label))
    if condition not in ARMS:raise ValueError('Unknown factorial condition')
    tokens=F.one_hot(torch.tensor(sequence[:-1]),4).float();cues=torch.tensor([[1.,0.,0.],[0.,1.,0.]])
    z,h=model(tokens,cues);a,b=h[0],h[1]
    ce=F.cross_entropy(z[0 if label=='A' else 1],torch.tensor(sequence[1:]))
    sep=weak_loss(a,b) if condition[1]=='0' else strong_loss(a,b,settings['margin'])
    close,target,q=closure_loss(model,sequence,a,b)
    total=ce+settings['lambda_sep']*sep+int(condition[2])*settings['lambda_close']*close
    return {'total':total,'ce':ce,'separation':sep,'closure':close,'target':target,'neutral_hidden':q}


def familiar_metrics(model,sequence=SEQUENCE,margin=2.5):
    a,ha=model.trajectory(sequence,[1,0,0]);b,hb=model.trajectory(sequence,[0,1,0]);y=torch.tensor(sequence[1:])
    ca=F.cross_entropy(a,y);cb=F.cross_entropy(b,y);close,_,_=closure_loss(model,sequence,ha,hb)
    wp=weak_loss(ha,hb);sp=strong_loss(ha,hb,margin)
    def gn(loss):
        grads=torch.autograd.grad(loss,tuple(model.parameters()),retain_graph=True,allow_unused=True)
        return float(torch.sqrt(sum(g.square().sum() for g in grads if g is not None)).detach())
    z1=a-a.mean(-1,keepdim=True);z2=b-b.mean(-1,keepdim=True)
    return {'A_ce':float(ca.detach()),'B_ce':float(cb.detach()),'familiar_mean_ce':float(((ca+cb)/2).detach()),
        'accuracy':float(torch.cat([a.argmax(-1)==y,b.argmax(-1)==y]).float().mean()),
        'nonconstant_prediction':bool(a.argmax(-1).unique().numel()>1 and b.argmax(-1).unique().numel()>1),
        'hidden_separation':float(torch.linalg.vector_norm(ha-hb,dim=-1).mean().detach()),
        'task_separation':float(torch.linalg.vector_norm(z1-z2,dim=-1).mean().detach()),
        'margin_attainment':float((torch.linalg.vector_norm(ha-hb,dim=-1)>=margin).float().mean()),
        'closure_fixed_H':float(close.detach()),'weak_gradient_norm':gn(wp),'strong_gradient_norm':gn(sp),'closure_gradient_norm':gn(close)}


def train_run(seed,condition,settings,sequence=SEQUENCE,*,save_states=True):
    model=initialize(seed);initial=familiar_metrics(model,sequence,settings['margin'])
    optimizer=torch.optim.Adam(model.parameters(),lr=.005);rows=[];states=[]
    def snapshot():return {k:v.detach().clone() for k,v in model.state_dict().items()}
    if save_states:states.append(snapshot())
    for step,label in enumerate(training_schedule(),1):
        optimizer.zero_grad();before=state_hash(model);p=objective(model,sequence,label,condition,settings)
        if not all(torch.isfinite(p[k]).all() for k in ('total','ce','separation','closure')):raise FloatingPointError('Nonfinite A/B-only training loss')
        p['total'].backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(grad):raise FloatingPointError('Nonfinite A/B-only gradient')
        optimizer.step();model.enforce_novel_neutral();after=state_hash(model)
        rows.append({'condition':condition,'seed':int(seed),'training_sequence_index':step,'training_context':label,
            'raw_ce_pre_update':float(p['ce'].detach()),'separation_loss_pre_update':float(p['separation'].detach()),
            'closure_loss_pre_update':float(p['closure'].detach()),'total_loss_pre_update':float(p['total'].detach()),
            'gradient_norm_pre_clip':float(grad),'parameter_hash_pre':before,'parameter_hash':after,'parameters_changed':before!=after,
            'initial_familiar_mean_ce':initial['familiar_mean_ce']})
        if save_states:states.append(snapshot())
    return model,pd.DataFrame(rows),states


def candidate_settings():
    return [dict(lambda_sep=s,lambda_close=c,margin=m) for s,c,m in itertools.product([.03,.1,.3,1.],[.03,.1,.3,1.],[1.5,2.5,3.5])]


def candidate_id(settings):
    return 'sep{lambda_sep:g}_close{lambda_close:g}_m{margin:g}'.format(**settings)


def pilot_choice(frame,rules):
    required={'candidate_id','condition','seed','A_ce','B_ce','familiar_mean_ce','hidden_separation','closure_fixed_H','improved','accuracy','nonconstant_prediction','weak_gradient_norm','strong_gradient_norm','closure_gradient_norm','lambda_sep','lambda_close','margin'}
    if not required.issubset(frame) or 'probe' in frame:raise ValueError('Pilot requires familiar-only complete factorial metrics')
    rows=[]
    for cid,f in frame.groupby('candidate_id',sort=True):
        if f.duplicated(['condition','seed']).any() or set(f.condition)!=set(ARMS) or set(f.seed)!=set(range(1000,1005)) or len(f)!=20:raise ValueError('Pilot arms/seeds incomplete')
        means=f.groupby('condition').mean(numeric_only=True)
        finite=np.isfinite(f.select_dtypes('number').to_numpy()).all()
        competence=bool((means[['A_ce','B_ce']]<=rules['endpoint_mean_ce_bound']).all().all() and means.familiar_mean_ce.le(rules['familiar_mean_ce_bound']).all())
        learning=bool(f.groupby('condition').improved.sum().ge(rules['minimum_improved_seeds_per_arm']).all() and means.accuracy.ge(rules['minimum_accuracy']).all() and f.nonconstant_prediction.all())
        separations=[means.loc[f'F1{j}','hidden_separation']-means.loc[f'F0{j}','hidden_separation'] for j in [0,1]]
        ratios=[means.loc[f'F1{j}','hidden_separation']/(means.loc[f'F0{j}','hidden_separation']+1e-12) for j in [0,1]]
        closure=[1-means.loc[f'F{i}1','closure_fixed_H']/(means.loc[f'F{i}0','closure_fixed_H']+1e-12) for i in [0,1]]
        gradients=bool((f.closure_gradient_norm>1e-10).all() and (f.loc[f.condition.str[1].eq('0'),'weak_gradient_norm']>1e-10).all() and (f.loc[f.condition.str[1].eq('1'),'strong_gradient_norm']>1e-10).all())
        separation=bool(min(separations)>=rules['minimum_strong_minus_weak_hidden_distance'] and min(ratios)>=rules['minimum_strong_weak_ratio'])
        closure_ok=bool(min(closure)>=rules['minimum_closure_relative_reduction'])
        setting=f.iloc[0][['lambda_sep','lambda_close','margin']].to_dict()
        rows.append({'candidate_id':cid,**setting,'finite':finite,'competence_passed':competence,'learning_passed':learning,'separation_passed':separation,'closure_passed':closure_ok,'gradient_passed':gradients,
            'min_separation_gap':min(separations),'min_separation_ratio':min(ratios),'min_closure_reduction':min(closure),'feasible':finite and competence and learning and separation and closure_ok and gradients})
    audit=pd.DataFrame(rows);valid=audit[audit.feasible].copy()
    if not len(valid):return None,audit
    valid['strength_sum']=valid.lambda_sep+valid.lambda_close
    chosen=valid.sort_values(['strength_sum','lambda_sep','lambda_close','margin']).iloc[0]
    return {k:float(chosen[k]) for k in ['lambda_sep','lambda_close','margin']},audit
