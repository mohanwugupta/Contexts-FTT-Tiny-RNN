"""Registered native operators. Arbitrary agent Python is never imported by a worker."""
import copy,types
import numpy as np,torch
from experiments.context_experience_prior import core
from experiments.joint_phenotype_hypothesis_screen import adapter
from confirmatory_srn.model import ConfirmatoryElmanSRN
from research_search.common import fingerprint

DEFAULT=dict(model=dict(hidden_dim=8,activation='linear',nominal_scale=5.,recurrent_norm=.9,centered=False,cue_timing='continuous',gate=False),prior=dict(episodes=1920,kind='P2',cue_rotation=False),target=dict(steps=330,readout_initial_multiplier=1.,update_rule='native'))

def validate(config):
    c=copy.deepcopy(config)
    if set(c)!=set(DEFAULT) or any(set(c[k])!=set(DEFAULT[k]) for k in DEFAULT):raise ValueError('Unregistered candidate fields/objective')
    m,p,t=c['model'],c['prior'],c['target']
    if any(type(m[k]) is not bool for k in ['centered','gate']) or type(p['cue_rotation']) is not bool:raise ValueError('Boolean mechanism flags required')
    if m['hidden_dim'] not in [8,32] or m['activation'] not in ['linear','tanh'] or m['cue_timing'] not in ['continuous','onset']:raise ValueError('Unregistered architecture')
    if not .1<=m['nominal_scale']<=10 or m['recurrent_norm'] is not None and not .1<=m['recurrent_norm']<=2:raise ValueError('Invalid fixed initialization')
    if m['activation']=='tanh' and (m['gate'] or m['cue_timing']!='continuous'):raise ValueError('This combination requires a separately tested native adapter')
    if m['activation']=='tanh' and p['cue_rotation']:raise ValueError('Tanh cue rotation needs a separately tested adapter')
    if m['gate'] and (m['hidden_dim']!=8 or t['update_rule']!='native'):raise ValueError('Combined mechanism not yet registered')
    if type(p['episodes']) is not int or not 0<=p['episodes']<=1920 or p['kind'] not in ['P1','P2']:raise ValueError('Unregistered ordinary background budget')
    if t['steps']!=330 or not .01<=t['readout_initial_multiplier']<=4 or t['update_rule'] not in ['native','slow_recurrence','reset_first_moment','freeze_context','readout_only']:raise ValueError('Target contract changed')
    return c

def build_model(config,seed):
    c=validate(config);spec=c['model'];h=spec['hidden_dim']
    with torch.random.fork_rng():
        torch.manual_seed(int(seed));m=core.ExperienceLinear()
        if h!=8:
            m.recurrent=torch.nn.RNN(7,h,nonlinearity='tanh',batch_first=True);m.sequence_head=torch.nn.Linear(h,4)
            with torch.no_grad():
                for name,v in m.named_parameters():torch.nn.init.normal_(v,0.,2.) if 'weight' in name else v.zero_()
        with torch.no_grad():
            m.recurrent.weight_ih_l0.mul_(spec['nominal_scale']/2);m.sequence_head.weight.mul_(spec['nominal_scale']/2*np.sqrt(8/h))
            if spec['recurrent_norm'] is not None:m.recurrent.weight_hh_l0.mul_(spec['recurrent_norm']/torch.linalg.matrix_norm(m.recurrent.weight_hh_l0,2))
            if spec['centered']:
                d=(m.recurrent.weight_ih_l0[:,4]-m.recurrent.weight_ih_l0[:,5])/2;m.recurrent.weight_ih_l0[:,4].copy_(d);m.recurrent.weight_ih_l0[:,5].copy_(-d)
        if spec['gate']:m.register_parameter('V',torch.nn.Parameter(torch.zeros(8,3)))
    m.onset=spec['cue_timing']=='onset';m.cue_basis=None;m.mask_cue=False;m.context_alpha=1.
    if spec['activation']=='tanh':m.forward=types.MethodType(ConfirmatoryElmanSRN.forward,m)
    elif spec['gate'] or spec['cue_timing']=='onset' or h!=8:m.forward=types.MethodType(adapter.flexible_forward,m)
    return m

def variant(config):return {'native':'A00','slow_recurrence':'A03','reset_first_moment':'A05','freeze_context':'A01','readout_only':'A02'}[config['target']['update_rule']]
def target_optimizer(model,policy,config):return adapter.optimizer(model,policy,variant(config))
def build_training_schedule(config,seed):
    validate(config)
    from experiments.joint_phenotype_hypothesis_screen.common import cue
    return [dict(update=k,context=cue(k)) for k in range(1,331)]

def run_pretraining(model,config,tasks):
    if not hasattr(model,'_research_runtime'):raise RuntimeError('Only trusted budgeted worker can run optimization')
    return model._research_runtime.pretrain(model,validate(config),tasks)
def run_target_learning(model,config,target_task):
    if not hasattr(model,'_research_runtime'):raise RuntimeError('Only trusted budgeted worker can run optimization')
    return model._research_runtime.target(model,validate(config),target_task)
def export_predictions(model,probe_contexts):
    path=probe_contexts['path'];assignment=probe_contexts.get('context_assignment',['A','B']);p=probe_contexts.get('policy');o=probe_contexts.get('optimizer');native=adapter.probes(model,p,o,path=path)
    return {c:native[assignment[0] if c=='A' else assignment[1] if c=='B' else c] for c in ['A','B','C','Mixed-average','NoContext']}
