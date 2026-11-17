from pathlib import Path
from context_sequence.provenance.hashing import object_hash
from experiments.output_compatible_trajectories.config import SEQUENCE,PROBES
ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'results/context_mixing_sequence_continuity/v1'
CONTRACT={'version':'v1','cohort':'canonical High_monotonic','initial_update':170,'seeds':list(range(20)),'sequence':list(SEQUENCE),
 'history_positions':[1,3,4,7],'diagnostic_positions':[2,5,6],'decoder_ridge':1.,'decoder_sd_floor':1e-6,'decoder_sd_ddof':0,'decoder_intercept':'unpenalized','decoder_transfer_min_correct':5,'decoder_min_pool':14,'decoder_min_centered_rank':7,
 'continuation_updates':160,'arms':['B','I','S'],'block_length':10,'within_switch_after':3,'smoke_seeds':[0,1],'smoke_reused_in_full':True,'maximum_scientific_updates':9600,
 'learning_rate':.005,'gradient_clip':1.,'lambda_execution':.20,'c_time':.5,'c_risk':1.,'gamma':2.,'optimizer':'source torch.optim.Adam defaults; lr=.005; same fresh empty moments in all arms if unavailable',
 'protected_cues':list(PROBES),'training_cues':['A','B'],'familiar_margin_nats':.20,'retained_raw_separation_fraction':.80,
 'bootstrap_seed':20260925,'bootstrap_draws':10000,'replay_ce_atol':.002,'replay_hidden_atol':.0002,'replay_logit_atol':.002,'same_source_csv_atol':1e-12,'exact_identity_atol':1e-11,
 'float32_forward_bound':'64*eps32*(1+absolute input/recurrent/bias summands); state deviation residual <= sum of relevant tanh roundoff bounds',
 'competitor':'highest incorrect natural-Mixed logit, fixed for all components at a matched position','decoder_optional_next_token_readout':False,
 'no_probe_selection':True,'final_update':330,'no_additional_search':True}
CONTRACT['sha256']=object_hash(CONTRACT)
