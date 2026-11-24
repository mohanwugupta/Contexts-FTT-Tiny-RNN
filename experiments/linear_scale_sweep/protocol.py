from .common import *
def scaled(d,s):
 if s not in SCALES:raise ValueError('Registered nominal scales only')
 q=copy.deepcopy(d)
 for k in ['recurrent.weight_ih_l0','sequence_head.weight']:q[k].mul_(s/2)
 return q
