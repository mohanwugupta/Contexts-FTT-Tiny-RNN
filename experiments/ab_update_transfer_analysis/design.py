"""Fixed analysis grid; no outcome-dependent selection."""
PRIMARY=[0,10,20,40,60]
SECONDARY=[100,170,330,990]
PROBES=['A','B','C','NoContext','Mixed-average']
SCALES=[4,5,6];SEEDS=list(range(6));REGIMES=['U','P'];OBJECTIVES=['T','B','O1','O2']
VERSION='ab_update_transfer_analysis_v1'
def maximum_steps():return 144*(5*2*10+4*2)+198*5*2+144*2*2
def gain(before,after):return before-after
def specificity(own,cross):return own-cross
def cue(k):
 if k<1:raise ValueError('Practice indices start at1')
 return 'A' if ((k-1)//10)%2==0 else 'B'
