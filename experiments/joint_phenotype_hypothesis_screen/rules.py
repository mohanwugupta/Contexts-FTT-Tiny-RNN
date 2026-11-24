"""Pure, fixed routing rules. Positive gains mean improvement; no outcome tuning."""
import numpy as np
import pandas as pd
ORDER=[f'A{i:02}' for i in range(1,8)]+[f'B{i:02}' for i in range(1,7)]

def useful(ce,correct):return bool(np.isfinite(ce) and ce<=.5 and int(correct)>=6)
def strict(ce,correct):return bool(np.isfinite(ce) and ce<=.1 and int(correct)==7)
def direction(start,own,cross,s):
 eligible=bool(np.isfinite([start,own,cross]).all() and start>.05 and own>1e-4 and own>.05*start)
 return {f'start_{s}':start,f'own_{s}':own,f'cross_{s}':cross,f'D_{s}':own-cross,f'Q_{s}':1-max(cross,0)/own if eligible else np.nan,f'Q_eligible_{s}':eligible,f'floor_{s}':start<=.05,f'nonlearning_{s}':not eligible and start>.05,f'interference_{s}':cross<0,f'net_{s}':own+cross}
def early_gate(r,ref):
 return bool(ref['resolved'] and np.isfinite([r['Q_A'],r['Q_B'],r['E_normalized']]).all() and r['Q_A']>0 and r['Q_B']>0 and r['net_A']>0 and r['net_B']>0 and r['E']>0 and r['Q_A']>=.5*ref['Q_A'] and r['Q_B']>=.5*ref['Q_B'] and r['E_normalized']>=.5*ref['E_normalized'])
def switches(frame):
 p=frame.pivot(index='update',columns='cue',values='ce');rows=[]
 if not np.isfinite(p.loc[[0,10,20,30,40,50,60],['A','B']].to_numpy()).all():raise ValueError('Missing frozen switch cell')
 for n in [10,20,30,40,50,60]:
  old='A' if ((n-1)//10)%2==0 else 'B';new='B' if old=='A' else 'A';raw=p.loc[n,new]-p.loc[n,old];initial=p.loc[0,new]-p.loc[0,old]
  rows.append(dict(update=n,outgoing=old,incoming=new,raw_switch=raw,initial_difference=initial,adjusted_switch=raw-initial))
 return pd.DataFrame(rows)
def summarize(frame):
 p=frame.pivot(index='update',columns='cue',values='ce');out={}
 for s,other,n,k in [('A','B',0,10),('B','A',10,20)]:out.update(direction(p.loc[n,s],p.loc[n,s]-p.loc[k,s],p.loc[n,other]-p.loc[k,other],s))
 b=switches(frame);h=max(.5,(p.loc[0,'A']+p.loc[0,'B'])/2)
 out.update(E=float(b.adjusted_switch.mean()),E_normalized=float(b.adjusted_switch.mean()/h),H0=h,early_cost=float(b.adjusted_switch.mean()),A_to_B=float(b[b.incoming.eq('B')].adjusted_switch.mean()),B_to_A=float(b[b.incoming.eq('A')].adjusted_switch.mean()),sustained_cost=float(b[b['update'].ge(30)].adjusted_switch.mean()),cycle=float(b[b['update'].le(20)].adjusted_switch.mean()))
 return out
def route(a,b,candidate,static=None):
 a=a.sort_values('seed').reset_index(drop=True);b=b.sort_values('seed').reset_index(drop=True);assert len(a)==len(b)==2 and a.seed.tolist()==b.seed.tolist()==[0,1]
 both=bool(a.joint.all() and a.C_correct.ge(6).all())
 transfer=bool((a.C_ce<b.C_ce).all() and b.C_ce.mean()-a.C_ce.mean()>=max(.05,.2*b.C_ce.mean()) and a.C_correct.ge(b.C_correct).all())
 split=bool((a.E_normalized>b.E_normalized).all() and a.E_normalized.mean()-b.E_normalized.mean()>=max(.01,.2*abs(b.E_normalized.mean())) and (a.Q_A>=b.Q_A-.05).all() and (a.Q_B>=b.Q_B-.05).all() and a.C_useful.all() and a.C_correct.ge(b.C_correct).all() and a.C_ce.mean()-b.C_ce.mean()<=.05)
 old=b.joint.astype(bool);rescue=bool(old.sum()<=1 and (a.loc[old,'C_correct']>=b.loc[old,'C_correct']).all() and (a.loc[old,'C_ce']<=b.loc[old,'C_ce']+.05).all() and (a.loc[old,'E_normalized']>=.8*b.loc[old,'E_normalized']).all())
 static_ok=True
 if candidate=='A06':
  assert static is not None;static=static.sort_values('seed').reset_index(drop=True)
  static_ok=bool(((a.C_correct>static.C_correct).any() and (a.C_correct>=static.C_correct).all()) or static.C_ce.mean()-a.C_ce.mean()>=.05)
 return dict(eligible=bool(both and (transfer or split or rescue) and static_ok),transfer=transfer,splitting=split,rescue=rescue,static_control_passed=static_ok,acquisition_plus_transfer=bool((a.familiar&~b.familiar).any()))
def rank(rows):
 return [r['id'] for r in sorted([r for r in rows if r['eligible']],key=lambda r:(-r['strict_count'],-r['retention'],r['C_ce'],ORDER.index(r['id'])))]
