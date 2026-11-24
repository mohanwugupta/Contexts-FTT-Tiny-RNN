from experiments.heldout_context_meta.report import QUESTIONS,primary_filter
import pandas as pd

def test_primary_filter_cannot_pick_favorable_context_or_checkpoint():
 f=pd.DataFrame([dict(condition=c,dose=d,update=t,cue=q,metric=m,contrast=a) for c in ['P1-I','P2-I','P1-C'] for d in [240,960] for t in [170,330] for q in ['A','C'] for m in ['ce','gap'] for a in ['Transfer-Seen','Transfer-Historical']]);p=primary_filter(f);assert len(p)==2 and set(p.condition)=={'P1-I','P2-I'} and p.metric.eq('gap').all();assert len(QUESTIONS)==7
