"""Table-only pilot exports, compact figure and owner status report."""
import json
import numpy as np,pandas as pd
from research_search.common import read,write,clean


def export(store,out):
    out.mkdir(parents=True,exist_ok=True)
    calls=[dict(r) for r in store.db.execute('SELECT id,activity,reserved,status,total,cached FROM calls')]
    pd.DataFrame(calls).to_csv(out/'agent_usage.csv',index=False)
    jobs=[dict(r) for r in store.db.execute('SELECT id,candidate,status,error FROM jobs')]
    pd.DataFrame(jobs).to_csv(out/'jobs.csv',index=False)
    result_file=store.directory/'development_result.json'
    if result_file.exists():
        r=read(result_file);write(out/'development_result.json',r,True)
        rows=[dict(condition=label,**x) for label,key in [('candidate','individual_results'),('control','matched_control_results')] for x in r[key]]
        pd.DataFrame(rows).to_csv(out/'seed_task_metrics.csv',index=False)
        pd.DataFrame(r['paired_seed_intervals']).to_csv(out/'paired_seed_intervals.csv',index=False)
        paired=read(store.directory/'paired_jobs.json');frames=[]
        for condition,ids in paired.items():
            for id in ids:
                f=pd.read_csv(store.directory/'jobs'/id/'metrics.csv');f['condition']=condition;f['task_id']=read(store.directory/'jobs'/id/'evaluation.json')['task_id'];frames.append(f)
        all_metrics=pd.concat(frames,ignore_index=True);all_metrics.to_csv(out/'all_frozen_metrics.csv',index=False)
        # Plot tasks within each trained seed before cohort means. No model or
        # scoring calls; all values are inherited frozen-evaluation table cells.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        g=all_metrics.groupby(['condition','seed','update','cue']).ce.mean().reset_index()
        g.to_csv(out/'figure_values.csv',index=False)
        fig,axes=plt.subplots(1,2,figsize=(10,3.5),constrained_layout=True)
        colors={'candidate':'#1967a3','control':'#d67d1a'}
        for condition in ['candidate','control']:
            f=g[g.condition.eq(condition)]
            fam=f[f.cue.isin(['A','B'])].groupby(['seed','update']).ce.mean().reset_index()
            for _,s in fam.groupby('seed'):axes[0].plot(s['update'],s.ce,color=colors[condition],alpha=.18,lw=.8)
            mean=fam.groupby('update').ce.mean();axes[0].plot(mean.index,mean,label=condition,color=colors[condition])
            for cue,style in [('C','-'),('Mixed-average','--')]:
                c=f[f.cue.eq(cue)].groupby('update').ce.mean();axes[1].plot(c.index,c,label=condition+' '+cue,color=colors[condition],ls=style)
        for ax,title in zip(axes,['Familiar A/B','C and physical Mixed']):
            ax.axhline(np.log(4),color='gray',ls=':',lw=1);ax.set(xlabel='Target update',ylabel='Cross-entropy (nats)',title=title);ax.legend(fontsize=8)
        fig.suptitle('Technical pilot: two seeds, four repeated development tasks')
        fig.savefig(out/'pilot_learning.png',dpi=160);fig.savefig(out/'pilot_learning.svg');plt.close(fig)
    status=store.state()|dict(phase=store.get('phase'),agent_calls=calls,jobs=jobs)
    write(out/('STATUS_'+store.get('phase')+'.json'),status,True)
    path=out/('REPORT_'+store.get('phase')+'.md')
    text=f'''# Controlled pilot v2 — {store.get('phase')}

Actual reported pilot service tokens: {store.totals()['tokens']}; native optimizer attempts: {store.totals()['optimizer_steps']}; active CPU: {store.totals()['cpu_seconds']:.3f}s. Current status: {store.get('status')}.

The pilot has50000 reported-token,6000-update,3600CPU-second and7200wall-second ceilings,2 proposal attempts and1 active branch. It uses two registered seeds and four unrelated tasks,330 A10/B10 updates, fresh zero-prior paired models and the unchanged full native scorer. No meta-training, competence updates, confirmation or full72h campaign. Individual manual approval is required per agent call; missing usage and failed jobs are preserved, never replayed. Provider hard in-turn token enforcement is unavailable.

The exact Code Mode-host-disabled pre-turn notice is retained as a notice by this versioned wrapper. Other errors/tools/unknown items still fail. Historical connection receipts and their old flags remain unchanged. The worker binds only a missing BudgetError control symbol, then delegates numeric execution/scoring to native code.

See agent_usage.csv, jobs.csv and the private campaign directory for actual calls, raw responses, full-state checkpoints and per-position probabilities/logits. If evaluation is complete, seed_task_metrics.csv preserves all16 seed/task/condition rows, all_frozen_metrics.csv all frozen scalar cells, and paired_seed_intervals.csv native paired trained-seed bootstrap contrasts. The figure is table-only; it does not substitute for exact values.

Two seeds and repeated tasks do not establish a human learning mechanism. The late functional splitting gate is unassessed; no geometry-based claim or favorable checkpoint selection is allowed. Zero-prior C results cannot establish acquired prior competence. Negative, failed and missing outcomes remain evidence.

Current phase requires {'separate approval before the result-critic call' if store.get('phase')=='critic_pending' else 'inspection of status and retained records'}. No additional calls, revisions, tier expansion or campaign launch follow automatically. Earlier known/unknown connection costs are separate and were not erased or zero-imputed.
'''
    if not path.exists():path.write_text(text)
    return dict(report=str(path),status=store.state(),phase=store.get('phase'))
