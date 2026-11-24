"""Reports from durable coordination records and saved table products only."""
import json,time,sqlite3
from pathlib import Path
import pandas as pd,numpy as np
from research_search.common import ROOT,read,write,sha
from .protocol import OUT,LIMITS,PHASES


def report(store):
    OUT.mkdir(parents=True,exist_ok=True)
    calls=[dict(r) for r in store.db.execute('SELECT * FROM calls')]
    for c in calls:c['raw']=json.loads(c['raw']) if c['raw'] else None
    write(OUT/'AGENT_USAGE.json',dict(calls=calls,actual_reported_tokens=store.totals()['tokens'],unknown_calls=[c['id'] for c in calls if c['status']!='reconciled'],UI_setup_usage=None,scope='Five integration service calls only; previous campaign costs remain separate',hard_in_turn_cap=False))
    pd.DataFrame(calls).to_csv(OUT/'agent_usage.csv',index=False)
    tree=dict(nodes=store.nodes(),events=[dict(r) for r in store.db.execute('SELECT * FROM events')],jobs=[dict(r) for r in store.db.execute('SELECT * FROM jobs')])
    write(OUT/'EXPERIMENT_TREE.json',tree)
    for role in PHASES:
        path=store.directory/f'{role}.json'
        if path.exists():write(OUT/f'{role}.json',read(path),True)
    status=store.state()|dict(phase=store.get('phase'),elapsed_seconds=(store.get('stopped_at') or time.time())-store.get('started') if store.get('started') else 0.,full_campaign_started=False,confirmation_accessed=False,new_model_forwards=0,new_optimizer_steps=0,complete_coordinated_cycle=store.get('phase')=='complete')
    write(OUT/'STATUS.json',status)
    product=store.directory/'jobs/integration11-cache'
    if (product/'complete.json').exists():
        import shutil
        for p in product.iterdir():
            if p.suffix in ['.csv','.json']:shutil.copyfile(p,OUT/p.name)
        figures(OUT)
    complete=status['complete_coordinated_cycle']
    text=f'''# Multi-agent integration v1 — {'complete and stopped' if complete else 'pending / incomplete'}

The separately versioned integration is in phase `{status['phase']}` with status `{status['status']}`. Actual reported service tokens: **{store.totals()['tokens']} /100000**. Recorded CPU:{store.totals()['cpu_seconds']:.3f}s /900; elapsed:{status['elapsed_seconds']:.3f}s /21600. New model forwards and optimizer updates: **0**. Full72h campaign and confirmation were not launched.

The frozen four-outcome criterion requires familiar A/B acquisition and the original early phenotype plus useful C, physical Mixed-average and NoContext in the same network/task at330. CE≤.5 and≥6/7 correct remain unchanged. Historical joint/screen flags are retained; unresolved screen references remain missing. Net gains, directional/balanced/sustained costs, starting errors/floors, margins and correctness remain separate. CE-only improvement never becomes transfer success.

Two independent specialist calls are planned (initialization/confidence and learning dynamics/interference), followed by coordinator selection, one isolated deterministic cached worker, critic, coordinator review and terminal stop. Each role sees a compact evidence-linked history and structured schema; no role can edit evaluator/source files or launch code. The two analysis families are preregistered. Scientific autonomy is bounded to hypothesis, prediction, comparison and selection within that grammar; arbitrary mechanism discovery/implementation is not tested.

The cached-only choice preserves the complete16 archived zero-prior histories, not a newly trained competent prior. This tests coordination and descriptive informativeness; a full pretrained scientific outcome remains inconclusive. Runtime/storage estimates use completed pilot receipts and actual file sizes; prior timing, storage and competence remain unmeasured. No prior dose was shortened. See COMPUTE_ESTIMATE.md/json.

Did coordination work? {'All five actual agent roles/turns and the deterministic worker completed; inspect logs and selection for quality.' if complete else 'Not yet established: required live role calls and evaluation/review remain pending.'}
Were hypotheses useful/nonredundant? See both specialist records and coordinator duplication assessment when available; confidence/endpoint interpretations have historical precedents and are never relabeled new discoveries.
Were all four outcomes evaluated? {'See seed_task_metrics.csv, outcome_counts.csv and evaluation_result.json; all16 planned rows retained.' if (product/'complete.json').exists() else 'Evaluator implementation tested; actual coordinated evaluation has not run.'}
Did critic find meaningful failures/alternatives? {'See critic.json and coordinator_review.json.' if complete else 'Pending the separately approved critic/review calls.'}
What did this cost? AGENT_USAGE.json, STATUS.json, private request/usage receipts and worker_receipt.json give actual measured scope. UI/setup token telemetry is unavailable, not zero. Agent time and manual approval are included in elapsed ceiling; these cannot be split retrospectively from the old pilot.
What remains before a larger campaign? Full-prior timing/competence validation, representative storage planning, more than two independent seeds, deliberate source-compatible mechanism implementations/ablations and separately authorized launcher integration. The historical72h launcher remains sealed and unchanged. Reservations are not provider hard token limits; missing telemetry stops this pilot. No automatic campaign escalation or follow-up exists.
'''
    (OUT/'REPORT.md').write_text(text)
    (OUT/'RESEARCHER_HANDOFF.md').write_text('''# Researcher handoff — integration v1

The actual native model is an additive Linear H8 SRN: h_t=Rh_(t−1)+U_x x_t+U_c c+b; z_t=Wh_t+b_o. Ordinary native mean sequence CE plus0.2 execution cost, detached confidence policy, Adam.005 and joint norm1 clipping created the archived source histories. This integration performs no model computation or learning. The C column is nonzero but fixed during A/B target practice; C is not NoContext. Physical Mixed-average is(.5,.5,0), NoContext(0,0,0).

The independent evaluator reuses saved native CE/positions and unchanged seed_summary/rules. Four-outcome feasibility is a row-level conjunction at330: adequate A/B, original early flag and useful C/Mixed/NoContext. It cannot borrow successful components across tasks or seeds. Original reference-normalized screen remains unresolved here, and the late counterfactual gate unassessed. Signs and floor0.05 are retained, never reinterpreted as persistent separate learning.

A frozen block gain is CE(c,t)−CE(c,t+10) at identical beginning/end weights for all cues, with actual practiced A10/B10 schedule. Positive means improved. Context specificity is practiced gain minus unpracticed gain. These changes are descriptive, not causal mediation. A static quarter-readout uses each saved checkpoint's unchanged output bias, so it is an expression control, not new learning. Every seven-position/four-logit endpoint is retained; loss-versus-correctness must be assessed jointly.

Tasks repeat within two initialization seeds. Average tasks within seed before the inherited10000-draw seed bootstrap; n=2 is a severe limit. Zero prior cannot assess human-like acquired priors. No new trained candidate, retrieval mechanism or confirmed cognitive explanation follows from this coordination test. Both unselected and selected hypotheses, negative results, unknown usage and failure records remain inspectable. Any follow-up is design only.

Reproduction: inspect REPRODUCIBILITY.md, EVALUATION_SPEC.md, SOURCE_AUDIT.md, COMPUTE_ESTIMATE.json, definition.json hashes and source pilot seals. Do not replay completed calls, queue work or training to verify. Read STATUS.json/REPORT.md for current completion; this handoff does not imply pending agent stages have run.
''')
    return status


def figures(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt']='integration-v1'
    f=pd.read_csv(out/'seed_task_metrics.csv');colors={'candidate':'#1768a0','control':'#cf741c'}
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),constrained_layout=True)
    cues=['A','B','C','Mixed','NoContext']
    for offset,(condition,g) in zip([-.08,.08],f.groupby('condition',sort=True)):
        for k,cue in enumerate(cues):
            x=g.groupby('seed')[cue+'_ce'].mean()
            axes[0].scatter(np.full(len(g),k+offset),g[cue+'_ce'],s=12,alpha=.25,color=colors[condition])
            axes[0].plot(k+offset,x.mean(),'o',color=colors[condition],label=condition if k==0 else None)
        for k,field in enumerate(['D_A','D_B','A_to_B','B_to_A','sustained_cost']):
            x=g.groupby('seed')[field].mean()
            axes[1].scatter(np.full(2,k+offset),x,color=colors[condition],s=20)
    axes[0].axhline(np.log(4),ls=':',color='gray');axes[0].set(xticks=range(5),xticklabels=cues,ylabel='CE (nats)',title='Archived endpoint330; all16 histories');axes[0].legend()
    axes[1].axhline(0,color='gray',lw=.7);axes[1].set(xticks=range(5),xticklabels=['D_A','D_B','A→B','B→A','sustained'],ylabel='Signed nats',title='Tasks averaged within two seeds')
    for ext in ['png','svg']:fig.savefig(out/('figure_1_outcomes.'+ext),dpi=160,metadata={'Date':None} if ext=='svg' else {})
    plt.close(fig)
    s=pd.read_csv(out/'selected_analysis.csv');analysis=read(out/'evaluation_result.json')['analysis']
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),constrained_layout=True)
    for ax,condition in zip(axes,['candidate','control']):
        g=s[s.condition.eq(condition)]
        if analysis=='acquisition_vs_interference':
            for cue,color in [('A','#1768a0'),('B','#cf741c'),('C','#398b3c'),('NoContext','#8250a3')]:
                h=g[g.cue.eq(cue)].groupby(['seed','end']).gain.mean().reset_index()
                for _,q in h.groupby('seed'):ax.plot(q.end,q.gain,color=color,alpha=.25,lw=.6)
                mean=h.groupby('end').gain.mean();ax.plot(mean.index,mean,label=cue,color=color)
            ax.set(xlabel='Block end update',ylabel='Signed CE gain')
        else:
            for cue,color in [('C','#398b3c'),('NoContext','#8250a3')]:
                h=g[g.cue.eq(cue)].groupby('update')[['ce_gain_static','correct_gain_static']].mean()
                ax.plot(h.index,h.ce_gain_static,label=cue+' CE gain',color=color)
                ax.plot(h.index,h.correct_gain_static,label=cue+' correct gain',color=color,ls='--')
            ax.set(xlabel='Fixed saved update',ylabel='Static-view gain (nats / positions)')
        ax.axhline(0,color='gray',lw=.7);ax.set_title(condition);ax.legend(fontsize=8)
    fig.suptitle(analysis+' — descriptive cached analysis')
    for ext in ['png','svg']:fig.savefig(out/('figure_2_selected.'+ext),dpi=160,metadata={'Date':None} if ext=='svg' else {})
    plt.close(fig)
