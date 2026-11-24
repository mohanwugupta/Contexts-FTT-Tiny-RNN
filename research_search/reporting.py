"""Compact periodic/final reports, with a full immutable event/tree export."""
import json,time
from .common import *
from .orchestration.experiment_tree import export,frontier

def report(store,final=False):
    tree=export(store);write(store.directory/'experiment_tree.json',tree)
    records=[r['value'] for r in tree['results']];development=[r for r in records if r.get('phase')=='development'];vectors=[dict(id=r['id'],**r['summary']) for r in development if 'summary' in r]
    from .orchestration.coordinator import DIRECTIONS
    front=frontier(vectors,DIRECTIONS);last=store.get('last_report') or 0;now=time.time();write(store.directory/'pareto.json',dict(all_nondominated=front,active_frontier=front[:8],criteria=DIRECTIONS,missing_outcomes_not_zero=True))
    compact=[{k:v for k,v in r.items() if k not in ['individual_results','matched_control_results','cohort','coverage']} for r in (development if final else development[-4:])]
    text=f'''# {'Final research report' if final else 'Campaign progress'}

## Central scientific question
Can ordinary learning combine context-specific A/B acquisition with C and physical Mixed transfer? NoContext is diagnostic.

## Experiment-search history
{len(tree['nodes'])} immutable candidates; {len(tree['events'])} retained events. See experiment_tree.json and jobs/. Duplicate and failed proposals remain in events.

## Strongest candidate models
Nondominated development IDs: {front}. These are exploratory and are not confirmed human mechanisms.

## Splitting and transfer results
{json.dumps(compact,indent=2)}
Full seed/task rows, negative signs and missing coverage are in candidate development JSON/CSVs and experiment_tree.json.

## Negative results and failed hypotheses
Failed/paused/uncertain jobs: {json.dumps([dict(r) for r in store.db.execute("SELECT id,candidate,status,error FROM jobs WHERE status!='completed'")])}. Missing values and inadequate background competence are not zero effects or hypothesis refutations.

## Mechanistic ablations
Only precommitted matched-control contrasts may support a mechanism. Development ablation flags above are separate from final confirmation and the unassessed late functional gate.

## Reproducibility and compute costs
{json.dumps(store.state(),indent=2)}
Reported token usage comes from the call ledger; cached input is a subset. Unknown usage retains its reservation and requires adjudication. This CLI cannot guarantee a hard token ceiling inside an unfinished call. Manual checkpoints are mandatory. Active CPU/update/wall ceilings and optimizer attempts are durable. No subscription allowance or API entitlement is inferred.
Implementation-session token telemetry is unavailable: actual setup usage is null, with a separate100000-token planning reservation, never reported as actual usage. The complete global ceiling therefore cannot be guaranteed until setup and each call have reliable aggregate telemetry.

## Remaining uncertainty
Development selection, small seed count, repeated tasks, acquisition floors, control competence, missing/nonfinite outputs and adapter coverage constrain conclusions. Confirmation outcomes remain under sealed/ and are not sent to research agents.

## Smallest recommended confirmatory experiment
Use a frozen replicated development family and its single-mechanism ablation once on the registered held-out task/seed bank. If none qualifies, retain the negative result and do not launch confirmation.
'''
    dest=store.directory/'reports'/('FINAL_RESEARCH_REPORT.md' if final else f'PROGRESS_{int(now)}.md');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text);store.set('last_report',now);return str(dest)
