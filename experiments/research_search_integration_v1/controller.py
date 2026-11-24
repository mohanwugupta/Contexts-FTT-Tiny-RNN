"""Five manual turns, one durable cached job, then terminal stop."""
import json,shutil,time
from pathlib import Path
import jsonschema
from research_search.common import ROOT,PACKAGE,read,write,sha,clean
from research_search.preflight import verify_artifacts
from research_search.orchestration.store import Store
from research_search.orchestration.queue import enqueue
from research_search.orchestration.budget import BudgetError
from .protocol import *
from .agent import invoke


def code_files():
    files=list(Path(__file__).parent.glob('*.py'))+list((ROOT/'tests/research_search_integration_v1').glob('*.py'))+[ROOT/'docs/research_search_integration_v1/PRD.md',ROOT/'docs/research_search_integration_v1/EVALUATION_SPEC.md']
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}


def source_files():
    files=[PUBLIC/'manifest.json',PUBLIC/'PRIVATE_EVIDENCE_FILES.json',SOURCE/'paired_jobs.json',SOURCE/'development_result.json',SOURCE/'development.json',PACKAGE/'audit/evaluator_seal.json',PACKAGE/'IMPLEMENTATION_MANIFEST.json']
    for j in sum(read(SOURCE/'paired_jobs.json').values(),[]):
        files.extend(SOURCE/'jobs'/j/n for n in ['metrics.csv','positions.csv','evaluation.json','static_readout_metrics.csv','static_readout_positions.csv','blocks.csv','switch_costs.csv','matched_acquisition.csv'])
    for n in ['HISTORY_INDEX_v2.json','OPEN_HYPOTHESES.json','SCREEN_HYPOTHESES_v2.json','KNOWN_FAILURES.md']:
        files.append(PACKAGE/'history'/n)
    return {str(p.relative_to(ROOT)):sha(p) for p in files}


def initialize():
    verify_artifacts()
    for rel,digest in read(PUBLIC/'manifest.json')['files'].items():
        if sha(ROOT/rel)!=digest:raise RuntimeError('Historical public artifact changed: '+rel)
    sources=source_files()
    # Verify every consumed private file against the historical seal.
    old=read(PUBLIC/'PRIVATE_EVIDENCE_FILES.json')['private_files']
    for rel,digest in sources.items():
        if rel in old and old[rel]['sha256']!=digest:raise RuntimeError('Historical private input changed')
    definition=dict(version='multi-agent-integration-v1',code=code_files(),sources=sources,limits=LIMITS,reservations=RESERVATIONS,roles=PHASES,only_cached_development=True,confirmation_authorized=False,full_campaign_authorized=False,training_authorized=False,manual_approval_every_call=True,hard_in_turn_token_cap=False,scope='two specialists, coordinator selection, critic, coordinator review; no automatic follow-up')
    write(DIRECTORY/'definition.json',definition,True);write(DIRECTORY/'budgets.json',LIMITS,True)
    s=Store(DIRECTORY,LIMITS)
    if s.get('phase') is None:
        s.set('phase',PHASES[0]);s.set('manual_checkpoints',True);s.event('registered',definition)
    return s


def integrity(store):
    d=read(store.directory/'definition.json')
    if d['code']!=code_files() or d['sources']!=source_files() or store.limits!=LIMITS:raise RuntimeError('Immutable integration identity changed')
    verify_artifacts()


def history():
    # Compact evidence-linked history; no confirmation inputs/outcomes.
    return dict(index='research_search/history/HISTORY_INDEX_v2.json',open=read(PACKAGE/'history/OPEN_HYPOTHESES.json'),qualifications=(PACKAGE/'history/KNOWN_FAILURES.md').read_text()[:1800],pilot=dict(source='results/research_search_pilot/v2/REPORT.md',design='four unrelated development sequences, two shared initialization seeds, native Linear H8, zero prior, half initial readout vs1, all331 frozen states',result='C CE66.23→32.44, correct3.625→3.25/7, no useful C; A/B/Mixed learned. NoContext CE5.67→2.94; previous late/screen unresolved. Position accounting0 corrected,3 newly wrong. Partial confidence signal, failed joint prediction. Do not repeat endpoint accounting as novel discovery.'),earlier_A06='results/a06_cached_readout_comparison/v1/REPORT.md and results/a06_novel_cue_followup/v1/REPORT.md: learned trajectory versus static quarter readout already studied; context robustness remains incomplete.',earlier_transfer='results/ab_update_transfer_analysis/v1/REPORT.md: early A/B practice can benefit C/NoContext, later deterioration occurs in some pretrained models; different cohort, do not pool.')


def prepare(store):
    integrity(store);role=store.get('phase')
    if role not in PHASES:raise RuntimeError('No pending approved agent stage')
    context=dict(history=history(),four_outcome='Same seed/task/network at330: A/B each CE≤.5 and≥6/7, unchanged original early phenotype, C/Mixed-average/NoContext each CE≤.5 and≥6/7. Switching/net/floors separate. Preserve old joint/screen and unresolved reference as missing.',budget=LIMITS,scope='Cached-only integration test; no new training/forwards/confirmation/canonical targets. Scientific outcome inconclusive for trained priors; preserve negative evidence. At most700 words, prefer250. No follow-up execution.',compute=read(OUT/'COMPUTE_ESTIMATE.json')['admission'])
    if role.startswith('specialist_'):
        area=role.split('_',1)[1];context['your_research_area']=area;context['analysis_contract']=ANALYSES[area]
        context['contract_description']='Compare archived native vs bias-preserving static quarter-readout at fixed0,10,20,40,80,120,170,330. Preserve per-position correctness and margins. Ask whether loss changes mainly reflect penalties on still-wrong outputs; distinguish expression from learning; endpoint portion duplicates known evidence.' if area=='initialization' else 'Use all33 A10/B10 block boundaries and five cues: signed loss/correct-count changes, starting errors and floor0.05 flags. Ask whether C/NoContext errors persist from initialization or appear after beneficial practice, without causal mediation claims; this developmental source analysis has not yet been executed here.'
        instruction='Generate one mechanistic hypothesis from your distinct research area and falsifiable cached analysis proposal within the assigned contract. Acknowledge prior equivalent analyses; do not invent novelty or claim a new learned candidate.'
    elif role=='coordinator_select':
        context['proposals']=[read(store.directory/f'{r}.json') for r in PHASES[:2]]
        context['estimated_cost']='Both0 updates/0 forwards; cached worker cap120CPU seconds. Full1920 prior timing/competence unvalidated; target-rate surrogate exceeds6h. Choose exactly one based on informativeness and duplication. Keep unselected proposal.'
        instruction='Compare independent specialist hypotheses, check named evidence for duplicates, choose one unchanged registered analysis, explain scientific cost and value. The trusted queue submits that choice. No threshold/config changes.'
    elif role=='critic':
        context['selection']=read(store.directory/'coordinator_select.json');context['result']=read(store.directory/'evaluation_result.json')
        instruction='Identify strongest four-outcome failure/tradeoff, compare prediction to evidence, give at least two competing explanations and one smallest design-only follow-up. Distinguish previously known evidence from new cached descriptive findings; no new mechanistic causal claim.'
    else:
        context['selection']=read(store.directory/'coordinator_select.json');context['result']=read(store.directory/'evaluation_result.json');context['critic']=read(store.directory/'critic.json')
        instruction='Review evidence and critic feedback, decide reject/revise-design-only/investigate-later/inconclusive and record why. Stop now; never queue another experiment. Be specific about missing competence and weak two-seed inference.'
    prompt='You are the '+role+' in a bounded scientific coordination test. Use no tools, files, network, shell, apps or nested agents. Return only schema JSON. '+instruction+'\n'+json.dumps(clean(context),separators=(',',':'))
    if len(prompt)>12000:raise ValueError('Compact context exceeded; no silent truncation of results')
    ticket='integration1-'+role;path=store.directory/'requests_agent'/f'{ticket}.json'
    request=dict(ticket=ticket,role=role,phase=role,prompt=prompt,schema=schema(role),reservation=RESERVATIONS[role],codex_sha256=sha(Path(shutil.which('codex')).resolve()),definition_sha=sha(store.directory/'definition.json'))
    write(path,request,True)
    approval=dict(ticket=ticket,role=role,approval_sha=sha(path),reservation=RESERVATIONS[role],prompt_characters=len(prompt),remaining_tokens=LIMITS['tokens']-store.totals()['tokens']-store.totals()['reserved_tokens'],command=f'PYTHONPATH=.:src python -m experiments.research_search_integration_v1 call --ticket {ticket} --approval {sha(path)}')
    write(OUT/f'APPROVAL_{role}.json',approval,True)
    return approval


def accept_reply(store,role,reply):
    if store.get('phase')!=role:raise ValueError('Out-of-order agent reply')
    try:jsonschema.validate(reply,schema(role))
    except jsonschema.ValidationError as ex:raise ValueError('Invalid role reply') from ex
    if role.startswith('specialist_'):validate_proposal(reply,role.split('_',1)[1])
    if role=='critic' and len(reply['competing_explanations'])<2:raise ValueError('Two critic alternatives required')
    write(store.directory/f'{role}.json',reply,True);store.event('role_reply',dict(role=role,reply=reply))
    if role=='coordinator_select':
        selected=reply['selected'];p=read(store.directory/('specialist_'+selected.removesuffix('-proposal')+'.json'))
        # Register both proposals as inactive history, then the selected analysis
        # as the only active immutable candidate: total3, active1.
        store.control('ready')
        for area in ANALYSES:
            proposal=read(store.directory/f'specialist_{area}.json')
            store.register(dict(id=proposal['id'],parent_candidate=None,config=dict(analysis=proposal['analysis']),proposal=proposal),'cached-spec-v1')
            store.db.execute("UPDATE nodes SET status='unselected_record' WHERE id=?",(proposal['id'],))
        store.register(dict(id='selected-cached-analysis',parent_candidate=selected,config=dict(analysis=p['analysis']),selection=reply),'cached-spec-v1')
        enqueue(store,'integration1-cache','selected-cached-analysis',dict(kind='cached_integration',analysis=p['analysis']))
        store.set('phase','compute_ready');store.control('manual_checkpoint')
    elif role=='coordinator_review':
        store.set('stopped_at',time.time());store.set('phase','complete');store.db.execute("UPDATE nodes SET status='closed' WHERE status='active'");store.control('stopped')
    else:
        store.set('phase',PHASES[PHASES.index(role)+1]);store.control('manual_checkpoint')


def call(store,ticket,approval):
    integrity(store);p=store.directory/'requests_agent'/f'{ticket}.json';r=read(p)
    if r['definition_sha']!=sha(store.directory/'definition.json') or r['phase']!=store.get('phase'):raise RuntimeError('Stale request')
    result=invoke(store,ticket,approval)
    if result['passed']:
        try:accept_reply(store,r['role'],result['reply'])
        except (ValueError,KeyError) as ex:
            store.set('stopped_at',time.time());store.set('phase','invalid_reply');store.control('stopped');store.event('invalid_reply',dict(error=str(ex),raw=result))
    else:
        store.set('stopped_at',time.time());store.set('phase','agent_failed_or_unknown');store.control('stopped')
    return result|dict(phase=store.get('phase'))
