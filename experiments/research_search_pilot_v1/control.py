"""Prepare the pilot; refuse initialization until all live-call gates pass.

This module never invokes an agent or scientific training. The failed connection
attempt must not be retried indirectly through a pilot command.
"""
import sqlite3
from pathlib import Path
from research_search.common import ROOT,PACKAGE,read,write,sha,fingerprint

OUTPUT=ROOT/'results/research_search_pilot/v1'
CAMPAIGN=PACKAGE/'database/controlled_pilot_v1'
PROBE=PACKAGE/'database/backend_validation_v2'
LIMITS=dict(tokens=50000,cpu_seconds=3600,optimizer_steps=6000,wall_seconds=7200,proposals=2,active_branches=1)


def definition():
    return dict(version='controlled-pilot-v1',status='prepared_not_initialized',limits=LIMITS.copy(),
        target_updates=330,development_seeds=[21000,21001],development_bank='registered development-v1, all4 unrelated tasks',
        development_bank_sha=read(PACKAGE/'audit/registered_bank_commitments.json')['files']['development.json'],
        native_evaluator_seal_sha=sha(PACKAGE/'audit/evaluator_seal.json'),
        native_runner_hash=read(PACKAGE/'audit/evaluator_seal.json')['runner_hash'],
        native_implementation_manifest_sha=sha(PACKAGE/'IMPLEMENTATION_MANIFEST.json'),
        campaign_directory=str(CAMPAIGN.relative_to(ROOT)),
        confirmation_authorized=False,canonical_target_authorized=False,full_campaign_authorized=False,
        manual_approval_each_agent_call=True,stop_on_missing_usage=True,stop_after_one_complete_iteration=True,
        prior_episodes=0,matched_control='same zero-prior native baseline; one registered initialization or target-learning change',
        ordinary_objective_unchanged=True,all_AB_targets_only=True,full_five_cue_evaluator_unchanged=True,
        tier_expansion_authorized=False,
        optimizer_accounting=dict(targets=5280,maximum_unrelated_competence=240,total_planned_maximum=5520,ceiling=6000),
        proposed_call_reservations=dict(proposal=4000,proposal_critic=3000,implementation_review=4000,result_critic=5000),
        proposed_token_allocations=dict(setup=0,hypothesis=10000,criticism=18000,implementation=10000,reporting=12000),
        token_scope='New pilot service calls only; earlier connection usage remains independently unresolved, never zero-imputed.',
        workflow=['manually approve hypothesis/proposal call','validate immutable candidate and same-dose control',
                  'manually approve proposal critic','manually approve implementation review',
                  'queue ordinary target learning on registered unrelated tasks','independent frozen native evaluation',
                  'manually approve result critic feedback','stop and owner report'],
        readiness='BLOCKED: original connection failed with unknown usage; no verified provider hard token cap',
        implementation_scope='Preparation and fail-closed initialization only. Live pilot dispatch needs versioned backend repair and enforcement, then manual approvals. No scientific jobs created.')


def unresolved_probe():
    p=PROBE/'campaign.sqlite'
    if not p.exists():return []
    db=sqlite3.connect(f'file:{p}?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    try:return [dict(r) for r in db.execute("SELECT id,status,reserved,total,cached FROM calls WHERE status!='reconciled'")]
    finally:db.close()


def gate(readiness,unresolved,hard_token_cap_verified):
    if not readiness.get('live_backend_verified'):raise RuntimeError('Live connection has not passed; no pilot may be initialized or dispatched')
    if unresolved:raise RuntimeError('Prior connection usage is unresolved; preserve reservations and stop actual agent calls')
    if hard_token_cap_verified is not True:raise RuntimeError('A hard reported-token ceiling cannot be guaranteed by the current native backend; do not silently weaken the requested50000 maximum')


def prepare(output=OUTPUT):
    output=Path(output);d=definition();write(output/'PILOT_DEFINITION.json',d,True)
    return dict(prepared=True,operational_campaign_created=False,definition=str(output/'PILOT_DEFINITION.json'),limits=d['limits'],execution_ready=False)


def status():
    from research_search.preflight import verify_artifacts,readiness
    verified=verify_artifacts();r=readiness();u=unresolved_probe()
    return dict(version='controlled-pilot-v1',sealed_source_verification=verified,live_backend_verified=r['live_backend_verified'],
        unresolved_probe=u,operational_campaign_created=CAMPAIGN.exists(),execution_ready=False,
        hard_provider_token_ceiling_verified=False,larger_campaign_ready=False,limits=LIMITS)


def initialize(directory=CAMPAIGN,*,readiness=None,unresolved=None,hard_token_cap_verified=False):
    # Injectable observations support local negative tests; CLI never supplies
    # overrides. A proof file fabricated by an agent is not accepted here.
    if readiness is None:
        from research_search.preflight import readiness as inspect
        readiness=inspect()
    if unresolved is None:unresolved=unresolved_probe()
    gate(readiness,unresolved,hard_token_cap_verified)
    # v1 deliberately has no scientific execution path: the native backend has
    # no verified hard token limiter. A future runnable adapter needs a new
    # version and retained tests, not an edit to this prepared definition.
    raise RuntimeError('Pilot dispatch adapter requires separately versioned backend repair/enforcement; no campaign created')
