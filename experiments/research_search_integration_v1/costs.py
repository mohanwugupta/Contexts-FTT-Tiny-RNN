"""Cost planning from completed receipts/stat sizes; no benchmark replay."""
from research_search.common import read,write,sha
from .protocol import SOURCE,PUBLIC,LIMITS,OUT


def estimate():
    pairs=read(SOURCE/'paired_jobs.json');jobs=sum(pairs.values(),[])
    receipts=[read(p) for p in (SOURCE/'logs').glob('job-*.json')]
    # Runtime receipts use native global accounting, not the six-update fixture.
    worker_cpu=sum(r['cpu_seconds'] for r in receipts)
    target_cpu=sum(read(p)['cpu_seconds'] for p in (SOURCE/'logs').glob('job-*.json') if any(j in p.name for j in jobs))
    sizes={j:sum(p.stat().st_size for p in (SOURCE/'jobs'/j).rglob('*') if p.is_file()) for j in jobs}
    prior_files=[p for p in (SOURCE/'logs').glob('job-*.json') if 'init' in p.name]
    startup_cpu=sum(read(p)['cpu_seconds'] for p in prior_files)/len(prior_files)
    state_bytes=sum(p.stat().st_size for j in jobs for p in (SOURCE/'jobs'/j).glob('*.pt'))
    status=read(PUBLIC/'FINAL_STATUS.json');usage=read(PUBLIC/'AGENT_USAGE.json')
    rate=target_cpu/5280
    # Two independent seed priors are shared across target-only interventions.
    # Prior-affecting changes require four priors. Actual prior throughput is
    # unknown: these are explicit target-rate surrogates, not measurements.
    rows=[]
    for prior_copies in [2,4]:
        updates=1920*20*prior_copies+5280
        rows.append(dict(prior_copies=prior_copies,ordinary_reptile_episodes_per_seed=1920,prior_optimizer_steps=1920*20*prior_copies,target_optimizer_steps=5280,total_steps=updates,target_rate_surrogate_CPU_seconds=updates*rate,conservative_3x_CPU_seconds=updates*rate*3,prior_runtime_measured=False,competence_established=False))
    result=dict(source_status_sha=sha(PUBLIC/'FINAL_STATUS.json'),source_usage_sha=sha(PUBLIC/'AGENT_USAGE.json'),historical_target_histories=16,historical_target_updates=5280,historical_target_worker_CPU_seconds=target_cpu,historical_worker_CPU_seconds=worker_cpu,target_CPU_seconds_per_update_including_startup_serialization_frozen_evaluation=rate,zero_update_process_CPU_mean=startup_cpu,historical_full_elapsed_seconds=status['full_elapsed_seconds'],historical_recorded_CPU_seconds=status['recorded_CPU_seconds'],elapsed_minus_CPU_not_identifiable_as_approval_time=status['full_elapsed_seconds']-status['recorded_CPU_seconds'],historical_service_reported_tokens=usage['total_reported_tokens'],historical_job_storage_bytes=sizes,target_storage_total_bytes=sum(sizes.values()),full_state_checkpoint_bytes=state_bytes,bytes_per_target_history=sum(sizes.values())/16,checkpoint_duplication='states.pt and progress.pt both retained in measured bytes',full_pretrained_candidate_and_control=rows,prior_storage='Unmeasured; depends on native prior checkpoint layout. Never extrapolate all-target-state bytes as measured prior storage.',unseparated_costs=['frozen evaluation vs target learning vs serialization not separately timed','Python startup included in each worker; zero-update jobs provide only an approximate startup cost','Reptile prior throughput/storage and background competence not measured in controlled pilot','manual approval time uncertain; included in six-hour elapsed ceiling','service latency not CPU time'],admission='cached_only_no_new_training',reason='No validated full-prior timing or competence evidence on these development tasks; target-rate surrogate already exceeds six hours. Do not shorten prior dose or interpret zero-prior results as competent-prior failure.',registered_limits=LIMITS,planned_service_calls=5,planned_new_optimizer_steps=0,planned_new_model_forwards=0,planned_cached_worker_CPU_cap=120,planned_agent_CPU_caps=450,remaining_CPU_headroom=330)
    write(OUT/'COMPUTE_ESTIMATE.json',result,True)
    text=f'''# Registered compute estimate\n\nCompleted pilot:16 target histories/5280 updates used {target_cpu:.3f} target-worker CPU seconds; {rate:.6f} CPU seconds per update includes full-state serialization, frozen probes and Python startup. Target directories occupy {sum(sizes.values()):,} bytes, including {state_bytes:,} bytes in retained .pt state bundles. A zero-update worker averaged {startup_cpu:.3f} CPU seconds. These replace the old six-update speed estimate.\n\nA complete1920-episode matched pair needs at least two shared seed priors for target-only changes:76800 prior updates plus5280 target updates. At the observed target rate the surrogate is {rows[0]['target_rate_surrogate_CPU_seconds']/3600:.2f} CPU hours; for prior-affecting changes four priors require {rows[1]['target_rate_surrogate_CPU_seconds']/3600:.2f} hours. Prior throughput, serialization/storage and background competence are unmeasured here: these are planning surrogates, not a measured lower bound or proof of impossibility. Conservative3x costs remain explicit in the JSON.\n\nThe admitted test is cached-only:0 updates,0 forwards,900 total CPU seconds maximum,120-second worker cap,21600 elapsed seconds including manual approvals,100000 service tokens across five individually approved calls. It tests coordination/scoring and produces no new trained candidate or adequate-prior negative evidence. Never silently reduce1920 episodes.\n\nPrevious pilot elapsed {status['full_elapsed_seconds']:.3f}s and CPU {status['recorded_CPU_seconds']:.3f}s do not identify approval time separately. Agent calls consumed35138 reported tokens; five planned calls retain100000 total reserves. Setup/UI usage is unavailable, separately disclosed; no unsupported hard in-turn cap or API fallback.\n'''
    (OUT/'COMPUTE_ESTIMATE.md').write_text(text)
    return result
