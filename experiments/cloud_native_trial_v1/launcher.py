"""Reuse accepted image, bounded Docker lifecycle, read-only same-identity export."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid
from .control import LIMITS,ROOT,definition,prepare,sha,save,validate_definition,verify_source
from .export import verify
from experiments.worker_isolation_preflight_v1.preflight import SOCKETS

CREATE_SECONDS=60
IMAGE=re.compile(r'sha256:[a-f0-9]{64}')
WORKER='/workspace/experiments/cloud_native_trial_v1/worker.py'

def pin_runtime(job,image):
    if not IMAGE.fullmatch(image):raise ValueError('Immutable image required')
    path=Path(job)/'runtime_identity.json';identity=dict(image=image)
    if path.exists():
        if json.loads(path.read_text())!=identity:raise ValueError('Runtime image changed across trial attempts')
    else:save(path,identity)

def create_command(docker,image,job,action,name):
    if not IMAGE.fullmatch(image) or action not in ['run','fixture','export','stop']:raise ValueError('Explicit immutable image and reviewed action required')
    job=Path(job)
    if ',' in str(job):raise ValueError('Unsupported comma in mount path')
    cap=LIMITS['cpu_seconds'] if action=='run' else 120
    command=[*docker,'create','--name',name,'--pull=never','--read-only','--network=none','--cap-drop=ALL','--user=65534:65534','--security-opt=no-new-privileges:true','--pids-limit=32','--memory=2g','--memory-swap=2g','--cpus=1',f'--ulimit=cpu={cap}:{cap}','--ipc=none','--tmpfs=/tmp:rw,noexec,nosuid,size=128m','--workdir=/trial']
    mounts=[('source','/workspace',True),('fixture/source','/worker',True),('fixture/private','/private',True),('native','/trial',action=='export'),(str(Path('exports')/name) if action=='export' else 'fixture/output','/output',False)]
    for rel,target,ro in mounts:
        command+=['--mount',f'type=bind,src={job/rel},dst={target}'+(',readonly' if ro else '')]
    return [*command,image,'python','-I','-B',WORKER,action]

def verify_config(config,command,job,action,image):
    c,h=config['Config'],config['HostConfig'];cap=LIMITS['cpu_seconds'] if action=='run' else 120
    expected=dict(ReadonlyRootfs=True,NetworkMode='none',Privileged=False,PidsLimit=32,Memory=2147483648,MemorySwap=2147483648,NanoCpus=1000000000,PidMode='',IpcMode='none',SecurityOpt=['no-new-privileges:true'],Ulimits=[dict(Name='cpu',Soft=cap,Hard=cap)],Tmpfs={'/tmp':'rw,noexec,nosuid,size=128m'})
    if config['Image']!=image:raise ValueError('Actual image ID differs from pinned runtime')
    if c['User']!='65534:65534' or c['Image']!=image or c.get('Entrypoint') or c['Cmd']!=['python','-I','-B',WORKER,action]:raise ValueError('Wrong worker command/image/identity')
    if any(h.get(k)!=v for k,v in expected.items()) or set(h.get('CapDrop') or [])!={'ALL'} or h.get('CapAdd'):raise ValueError('Effective isolation differs')
    wanted={}
    for i,arg in enumerate(command):
        if arg=='--mount':
            parts=command[i+1].split(',');values=dict(p.split('=',1) for p in parts if '=' in p);wanted[values['dst']]=(values['src'],'readonly' not in parts)
    binds=[m for m in config['Mounts'] if not(m['Type']=='tmpfs' and m['Destination']=='/tmp')]
    if len(binds)!=len(wanted) or set(m['Destination'] for m in binds)!=set(wanted):raise ValueError('Unexpected mount')
    for m in binds:
        if m['Type']!='bind' or (m['Source'],m['RW'])!=wanted[m['Destination']]:raise ValueError('Mount protection mismatch')

def launch(job,image,action,socket):
    job=Path(job).resolve();validate_definition(json.loads((job/'definition.json').read_text()))
    if not IMAGE.fullmatch(image) or socket not in SOCKETS:raise ValueError('Reviewed image/socket required')
    unresolved=job.parent/'.native_trial_unresolved.json'
    if unresolved.exists():raise RuntimeError('Unresolved container cleanup requires adjudication; see '+str(unresolved))
    pin_runtime(job,image)
    expected=verify_source()
    staged=job/'source';hashes=json.loads((staged/'STAGED_FILES.json').read_text())
    for rel,h in expected.items():
        if hashes.get(rel)!=h:raise ValueError('Staged scientific source differs from pinned release: '+rel)
    for rel,h in hashes.items():
        if sha(staged/rel)!=h:raise ValueError('Staged source changed: '+rel)
    if json.loads((staged/'definition.json').read_text())!=definition():raise ValueError('Staged scope changed')
    (job/'fixture/output').mkdir(parents=True,exist_ok=True);(job/'fixture/output').chmod(0o777)
    name='srn-native-'+uuid.uuid4().hex;logs=[];started=time.monotonic();cid=None
    docker=[shutil.which('docker') or 'docker','--host=unix://'+socket]
    env={k:os.environ[k] for k in ['PATH','HOME','LANG'] if k in os.environ}
    def call(argv,timeout=60,check=True):
        try:r=subprocess.run(argv,env=env,text=True,capture_output=True,timeout=timeout,close_fds=True)
        except subprocess.TimeoutExpired:
            logs.append(dict(command=argv,timed_out=True,timeout=timeout));raise
        logs.append(dict(command=argv,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
        if check and r.returncode:raise RuntimeError('Docker command failed: '+str(argv[len(docker)]))
        return r
    receipt=dict(passed=False,action=action,image=image,new_agent_calls=0,provider_usage=None,parent_agent_isolation_certified=False)
    if action=='export':
        destination=job/'exports'/name;destination.mkdir();destination.chmod(0o777)
        receipt['export_directory']=str(destination/'bundle')
    try:
        if not shutil.which('docker'):raise RuntimeError('Docker CLI unavailable; no automatic installation')
        command=create_command(docker,image,job,action,name)
        # Always clean by unique name, even if create times out before returning its ID.
        cid=call(command,timeout=CREATE_SECONDS).stdout.strip()
        if not re.fullmatch('[a-f0-9]{64}',cid):raise ValueError('Invalid container ID')
        config=json.loads(call([*docker,'inspect',cid]).stdout)[0];verify_config(config,command,job,action,image)
        receipt['effective_configuration_verified']=True
        timeout=LIMITS['wall_seconds']+30 if action=='run' else 180
        result=call([*docker,'start','--attach',cid],timeout=timeout,check=False)
        state=json.loads(call([*docker,'inspect','--format','{{json .State}}',cid]).stdout)
        receipt.update(returncode=result.returncode,worker_exited=not state.get('Running'),image_id=config['Image'])
        if result.returncode or state.get('ExitCode')!=0 or state.get('Running'):raise RuntimeError('Worker stopped/incomplete; preserve outputs, no automatic retry')
        receipt['passed']=True
    except Exception as error:receipt.update(reason=str(error),error_type=type(error).__name__)
    finally:
        try:
            cleanup=call([*docker,'rm','--force',name],timeout=60,check=False)
            receipt['container_removed']=cleanup.returncode==0
            if cid and not receipt['container_removed']:receipt['passed']=False
        except Exception as error:receipt.update(passed=False,cleanup_error=str(error))
        if receipt.get('container_removed') is not True:
            save(unresolved,dict(name=name,container_id=cid,job=str(job),manual_adjudication_required=True))
        receipt['wall_seconds']=time.monotonic()-started
        audit=job/'launches'/name;audit.mkdir(parents=True)
        save(audit/'docker_commands.json',logs);save(audit/'RECEIPT.json',receipt)
    return receipt

def run_and_export(job,image,action,socket):
    # One slot spans numerical execution AND checksum export. Stop only writes a flag.
    lock=None
    try:
        if action!='stop':
            lock=(Path(job).parent/'.native_trial.lock').open('a+')
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        receipt=launch(job,image,action,socket)
        if action in ['run','fixture'] and receipt.get('container_removed') is True:
            export=launch(job,image,'export',socket)
            receipt['export']=export
            if export['passed']:receipt['verified_export']=verify(export['export_directory'])
            if not export['passed']:receipt['passed']=False
            save(Path(job)/'LATEST_LAUNCH.json',receipt)
        elif action=='export' and receipt['passed']:
            receipt['verified_export']=verify(receipt['export_directory'])
        return receipt
    finally:
        if lock:lock.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run','fixture','export','stop','status'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--image');p.add_argument('--socket',choices=SOCKETS,default=SOCKETS[0]);a=p.parse_args()
    if a.action=='prepare':result=prepare(a.out)
    elif a.action=='status':
        path=a.out/'LATEST_LAUNCH.json';result=json.loads(path.read_text()) if path.exists() else dict(status='prepared_or_running',hint='Native state is container-owned; do not infer completion from an absent host receipt')
    else:
        if not a.image:p.error('--image must be the accepted immutable image ID')
        result=run_and_export(a.out,a.image,a.action,a.socket)
    print(json.dumps(result,indent=2));return 0 if result.get('passed',result.get('prepared',False)) else 3

if __name__=='__main__':raise SystemExit(main())
