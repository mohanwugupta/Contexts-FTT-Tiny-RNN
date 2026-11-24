import json,sys
from pathlib import Path
from research_search.common import ROOT
from experiments.research_backend_repair_v1.backend import create_runtime,run


def test_actual_repaired_sandbox_preserves_all_boundaries(tmp_path):
    original=tmp_path/'original';original.mkdir()
    auth=original/'auth.json';auth.write_text('fixture-only-credential')
    protected=original/'evaluator.py';protected.write_text('fixture evaluator')
    rt=create_runtime(tmp_path/'runtime',auth)
    (rt.workspace/'leak').symlink_to(protected)
    code='''import pathlib,json,sys,os,socket,subprocess
w,source,protected,repo=map(pathlib.Path,sys.argv[1:]);r={}
r['auth_read_through_link']=(w/'codex-home/auth.json').read_text()=='fixture-only-credential'
for name,p,action in [
 ('old_home_read_denied',protected,'read'),('repo_read_denied',repo/'research_search/evaluation/scoring.py','read'),
 ('confirmation_read_denied',repo/'research_search/database/benchmark_banks/sealed/confirmation.json','read'),
 ('symlink_escape_denied',w/'leak','read'),('auth_write_denied',source,'write'),
 ('auth_link_write_denied',w/'codex-home/auth.json','write'),('evaluator_write_denied',protected,'write'),
 ('old_home_runtime_write_denied',source.parent/'new-runtime','write')]:
 try:
  p.read_text() if action=='read' else p.write_text('tampered');r[name]=False
 except PermissionError:r[name]=True
try:os.unlink(w/'codex-home/auth.json');r['auth_unlink_denied']=False
except PermissionError:r['auth_unlink_denied']=True
try:os.link(source,w/'hardlink');r['hardlink_denied']=False
except PermissionError:r['hardlink_denied']=True
try:list(source.parent.iterdir());r['old_home_listing_denied']=False
except PermissionError:r['old_home_listing_denied']=True
try:subprocess.run(['/bin/sh','-c','true'],check=True);r['shell_exec_denied']=False
except PermissionError:r['shell_exec_denied']=True
try:socket.create_connection(('127.0.0.1',9),timeout=.2);r['network_denied']=False
except PermissionError:r['network_denied']=True
for part in ['tmp','state','logs','codex-home']:(w/part/'allowed').write_text('allowed')
r['runtime_writes_allowed']=True
print(json.dumps(r))
'''
    result=run([sys.executable,'-c',code,str(rt.workspace),str(auth),str(protected),str(ROOT)],rt,network=False)
    assert result['returncode']==0,result['stderr']
    assert all(json.loads(result['stdout']).values()),result
    assert auth.read_text()=='fixture-only-credential' and protected.read_text()=='fixture evaluator'


def test_timeout_and_cancellation_do_not_leave_process_running(tmp_path):
    h=tmp_path/'original';h.mkdir();a=h/'auth.json';a.write_text('{}')
    rt=create_runtime(tmp_path/'runtime',a)
    r=run([sys.executable,'-c','import time; time.sleep(20)'],rt,timeout=.1)
    assert r['timed_out'] and r['returncode']!=0 and r['wall_seconds']<3
    r=run([sys.executable,'-c','import time; time.sleep(20)'],rt,timeout=5,cancel_check=lambda:True)
    assert r['cancelled'] and r['returncode']!=0
