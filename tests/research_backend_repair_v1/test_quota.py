import sys
from experiments.research_backend_repair_v1.backend import create_runtime,rpc_quota


def test_bounded_quota_rpc_is_readonly_and_drains_notifications(tmp_path):
    h=tmp_path/'original';h.mkdir();a=h/'auth.json';a.write_text('{}');rt=create_runtime(tmp_path/'runtime',a)
    fake=rt.workspace/'fake_rpc.py'
    fake.write_text('''import json,sys,os
for line in sys.stdin:
 r=json.loads(line)
 if r['method']=='initialize':
  os.write(1,b'{"id":0,"result":{}}\\n{"method":"notification"}\\n')
 elif r['method']=='initialized':pass
 elif r['method']=='account/rateLimits/read':
  assert r['params']=={'supportsLunaReserve':False,'excludeResetCreditDetails':True}
  print(json.dumps({'id':1,'result':{'ordinaryUsageAllowed':True,'accountPrivate':'must-not-export','rateLimits':{'primary':{'usedPercent':1}}}}),flush=True)
 else:raise RuntimeError('Unexpected RPC; not read only')
''')
    r=rpc_quota(rt,command=[sys.executable,str(fake)],network=False)
    assert r['available'] and r['ordinary_usage_allowed'] is True
    assert 'must-not-export' not in str(r) and r['paid_fallback_requested'] is False
    assert r['CPU_seconds']>=0
