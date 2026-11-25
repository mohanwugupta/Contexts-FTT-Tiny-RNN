import copy,json,tempfile,unittest
from pathlib import Path
from experiments.cloud_role_runner_v1 import contract as c

class Contracts(unittest.TestCase):
    def test_hosts_and_authorities(self):
        self.assertEqual(c.authority('chatgpt.com:443'),('chatgpt.com',443))
        for s in ['api.openai.com:443','evil.test:443','chatgpt.com.evil:443','chatgpt.com:80','127.0.0.1:443','chatgpt.com.:443','user@chatgpt.com:443','chatgpt.com:443/path']:
            with self.subTest(s=s),self.assertRaises(ValueError):c.authority(s)
    def test_auth_never_api(self):
        good={'auth_mode':'chatgpt','tokens':{'access_token':'SECRET_ACCESS','refresh_token':'SECRET_REFRESH','id_token':'SECRET_ID'}}
        self.assertEqual(c.auth_bytes(json.dumps(good).encode()),json.dumps(good).encode())
        for a in [{'OPENAI_API_KEY':'key'},good|{'OPENAI_API_KEY':'key'},{'auth_mode':'api','tokens':good['tokens']}]:
            with self.assertRaises(ValueError):c.auth_bytes(json.dumps(a).encode())
    def test_argv_no_tools_and_no_resume(self):
        a=c.codex_argv('exec')
        for k in ['--ephemeral','--ignore-user-config','--ignore-rules','--json','--output-schema']:self.assertIn(k,a)
        self.assertIn('forced_login_method="chatgpt"',a)
        self.assertIn('web_search="disabled"',a)
        self.assertNotIn('resume',a);self.assertNotIn('--dangerously-bypass-approvals-and-sandbox',a)
    def test_deadline_is_not_started_or_reset(self):
        self.assertEqual(c.remaining(100,21600,200),21500)
        for start in [None,True,300]:
            with self.assertRaises(ValueError):c.remaining(start,21600,200)
        with self.assertRaises(ValueError):c.remaining(100,10,200)
    def test_redaction_does_not_mutate_usage(self):
        value={'auth_mode':'chatgpt','tokens':{'access_token':'super-private-token-string'}}
        text='super-private-token-string {"input_tokens":7,"output_tokens":3}'
        red=c.redact(text,json.dumps(value).encode());self.assertNotIn('super-private',red);self.assertIn('"input_tokens":7',red)
    def test_provider_usage_unknown_and_multiple_turns(self):
        from experiments.cloud_role_runner_v1.accounting import assess
        good=[{'type':'turn.started'},{'type':'item.completed','item':{'type':'agent_message','text':'{"ok":true}'}},{'type':'turn.completed','usage':dict(input_tokens=100,output_tokens=20,cached_input_tokens=80)}]
        r=assess(good,0,200);self.assertTrue(r['passed']);self.assertEqual(r['reported_tokens'],120)
        self.assertFalse(assess(good[:-1],0,200)['passed'])
        self.assertFalse(assess(good+[good[-1]],0,200)['passed'])
        self.assertFalse(assess(good,0,119)['passed'])
    def test_tool_calls_rejected(self):
        from experiments.cloud_role_runner_v1.accounting import assess
        ev=[{'type':'item.completed','item':{'type':'command_execution','text':'x'}},{'type':'turn.completed','usage':dict(input_tokens=1,output_tokens=1,cached_input_tokens=0)}]
        self.assertFalse(assess(ev,0,100)['passed'])
    def test_mount_contract(self):
        from experiments.cloud_role_runner_v1.docker import command
        a=command(['docker'],'sha256:'+'a'*64,Path('/tmp/job'),'role','n','internal-net','172.20.0.2')
        self.assertIn('--network=internal-net',a);self.assertIn('--dns=127.0.0.1',a)
        self.assertIn('--user=65534:65534',a);self.assertIn('--read-only',a)
        for s in a:self.assertFalse('/.codex' in s or 'docker.sock' in s or 'dst=/workspace' in s)
    def test_process_filter_has_no_socket_or_tls_bypass(self):
        from experiments.cloud_role_runner_v1.bootstrap import RULES
        self.assertIn('fork',RULES);self.assertIn('clone3',RULES);self.assertNotIn('socket',RULES)

class ApprovalTests(unittest.TestCase):
 def fixture(self,d):
  import time
  from research_search.orchestration.store import Store
  from experiments.cloud_role_runner_v1.contract import save,sha
  root=Path(d).resolve();s=Store(root/'campaign',dict(tokens=100000,cpu_seconds=900,optimizer_steps=0,wall_seconds=21600,proposals=3,active_branches=1));s.set('started',time.time());s.set('phase','specialist_initialization');s.set('manual_checkpoints',True)
  save(s.directory/'definition.json',dict(version='fixture'))
  runtime=dict(directory=str(root/'runtime'),synthetic=False,codex_sha256='a'*64)
  save(Path(runtime['directory'])/'RUNTIME.json',runtime)
  req=dict(ticket='t1',role='specialist_initialization',phase='specialist_initialization',definition_sha=sha(s.directory/'definition.json'),codex_sha256='a'*64,prompt='Proposal only',schema={'type':'object'},reservation=19000)
  save(s.directory/'requests_agent/t1.json',req);return s,runtime
 def test_old_request_approval_cannot_approve_new_transport(self):
  from experiments.cloud_role_runner_v1.agent import validate_request,prepare_approval
  with tempfile.TemporaryDirectory() as d:
   s,r=self.fixture(d);p=prepare_approval(s,'t1',r)
   with self.assertRaises(PermissionError):validate_request(s,'t1',c.sha(s.directory/'requests_agent/t1.json'),r)
   self.assertEqual(validate_request(s,'t1',p['approval_sha'],r)['ticket'],'t1');s.close()
 def test_runtime_change_invalidates_approval(self):
  from experiments.cloud_role_runner_v1.agent import validate_request,prepare_approval
  with tempfile.TemporaryDirectory() as d:
   s,r=self.fixture(d);p=prepare_approval(s,'t1',r);c.save(Path(r['directory'])/'RUNTIME.json',r|{'changed':True})
   with self.assertRaises((PermissionError,ValueError)):validate_request(s,'t1',p['approval_sha'],r)
   s.close()
 def test_prepare_does_not_start_missing_clock(self):
  from experiments.cloud_role_runner_v1.agent import prepare_approval
  with tempfile.TemporaryDirectory() as d:
   s,r=self.fixture(d);s.set('started',None)
   with self.assertRaises(ValueError):prepare_approval(s,'t1',r)
   self.assertIsNone(s.get('started'));s.close()

class LedgerTests(ApprovalTests):
 def setup_call(self,d):
  s,r=self.fixture(d)
  req=json.loads((s.directory/'requests_agent/t1.json').read_text());req['schema']={'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok'],'additionalProperties':False};c.save(s.directory/'requests_agent/t1.json',req)
  auth=Path(d)/'auth.json';auth.write_text(json.dumps({'auth_mode':'chatgpt','tokens':{'access_token':'synthetic-access-token'}}))
  from experiments.cloud_role_runner_v1.agent import prepare_approval
  p=prepare_approval(s,'t1',r);return s,r,auth,p
 def receipts(self,complete=True):
  usage=dict(input_tokens=100,output_tokens=20,cached_input_tokens=80)
  ev=[dict(type='turn.started'),dict(type='item.completed',item=dict(type='agent_message',text='{"wrong":true}'))]
  if complete:ev.append(dict(type='turn.completed',usage=usage))
  common=dict(passed=True,CPU_seconds=1,relay_CPU_seconds=1,returncode=0,stderr='')
  return [common|dict(stdout='Logged in using ChatGPT'),common|dict(stdout='',process=dict(quota=dict(ordinary_usage_allowed=True,buckets={}))),common|dict(stdout='\n'.join(json.dumps(e) for e in ev),isolation=dict(passed=True))]
 def test_invalid_reply_retains_real_usage(self):
  from experiments.cloud_role_runner_v1.agent import invoke
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as d:
   s,r,a,p=self.setup_call(d)
   with patch('experiments.cloud_role_runner_v1.agent.launch',side_effect=self.receipts()):result=invoke(s,'t1',p['approval_sha'],r,a)
   self.assertFalse(result['passed']);self.assertEqual(result['reported_tokens'],120);self.assertEqual(s.totals()['tokens'],120);s.close()
 def test_unknown_usage_keeps_reservation_and_stops(self):
  from experiments.cloud_role_runner_v1.agent import invoke
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as d:
   s,r,a,p=self.setup_call(d)
   with patch('experiments.cloud_role_runner_v1.agent.launch',side_effect=self.receipts(False)):result=invoke(s,'t1',p['approval_sha'],r,a)
   self.assertFalse(result['passed']);self.assertIsNone(result['reported_tokens']);self.assertEqual(s.totals()['reserved_tokens'],19000);self.assertEqual(s.get('status'),'manual_checkpoint')
   with self.assertRaises(RuntimeError):invoke(s,'t1',p['approval_sha'],r,a)
   s.close()
