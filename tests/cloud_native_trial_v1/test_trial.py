import copy, importlib.util, json, tempfile, unittest, zipfile
from pathlib import Path
from experiments.cloud_native_trial_v1 import control as c

class TrialTests(unittest.TestCase):
    def test_fixed_scope(self):
        self.assertEqual(c.LIMITS['optimizer_steps'],159120)
        self.assertEqual(c.META_CEILING,7680)
        a,b=c.configs()
        self.assertFalse(a['prior']['cue_rotation']);self.assertTrue(b['prior']['cue_rotation'])
        b['prior']['cue_rotation']=False;self.assertEqual(a,b)
        self.assertEqual(a['prior']['episodes'],1920);self.assertEqual(a['target']['steps'],330)
    def test_no_threshold_or_scope_edit(self):
        s=c.definition();s['limits']['wall_seconds']=9
        with self.assertRaises(ValueError):c.validate_definition(s)
        s=c.definition();s['seeds']=[21000]
        with self.assertRaises(ValueError):c.validate_definition(s)
    def test_identity_and_confirmation_not_in_source(self):
        files=c.source_files()
        self.assertIn('research_search/runners/train.py',files)
        self.assertIn('execution_objective_search/objectives.py',files)
        self.assertFalse(any('database' in Path(p).parts or p.endswith('.pt') for p in files))
    def test_export_preserves_0600(self):
        from experiments.cloud_native_trial_v1.export import collect, verify
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'native';root.mkdir();p=root/'result.json';p.write_text('{"x":2}');p.chmod(0o600)
            out=Path(d)/'export';collect(root,out);self.assertEqual(p.stat().st_mode&0o777,0o600)
            verify(out);self.assertEqual((out/'results.zip').stat().st_mode&0o777,0o644)
            with zipfile.ZipFile(out/'results.zip') as z:self.assertEqual(z.read('result.json'),p.read_bytes())
    def test_export_rejects_symlinks_and_overwrite(self):
        from experiments.cloud_native_trial_v1.export import collect
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'native';root.mkdir();(root/'link').symlink_to('/etc/passwd')
            with self.assertRaises(ValueError):collect(root,Path(d)/'export')
    def test_export_detects_tamper(self):
        from experiments.cloud_native_trial_v1.export import collect, verify
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'native';root.mkdir();(root/'a').write_text('yes');out=Path(d)/'export';collect(root,out)
            (out/'results.zip').write_bytes(b'bad')
            with self.assertRaises(ValueError):verify(out)
    def test_create_timeout_and_protections(self):
        from experiments.cloud_native_trial_v1.launcher import create_command, CREATE_SECONDS
        self.assertEqual(CREATE_SECONDS,60)
        cmd=create_command(['docker'], 'sha256:'+'a'*64, Path('/tmp/job'), 'run','trial-test')
        for x in ['--network=none','--read-only','--user=65534:65534','--cap-drop=ALL','--pull=never']:self.assertIn(x,cmd)
        self.assertIn('type=bind,src=/tmp/job/source,dst=/workspace,readonly',cmd)
        self.assertFalse(any('docker.sock' in p or 'HOME=' in p for p in cmd))
    def test_collector_mount_is_readonly(self):
        from experiments.cloud_native_trial_v1.launcher import create_command
        cmd=create_command(['docker'],'sha256:'+'a'*64,Path('/tmp/job'),'export','trial-export')
        self.assertIn('type=bind,src=/tmp/job/native,dst=/trial,readonly',cmd)
    def test_no_agent_or_confirmation_launch(self):
        text=(c.ROOT/'experiments/cloud_native_trial_v1/worker.py').read_text()
        self.assertNotIn('agent_backend.invoke',text);self.assertNotIn('ensure_jobs(',text)
        self.assertNotIn('confirmation.json',text)

if __name__=='__main__':unittest.main()

class LifecycleTests(unittest.TestCase):
    def test_export_attempts_have_distinct_mounts(self):
        from experiments.cloud_native_trial_v1.launcher import create_command
        a=create_command(['docker'],'sha256:'+'a'*64,Path('/tmp/job'),'export','one')
        b=create_command(['docker'],'sha256:'+'a'*64,Path('/tmp/job'),'export','two')
        self.assertIn('type=bind,src=/tmp/job/exports/one,dst=/output',a)
        self.assertIn('type=bind,src=/tmp/job/exports/two,dst=/output',b)
    def test_deadline_guard_does_not_update_or_restore(self):
        from experiments.cloud_native_trial_v1.worker import guarded_runtime
        from research_search.orchestration.budget import BudgetError
        class Base:pass
        class Store:
            def check(self):pass
        r=guarded_runtime(Base)();r.store=Store();r.cpu_deadline=0;r.deadline=None
        with self.assertRaises(BudgetError):r.check()
    def test_unknown_or_failed_job_not_replayed(self):
        from research_search.orchestration.store import Store
        from research_search.orchestration.coordinator import _enqueue
        with tempfile.TemporaryDirectory() as d:
            s=Store(d,c.LIMITS)
            _enqueue(s,'prior-test','control',dict(kind='prior',tasks=[]))
            s.db.execute("UPDATE jobs SET status='uncertain'")
            with self.assertRaises(RuntimeError):_enqueue(s,'prior-test','control',dict(kind='prior',tasks=[]))
            s.close()

class ResumeAndBudgetTests(unittest.TestCase):
    def test_total_job_budget_arithmetic(self):
        self.assertEqual(4*1920*20+12*20+16*330,c.LIMITS['optimizer_steps'])
        self.assertEqual(4*1920,c.META_CEILING)
    def test_resource_guard_preserves_store_wall_gate(self):
        from experiments.cloud_native_trial_v1.worker import guarded_runtime
        from research_search.orchestration.budget import BudgetError
        class Base:pass
        class Store:
            def check(self):raise BudgetError('original deadline elapsed')
        r=guarded_runtime(Base)();r.store=Store();r.cpu_deadline=float('inf');r.deadline=None
        with self.assertRaisesRegex(BudgetError,'original deadline'):r.check()
    def test_unknown_usage_cannot_dispatch(self):
        from research_search.orchestration.store import Store
        from research_search.orchestration.budget import BudgetError
        with tempfile.TemporaryDirectory() as d:
            s=Store(d,c.LIMITS)
            with self.assertRaises(BudgetError):s.reserve_call('no','hypothesis',1)
            s.close()
    def test_native_default_and_candidate_are_valid(self):
        from research_search.runners.candidate import validate,DEFAULT
        a,b=c.configs();self.assertEqual(a,DEFAULT);self.assertEqual(validate(a),a);self.assertEqual(validate(b),b)
    def test_source_asset_manifest_is_anchored(self):
        self.assertIn('experiments/cloud_native_trial_v1/assets/FILES.json',c.source_files())

class PureRegistrationTests(unittest.TestCase):
    def test_cached_exclusions_exactly_match_native(self):
        import itertools
        from experiments.cloud_native_trial_v1.register_inputs import exclusion_lookup
        from research_search.evaluation.validity import background_allowed
        forbidden=[[0,1,2,0,3,1,2,3],[3,0,0,2,3,1,2,1]]
        fast=exclusion_lookup(forbidden)
        for p in itertools.islice(itertools.product(range(4),repeat=8),0,None,61):
            self.assertEqual(fast(list(p),forbidden),background_allowed(list(p),forbidden))
    def test_cached_generator_retains_native_rng_and_order(self):
        from experiments.cloud_native_trial_v1.register_inputs import cached_background
        from research_search.runners.tasks import background
        forbidden=[[0,1,2,0,3,1,2,3]]
        self.assertEqual(cached_background(21000,8,forbidden,'P2'),background(21000,8,forbidden,'P2'))

class CompetenceRegistrationTests(unittest.TestCase):
    def test_shared_competence_is_fixed_and_excludes_background(self):
        from experiments.cloud_native_trial_v1.register_inputs import shared_competence
        forbidden=[[0,1,2,0,3,1,2,3]]
        a=shared_competence(21000,forbidden);b=shared_competence(21000,forbidden)
        self.assertEqual(a,b);self.assertEqual(len(a),3)
        from research_search.evaluation.validity import background_allowed
        for task,pair in zip(a,['AB','AC','BC']):
            self.assertEqual(task['kind'],'shared');self.assertEqual(task['pair'],pair)
            self.assertEqual(len(task['paths']),1);self.assertTrue(background_allowed(task['paths'][0],forbidden))
            self.assertEqual([r['context'] for r in task['updates']],[pair[(k//5)%2] for k in range(20)])
            self.assertTrue(all(r['sequence']==0 for r in task['updates']))

class QueueRegistrationTests(unittest.TestCase):
    def test_four_priors_then_exact_remaining_jobs(self):
        from experiments.cloud_native_trial_v1.worker import schedule
        from research_search.orchestration.store import Store
        assets=c.HERE/'assets'
        with tempfile.TemporaryDirectory() as d:
            s=Store(d,c.LIMITS);schedule(s,assets,'fixture-identity')
            jobs=s.db.execute('SELECT * FROM jobs').fetchall()
            self.assertEqual(len(jobs),4);self.assertEqual(s.totals()['reserved_optimizer_steps'],153600)
            for job in jobs:
                folder=Path(d)/'jobs'/job['id'];folder.mkdir(parents=True);(folder/'pretrained.pt').write_bytes(b'NOT_A_MODEL_FIXTURE')
                s.db.execute("UPDATE jobs SET status='completed' WHERE id=?",(job['id'],))
            targets,competence=schedule(s,assets,'fixture-identity')
            self.assertEqual(len(targets),16);self.assertEqual(len(competence),12)
            self.assertEqual(s.totals()['reserved_optimizer_steps'],159120)
            self.assertEqual(s.totals()['optimizer_steps'],0)
            before=s.db.execute('SELECT count(*) FROM jobs').fetchone()[0]
            schedule(s,assets,'fixture-identity')
            self.assertEqual(s.db.execute('SELECT count(*) FROM jobs').fetchone()[0],before)
            s.close()
    def test_create_timeout_cleans_by_name(self):
        import subprocess
        from unittest.mock import patch
        from experiments.cloud_native_trial_v1.launcher import launch
        calls=[]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c.save(root/'definition.json',c.definition());(root/'source').mkdir();c.save(root/'source/STAGED_FILES.json',{});c.save(root/'source/definition.json',c.definition())
            def fake(cmd,**kw):
                calls.append((cmd,kw))
                if 'create' in cmd:raise subprocess.TimeoutExpired(cmd,kw['timeout'])
                return subprocess.CompletedProcess(cmd,0,'','')
            with patch('experiments.cloud_native_trial_v1.launcher.verify_source',return_value={}),patch('shutil.which',return_value='/bin/docker'),patch('subprocess.run',side_effect=fake):
                r=launch(root,'sha256:'+'a'*64,'run','/var/run/docker.sock')
            self.assertFalse(r['passed']);self.assertTrue(r['container_removed'])
            self.assertEqual(calls[0][1]['timeout'],60);self.assertIn('rm',calls[-1][0])

class CleanStagingTests(unittest.TestCase):
    def test_clean_staging_imports_native_without_models(self):
        import subprocess,sys
        with tempfile.TemporaryDirectory() as d:
            path=Path(d).resolve()/'trial';c.prepare(path)
            code="import sys;sys.path[:0]="+repr([str(path/'source'),str(path/'source/src')])+";import torch;torch.nn.RNN.__init__=lambda *a,**k:(_ for _ in ()).throw(AssertionError('model construction forbidden'));from research_search.runners.evaluate import execute;from research_search.runners.candidate import DEFAULT,validate;from experiments.cloud_native_trial_v1.control import configs;assert DEFAULT==configs()[0];validate(configs()[1])"
            r=subprocess.run([sys.executable,'-I','-c',code],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)

class OwnerStopTests(unittest.TestCase):
    def test_stop_checked_before_each_update(self):
        from experiments.cloud_native_trial_v1.worker import guarded_runtime
        from research_search.orchestration.budget import BudgetError
        class Base:pass
        with tempfile.TemporaryDirectory() as d:
            class Store:
                directory=Path(d)
                def check(self):pass
            r=guarded_runtime(Base)();r.store=Store();r.cpu_deadline=float('inf');r.deadline=None
            (Path(d)/'STOP_REQUESTED').write_text('stop')
            with self.assertRaises(BudgetError):r.check()

class FrozenRuntimeTests(unittest.TestCase):
    def test_resume_refuses_different_image(self):
        from experiments.cloud_native_trial_v1.launcher import pin_runtime
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);pin_runtime(root,'sha256:'+'a'*64)
            pin_runtime(root,'sha256:'+'a'*64)
            with self.assertRaises(ValueError):pin_runtime(root,'sha256:'+'b'*64)
    def test_failed_cleanup_does_not_collect_live_files(self):
        from experiments.cloud_native_trial_v1.launcher import run_and_export
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'trial';root.mkdir()
            with patch('experiments.cloud_native_trial_v1.launcher.launch',return_value=dict(passed=False,container_removed=False)) as start:
                result=run_and_export(root,'sha256:'+'a'*64,'run','/var/run/docker.sock')
            self.assertFalse(result['passed']);self.assertEqual(start.call_count,1)

class UncertainOperationTests(unittest.TestCase):
    def test_uncertain_step_does_not_claim_exact_outer_count(self):
        import torch
        from research_search.orchestration.store import Store
        from research_search.orchestration.queue import enqueue
        from experiments.cloud_native_trial_v1.worker import report
        with tempfile.TemporaryDirectory() as d:
            s=Store(d,c.LIMITS);enqueue(s,'prior-test','control',dict(kind='prior',seed=21000,tasks=[{}]))
            s.charge_step('prior-test',1)
            folder=Path(d)/'jobs/prior-test';folder.mkdir(parents=True)
            torch.save(dict(extra=dict(episode=1)),folder/'progress.pt')
            result=report(s)
            self.assertIsNone(result['reptile_interpolations'])
            self.assertEqual(result['confirmed_reptile_interpolations'],1)
            self.assertIsNone(result['combined_optimizer_and_meta_operations'])
            s.close()
