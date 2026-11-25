import ast
import copy
import datetime
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.cloud_confidence_container_v2.contract import (
    validate_authorization, checked_relative, compare_scientific_body,
    verify_bundle, evidence_entries,
)
from experiments.cloud_confidence_container_v2.launcher import command, run, build_context
from experiments.cloud_confidence_container_v2.staging import stage
from unittest.mock import patch


class AdapterContracts(unittest.TestCase):
    def authorization(self):
        return dict(version='container-execution-v2', owner_approved=True,
                    scope='one_cached_confidence_execution',
                    deadline_utc='2027-01-01T01:00:00Z', CPU_cap_seconds=120,
                    wall_cap_seconds=300, new_model_forwards=0,
                    new_optimizer_steps=0, new_agent_calls=0,
                    proposal_sha256='a'*64, selection_sha256='b'*64)

    def test_authorization_has_no_implicit_budget_or_deadline_reset(self):
        a=self.authorization();now=datetime.datetime(2027,1,1,tzinfo=datetime.timezone.utc)
        self.assertEqual(validate_authorization(a,now),300)
        for key,value in [('owner_approved',False),('deadline_utc','2026-11-25T09:00:00Z'),
                          ('CPU_cap_seconds',121),('new_agent_calls',1),
                          ('scope','campaign'),('wall_cap_seconds',301)]:
            bad=a|{key:value}
            with self.subTest(key=key),self.assertRaises(ValueError):validate_authorization(bad,now)
        a['deadline_utc']='2027-01-01T00:00:45Z'
        self.assertEqual(validate_authorization(a,now),45)

    def test_paths_forbid_credentials_checkpoints_and_private_banks(self):
        for value in ['/etc/passwd','../escape','a/../escape','.codex/auth.json',
                      'research_search/database/benchmark_banks/sealed/x.json',
                      'anything.pt','data/raw/x.csv','x/.env','x/link.key','a\\b.py']:
            with self.subTest(value=value),self.assertRaises(ValueError):checked_relative(value)
        self.assertEqual(checked_relative('experiments/ok.py'),'experiments/ok.py')

    def test_only_manifest_files_and_selected_extensions_are_admitted(self):
        m={'files':{'experiments/ok.py':'a'*64},'native_consumed_inputs':{}}
        self.assertEqual(evidence_entries(m),m['files'])
        for mutation in [m|{'files':{'secret.pem':'a'*64}},
                         m|{'native_consumed_inputs':{'x.csv':'b'*64}},
                         m|{'files':{'experiments/ok.py':'wrong'}}]:
            with self.assertRaises(ValueError):evidence_entries(mutation)

    def test_bundle_hashes_and_symlink_escapes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'x.py';p.write_text('fixture')
            hashes={'x.py':hashlib.sha256(p.read_bytes()).hexdigest()}
            verify_bundle(root,hashes)
            p.write_text('changed')
            with self.assertRaises(ValueError):verify_bundle(root,hashes)
            p.unlink();p.symlink_to('/etc/passwd')
            with self.assertRaises(ValueError):verify_bundle(root,hashes)

    def test_scientific_algorithm_matches_original_ast(self):
        compare_scientific_body()

    def test_container_command_has_frozen_budget_and_readonly_evidence(self):
        c=command(['docker'],'sha256:'+'a'*64,Path('/job'),'execute')
        for item in ['--network=none','--read-only','--cap-drop=ALL',
                     '--ulimit=cpu=120:120','--memory=2g','--cpus=1']:
            self.assertIn(item,c)
        self.assertIn('type=bind,src=/job/source,dst=/workspace,readonly',c)
        self.assertEqual(c[-2:],['--mode','execute'])
        self.assertNotIn('--privileged',c)

    def test_smoke_cannot_receive_scientific_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp).resolve()/'out'
            with self.assertRaises(ValueError):
                run(out,'sha256:'+'a'*64,'smoke',evidence_root=Path(tmp))
            self.assertFalse(out.exists())

    def test_execution_without_new_approval_never_contacts_docker(self):
        with tempfile.TemporaryDirectory() as tmp,patch('subprocess.run') as execute:
            out=Path(tmp).resolve()/'out'
            with self.assertRaises(ValueError):run(out,'sha256:'+'a'*64,'execute')
            execute.assert_not_called();self.assertFalse(out.exists())

    def test_staging_and_build_context_contain_no_scientific_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            files=stage(root/'source')
            self.assertTrue(all(n.endswith('.py') for n in files))
            build_context(root/'build')
            self.assertEqual({p.name for p in (root/'build').iterdir()},
                             {'Dockerfile','requirements.txt','BUILD_CONTEXT.json'})
            with self.assertRaises(ValueError):stage(root/'source')
            with self.assertRaises(ValueError):build_context(root/'build')

    def test_cpu_receipt_is_written_after_the_scientific_or_smoke_branch(self):
        worker=Path('experiments/cloud_confidence_container_v2/worker.py')
        nodes=next(n for n in ast.parse(worker.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main').body
        branch=next(i for i,n in enumerate(nodes) if isinstance(n,ast.If) and 'args.mode' in ast.unparse(n.test))
        write=next(i for i,n in enumerate(nodes) if 'PROCESS_USAGE.json' in ast.unparse(n))
        self.assertGreater(write,branch)


if __name__=='__main__':unittest.main()
