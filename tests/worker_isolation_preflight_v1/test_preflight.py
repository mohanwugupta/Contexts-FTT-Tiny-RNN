import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from experiments.worker_isolation_preflight_v1.preflight import (
    IMAGE_RE, create_command, verify_config, verify_checks, REQUIRED_CHECKS, run,
)

IMAGE = 'python@sha256:' + 'a' * 64


class Contracts(unittest.TestCase):
    def config(self):
        return {
            'Config': {'User': '65534:65534', 'Image': IMAGE,
                       'Entrypoint':None, 'Cmd':['python','-I','-B','/worker/probe.py']},
            'HostConfig': {'ReadonlyRootfs': True, 'NetworkMode': 'none',
                'CapDrop': ['ALL'], 'CapAdd': None,
                'SecurityOpt': ['no-new-privileges:true'], 'Privileged': False,
                'PidsLimit': 32, 'Memory': 268435456, 'MemorySwap':268435456, 'NanoCpus': 1000000000,
                'Ulimits': [{'Name':'cpu','Soft':20,'Hard':20}],
                'Tmpfs': {'/tmp':'rw,noexec,nosuid,size=16m'},
                'PidMode':'', 'IpcMode':'none'},
            'Mounts': [
                {'Source':'/fixture/source','Destination':'/worker','RW':False,'Type':'bind'},
                {'Source':'/fixture/private','Destination':'/private','RW':False,'Type':'bind'},
                {'Source':'/fixture/output','Destination':'/output','RW':True,'Type':'bind'},
            ],
        }

    def test_image_must_be_immutable(self):
        self.assertIsNotNone(IMAGE_RE.fullmatch(IMAGE))
        for bad in ['python:latest', IMAGE+'\n', 'bad;'+IMAGE, 'https://'+IMAGE]:
            self.assertIsNone(IMAGE_RE.fullmatch(bad))

    def test_command_has_explicit_restrictions_and_no_host_credentials(self):
        cmd=create_command(['docker','--host=unix:///var/run/docker.sock'],IMAGE,Path('/fixture'))
        self.assertIn('--read-only',cmd)
        self.assertIn('--network=none',cmd)
        self.assertIn('--cap-drop=ALL',cmd)
        self.assertIn('--user=65534:65534',cmd)
        self.assertIn('--security-opt=no-new-privileges:true',cmd)
        self.assertNotIn('--privileged',cmd)
        self.assertNotIn('--env',cmd)
        self.assertNotIn('--pull=always',cmd)
        self.assertIn('--pull=never',cmd)
        self.assertEqual(cmd[-4:],['python','-I','-B','/worker/probe.py'])

    def test_effective_config_must_match_every_restriction(self):
        good=self.config();verify_config(good,Path('/fixture'),IMAGE)
        for key,value in [('ReadonlyRootfs',False),('NetworkMode','host'),('Privileged',True),
                          ('CapAdd',['SYS_ADMIN']),('CapDrop',[]),('SecurityOpt',[]),
                          ('PidsLimit',0),('Memory',0),('NanoCpus',0),('Ulimits',[]),
                          ('MemorySwap',-1),
                          ('PidMode','host'),('IpcMode','host'),('Tmpfs',{})]:
            bad=copy.deepcopy(good);bad['HostConfig'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                verify_config(bad,Path('/fixture'),IMAGE)
        for field,value in [('RW',True),('Source','/unexpected'),('Type','volume')]:
            bad=copy.deepcopy(good);bad['Mounts'][0][field]=value
            with self.assertRaises(ValueError):verify_config(bad,Path('/fixture'),IMAGE)
        bad=copy.deepcopy(good);bad['Mounts'].append({'Destination':'/host','Source':'/','RW':False,'Type':'bind'})
        with self.assertRaises(ValueError):verify_config(bad,Path('/fixture'),IMAGE)
        for user in ['0','root','65534:0']:
            bad=copy.deepcopy(good);bad['Config']['User']=user
            with self.assertRaises(ValueError):verify_config(bad,Path('/fixture'),IMAGE)
        bad=copy.deepcopy(good);bad['Config']['Entrypoint']=['/bin/sh']
        with self.assertRaises(ValueError):verify_config(bad,Path('/fixture'),IMAGE)

    def test_missing_false_or_nonboolean_checks_never_pass(self):
        checks=dict.fromkeys(REQUIRED_CHECKS,True)
        verify_checks(checks)
        for key in REQUIRED_CHECKS:
            for value in [False,None,1,'true']:
                bad=checks|{key:value}
                with self.assertRaises(ValueError):verify_checks(bad)
            bad=checks.copy();bad.pop(key)
            with self.assertRaises(ValueError):verify_checks(bad)

    def test_missing_docker_retains_blocked_receipt_without_installing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp).resolve()/'job'
            with patch('shutil.which',return_value=None),patch('subprocess.run') as execute:
                receipt=run(out,IMAGE)
            self.assertFalse(receipt['passed']);execute.assert_not_called()
            self.assertEqual(receipt['new_agent_calls'],0)
            self.assertTrue((out/'FILES.json').is_file())
            self.assertIn('Docker CLI unavailable',receipt['reason'])
            with self.assertRaises(ValueError):run(out,IMAGE)

    def test_timeout_always_removes_container_and_preserves_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp).resolve()/'job';config=self.config()
            for mount in config['Mounts']:
                mount['Source']=str(out/'fixture'/Path(mount['Source']).name)
            calls=[];container='c'*64
            def command(args,**kwargs):
                calls.append(args)
                action=args[2]
                if action=='start':raise subprocess.TimeoutExpired(args,35)
                stdout={'info':'28.0.0','create':container,'inspect':json.dumps([config]),'rm':container}[action]
                return subprocess.CompletedProcess(args,0,stdout,'')
            with patch('shutil.which',return_value='/usr/bin/docker'),patch('subprocess.run',side_effect=command):
                receipt=run(out,IMAGE)
            self.assertFalse(receipt['passed']);self.assertTrue(receipt['container_removed'])
            self.assertEqual(calls[-1][2:4],['rm','--force'])
            self.assertEqual(receipt['error_type'],'TimeoutExpired')

    def test_no_daemon_retry_no_image_pull_and_no_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('shutil.which',return_value='/usr/bin/docker'),patch('subprocess.run',return_value=subprocess.CompletedProcess([],1,'','unavailable')) as execute:
                receipt=run(Path(tmp).resolve()/'job',IMAGE)
            self.assertFalse(receipt['passed']);self.assertEqual(execute.call_count,1)
            self.assertIsNone(receipt.get('container_removed'))


if __name__=='__main__':unittest.main()
