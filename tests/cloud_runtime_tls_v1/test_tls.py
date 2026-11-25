import hashlib
import os
from pathlib import Path
import ssl
import tempfile
import subprocess
import unittest
from unittest.mock import patch

from experiments.cloud_runtime_tls_v1.build import (
    certificates, combined_bundle, docker_build_command, prepare_context, run,
)


class TLSContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        roots=ssl.create_default_context().get_ca_certs(binary_form=True)
        if not roots:raise RuntimeError('Test host has no system roots')
        cls.pem=ssl.DER_cert_to_PEM_cert(roots[0]).encode()
        cls.other=ssl.DER_cert_to_PEM_cert(roots[-1]).encode()

    def test_validates_certificates_and_rejects_keys_or_junk(self):
        self.assertEqual(len(certificates(self.pem)),1)
        for bad in [b'',b'not a certificate',self.pem+b'-----BEGIN PRIVATE KEY-----\nsecret',
                    b'-----BEGIN CERTIFICATE-----\ninvalid\n-----END CERTIFICATE-----\n']:
            with self.assertRaises(ValueError):certificates(bad)

    def test_combines_system_and_session_roots_without_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);system=root/'system.pem';session=root/'session.pem'
            system.write_bytes(self.pem+self.other);session.write_bytes(self.pem)
            bundle,metadata=combined_bundle(system,session)
            self.assertEqual(len(certificates(bundle)),2)
            self.assertEqual(metadata['unique_certificates'],2)
            self.assertEqual(metadata['sha256'],hashlib.sha256(bundle).hexdigest())
            self.assertNotIn('certificate_contents',metadata)
            ctx=ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ctx.load_verify_locations(cadata=bundle.decode())

    def test_context_has_no_ca_and_every_network_step_mounts_secret(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'build';prepare_context(out)
            self.assertEqual({p.name for p in out.iterdir()},
                             {'Dockerfile','requirements.txt','BUILD_CONTEXT.json'})
            text=(out/'Dockerfile').read_text()
            runs=[line for line in text.splitlines() if line.startswith('RUN') and 'pip install' in line]
            self.assertEqual(len(runs),2)
            self.assertTrue(all('--mount=type=secret,id=proxy_ca,required=true' in line for line in runs))
            self.assertEqual(text.count('PIP_CERT=/run/secrets/proxy_ca'),2)
            self.assertNotIn('trusted-host',text)
            self.assertNotIn('ENV PIP_CERT',text)
            self.assertNotIn('COPY proxy',text)
            self.assertIn("assert not os.path.exists('/run/secrets/proxy_ca')",text)
            with self.assertRaises(ValueError):prepare_context(out)

    def test_ca_is_passed_by_file_secret_not_build_argument(self):
        cmd=docker_build_command('/var/run/docker.sock',Path('/new/build'),Path('/new/id'),Path('/private/ca.pem'))
        self.assertIn('--secret',cmd)
        self.assertIn('id=proxy_ca,src=/private/ca.pem',cmd)
        self.assertNotIn('--build-arg',cmd)
        self.assertNotIn('--network=host',cmd)
        self.assertIn('--no-cache',cmd)

    def test_failed_build_removes_ca_and_never_runs_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();ca=root/'system.pem';ca.write_bytes(self.pem)
            seen=[]
            def fail(command,**kwargs):
                arg=command[command.index('--secret')+1]
                secret=Path(arg.split('src=',1)[1]);seen.append(secret)
                self.assertTrue(secret.is_file())
                self.assertEqual(secret.stat().st_mode & 0o777,0o600)
                self.assertEqual(kwargs['env']['HTTPS_PROXY'],'https://proxy.example:8443')
                self.assertNotIn('DOCKER_HOST',kwargs['env'])
                return subprocess.CompletedProcess(command,1)
            with patch.dict(os.environ,{'HTTPS_PROXY':'https://proxy.example:8443','DOCKER_HOST':'tcp://wrong'}),patch('subprocess.run',side_effect=fail),patch('experiments.cloud_confidence_container_v2.launcher.run') as smoke:
                result=run(root/'new',ca_bundle=ca)
            self.assertFalse(result['passed']);smoke.assert_not_called()
            self.assertTrue(result['temporary_CA_removed'])
            self.assertTrue(all(not p.exists() for p in seen))
            self.assertTrue((root/'new/TLS_BUILD_RECEIPT.json').is_file())
            with self.assertRaises(ValueError):run(root/'new',ca_bundle=ca)

    def test_invalid_ca_stops_before_any_docker_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();ca=root/'wrong.pem';ca.write_text('not a certificate')
            with patch('subprocess.run') as execute:
                result=run(root/'new',ca_bundle=ca)
            self.assertFalse(result['passed']);execute.assert_not_called()

    def test_temporary_ca_cleanup_even_on_build_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();ca=root/'system.pem';ca.write_bytes(self.pem)
            seen=[]
            def timeout(command,**kwargs):
                seen.append(Path(command[command.index('--secret')+1].split('src=',1)[1]))
                raise subprocess.TimeoutExpired(command,600)
            with patch('subprocess.run',side_effect=timeout):result=run(root/'new',ca_bundle=ca)
            self.assertFalse(result['passed']);self.assertEqual(result['error_type'],'TimeoutExpired')
            self.assertTrue(result['temporary_CA_removed']);self.assertFalse(seen[0].exists())


if __name__=='__main__':unittest.main()
