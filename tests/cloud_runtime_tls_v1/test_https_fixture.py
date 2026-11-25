"""Actual OpenSSL TLS handshakes over memory BIOs; no network permissions."""
from pathlib import Path
import ssl
import subprocess
import tempfile
import unittest

from experiments.cloud_runtime_tls_v1.build import combined_bundle


def handshake(client_context, server_context, hostname='localhost'):
    incoming_client,outgoing_client=ssl.MemoryBIO(),ssl.MemoryBIO()
    incoming_server,outgoing_server=ssl.MemoryBIO(),ssl.MemoryBIO()
    client=client_context.wrap_bio(incoming_client,outgoing_client,server_hostname=hostname)
    server=server_context.wrap_bio(incoming_server,outgoing_server,server_side=True)
    client_done=server_done=False
    for _ in range(100):
        if not client_done:
            try:client.do_handshake();client_done=True
            except ssl.SSLWantReadError:pass
        data=outgoing_client.read()
        if data:incoming_server.write(data)
        if not server_done:
            try:server.do_handshake();server_done=True
            except ssl.SSLWantReadError:pass
        data=outgoing_server.read()
        if data:incoming_client.write(data)
        if client_done and server_done:
            client.write(b'synthetic TLS fixture')
            incoming_server.write(outgoing_client.read())
            return server.read()
    raise AssertionError('TLS fixture did not finish within100 handshake rounds')


class ActualTLSFixture(unittest.TestCase):
    def test_untrusted_chain_fails_but_explicit_ca_preserves_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'root.cnf').write_text('[req]\ndistinguished_name=dn\nx509_extensions=ext\nprompt=no\n[dn]\nCN=SRN Synthetic Test CA\n[ext]\nbasicConstraints=critical,CA:TRUE\nkeyUsage=critical,keyCertSign,cRLSign\n')
            (root/'server.cnf').write_text('[req]\ndistinguished_name=dn\nprompt=no\n[dn]\nCN=localhost\n')
            (root/'extensions.cnf').write_text('subjectAltName=DNS:localhost\nbasicConstraints=CA:FALSE\nkeyUsage=digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\n')
            def openssl(*args):
                subprocess.run(['openssl',*args],cwd=root,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            openssl('req','-x509','-newkey','rsa:2048','-nodes','-sha256','-days','1','-config','root.cnf','-keyout','root.key','-out','root.pem')
            openssl('req','-newkey','rsa:2048','-nodes','-config','server.cnf','-keyout','server.key','-out','server.csr')
            openssl('x509','-req','-in','server.csr','-CA','root.pem','-CAkey','root.key','-CAcreateserial','-out','server.pem','-days','1','-sha256','-extfile','extensions.cnf')
            server=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            server.load_cert_chain(root/'server.pem',root/'server.key')
            with self.assertRaises(ssl.SSLCertVerificationError):
                handshake(ssl.create_default_context(),server)
            ca,_=combined_bundle(root/'root.pem')
            trusted=ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            trusted.load_verify_locations(cadata=ca.decode())
            self.assertTrue(trusted.check_hostname)
            self.assertEqual(trusted.verify_mode,ssl.CERT_REQUIRED)
            self.assertEqual(handshake(trusted,server),b'synthetic TLS fixture')
            with self.assertRaises(ssl.SSLCertVerificationError):
                handshake(trusted,server,'wrong-hostname.example')


if __name__=='__main__':unittest.main()
