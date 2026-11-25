# Build-time TLS trust for the cached runtime

This separate build wrapper repairs CA propagation into the dependency image.
It does not modify the frozen v2 worker, scientific code, requirements, failed
outputs, research budgets or deadlines. It performs infrastructure/synthetic work
only, with no model call or archived observations.

The wrapper combines the host's existing system CA bundle with explicitly
configured local trust files (`CODEX_PROXY_CERT`, `SSL_CERT_FILE`,
`REQUESTS_CA_BUNDLE`, `PIP_CERT`). An explicit `--ca-bundle FILE` replaces that
selection. It never fetches a CA or changes a global trust store. Input must
contain valid X509 certificates; private keys and malformed data are rejected.
Metadata records fingerprints/hashes, not certificate contents.

Each networked pip step receives the combined bundle as a required BuildKit file
secret and uses `PIP_CERT`, `REQUESTS_CA_BUNDLE` and `SSL_CERT_FILE` for that step
only. TLS chain and hostname validation remain enabled. There is no trusted-host
or verification bypass. The bundle stays outside the build context, is removed
on success/failure/timeout, and is never copied into image layers or persistent
certificate environment variables. A later build step asserts that the secret
path and those environment variables are absent. `--no-cache` ensures a changed
secret cannot silently reuse an earlier install layer.

From a normal verified repository checkout, the infrastructure command is:

```bash
python3 -m unittest discover -s tests/cloud_runtime_tls_v1 -v
python3 -m experiments.cloud_runtime_tls_v1.build --out /tmp/srn-cached-adapter-tls-v1
```

For a conflicting or older checkout, use the separately extracted code-only kit:

```bash
bash /PATH/TO/FRESH/KIT/cloud_tls_smoke.sh --out /tmp/srn-cached-adapter-tls-v1
```

The bootstrap runs isolated Python, checks every package file hash, and imports
only this kit's source. It does not read or overwrite the conflicting checkout.
Use the same endpoint as the successful preflight, optionally specifying
`--socket /run/codex-docker/docker.sock`. Preserve Docker's registry configuration
and managed proxy settings; only inherited remote Docker selectors are removed.
BuildKit is required. The wrapper never starts or reconfigures a daemon.

The image build has a600-second limit and saves image-build.log plus
TLS_BUILD_RECEIPT.json even on failure. A successful image proceeds to the
unchanged v2 synthetic smoke, with its120CPU/300wall limits and all original
restriction checks. No automatic retry or scientific-execution action exists
here. The old scientific pilot remains stopped; parent-agent isolation and
provider token usage remain unresolved.

Tests use actual OpenSSL TLS handshakes in memory: an untrusted chain and wrong
hostname fail; explicitly supplied CA trust succeeds with CERT_REQUIRED. This
requires no open sockets and is distinct from actual Cloud-proxy acceptance.
Linux CI additionally builds the real CPU image using a file secret, verifies
its secret/env absence assertion and runs synthetic tables/plotting under the
container restrictions. It supplies no research data.

References: [Docker build secrets](https://docs.docker.com/build/building/secrets/),
[pip verified certificate stores](https://pip.pypa.io/en/stable/topics/https-certificates/).
