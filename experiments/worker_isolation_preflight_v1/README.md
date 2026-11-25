# Synthetic worker isolation preflight

This is a standalone infrastructure check. It never reads research evidence,
loads a checkpoint, evaluates a network or calls an AI backend. No prior pilot
budget, deadline, model or evaluator is changed.

It validates a Docker container backend without Landlock. Docker must already
be available and accessible on the explicitly selected local daemon socket.
The program never installs or starts a daemon, changes its configuration, or
falls back to an unrestricted process. A remote host must pass its own check;
success on a GitHub runner does not establish availability in a different
managed Cloud runtime.

## Run without a research-agent call

In an existing terminal on the intended Linux host:

```bash
python3 -m unittest discover -s tests/worker_isolation_preflight_v1 -v
docker --host=unix:///var/run/docker.sock info --format '{{.ServerVersion}}'
docker --host=unix:///var/run/docker.sock pull python:3.12-slim
image_ref=$(docker --host=unix:///var/run/docker.sock image inspect python:3.12-slim --format '{{index .RepoDigests 0}}')
python3 -m experiments.worker_isolation_preflight_v1.preflight --image "$image_ref" --out /tmp/new-isolation-fixture
```

The image tag is resolved to its actual immutable digest before the preflight;
the worker uses `--pull=never` and records that digest and image ID. Use a fresh
output directory once. Preserve failed receipts. If the standard managed socket
is unavailable, the documented alternative `/run/codex-docker/docker.sock` can be
chosen explicitly with `--socket`; use that same endpoint for image setup.
Do not restart daemons, enable privileged mode or weaken controls.

The source-only `Synthetic worker isolation preflight` GitHub Actions workflow
runs the same checks on an Ubuntu24.04 VM, with a five-minute job ceiling.
It checks out only this namespace/tests; no evidence, scientific results,
hypotheses or secrets are passed to the container. Its public logs contain only
synthetic fixtures and infrastructure metadata. It uses no research-agent calls.

## Contract and receipt

The parent verifies the effective container configuration before starting it:
read-only root/source, exact synthetic mounts, nonroot UID/GID65534, all
capabilities dropped, no-new-privileges, network disabled, IPC disabled,32PIDs,
256MiB memory/no swap, one CPU,20CPU seconds and a35-second attached-run timeout.
All Docker CLI commands have finite timeouts. Cleanup force-removes the one
created container even after timeout or failed checks. No Docker socket is
mounted into it. `/output` and a16MiB non-executable `/tmp` hold synthetic writes.
Only dummy private files are mounted to prove denial; real private inputs are
never supplied. No credentials or parent runtime variables are forwarded.

Inside the container, a seccomp filter additionally denies internet/local
sockets, exec and escape syscalls. Nineteen actual checks test capabilities,
identity, source reads and write/truncate/unlink denial, root writes, private
reads, link escapes, sockets, exec and allowed writes. Protected fixture hashes
must remain unchanged. Any missing/false check fails. `FILES.json` records exact
receipt/log/fixture hashes. A missing CLI/daemon produces a blocked receipt,
with no automatic retry or alternate backend selection.

A pass establishes this synthetic container's controls on this daemon and
image only. It does not certify the parent Cloud agent, independent research
roles, provider token accounting, numerical dependency compatibility, or the
unchanged Landlock-dependent scientific driver. A separately reviewed adapter
and new execution authorization would be needed before scientific execution.
The old scientific driver must not be launched from this preflight.

References: [Docker run controls](https://docs.docker.com/reference/cli/docker/container/run/),
[seccomp](https://www.kernel.org/doc/html/latest/userspace-api/seccomp_filter.html),
[GitHub-hosted VM runners](https://docs.github.com/en/actions/concepts/runners/github-hosted-runners).
