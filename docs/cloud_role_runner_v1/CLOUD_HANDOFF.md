# One Cloud handoff: Linux role runner

Use the **existing successful Cloud environment**. Fetch the new source in a clean detached worktree to preserve all rehearsal edits and prior results. Do not repeat TLS installation, rebuild the accepted numerical image or rerun scientific models.

Preparation and offline role fixture (no credential read, provider request or model turn):

```bash
git fetch origin main
git worktree add --detach /tmp/srn-linux-role-code-v1 origin/main
cd /tmp/srn-linux-role-code-v1
export PYTHONPATH=.:src
python3 -m experiments.cloud_role_runner_v1 prepare \
  --out /tmp/srn-linux-role-runtime-v1 \
  --image "$(cat /tmp/srn-cached-adapter-tls-v1/image-id)"
python3 -m experiments.cloud_role_runner_v1 fixture \
  --runtime /tmp/srn-linux-role-runtime-v1/RUNTIME.json \
  --out /tmp/srn-linux-role-fixture-v1
```

Default socket is /var/run/docker.sock. If only the already accepted /run/codex-docker/docker.sock exists, pass --socket /run/codex-docker/docker.sock to prepare. `--codex` can identify an exact installed Linux ELF or npm launcher; nothing is downloaded. Existing configured CA stores are combined automatically; `--ca` selects an explicit certificate-only trust bundle. Fresh output directories are mandatory; preserve failures instead of overwriting or auto-retrying.

In this Cloud task, locate the blocked rehearsal's new driver,18 tests and existing operational ledger. They were not pushed with main26cb0f0. Preserve its original files, original started/deadline/limits and old token holds. Add the transport in a separately versioned adapter using README.md's invoke API. Keep its existing independent evaluator and accept_reply phase logic. Run its18 tests plus tests/cloud_role_runner_v1; retain both logs. Do not recreate its state or reuse the numerical-trial ledger.

Generate a reviewable approval envelope from its **existing** pending ticket. Replace REHEARSAL and TICKET below with the actual saved paths/ID (do not invent a start time or ticket):

```bash
python3 -m experiments.cloud_role_runner_v1 approval \
  --runtime /tmp/srn-linux-role-runtime-v1/RUNTIME.json \
  --campaign REHEARSAL --ticket TICKET
```

Return the exact approval envelope/sha, pending role, unchanged token reservation, actual existing deadline, binary/image identities and offline fixture receipt. **Stop for owner approval before call.** Old Mac/request-only approval hashes are invalid for this transport.

After explicit approval of that one envelope, the transport command is:

```bash
python3 -m experiments.cloud_role_runner_v1 call \
  --runtime /tmp/srn-linux-role-runtime-v1/RUNTIME.json \
  --campaign REHEARSAL --ticket TICKET --approval APPROVED_ENVELOPE_SHA
```

The standalone transport returns its result; the registered rehearsal controller must then accept that reply through its unchanged phase logic. Alternatively, set SRN_LINUX_ROLE_RUNTIME=/tmp/srn-linux-role-runtime-v1/RUNTIME.json and use the new adapter's controller call entry point. Never run both for the same ticket. Each later role needs its own new envelope/manual approval.

If a call lacks provider usage, stop with the reservation held. Do not paste a Cloud chat token estimate as a receipt. If no existing clock/ledger can be located or it has expired, report that precise blocker; no automatic six-hour reset. Do not start training, a new research campaign, confirmation or the sealed72h campaign.
