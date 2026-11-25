# Cached-readout container adapter v2

This separately versioned adapter removes the execution dependency on Landlock.
The frozen scientific driver and evaluator remain unchanged. `compute.py` retains
the original scientific function body; a normalized-AST test permits only
relocation of its two relative imports. It uses the original cached evaluator,
analysis, figures, thresholds and bootstrap.

## Infrastructure-only setup

From the repository root in the intended Docker-enabled Linux environment:

```bash
python3 -m unittest discover -s tests/cloud_confidence_container_v2 -v
python3 -m experiments.cloud_confidence_container_v2.launcher build-context --out /tmp/srn-cached-runtime-build-v2
docker --host=unix:///var/run/docker.sock build --iidfile /tmp/srn-cached-runtime-id-v2 /tmp/srn-cached-runtime-build-v2
image_id=$(cat /tmp/srn-cached-runtime-id-v2)
python3 -m experiments.cloud_confidence_container_v2.launcher smoke --image "$image_id" --out /tmp/srn-cached-adapter-smoke-v2
```

These commands do not read scientific observations or call an AI backend. Use a
new output/build directory once, retain failures, and do not retry failed checks
or weaken protections. Select the same explicit managed Docker socket for build
and run; `--socket /run/codex-docker/docker.sock` is the documented alternative.
Do not install/start a daemon or disable TLS verification to repair a build.

The base Python image is digest-pinned. The generated build context contains only
the Dockerfile, existing pinned Cloud requirements and build metadata. CPU
torch2.8.0 is installed from the official CPU index. No research data, source
checkout or credentials enter image layers. The local immutable image ID is
recorded at launch; no registry push is needed. Image build time is separate from
the120CPU/300wall worker ceiling. The synthetic CI job is bounded by15minutes.

Smoke runs all19 real container denial checks plus denial of writes to the actual
staged workspace. It then imports numpy/pandas/torch/matplotlib/jsonschema, applies
the original table-accounting functions to seven artificial positions, and saves
a synthetic plot. No network/model construction/forward/optimizer use occurs.
Full parent-agent isolation and provider token accounting remain unverified.

## Scientific execution is a distinct action

The stopped pilot's deadline and exhausted allocations are not reset. Preparing
this adapter or passing smoke does not authorize a scientific run, research-agent
call, critic, final coordinator or72-hour campaign. Do not invoke the old driver
inside the container expecting its Landlock gate to disappear.

Any later specifically approved execution must supply an owner authorization
record with `version=container-execution-v2`, `owner_approved=true`,
`scope=one_cached_confidence_execution`, a new explicit timezone-aware
`deadline_utc`, `CPU_cap_seconds=120`, `wall_cap_seconds=300`, zero
`new_model_forwards`, `new_optimizer_steps`, `new_agent_calls`, and the exact
`proposal_sha256`/`selection_sha256`. The record attests the owner's separately
given approval; it is not a cryptographic identity proof or parent-agent sandbox.
Do not fabricate such a record or infer approval from setup. Any enclosing model
call also needs its own token allocation/approval under the user's existing rule.

```bash
python3 -m experiments.cloud_confidence_container_v2.launcher execute \
  --evidence-root RESTORED_ROOT --selection PRIVATE_SELECTION.json \
  --proposal PRIVATE_PROPOSAL.json --authorization NEW_OWNER_APPROVAL.json \
  --image IMMUTABLE_IMAGE_ID --out NEW_PRIVATE_JOB_DIRECTORY
```

Before launching, the adapter verifies the original1609-file manifest identity
and all listed hashes. It stages those exact files, pinned selected-analysis
code and the selected request, refusing unsafe paths, symlinks, checkpoints,
credentials and confirmation banks. The host repository/home are never mounted.
The Docker socket is not mounted. `/workspace` and synthetic source/private
fixtures are read-only; only a new output and temporary directory are writable.

The parent inspects effective settings before start: nonroot UID/GID65534,
all capabilities dropped, no-new-privileges, no network, IPC disabled,32PIDs,
one CPU,2GiB memory/no swap,120CPU seconds and a wall ceiling of the smaller of
300seconds and the explicit remaining deadline. A file lock spans execution and
container removal. The child passes actual restrictions before reading source
observations. The original no-scientific-execution/torch-load guards remain.
Hash verification repeats after computation. CPU receipts are written after the
work, and a missing receipt, failed control, deadline, timeout or changed hash
blocks completion. No retries or fallback. `FILES.json` covers the exact new
output tree; all prior failed runs and sealed evidence remain unchanged.
