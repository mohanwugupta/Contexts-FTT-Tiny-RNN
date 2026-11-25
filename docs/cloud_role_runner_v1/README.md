# Linux Codex role transport v1

This replaces the macOS-only role **transport** in a separately versioned rehearsal. It does not change scientific scoring, the numerical worker, historical results, the72h campaign or any campaign budget. No live model call has been made by this implementation.

## What it enforces

Each invocation creates a fresh nonroot Docker role container with a read-only root and only three ordinary mounts: pinned runtime source/CLI/CA (read-only), that request's schema (read-only), and a unique writable output directory. Optional system managed-policy files are preserved as additional read-only mounts. The host checkout, evaluator, confirmation data, other role workspaces, parent Codex configuration, history, plugins and Docker socket are absent. An inherited seccomp filter denies fork/vfork and non-thread clone, while permitting threads; clone3 returnsENOSYS for libc fallback. This needs neither sandbox-exec, Landlock nor user namespace creation by the child.

A fresh internal bridge gives the role no external route. Its only egress is a trusted CONNECT relay restricted to chatgpt.com:443 and auth.openai.com:443. The relay honors the existing managed HTTP(S) proxy and its authentication, without logging the proxy address/secret. Only the trusted relay has an external network. TLS verification remains enabled; the runtime combines existing system/configured certificates using the already accepted TLS helper. No new CA is downloaded. Unsupported proxies fail; no direct or unrestricted fallback is added.

The installed x86_64 Linux Codex ELF is discovered and hash-pinned. An installed npm codex.js launcher is resolved to its existing vendor binary, with both identities recorded; missing or ambiguous binaries stop preparation. No installation, upgrade or API-key billing fallback occurs. A synthetic fixture can also run the actual staged CLI's exec --help without authentication or provider access. This checks basic binary compatibility without a model turn.

Each real call uses only the existing ChatGPT login. Authentication is delivered in a bounded private stdin frame and exists in a read-only file inside a temporary memory filesystem. It never enters the image, host staging, requests or receipts; host authentication is hashed before/after and must remain unchanged. Temporary auth cannot persist a refresh; an expired session fails closed rather than modifying the original login. Known auth strings are redacted before event/stderr persistence. Runtime/evidence directories must stay private and must not be committed.

CLI settings explicitly disable tools, hooks, apps, plugins, web search, memories and nested agents. A fresh ephemeral conversation has no other role's context. The transport preserves any existing /etc/codex/requirements.toml and managed_config.toml; it does not remove managed requirements to make a call work.

## Approvals, clocks and accounting

`approval` writes a new envelope binding the existing request, runtime/binary identity, rehearsal definition, limits, original start and deadline. Old request-SHA approvals do not authorize the new transport. `call` requires exact envelope SHA approval. Preparation and fixtures do not read credentials or contact providers. Each live invocation first checks the existing login and ordinary quota inside the restricted container, then runs one exec turn. All three use the same isolation transport. Quota uses only initialize/initialized/account/rateLimits/read; never turn/start or reserve-credit methods.

The original Store reserves the request's unchanged token allowance and540 CPU seconds (conservative maximum for login/quota/exec wrappers, CLI children and relays). It reconciles actual process/relay CPU and provider turn.completed usage. Input+output are aggregated once, cached input tracked as a subset; no fabricated zero for missing usage. Invalid replies still retain and charge available usage. Missing usage, unknown CPU, over-reservation usage, tools, multiple turns, schema failure, cancellation or cleanup failure stop advancement. Unresolved resources prevent relaunch. No automatic retry is performed.

This runner never starts a missing clock, resets the six-hour ceiling, replaces limits or releases old holds. The existing rehearsal must already have an initialized Store, definition, budgets and request. The sealed72h campaign and historical result paths are rejected. A missing/expired clock requires explicit owner adjudication outside this adapter. A model-token reservation is not a provider-enforced in-turn cap; per-call manual approval remains required.

`new_agent_calls=0` in fixtures refers to actual model calls: fake usage events are parser fixtures, not provider receipts. Live usage stays unverified until an approved real turn returns it. Actual role container checks do not certify isolation of the enclosing Cloud chat agent.

## Existing controller integration

The public `invoke(store, ticket, approval, runtime=None, auth_file=None)` returns the same `passed`, `reply`, `reported_usage` and ledger fields used by the published integration runner. With runtime omitted it reads the explicit SRN_LINUX_ROLE_RUNTIME path. It never advances phases or queues scientific work. The rehearsal's existing accept_reply logic remains responsible for ordering and independent deterministic evaluation.

In a **new version of the blocked rehearsal adapter**, use:

```python
from experiments.cloud_role_runner_v1.agent import invoke, prepare_approval
# Display prepare_approval(store, ticket, runtime) to the owner.
result = invoke(store, ticket, approved_linux_envelope_sha, runtime)
if result['passed']:
    accept_reply(store, request['role'], result['reply'])  # existing registered controller
```

Keep the existing integrity checks; do not monkeypatch them out, edit the old sealed package, initialize another ledger, reset its clock, rerun completed compute or auto-call the next role. Retain the blocked rehearsal's source/18 tests/deadline and add a versioned transport manifest. The unpublished rehearsal files were not available in this local checkout or public main; their exact import wiring cannot be claimed tested here. The adapter API and standalone commands are implemented and tested.

## Validation

See SCRATCHPAD.md and validation receipts for actual status. Tests cover destination/port injection, API auth rejection, disabled tools, unchanged clocks, new-envelope approval, runtime mutation, malformed/multiple-turn/missing usage, known usage retained after invalid reply, reservation holds, ambiguous executables, forbidden historical paths and independent campaign/runtime locks. A Linux GitHub Action additionally exercises actual filesystem, symlink/hardlink, process/thread and network restrictions with synthetic events. No scientific model or real Codex turn is run in CI.

## Source map

- contract.py: fixed endpoints/CLI options, auth validation, redaction and clock semantics.
- runtime.py: local installed executable resolution, CA/managed-policy pins; no provider calls.
- proxy.py: trusted bounded CONNECT relay; no model requests of its own.
- bootstrap.py: actual child restrictions, ephemeral auth, one CLI/RPC process, private outputs/CPU receipts.
- docker.py: internal network, effective configuration checks, bounded lifecycle, cleanup lock/receipts.
- accounting.py: unchanged repaired native event assessor.
- agent.py: existing Store approval, quota, reservation, actual usage and manual checkpoint adapter.
- __main__.py: prepare/fixture/approval/call only; no launch-campaign or training command.

Current official CLI behavior is documented at https://learn.chatgpt.com/docs/non-interactive-mode and https://learn.chatgpt.com/docs/config-file/config-reference . Local installed exec/login/app-server help was inspected without dispatch. Synthetic acceptance is not a substitute for one separately approved live role receipt.
