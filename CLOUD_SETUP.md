# Codex Cloud setup

The source can be prepared in a cloud environment while your laptop is off. This repository is currently public. Credentials, checkpoint histories, raw human data, private operational evidence and unpublished result reports are excluded from this source refresh.

1. Create or select a Codex Cloud environment for mohanwugupta/Contexts-FTT-Tiny-RNN using the updated main branch.
2. Privately transfer the separately supplied evidence.tar.gz and its checksum file. Do not commit the archive to this public repository. Verify the checksum and restore its paths at the cloud checkout root.
3. Use [the install script](docs/research_search_cloud_v1/install.sh) to install CPU dependencies.
4. Review [the bounded preflight task](docs/research_search_cloud_v1/CLOUD_TASK.md) and approve that single cloud call. Return the exact JSON, tests, errors and actual aggregate model-token receipt.
5. Stop after preflight. Missing token telemetry or unvalidated cloud isolation blocks research-agent dispatch. Do not launch a scientific experiment, model update or the original72-hour campaign.

The current build was tested locally in an isolated directory without imports falling back to the original checkout. Linux/cloud acceptance still requires the actual remote preflight. Existing local readiness does not establish remote readiness. Each real agent call still requires manual approval; the cloud environment can wait independently of your laptop between calls.

No local Codex auth file or Mac-only executable is transferred. No API billing fallback or automatic credit purchase is configured. The private bundle contains only the explicit saved development tables and required source, not confirmation tasks or model checkpoints. See [the source map](CLOUD_REPO_MAP.md).
