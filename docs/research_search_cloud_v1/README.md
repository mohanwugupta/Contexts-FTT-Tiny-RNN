# Codex Cloud preparation v1

Cloud tasks can run independently of the laptop, but the local macOS backend is not portable as written. This namespace packages the exact saved evidence and a portable deterministic preflight without using macOS sandbox-exec, local VS Code executables, auth.json or API billing fallback. All original experiments/seals remain unchanged.

## Local export

```bash
PYTHONPATH=.:src python -m experiments.research_search_cloud_v1 export --destination /private/tmp/srn_cloud_evidence_v1
```

The resulting tar.gz is PRIVATE: it contains only reviewed scientific source and the specific saved development tables required by the evaluator, no checkpoints, raw human data, confirmation bank, auth files, agents' raw traces or operational databases. The manifest pins every file; private source tables retain exact original paths/hashes. Never commit this bundle to the public repository. Supply it privately to your cloud task/environment.

## Cloud environment and first task

Connect the updated GitHub repository in Codex Cloud, install with install.sh, then restore the private bundle preserving paths. Use CLOUD_TASK.md as the separately approved preflight task. The deterministic preflight verifies hashes and imports the existing saved-results evaluator, recording the actual cloud platform/software separately. It starts no model, training, research-role call or72h campaign.

Local acceptance does not validate Linux/cloud isolation or provider token telemetry. The full research cycle is deliberately blocked until these pass. Cloud task/model usage itself must be counted; missing usage stops dispatch, and manual approval remains required for every real role call. A cloud task is not a mechanism for bypassing that contract. No cloud setup was remotely launched by preparing these files.

The cloud account UI must create/publish the environment and start the approved first task. The current agent session has no cloud-environment launch tool and its browser host is unavailable; it cannot perform those account-side actions. Exact artifact hash/size and local acceptance are in results/research_search_cloud/v1/. The original public repository does not contain private historical evidence merely because source code has been updated.
