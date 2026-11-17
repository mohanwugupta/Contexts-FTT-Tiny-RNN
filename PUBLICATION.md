# GitHub publication and evidence restoration

Local Git history remains untouched; the GitHub repository begins with a publication snapshot commit.

Private repository: https://github.com/mohanwugupta/Contexts-FTT-Tiny-RNN

## What is included

The Git repository publishes a current working-tree snapshot of source/tests/protocols, `AGENTS.md`, `REPO_MAP.md`, and the complete latest sealed `results/context_failure_selective_repair/v1_1/` results. The20 original High checkpoints listed by that program are also included at their original paths. Existing `.gitignore` continues to exclude bulk outputs by default; only these explicit scientific artifacts are force-added.

The `research-snapshot-2026-11-17` GitHub release contains `research-evidence.tar.gz` and `research-evidence.sha256`. The archive contains all798 exact source/evidence dependencies listed in the latest sealed manifest, all101 sealed latest result files, the seal itself and the16 current implementation/test files. This includes large predecessor tables that exceed GitHub's ordinary single-file limit. Paths and scientific file bytes are preserved. The archive is an evidence restoration package, not another experimental run.

Unrelated historical outputs, raw human data, virtual environments, bytecode caches and local credentials are excluded. The whole local multi-gigabyte workspace is not claimed to be present in a Git clone. Optional cloud-only historical files remain unavailable exactly as recorded in the reports; no content/hash was invented.

## Restore full evidence

Clone the private repository using an account with access, then from its root:

```bash
gh release download research-snapshot-2026-11-17 --repo mohanwugupta/Contexts-FTT-Tiny-RNN --pattern 'research-evidence*'
shasum -a 256 -c research-evidence.sha256
tar -xzf research-evidence.tar.gz
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
PYTHONPATH=.:src python -m experiments.context_failure_selective_repair.verify --output results/context_failure_selective_repair/v1_1
```

Restore into a clean clone. The release files duplicate a subset of already tracked files using the same hashes; do not extract over a directory containing scientific edits. Verification checks101 scientific outputs,798 dependencies, implementation identities and20 original checkpoint identities, without training. Table-only regeneration instructions are in `REPO_MAP.md` and the final report.

`publication/evidence_inventory.json` records every archived relative path, size and SHA256. The release checksum verifies the complete compressed archive. Repo-map additions do not alter sealed scientific code or outputs.
