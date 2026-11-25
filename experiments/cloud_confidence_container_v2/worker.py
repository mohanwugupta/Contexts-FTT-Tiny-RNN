"""Container entry point: enforce proofs before imports or cached calculations."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import resource
import time
from types import SimpleNamespace


def main():
    started_cpu = time.process_time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['smoke', 'execute'], required=True)
    args = parser.parse_args()
    # Discard image/client-supplied proxies and runtime variables, not just keys
    # whose names look like credentials. Never print original environment.
    os.environ.clear()
    os.environ.update(PATH='/usr/local/bin:/usr/bin:/bin', HOME='/tmp', TMPDIR='/tmp',
                      LANG='C.UTF-8', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', MPLCONFIGDIR='/tmp/mpl',
                      PYTHONDONTWRITEBYTECODE='1')
    sys.path[:0] = ['/workspace', '/workspace/src']
    from experiments.worker_isolation_preflight_v1.container_probe import probe
    from experiments.worker_isolation_preflight_v1.preflight import verify_checks
    from experiments.cloud_confidence_container_v2.contract import verify_bundle, compare_scientific_body, validate_authorization, sha
    out = Path('/output')
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        probe()
    proof = json.loads(stream.getvalue())
    verify_checks(proof['checks'])
    # Test the *actual* evidence mount too, not only the synthetic source mount.
    try:
        with Path('/workspace/STAGED_FILES.json').open('a') as f:
            f.write('MUST_BE_DENIED')
    except PermissionError:
        pass
    except OSError as error:
        if error.errno != 30:
            raise
    else:
        raise RuntimeError('Evidence mount was writable')
    proof['evidence_write_denied'] = True
    (out / 'WORKER_ISOLATION.json').write_text(json.dumps(proof, indent=2) + '\n')
    root = Path('/workspace')
    hashes = json.loads((root / 'STAGED_FILES.json').read_text())
    verify_bundle(root, hashes)
    compare_scientific_body()
    if args.mode == 'smoke':
        import numpy as np
        import pandas as pd
        import torch
        import matplotlib
        import jsonschema
        from experiments.cloud_confidence_selected_v1.analysis import classify, accounting
        # Seven artificial position losses. No archived observation or model.
        p = pd.DataFrame(dict(position=range(1, 8), group=['synthetic'] * 7,
                              correct_reference=[False] * 7, correct_alternative=[False] * 7,
                              ce_reference=np.arange(7.) + 2, ce_alternative=np.arange(7.) + 1))
        result = accounting(classify(p), ['group'])
        if float(result.CE_gain_contribution.sum()) != 1 or result.positions.sum() != 7:
            raise ValueError('Synthetic signed accounting failed')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(); ax.plot([0, 1], [0, 1]); fig.savefig(out / 'synthetic.png'); plt.close(fig)
        status = dict(passed=True, mode='synthetic_dependency_smoke',
                      software=dict(numpy=np.__version__, pandas=pd.__version__, torch=torch.__version__, matplotlib=matplotlib.__version__),
                      scientific_inputs_read=0, new_model_forwards=0, new_optimizer_steps=0,
                      parent_agent_isolation_certified=False)
        (out / 'SMOKE.json').write_text(json.dumps(status, indent=2) + '\n')
    else:
        request = root / 'selected_request'
        authorization = json.loads((request / 'authorization.json').read_text())
        validate_authorization(authorization)
        for name in ['proposal', 'selection']:
            if sha(request / (name + '.json')) != authorization[name + '_sha256']:
                raise ValueError('Selected request hash differs from approval')
        from experiments.cloud_confidence_container_v2.compute import compute
        manifest = json.loads((root / 'CLOUD_INPUT_MANIFEST.json').read_text())
        compute(SimpleNamespace(selection=request / 'selection.json', proposal=request / 'proposal.json'), out, manifest)
    verify_bundle(root, hashes)
    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    usage = dict(CPU_seconds=time.process_time() - started_cpu + child_usage.ru_utime + child_usage.ru_stime,
                 CPU_cap_seconds=120)
    (out / 'PROCESS_USAGE.json').write_text(json.dumps(usage, indent=2) + '\n')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        Path('/output/BLOCKED.json').write_text(json.dumps(dict(error_type=type(error).__name__,
            reason=str(error), no_automatic_retry=True, scientific_outcome=None), indent=2) + '\n')
        raise
