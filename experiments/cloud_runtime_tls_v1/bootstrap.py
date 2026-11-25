"""Transport-only entry point: ignore external PYTHONPATH and verify kit bytes."""
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]


def main():
    manifest = ROOT / 'PACKAGE_FILES.json'
    if not manifest.is_file():
        raise RuntimeError('This entry point requires the independently extracted verified kit')
    for name, expected in json.loads(manifest.read_text()).items():
        path = ROOT / name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError('Unsafe kit entry')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Different kit file: ' + name)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    if sys.argv[1:] == ['--tests-only']:
        suite = unittest.TestSuite()
        for program in ['cloud_runtime_tls_v1', 'cloud_confidence_container_v2']:
            suite.addTests(unittest.TestLoader().discover(str(ROOT / 'tests' / program)))
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    from experiments.cloud_runtime_tls_v1.build import main as build
    return build()


if __name__ == '__main__':raise SystemExit(main())
