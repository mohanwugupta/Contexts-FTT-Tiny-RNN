"""Discover only the hash-pinned installed CLI; never install, run or upgrade."""
import os,shutil
from pathlib import Path
from research_search.common import ROOT,read,sha


def resolve_codex(expected_sha,*,environ=None,home=None,explicit=None):
    environ=os.environ if environ is None else environ
    home=Path.home() if home is None else Path(home)
    override=explicit or environ.get('SRN_CODEX_EXECUTABLE')
    candidates=[]
    if override:candidates=[Path(override).expanduser()]
    else:
        found=shutil.which('codex',path=environ.get('PATH',''))
        if found:candidates.append(Path(found))
        candidates.extend(sorted((home/'.vscode/extensions').glob('openai.chatgpt-*/bin/macos-*/codex')))
    mismatched=[];invalid=[]
    for candidate in candidates:
        path=candidate.resolve()
        if not path.is_file() or not os.access(path,os.X_OK):invalid.append(str(path));continue
        if sha(path)==expected_sha:return path
        mismatched.append(str(path))
    detail=(' Installed candidates failed the pinned SHA256 check: '+', '.join(mismatched)) if mismatched else (' Path is not an executable file: '+', '.join(invalid)) if invalid else ' No executable was found on PATH or in installed VS Code extensions.'
    raise RuntimeError('Cannot locate the approved Codex CLI.'+detail+' Set SRN_CODEX_EXECUTABLE to the absolute path of the previously tested executable; a different version requires separate validation.')


def pinned_codex():
    pin=read(ROOT/'docs/research_search_integration_v1_1/BACKEND_PIN.json')
    return resolve_codex(pin['codex_sha256'])
