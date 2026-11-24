from pathlib import Path
import pytest
from research_search.common import sha
from experiments.research_search_integration_v1_1.executable import resolve_codex


def binary(path,data=b'fixture executable'):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);path.chmod(0o700);return path


def test_empty_path_discovers_pinned_extension_not_newest(tmp_path):
    old=binary(tmp_path/'.vscode/extensions/openai.chatgpt-1/bin/macos-aarch64/codex')
    binary(tmp_path/'.vscode/extensions/openai.chatgpt-2/bin/macos-aarch64/codex',b'changed')
    assert resolve_codex(sha(old),environ={'PATH':''},home=tmp_path)==old.resolve()


def test_override_requires_pinned_hash_and_executable(tmp_path):
    p=binary(tmp_path/'explicit/codex')
    assert resolve_codex(sha(p),environ={'PATH':'','SRN_CODEX_EXECUTABLE':str(p)},home=tmp_path)==p.resolve()
    with pytest.raises(RuntimeError,match='SHA256'):resolve_codex('0'*64,environ={'SRN_CODEX_EXECUTABLE':str(p)},home=tmp_path)
    p.chmod(0o600)
    with pytest.raises(RuntimeError,match='executable'):resolve_codex(sha(p),environ={'SRN_CODEX_EXECUTABLE':str(p)},home=tmp_path)


def test_missing_actionable_error_not_none_type(tmp_path):
    with pytest.raises(RuntimeError,match='SRN_CODEX_EXECUTABLE'):resolve_codex('0'*64,environ={'PATH':''},home=tmp_path)


def test_path_fallback_is_hash_checked(tmp_path):
    p=binary(tmp_path/'bin/codex')
    assert resolve_codex(sha(p),environ={'PATH':str(p.parent)},home=tmp_path)==p.resolve()
    with pytest.raises(RuntimeError,match='SHA256'):resolve_codex('0'*64,environ={'PATH':str(p.parent)},home=tmp_path)


def test_live_adapter_uses_identical_explicit_binary_for_quota():
    import inspect
    from experiments.research_search_integration_v1_1 import agent
    text=inspect.getsource(agent.invoke)
    assert "resolve_codex(request['codex_sha256'],explicit=request['codex_path'])" in text
    assert 'rpc_quota(rt,command=app_server_argv(codex,rt)' in text
    assert "shutil.which('codex')" not in text
