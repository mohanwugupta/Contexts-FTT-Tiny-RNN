import json
from pathlib import Path
import pytest
from research_search.common import ROOT
from experiments.research_backend_repair_v1.backend import create_runtime,environment,profile,exec_argv


def test_private_external_runtime_and_readonly_login_link(tmp_path):
    home=tmp_path/'original';home.mkdir();auth=home/'auth.json';auth.write_text('fixture credential')
    rt=create_runtime(tmp_path/'runtime',auth)
    link=rt.home/'auth.json'
    assert link.is_symlink() and link.samefile(auth)
    assert rt.workspace.stat().st_mode & 0o777 == 0o700
    assert rt.home != home and rt.tmp.parent==rt.home.parent
    assert not rt.home.is_relative_to(rt.tmp)
    assert auth.read_text()=='fixture credential'


def test_clean_environment_strips_parent_client_and_billing_routes(tmp_path):
    home=tmp_path/'original';home.mkdir();auth=home/'auth.json';auth.write_text('{}');rt=create_runtime(tmp_path/'runtime',auth)
    env=environment(rt,{'PATH':'/usr/bin','HOME':str(tmp_path),'CODEX_THREAD_ID':'private','CODEX_EXEC_SERVER_NOISE_AUTH_TOKEN':'private','VSCODE_IPC_HOOK':'private','OPENAI_API_KEY':'private','OPENAI_BASE_URL':'private','ANTHROPIC_API_KEY':'private','PYTHONPATH':'private'})
    assert env['CODEX_HOME']==str(rt.home) and env['TMPDIR']==str(rt.tmp)
    assert all(x!='private' for x in env.values())
    assert env['PATH']=='/usr/bin'


def test_nested_and_symlink_runtime_paths_refused(tmp_path):
    with pytest.raises(ValueError,match='repository'):
        create_runtime(ROOT/'research_search/runtime_validation/forbidden',tmp_path/'auth')
    a=tmp_path/'a';a.mkdir();link=tmp_path/'link';link.symlink_to(a,target_is_directory=True)
    with pytest.raises(ValueError):create_runtime(link,tmp_path/'auth')


def test_profile_retains_repository_auth_and_network_boundaries(tmp_path):
    home=tmp_path/'original';home.mkdir();auth=home/'auth.json';auth.write_text('{}');rt=create_runtime(tmp_path/'runtime',auth)
    p=profile(rt,'/usr/bin/true',network=False)
    assert '(deny network*)' in p and str(ROOT) in p
    assert '(deny file-link)' in p and str(auth) in p
    assert str(rt.home/'auth.json') in p


def test_exec_has_no_user_tools_and_routes_runtime_config(tmp_path):
    home=tmp_path/'original';home.mkdir();auth=home/'auth.json';auth.write_text('{}');rt=create_runtime(tmp_path/'runtime',auth)
    args=exec_argv('/codex',rt)
    for value in ['--ignore-user-config','--ignore-rules','--ephemeral','--json','shell_tool','multi_agent','plugins','hooks','apps']:
        assert value in args
    assert any(str(rt.state) in x for x in args) and any(str(rt.logs) in x for x in args)


def test_builtin_provider_is_not_overridden(tmp_path):
    h=tmp_path/'original';h.mkdir();a=h/'auth.json';a.write_text('{}');rt=create_runtime(tmp_path/'runtime',a)
    assert not any('model_providers.openai.' in arg for arg in exec_argv('/codex',rt))
