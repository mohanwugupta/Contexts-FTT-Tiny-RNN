import os,subprocess,sys
from pathlib import Path
import pytest
from research_search.orchestration.isolation import profile,run,certify
from research_search.orchestration.queue import worker_lock

def test_single_worker_lock_released_after_exception(tmp_path):
 with worker_lock(tmp_path):
  with pytest.raises(RuntimeError):
   with worker_lock(tmp_path):pass
 with worker_lock(tmp_path):pass

def test_paths_cannot_escape_through_symlinks(tmp_path):
 from research_search.orchestration.isolation import safe_workspace
 p=tmp_path/"x";p.symlink_to("/tmp",target_is_directory=True)
 with pytest.raises(ValueError):safe_workspace(p,tmp_path)

def test_actual_OS_denies_private_read_and_evaluator_write(tmp_path):
 result=certify(tmp_path)
 if not result["sandbox_available"]:pytest.skip("Host sandbox unavailable; readiness must block, never pass silently")
 assert result["private_read_denied"] and result["evaluator_write_denied"] and result["workspace_write_allowed"] and result["symlink_read_denied"]

def test_missing_sandbox_never_falls_back(monkeypatch,tmp_path):
 import research_search.orchestration.isolation as i
 monkeypatch.setattr(i.shutil,"which",lambda _:None)
 with pytest.raises(RuntimeError):i.run([sys.executable,"-c","print(1)"],tmp_path,[])
