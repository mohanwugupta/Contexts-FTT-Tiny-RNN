import torch
import pytest
from experiments.a06_novel_cue_followup import execution as e
def test_only_registered_new_target_updates_can_execute():
 for seed,condition,k,cue in [(0,'A06',1,'A'),(2,'A00',1,'A'),(2,'A06',1,'C'),(2,'A06',11,'A'),(2,'A06',331,'B')]:
  with pytest.raises(ValueError):e.authorized_update(seed,condition,k,cue)
 assert e.authorized_update(2,'A06',11,'B')
def test_resume_never_repeats_uncertain_updates():
 assert e.validate_cursor(30,30,'x','x')
 for args in [(29,30,'x','x'),(30,30,'x','y')]:
  with pytest.raises(ValueError):e.validate_cursor(*args)
