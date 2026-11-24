import json,time
import pytest
from research_search.orchestration.store import Store
from research_search.orchestration.budget import BudgetError
from research_search.evaluation.development import make_banks,orbit

def limits(**kw):return dict(tokens=1000,cpu_seconds=10,optimizer_steps=3,wall_seconds=100,proposals=2,active_branches=2,**kw)
def test_reservation_survives_restart_and_unknown_usage_pauses(tmp_path):
 s=Store(tmp_path,limits());s.reserve_call("a","hypothesis",200);s.close();s=Store(tmp_path,limits())
 with pytest.raises(BudgetError):s.reserve_call("b","hypothesis",900)
 s.reconcile_call("a",None);assert s.state()["status"]=="manual_checkpoint"
 with pytest.raises(BudgetError):s.reserve_call("b","hypothesis",1)

def test_usage_no_cached_double_count_no_unknown_reasoning(tmp_path):
 s=Store(tmp_path,limits());s.reserve_call("a","hypothesis",200);s.reconcile_call("a",dict(input_tokens=100,cached_input_tokens=30,output_tokens=50));assert s.totals()["tokens"]==150 and s.totals()["cached_input_tokens"]==30
 with pytest.raises(ValueError):s.reconcile_call("a",dict(input_tokens=100,output_tokens=50,cached_input_tokens=30))

def test_atomic_budget_and_uncertain_steps_never_replay(tmp_path):
 s=Store(tmp_path,limits());s.charge_step("job",1);s.close();s=Store(tmp_path,limits())
 with pytest.raises(BudgetError):s.charge_step("job",1)
 with pytest.raises(BudgetError):s.recover("job",0)
 s.complete_step("job",1,"abc");assert s.recover("job",1)==1

def test_immutable_proposals_duplicates_parent_and_caps(tmp_path):
 s=Store(tmp_path,limits());p=dict(id="p1",parent_candidate=None,mechanism={"readout_scale":.25},config={"x":1})
 s.register(p,"code");q=dict(p,id="p2")
 with pytest.raises(ValueError):s.register(q,"code")
 with pytest.raises(ValueError):s.register(dict(p,config={"x":2}),"code")
 assert len(s.nodes())==1

def test_banks_deterministic_no_dihedral_leakage():
 a,b=make_banks();aa,bb=make_banks();assert a==aa and b==bb
 for x in a["tasks"]:
  assert all(not orbit(x["path"])&orbit(y["path"]) for y in b["tasks"])
 assert set(a["seeds"]).isdisjoint(b["seeds"]) and len(a["tasks"])>=3
