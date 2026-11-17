"""Cue definitions and behavior-blind familiar-cue decomposition."""

from __future__ import annotations

import inspect
import json
import numpy as np
import pandas as pd


SQRT3=float(np.sqrt(3.0))
CUES={
    "A":np.array([1.,0.,0.],dtype=np.float32),
    "B":np.array([0.,1.,0.],dtype=np.float32),
    "Mixed":np.array([.5,.5,0.],dtype=np.float32),
    "NoContext":np.zeros(3,dtype=np.float32),
    "Novel1":np.array([1.,1.,1.],dtype=np.float32)/SQRT3,
    "Novel2":np.array([1.,-1.,1.],dtype=np.float32)/SQRT3,
    "Novel3":np.array([-1.,1.,1.],dtype=np.float32)/SQRT3,
    "Novel4":np.array([-1.,-1.,1.],dtype=np.float32)/SQRT3,
}
PROBES=("Mixed","NoContext","Novel1","Novel2","Novel3","Novel4")
TARGET_CONSTRUCTION_CUES=("A","B","NoContext")


def solve_cue_coefficients(cue_a,cue_b,cue_mixed,cue_zero,tolerance=1e-6):
    """Solve from cue vectors only; behavioral arguments are deliberately impossible."""
    a=np.asarray(cue_a,float)-np.asarray(cue_zero,float);b=np.asarray(cue_b,float)-np.asarray(cue_zero,float)
    target=np.asarray(cue_mixed,float)-np.asarray(cue_zero,float)
    coefficients,*_=np.linalg.lstsq(np.column_stack([a,b]),target,rcond=None)
    reconstructed=np.asarray(cue_zero,float)+coefficients[0]*a+coefficients[1]*b
    residual=float(np.linalg.norm(np.asarray(cue_mixed,float)-reconstructed))
    if residual>=tolerance:raise AssertionError("FAIL: MIXED_NOT_IN_FAMILIAR_CUE_SPAN")
    return float(coefficients[0]),float(coefficients[1]),residual


def cue_coefficients(tolerance=1e-6):
    return solve_cue_coefficients(CUES["A"],CUES["B"],CUES["Mixed"],CUES["NoContext"],tolerance)


def assert_coefficient_firewall():
    allowed={"cue_a","cue_b","cue_mixed","cue_zero","tolerance"}
    if set(inspect.signature(solve_cue_coefficients).parameters)!=allowed:
        raise AssertionError("FAIL: COEFFICIENT_BEHAVIOR_LEAKAGE")
    return True


def composite(base,a,b,alpha_a,alpha_b):
    return base+alpha_a*(a-base)+alpha_b*(b-base)


def cue_audit_table(preactivation_residual_max=np.nan):
    alpha_a,alpha_b,residual=cue_coefficients();rows=[]
    for name,value in CUES.items():
        active=np.flatnonzero(np.abs(value)>1e-12).tolist()
        rows.append({"condition":name,"cue_vector":json.dumps([float(x) for x in value]),"cue_dimensionality":len(value),
                     "cue_norm":float(np.linalg.norm(value)),"active_dimensions":json.dumps(active),
                     "is_normalized":bool(np.isclose(np.linalg.norm(value),1.)),
                     "contains_both_familiar_channels":bool(name=="Mixed" and value[0]!=0 and value[1]!=0),
                     "is_exact_zero":bool(np.array_equal(value,np.zeros_like(value))),
                     "alpha_A":alpha_a,"alpha_B":alpha_b,"AB_span_residual":residual,
                     "is_exact_AB_composition":residual<1e-6,
                     "preactivation_composition_residual_max":preactivation_residual_max,
                     "used_for_coefficient_fit":name in ("A","B","Mixed","NoContext"),
                     "used_for_training":name in ("A","B")})
    return pd.DataFrame(rows)
