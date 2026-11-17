"""Three prediction-only recurrent architectures with a common interface."""

from __future__ import annotations

import hashlib
import torch
from torch import nn
from torch.nn import functional as F


class GRU150(nn.Module):
    architecture_id="A"; context_size=0; hidden_size=150
    def __init__(self):
        super().__init__(); self.gru=nn.GRU(46,150,batch_first=True); self.sequence_head=nn.Linear(150,4)
    def forward(self,tokens,evidence,state=None,overrides=None):
        hidden,terminal=self.gru(torch.cat([tokens,evidence],-1),state)
        return {"logits":self.sequence_head(hidden),"hidden":hidden,"context_state":None,"gate":None,"terminal":terminal}
    def shared_core_parameters(self): return list(self.gru.parameters())


class ContextControlledElman(nn.Module):
    hidden_size=128; context_size=8
    def __init__(self,*,gated=False):
        super().__init__(); self.gated=bool(gated); self.architecture_id="C" if gated else "B"
        self.context_gru=nn.GRUCell(46,8); self.input_to_core=nn.Linear(12,128); self.recurrent_core=nn.Linear(128,128,bias=False)
        self.context_to_core=nn.Linear(8,128,bias=False); self.core_bias=nn.Parameter(torch.zeros(128)); self.sequence_head=nn.Linear(128,4)
        self.gate_head=nn.Linear(8,128) if gated else None
    def forward(self,tokens,evidence,state=None,overrides=None):
        batch,length,_=tokens.shape
        if state is None: h=tokens.new_zeros(batch,128); c=tokens.new_zeros(batch,8)
        else: h,c=state
        hidden=[]; contexts=[]; gates=[]; overrides=overrides or {}
        forced_context=overrides.get("context_state"); forced_gate=overrides.get("gate")
        for step in range(length):
            controller_input=torch.cat([tokens[:,step],evidence[:,step]],-1); c=self.context_gru(controller_input,c)
            if forced_context is not None: c=forced_context[:,step] if forced_context.ndim==3 else forced_context
            gate=torch.sigmoid(self.gate_head(c)) if self.gated else torch.ones_like(h)
            if forced_gate is not None: gate=forced_gate[:,step] if forced_gate.ndim==3 else forced_gate
            recurrent_input=gate*h if self.gated else h
            task_input=torch.cat([tokens[:,step],evidence[:,step,:8]],-1)
            h=torch.tanh(self.input_to_core(task_input)+self.recurrent_core(recurrent_input)+self.context_to_core(c)+self.core_bias)
            hidden.append(h); contexts.append(c); gates.append(gate)
        hseq=torch.stack(hidden,1); cseq=torch.stack(contexts,1); gseq=torch.stack(gates,1)
        return {"logits":self.sequence_head(hseq),"hidden":hseq,"context_state":cseq,"gate":gseq if self.gated else None,"terminal":(h,c)}
    def shared_core_parameters(self): return list(self.input_to_core.parameters())+list(self.recurrent_core.parameters())+[self.core_bias]
    def controller_parameters(self):
        values=list(self.context_gru.parameters())+list(self.context_to_core.parameters())+list(self.sequence_head.parameters())
        if self.gate_head is not None: values+=list(self.gate_head.parameters())
        return values


def build_model(architecture_id,seed):
    torch.manual_seed(int(seed))
    if architecture_id=="A": model=GRU150()
    elif architecture_id=="B": model=ContextControlledElman(gated=False)
    elif architecture_id=="C": model=ContextControlledElman(gated=True)
    else: raise ValueError(architecture_id)
    for name,parameter in model.named_parameters():
        if parameter.ndim>=2: nn.init.xavier_uniform_(parameter)
        elif "bias" in name or name=="core_bias": nn.init.zeros_(parameter)
    return model


def model_hash(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()): digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def parameter_count(model): return sum(value.numel() for value in model.parameters())
