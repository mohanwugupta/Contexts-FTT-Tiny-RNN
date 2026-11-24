import pytest
from experiments.recurrent_learning_transfer.verify import verify_hashes
from experiments.recurrent_learning_transfer.design import hash_file

def test_integrity_rejects_changed_missing_and_escaping_files(tmp_path):
 p=tmp_path/'data';p.write_text('original');expected={'data':hash_file(p)}
 assert verify_hashes(tmp_path,expected)==1
 p.write_text('corrupt')
 with pytest.raises(ValueError):verify_hashes(tmp_path,expected)
 p.unlink()
 with pytest.raises(ValueError):verify_hashes(tmp_path,expected)
 with pytest.raises(ValueError):verify_hashes(tmp_path,{'../outside':'bad'})

def test_future_tokens_cannot_change_earlier_predictions():
 import torch
 from experiments.recurrent_learning_transfer.network import pair
 from experiments.recurrent_learning_transfer.design import tokens,CUES
 x=tokens();changed=x.clone();changed[4:]=torch.roll(changed[4:],1,dims=1)
 for model in pair(0).values():
  z,h=model(x,torch.tensor(CUES['A']))
  zz,hh=model(changed,torch.tensor(CUES['A']))
  assert torch.equal(z[:4],zz[:4]) and torch.equal(h[:4],hh[:4])
