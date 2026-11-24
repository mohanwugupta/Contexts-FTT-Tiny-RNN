"""Only extend the source switch loop horizon; all definitions unchanged."""
import inspect
from .protocol import target_cue
from experiments.recurrent_learning_transfer import measures as native
s=inspect.getsource(native.boundaries).replace('def boundaries(learning):','def extended_boundaries(learning):').replace("p=f.pivot(index='update',columns='cue',values='ce')","p=f.pivot(index='update',columns='cue',values='ce');last=int(p.index.max())").replace('range(10,331,10)','range(10,last+1,10)').replace('min(10,330-k)','min(10,last-k)')
exec(compile(s,__file__,'exec'),globals() | vars(native) | {'training_cue':target_cue},_ns:={})
extended_boundaries=_ns['extended_boundaries']
