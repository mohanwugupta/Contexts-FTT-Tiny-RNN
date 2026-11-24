"""Semantic numeric cache keys, without editing any historical hash function."""
import copy, hashlib, json, math
from decimal import Decimal
from research_search.runners.candidate import DEFAULT, validate


def _canonical(value):
    if value is None or type(value) in (bool,str):
        return [type(value).__name__,value]
    if type(value) in (int,float):
        if type(value) is float and not math.isfinite(value):
            raise ValueError('Nonfinite configuration number')
        number=Decimal(str(value))
        # Decimal normalization uses context precision; fixed-point formatting
        # followed by trimming preserves arbitrary integer/float identities.
        text=format(number,'f')
        if '.' in text:text=text.rstrip('0').rstrip('.')
        if number==0:text='0'
        return ['number',text]
    if isinstance(value,dict):
        if any(type(k) is not str for k in value):raise ValueError('String keys required')
        return ['object',[[k,_canonical(v)] for k,v in sorted(value.items())]]
    if isinstance(value,(list,tuple)):return ['array',[_canonical(v) for v in value]]
    raise ValueError('Unsupported configuration type')


def canonical_hash(value):
    data=json.dumps(_canonical(value),ensure_ascii=True,separators=(',',':')).encode()
    return hashlib.sha256(data).hexdigest()


def normalize_config(config):
    c=copy.deepcopy(config)
    if set(c)!=set(DEFAULT) or any(set(c[g])!=set(DEFAULT[g]) for g in DEFAULT):raise ValueError('Unknown configuration fields')
    for group,fields in DEFAULT.items():
        for key,default in fields.items():
            value=c[group][key]
            if type(default) in (bool,int,str) and type(value) is not type(default):raise ValueError('Incorrect type: '+group+'.'+key)
            if type(default) is float:
                if value is None and (group,key)==('model','recurrent_norm'):continue
                if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('Finite numeric field required')
                c[group][key]=float(value)
    return validate(c)


def prior_identity(config,seed,provenance):
    c=normalize_config(config)
    if type(seed) is not int:raise ValueError('Integer seed required')
    required={'source_sha','code_sha','task_exclusion_sha','rng_contract','adam_contract'}
    if not required<=set(provenance) or not all(provenance[k] for k in required):raise ValueError('Complete provenance required')
    # Target-only changes do not require a second identical prior. Every
    # source/task exclusion/RNG/Adam distinction remains part of the key.
    return canonical_hash(dict(version='semantic-prior-v1',model=c['model'],prior=c['prior'],seed=seed,provenance=provenance))
