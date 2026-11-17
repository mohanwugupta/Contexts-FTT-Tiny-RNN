import numpy as np
import pandas as pd
from context_sequence.task.novelty import GROUPS,VARIANTS


def novel_means(frame,keys,metrics):
    if frame.probe.isin(GROUPS).any():raise ValueError('Aggregate labels cannot be reaggregated')
    novel=frame[frame.probe.isin(VARIANTS)]
    if novel.duplicated(keys+['probe']).any():raise ValueError('Duplicate variant weights')
    if not novel.groupby(keys,dropna=False).probe.nunique().eq(4).all():raise ValueError('Missing Novel variant')
    parts=[frame]
    for label,members in GROUPS.items():
        mean=novel[novel.probe.isin(members)].groupby(keys,as_index=False,dropna=False)[metrics].mean();mean['probe']=label;parts.append(mean)
    return pd.concat(parts,ignore_index=True)


def paired_effect(left,right):
    if left.index.has_duplicates or right.index.has_duplicates or set(left.index)!=set(right.index):raise ValueError('Paired seeds differ')
    return left.sort_index()-right.sort_index()
