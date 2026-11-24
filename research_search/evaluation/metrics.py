"""Thin bindings to inherited signed functional measures, not geometric proxies."""
from experiments.joint_phenotype_hypothesis_screen import rules
from experiments.recurrent_learning_transfer.measures import boundaries
from experiments.centered_context_linear.analysis import block_table
from experiments.meta_learning_scaling.analysis import matched_acquisition

def first_blocks(frame):return rules.summarize(frame)
def learning_tables(frame):
    f=frame[frame.cue.isin(['A','B'])]
    blocks=block_table(f.assign(arm='candidate'))
    switches=boundaries(f.assign(architecture='candidate'))
    matched=matched_acquisition(blocks.assign(condition='candidate',dose=0))
    return blocks,switches,matched
