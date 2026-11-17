"""Repeated observations first collapse within trained seed."""
from experiments.output_compatible_trajectories.nullspace import complete_bootstrap,strict_novel_means


def seed_bootstrap(frame,metrics,keys):return complete_bootstrap(frame,metrics,keys)


def novel_means(frame,keys,metrics):return strict_novel_means(frame,['seed',*keys],metrics)
