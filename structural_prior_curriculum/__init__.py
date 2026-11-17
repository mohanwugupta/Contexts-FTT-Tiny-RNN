"""Sequentially gated structural-prior curriculum."""

from .config import CurriculumConfig, load_curriculum_config
from .pipeline import run_structural_prior_curriculum

__all__ = ["CurriculumConfig", "load_curriculum_config", "run_structural_prior_curriculum"]
