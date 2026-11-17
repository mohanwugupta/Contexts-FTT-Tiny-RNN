"""Frozen architecture/schedule and modest execution-objective search grid."""

from __future__ import annotations

from dataclasses import dataclass

from ftt_srn_execution_v2.config import FIXED_SEQUENCE


FAMILY_LABELS = {
    "A": "confidence risk, quadratic",
    "B": "confidence risk, steeper penalty",
    "C": "predictive-entropy risk",
    "D": "probability-margin risk",
    "E": "risk-optimal policy cross-entropy",
}


def _number_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class ObjectiveCandidate:
    family: str
    c_time: float
    c_risk: float
    gamma: float
    lambda_exec: float = 0.20

    @property
    def candidate_id(self) -> str:
        return (
            f"{self.family}_ct{_number_tag(self.c_time)}_"
            f"cr{_number_tag(self.c_risk)}_g{_number_tag(self.gamma)}"
        )

    @property
    def uncertainty_type(self) -> str:
        return {
            "A": "one_minus_correct_confidence",
            "B": "one_minus_correct_confidence",
            "C": "normalized_predictive_entropy",
            "D": "one_minus_top_two_probability_margin",
            "E": "one_minus_correct_confidence",
        }[self.family]

    @property
    def equation(self) -> str:
        if self.family == "E":
            return (
                "BCE(v, stopgrad(clip(("
                f"{self.c_time:g}/(2*{self.c_risk:g}*(1-p_correct))), 0, 1)))"
            )
        uncertainty = {
            "A": "(1 - p_correct)",
            "B": "(1 - p_correct)",
            "C": "H(P(next))/log(4)",
            "D": "(1 - (p_top1 - p_top2))",
        }[self.family]
        return (
            f"{self.c_time:g}*(1-v) + {self.c_risk:g}*{uncertainty}*"
            f"v^{self.gamma:g}"
        )

    def validate(self):
        if self.family not in FAMILY_LABELS:
            raise ValueError(f"Unknown objective family {self.family!r}.")
        if self.c_time <= 0 or self.c_risk <= 0:
            raise ValueError("Time and risk costs must be positive.")
        if self.gamma <= 1:
            raise ValueError("The risk exponent must be greater than one.")
        if not 0 < self.lambda_exec <= 1:
            raise ValueError("lambda_exec must be in (0, 1].")
        if self.family == "A" and self.gamma != 2:
            raise ValueError("Candidate A is the quadratic confidence-risk objective.")
        if self.family in {"C", "D", "E"} and self.gamma != 2:
            raise ValueError(
                "Entropy, margin, and risk-optimal policy candidates use the minimal "
                "quadratic normative risk term."
            )


@dataclass(frozen=True)
class SearchConfig:
    hidden_dim: int = 8
    context_alpha: float = 1.0
    high_init_scale: float = 2.0
    learning_rate: float = 0.005
    gradient_clip: float = 1.0
    fixed_sequence: tuple[int, ...] = FIXED_SEQUENCE
    block_length: int = 10
    n_blocks: int = 17
    seeds: tuple[int, ...] = tuple(range(5))
    c_time_values: tuple[float, ...] = (0.05, 0.10)
    c_risk_values: tuple[float, ...] = (0.50, 1.00)
    lambda_exec: float = 0.20
    calibration_grid_size: int = 201
    central_confidence_low: float = 0.30
    central_confidence_high: float = 0.90
    minimum_calibration_dynamic_range: float = 0.20
    maximum_calibration_saturation_fraction: float = 0.50
    direction_window_sequences: int = 20
    minimum_direction_seed_fraction: float = 0.80
    saturation_epsilon: float = 0.02
    maximum_trained_saturation_fraction: float = 0.80
    minimum_trained_speed_range: float = 0.05
    minimum_switch_jump: float = 0.002
    minimum_context_coefficient: float = 0.002
    screen_only_calibration_passers: bool = True
    make_plots: bool = True
    show_progress: bool = True
    profile: str = "screen"

    @property
    def training_sequences(self) -> int:
        return int(self.n_blocks * self.block_length)

    def validate(self):
        if self.hidden_dim != 8:
            raise ValueError("The objective search fixes the Elman hidden layer at eight units.")
        if self.context_alpha != 1.0:
            raise ValueError("Context strength is fixed at one.")
        if self.high_init_scale != 2.0:
            raise ValueError("The objective search fixes High scale at 2.0.")
        if self.learning_rate != 0.005:
            raise ValueError("The established SRN learning rate is fixed at .005.")
        if self.block_length != 10:
            raise ValueError("The A10/B10 curriculum requires ten sequences per block.")
        if self.n_blocks < 2:
            raise ValueError("At least one A and B block is required.")
        if len(self.fixed_sequence) != 8 or set(self.fixed_sequence) - set(range(4)):
            raise ValueError("The fixed eight-item/four-key sequence must be preserved.")
        if not self.seeds:
            raise ValueError("At least one seed is required.")
        if not 0 < self.lambda_exec <= 1:
            raise ValueError("lambda_exec must be in (0, 1].")
        if self.calibration_grid_size < 21:
            raise ValueError("Calibration needs at least 21 certainty values.")
        if not 0 < self.central_confidence_low < self.central_confidence_high < 1:
            raise ValueError("The central certainty interval must lie inside (0, 1).")
        if self.profile == "screen":
            if self.n_blocks != 17 or self.training_sequences != 170:
                raise ValueError("A scientific screen must use the full 170-sequence curriculum.")
            if self.seeds != tuple(range(5)):
                raise ValueError("A scientific screen must use the prespecified five seeds.")


def build_candidate_grid(config: SearchConfig) -> tuple[ObjectiveCandidate, ...]:
    candidates = []
    for c_time in config.c_time_values:
        for c_risk in config.c_risk_values:
            candidates.append(ObjectiveCandidate(
                "A", c_time, c_risk, 2.0, config.lambda_exec
            ))
            # Candidate A already covers gamma=2; B adds only the steeper variants.
            for gamma in (3.0, 4.0):
                candidates.append(ObjectiveCandidate(
                    "B", c_time, c_risk, gamma, config.lambda_exec
                ))
            candidates.append(ObjectiveCandidate(
                "C", c_time, c_risk, 2.0, config.lambda_exec
            ))
            candidates.append(ObjectiveCandidate(
                "D", c_time, c_risk, 2.0, config.lambda_exec
            ))
    for candidate in candidates:
        candidate.validate()
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise AssertionError("Objective candidate identifiers are not unique.")
    order = {family: index for index, family in enumerate(("A", "B", "C", "D"))}
    return tuple(sorted(
        candidates,
        key=lambda item: (order[item.family], item.gamma, item.c_time, item.c_risk),
    ))


def stronger_primary_candidates(lambda_exec: float = .20) -> tuple[ObjectiveCandidate, ...]:
    """Targeted Candidate-A follow-up after every original family fails.

    Common multiplication of ``c_time`` and ``c_risk`` preserves the analytical
    optimum but strengthens gradients reaching a scale-2 saturated speed head.
    Ratios .05, .10, and .20 retain distinct calibrated policies.
    """
    candidates = tuple(
        ObjectiveCandidate("A", c_time, c_risk, 2.0, lambda_exec)
        for c_time, c_risk in (
            (.25, 5.0), (.50, 5.0), (1.00, 5.0),
            (.50, 10.0), (1.00, 10.0), (2.00, 10.0),
        )
    )
    for candidate in candidates:
        candidate.validate()
    return candidates


def post_failure_policy_candidates(
    lambda_exec: float = .20,
) -> tuple[ObjectiveCandidate, ...]:
    """Minimal expansion allowed only after the prespecified A--D families fail.

    Candidate E derives the same risk-optimal speed as Candidate A, but uses it
    as a soft Bernoulli policy target.  Its logit gradient is ``v - v*`` and so
    does not acquire the extra sigmoid-saturation factor in the direct economic
    cost.  The target is computed only from the network's detached uncertainty;
    it is not a human RT or externally supplied competence state.
    """
    # This bounded range spans conservative through near-saturation mappings.
    # .75 is the largest ratio that still passes the prespecified central-range
    # calibration gate; larger ratios cease to have an intermediate optimum.
    candidates = tuple(
        ObjectiveCandidate("E", c_time, 1.0, 2.0, lambda_exec)
        for c_time in (.05, .10, .20, .30, .40, .50, .60, .65, .70, .75)
    )
    for candidate in candidates:
        candidate.validate()
    return candidates


def smoke_config() -> SearchConfig:
    return SearchConfig(
        n_blocks=4,
        seeds=(0,),
        c_time_values=(0.10,),
        c_risk_values=(1.00,),
        calibration_grid_size=101,
        make_plots=False,
        profile="smoke",
    )


def screen_config() -> SearchConfig:
    return SearchConfig()
