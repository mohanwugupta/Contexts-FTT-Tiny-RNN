"""Cohort-level hard prerequisite for probe interpretation."""


def splitting_gate(*, high_separation, low_separation, high_transition_cost,
                   low_transition_cost, high_mastery, low_mastery,
                   high_context_weight_norm, high_transfer_cost, low_transfer_cost):
    checks = {
        "G1_sequence_mastery": bool(high_mastery >= .80 and low_mastery >= .80),
        "G2_high_exceeds_low_separation": bool(
            high_separation > max(low_separation * 1.25, low_separation + .05)
        ),
        "G3_high_exceeds_low_transition_cost": bool(high_transition_cost > low_transition_cost),
        "G4_high_exceeds_low_transfer_cost": bool(high_transfer_cost > low_transfer_cost),
        "G5_context_input_used": bool(high_context_weight_norm > .05),
    }
    # G4 is diagnostic; the PRD's hard equations preregister G1/G2/G3/G5.
    hard = ["G1_sequence_mastery", "G2_high_exceeds_low_separation",
            "G3_high_exceeds_low_transition_cost", "G5_context_input_used"]
    return {**checks, "passed": bool(all(checks[name] for name in hard))}

