"""Discovery/confirmation seed separation."""


def validate_fresh_seeds(discovery_seeds, confirmation_seeds):
    overlap = set(discovery_seeds) & set(confirmation_seeds)
    if overlap:
        raise ValueError(f"Discovery and confirmation seeds overlap: {sorted(overlap)}")
    return True

