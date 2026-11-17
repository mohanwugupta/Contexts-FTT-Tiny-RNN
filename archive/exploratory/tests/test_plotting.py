from probe_modeling.metrics import compute_behavioral_probe_metrics
from probe_modeling.plotting import plot_mixture_indices
from probe_modeling.skill_states import fit_counterfactual_states


def test_mixture_figure_handles_sem_column(tmp_path, normalized_schedule):
    counterfactuals, _ = fit_counterfactual_states(normalized_schedule)
    metrics = compute_behavioral_probe_metrics(counterfactuals)
    destination = tmp_path / "mixture.png"
    plot_mixture_indices(metrics, destination)
    assert destination.exists()
    assert destination.stat().st_size > 0

