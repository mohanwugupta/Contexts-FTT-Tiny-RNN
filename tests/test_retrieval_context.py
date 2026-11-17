import numpy as np

from retrieval_modeling.context_features import (
    CONTEXT_VECTORS,
    canonical_retrieval_context,
    context_feature_table,
)
from retrieval_modeling.context_geometry import context_geometry
from retrieval_modeling.retrieval_weights import similarity_responsibility


def test_task_derived_context_feature_contract_includes_no_context():
    assert CONTEXT_VECTORS == {
        "A": (1.0, 0.0, 0.0),
        "B": (0.0, 1.0, 0.0),
        "Both": (1.0, 1.0, 0.0),
        "Novel": (0.0, 0.0, 1.0),
        "Default": (0.0, 0.0, 0.0),
    }
    table = context_feature_table()
    assert set(table.context) == set(CONTEXT_VECTORS)
    assert not any(column.lower().endswith("rt") for column in table.columns)


def test_geometry_separates_relative_from_absolute_evidence():
    geometry = context_geometry(beta=1.0).set_index("context")
    assert geometry.loc["Both", "d_A"] == geometry.loc["Both", "d_B"]
    assert geometry.loc["Novel", "d_A"] == geometry.loc["Novel", "d_B"]
    assert geometry.loc["Novel", "d_A"] > geometry.loc["Both", "d_A"]
    assert geometry.loc["Both", "relative_evidence_R"] == 0
    assert geometry.loc["Novel", "relative_evidence_R"] == 0
    assert geometry.loc["Both", "absolute_evidence_E"] > geometry.loc["Novel", "absolute_evidence_E"]
    # No Context is a third symmetric target with a different absolute match.
    assert geometry.loc["Default", "q_A"] == 0.5


def test_actual_three_context_export_labels_are_canonicalized_explicitly():
    assert canonical_retrieval_context("Space Training") == "Novel"
    assert canonical_retrieval_context("test spaceA") == "Novel"
    assert canonical_retrieval_context("No Context Training") == "Default"
    assert canonical_retrieval_context("No Context Test") == "Default"


def test_similarity_retrieval_and_large_beta_limit():
    result = similarity_responsibility(["A", "B", "Both", "Novel", "Default"], beta=2.0)
    assert result.q_A.iloc[0] > 0.5
    assert result.q_A.iloc[1] < 0.5
    np.testing.assert_allclose(result.q_A.iloc[2:], 0.5)
    sharp = similarity_responsibility(["A", "B"], beta=20.0)
    assert sharp.q_A.iloc[0] > 0.999
    assert sharp.q_A.iloc[1] < 0.001
