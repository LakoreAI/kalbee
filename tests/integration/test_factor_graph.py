import numpy as np

from kalbee.modules.integration.factor_graph import FactorGraphExporter


def test_factor_graph_exporter(tmp_path):
    """FactorGraphExporter should output graph nodes/factors as JSON."""
    exporter = FactorGraphExporter()
    exporter.add_state_node(0, np.zeros((2, 1)), np.eye(2))
    exporter.add_motion_factor(0, 1, np.eye(2), np.eye(2) * 0.01)
    exporter.add_measurement_factor(
        0, np.array([[1.0]]), np.array([[1, 0]]), np.array([[0.1]])
    )

    graph_dict = exporter.to_dict()
    assert len(graph_dict["nodes"]) == 1
    assert len(graph_dict["factors"]) == 2

    filepath = tmp_path / "graph.json"
    exporter.save_json(str(filepath))
    assert filepath.exists()
