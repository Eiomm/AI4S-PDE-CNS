from pathlib import Path

import h5py
import numpy as np

from mlevolve_lite.evaluator import Evaluator
from mlevolve_lite.graph_db import GraphDB
from mlevolve_lite.node_schema import Metrics, Node, Operator
from mlevolve_lite.selector import select_parent


def test_node_round_trip_preserves_metrics(tmp_path: Path) -> None:
    node = Node(
        node_id="n1",
        signature="sig",
        parent_ids=[],
        operator=Operator.DRAFT,
        hypothesis="test",
        code_dir=str(tmp_path / "code"),
        artifact_dir=str(tmp_path / "artifacts"),
        metrics=Metrics(overall_mse=0.25, shape_pass=True, first_10_pass=True, compliance_pass=True, reward=0.5),
    )

    restored = Node.from_dict(node.to_dict())

    assert restored.node_id == "n1"
    assert restored.metrics is not None
    assert restored.metrics.reward == 0.5


def test_graph_db_writes_graph_and_leaderboard(tmp_path: Path) -> None:
    db = GraphDB(tmp_path)
    node = Node(
        node_id="n1",
        signature="sig",
        parent_ids=[],
        operator=Operator.DRAFT,
        hypothesis="test",
        code_dir=str(tmp_path / "nodes/n1/code"),
        artifact_dir=str(tmp_path / "nodes/n1/artifacts"),
        metrics=Metrics(overall_mse=0.1, shape_pass=True, first_10_pass=True, compliance_pass=True, reward=1.0),
    )

    graph = db.load_graph()
    db.upsert_node(graph, node)
    db.save_graph(graph)
    db.update_leaderboard(graph)

    assert (tmp_path / "graph.json").exists()
    assert db.load_graph()["nodes"]["n1"]["metrics"]["reward"] == 1.0
    assert db.load_leaderboard()[0]["node_id"] == "n1"


def test_selector_prefers_unvisited_compliant_node(tmp_path: Path) -> None:
    good = Node(
        node_id="good",
        signature="good",
        parent_ids=[],
        operator=Operator.DRAFT,
        hypothesis="good",
        code_dir=str(tmp_path / "good/code"),
        artifact_dir=str(tmp_path / "good/artifacts"),
        status="preflight_passed",
    )
    bad = Node(
        node_id="bad",
        signature="bad",
        parent_ids=[],
        operator=Operator.DRAFT,
        hypothesis="bad",
        code_dir=str(tmp_path / "bad/code"),
        artifact_dir=str(tmp_path / "bad/artifacts"),
        status="failed",
    )

    assert select_parent([bad, good]).node_id == "good"


def test_shape_check_requires_first_ten_copy(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "artifacts"
    data_dir.mkdir()
    out_dir.mkdir()
    test_tensor = np.zeros((1000, 10, 256), dtype=np.float32)
    pred_tensor = np.zeros((1000, 200, 256), dtype=np.float32)
    pred_tensor[:, :10, :] = test_tensor

    with h5py.File(data_dir / "task2_test.h5", "w") as f:
        f.create_dataset("tensor", data=test_tensor)
    with h5py.File(out_dir / "task2_pred.hdf5", "w") as f:
        f.create_dataset("tensor", data=pred_tensor)

    metrics = Evaluator(data_dir).shape_check(out_dir / "task2_pred.hdf5")

    assert metrics.shape_pass
    assert metrics.first_10_pass
    assert metrics.compliance_pass
