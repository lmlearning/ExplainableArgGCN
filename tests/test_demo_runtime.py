import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("torch_scatter")

from examples.cpu_demo import run_demo
from train_consolidated import load_pyg_baselines


def test_real_cpu_demo():
    result = run_demo()
    assert result["accepted_targets"] == [1, 0, 1]
    assert result["logits_shape"] == result["ranking_shape"] == [3]
    assert result["finite_loss"] and result["finite_gradients"]


def test_baseline_loading_is_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    builders = load_pyg_baselines()
    assert set(builders) == {"afgcn", "gcn", "gat", "graphsage", "gin", "randalign"}
    assert all(isinstance(build(16, 2), torch.nn.Module) for build in builders.values())
