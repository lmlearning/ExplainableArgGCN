import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("torch_scatter")

from train_refined_afgcn import AFGraphDataset, RefinedAFGCN as OriginalModel
from train_consolidated import RefinedAFGCN


@pytest.mark.parametrize("graph,solution,expected", [
    ("a\n#\n", "[[a]]", [1]),
    ("a\nb\n#\n", "[[a,b]]", [1, 1]),
    ("a\nb\n#\na b\nb a\n", "[[a],[b]]", [1, 1]),
    ("a\n#\na a\n", "[]", [0]),
])
def test_dataset_labels_edges_and_real_model_backward(tmp_path, graph, solution, expected):
    torch.set_num_threads(1)
    (tmp_path / "tiny.tgf").write_text(graph)
    (tmp_path / "tiny.EE-PR").write_text(solution)
    data = AFGraphDataset(str(tmp_path))[0]
    assert data.y.tolist() == expected
    assert data.edge_index.shape[0] == 2
    for model in (OriginalModel(hidden=8), RefinedAFGCN(hidden=8)):
        output = model(data)
        logits = output[0] if isinstance(output, tuple) else output
        assert logits.shape == (len(expected),)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, data.y.float())
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_undeclared_solution_argument_is_rejected(tmp_path):
    (tmp_path / "tiny.tgf").write_text("a\n#\n")
    (tmp_path / "tiny.EE-PR").write_text("[[missing]]")
    with pytest.raises(ValueError, match="undeclared solution arguments"):
        AFGraphDataset(str(tmp_path))[0]
