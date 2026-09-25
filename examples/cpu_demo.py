"""Exercise the dataset loader and refined model on an illustrative tiny graph."""
import json
from pathlib import Path
import tempfile

import torch

from train_refined_afgcn import AFGraphDataset
from train_consolidated import RefinedAFGCN


def run_demo():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "example.tgf").write_text("a\nb\nc\n#\na b\nb c\n", encoding="utf-8")
        (root / "example.EE-PR").write_text("[[a,c]]\n", encoding="utf-8")
        data = AFGraphDataset(directory)[0]
        model = RefinedAFGCN(hidden=8, layers=2)
        logits, ranking = model(data)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, data.y.float())
        loss.backward()
        return {"example": "untrained CPU model; not a benchmark score",
                "arguments": data.num_nodes, "accepted_targets": data.y.tolist(),
                "logits_shape": list(logits.shape), "ranking_shape": list(ranking.shape),
                "finite_loss": bool(torch.isfinite(loss)),
                "finite_gradients": all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)}


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))
