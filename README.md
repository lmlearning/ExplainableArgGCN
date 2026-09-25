# ExplainableArgGCN

Research code for **explainable graph neural networks in abstract argumentation**, with training, baseline evaluation, ablation studies and neighborhood visualizations.

## What is included

- Training implementations for refined argumentation GCNs and comparison models.
- Saved checkpoints, including ablation and ranking-loss sweeps.
- Intrinsic and baseline evaluation scripts.
- Results and graph-neighborhood visualizations.

## Start here

| Task | Entry point |
| --- | --- |
| Inspect dependencies | [requirements.txt](requirements.txt) and [setup_environment.sh](setup_environment.sh) |
| Understand training | [train_refined_afgcn.py](train_refined_afgcn.py) and [train_consolidated.py](train_consolidated.py) |
| Inspect evaluation | [evaluate_intrinsic.py](evaluate_intrinsic.py) and [evaluate_pyg_baselines.py](evaluate_pyg_baselines.py) |
| Review experiment orchestration | [run_main_and_baselines.sh](run_main_and_baselines.sh), [run_ablations.sh](run_ablations.sh), [run_rank_loss_sweep.sh](run_rank_loss_sweep.sh) |
| Explore outputs | [results](results/) and [visualizations](visualizations/) |
| Understand visualizations | [visualize_neighbourhoods.py](visualize_neighbourhoods.py) |

The shell scripts document the experiment workflows. Review their data paths, model paths and hardware settings before reproducing a run. Saved checkpoints and outputs should be interpreted together with the configuration that produced them.

## Related work

[AFGCN](https://github.com/lmlearning/AFGCN) · [AFGraphLib](https://github.com/lmlearning/AFGraphLib) · [FastAFGCN](https://github.com/lmlearning/FastAFGCN)

## License

See [LICENSE](LICENSE).
