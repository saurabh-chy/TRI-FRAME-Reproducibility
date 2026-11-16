`results/full` is the `output/full` folder of the reported run without the trained model files
(`models/`, about 100 MB, available on request). It contains every per-run result (results_main.csv,
results_ablation.csv, results_sensitivity.csv), tuning results, logs, training curves, the post-hoc
attribution and hypervolume points, the flip shares and certified bounds (bounds/), and the tables and figures produced by the pipeline. Running
`analysis/run_analysis` on it regenerates the statistics, the paper's tables and figures, and the supplement.
