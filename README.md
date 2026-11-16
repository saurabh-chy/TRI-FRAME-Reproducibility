# TRI-FRAME: reproducibility package

Code for **"Fairness Under Feasible Edits: Realistic Record Manipulation Breaks Group-Fairness Audits of Tabular
Classifiers, and How to Train Against It"** (Choudhary, Malhotra and Janarthanan).

The package reproduces every number, table and figure in the paper and the supplementary material: the feasible
threat model and attacks, the six training methods (ERM, Fair-Only, Robust-Only, fair adversarial training,
the Chai & Wang-style baseline and TRI-FRAME), tuning, evaluation, per-feature attribution, hypervolume and the
statistical tests, and the exact worst-case analysis of Proposition 1 (per-record flip attacks and certified
interval bounds).

## Folder layout

```
TRI-FRAME_reproducibility/
├── download/            <- you create this and put the datasets in it (see data/README.md)
├── output/full/         <- created by the pipeline: all results, tables, logs, models
├── experiments/
│   ├── TRI_FRAME_full.py            full pipeline (the notebook exported with MODE = "full")
│   ├── TRI_FRAME_posthoc.py         per-feature attribution and hypervolume over all tuned configurations
│   ├── run_full.bat / run_full.sh   run the pipeline with automatic restart
│   ├── run_posthoc.bat / .sh        run attribution + hypervolume (all runs) with automatic restart
│   ├── TRI_FRAME_bounds.py          flip shares, exact worst case (Proposition 1) and certified bounds
│   ├── run_bounds.bat / .sh         run the bounds analysis with automatic restart
│   └── run_hypervolume_all_runs.bat hypervolume only (all runs)
├── analysis/
│   ├── statistical_tests.py         paired and corrected resampled t-tests, Holm correction, Wilcoxon
│   ├── make_figures_tables.py       Figures 1–6 and Tables 5–7 of the paper
│   ├── bounds_analysis.py           Figure 7 and the numbers of Section 6.5
│   ├── make_supplement.py           supplementary material (Tables S1–S29)
│   └── run_analysis.bat / .sh       runs the scripts above
├── notebook/TRI_FRAME_v2_pipeline.ipynb   the same pipeline as a notebook (MODE "smoke" / "pilot" / "full")
├── notebook/TRI_FRAME_check_files.ipynb   checks that the dataset files (and, later, saved models) are in place
├── notebook/TRI_FRAME_bounds.ipynb        the bounds analysis of step 5 as a notebook
├── results/             per-run results of the reported experiments (CSV), so the analysis can be rerun without training
├── data/README.md       dataset sources and file names
├── requirements.txt
├── LICENSE, CITATION.cff
```

## Environment

The reported run used Python 3.8.8, PyTorch 2.4.1 (CPU), NumPy 1.24.4, pandas 1.5.3, scikit-learn 1.2.2,
SciPy 1.10.1 and matplotlib 3.3.4 on Windows (Anaconda) with a
multi-core CPU and no GPU.

```
conda create -n triframe python=3.8.8
conda activate triframe
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Reproducing the results

1. **Data.** Create `download/` in this folder and add the four datasets as described in `data/README.md`.
2. **Quick check (a few minutes).** Open `notebook/TRI_FRAME_v2_pipeline.ipynb`, keep `MODE = "smoke"` and
   run all cells. Results go to `output/smoke/`.
3. **Full run (about two days on a multi-core laptop CPU).** In `experiments/`, run `run_full.bat` (Windows) or
   `bash run_full.sh`. The pipeline tunes every method (nine configurations each), trains 10 hold-out runs per
   dataset (25 cross-validation folds for South German Credit), runs all attacks at all budgets, the ablations,
   the λ_fa sensitivity study, the interval bounds and the GBDT reference. It is resumable: finished units are
   recorded in `output/full/done.json` and skipped on restart.
4. **Post-hoc analyses (roughly half a day to a day).** In `experiments/`, run `run_posthoc.bat` or `bash run_posthoc.sh`.
   This computes the per-feature attribution and evaluates every tuned configuration of every method on the
   test set of every run, for the hypervolume. Also resumable (`output/full/posthoc/done_posthoc.json`).
5. **Exact worst case and certified bounds (several hours).** In `experiments/`, run `run_bounds.bat` or
   `bash run_bounds.sh` (or `notebook/TRI_FRAME_bounds.ipynb`). It reloads the saved models, computes for every
   selected model, run and budget, and for every tuned configuration on the first run, the share of each group
   that a feasible edit can flip (per-record attack: lower bound; interval bound propagation: upper bound), and
   writes them to `output/full/bounds/`. Resumable (`output/full/bounds/done_bounds.json`).
6. **Statistics, tables, figures and supplement (a few minutes).** In `analysis/`, run
   `run_analysis.bat ..\output\full` or `bash run_analysis.sh ../output/full`. Outputs go to `reproduced/`.

To rerun only step 6 on the reported results, point it at `results/full` instead of `output/full`.

## Determinism

All splits and training runs use fixed seeds (`set_seed` in the pipeline; seeds derived from dataset, run,
method and configuration in the post-hoc script). PyTorch on CPU is deterministic for this model up to
floating-point summation order, so reruns on other hardware or thread counts may differ in the third decimal.

## Settings used in the paper

All settings are in Section 1 of the notebook and are written to `output/full/config.json` by every run. The
main ones: MLP 128–64 with ReLU; Adam, learning rate 10⁻³, batch size 256, at most 50 epochs, patience 5;
training attacks 5 PGD steps; evaluation attacks 50 steps, 5 restarts; budget multipliers 0.25–4; nine tuning
configurations per method, selection within one point of ERM's validation accuracy; 10 hold-out runs
(60/20/20) and 5 × 5-fold cross-validation for South German Credit; ablations on Adult with 5 runs; λ_fa
sensitivity with 3 runs.

## Notes

- The ACS cross-state experiment in the notebook (`run_acs_shift`) requires additional ACS state files and was
  not part of the reported results.
- Tree ensembles are evaluated on clean data and under transfer attacks only.

## Licence and citation

Code under the MIT licence (`LICENSE`). Please cite the paper (`CITATION.cff`). The datasets keep their own
licences and are not redistributed.
