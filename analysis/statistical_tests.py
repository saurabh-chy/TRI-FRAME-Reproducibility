"""Paired tests of TRI-FRAME against every baseline, as reported in the paper and supplement.

For each dataset and each of nine headline metrics at the base budget, TRI-FRAME is compared with each
baseline, paired by run. Two t-tests are reported:
  - the standard paired t-test;
  - the corrected resampled t-test of Nadeau and Bengio (2003), which inflates the variance of the
    paired differences by (1/k + n_test/n_train) to account for training sets shared between runs.
    n_test/n_train = 1/3 for every dataset (60/20/20 hold-out splits; for South German Credit, 5-fold
    cross-validation with 25% of each training fold held out for validation: 200 test vs 600 training).
Holm's correction is applied over the 45 tests (9 metrics x 5 baselines) of each dataset. For South German
Credit the corrected test is the one reported in the paper, and the Wilcoxon signed-rank test is applied to
the means of the five cross-validation repetitions. The same two tests are applied to the normalised
hypervolume (TRI-FRAME minus each baseline, over every run available in posthoc/hv_per_run.csv).

Usage: python statistical_tests.py <output/full folder> <out folder>
Writes paired_tests.csv and hypervolume_tests.csv to <out folder>.
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats

R, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
RATIO = 1 / 3
DS = ["adult", "acs_income", "compas", "south_german_credit"]
BASE = ["ERM", "FAIR_ONLY", "ROBUST_ONLY", "FAIR_ADV", "FAIR_ADV_CW"]
MET = [("clean", 0, "acc"), ("clean", 0, "dp"), ("clean", 0, "eo"), ("pgd_acc", 1, "acc"), ("fair_dp", 1, "dp"),
       ("fair_dp_targeted", 1, "dp"), ("fair_eo", 1, "eo"), ("combined_b1.0", 1, "dp"), ("combined_b1.0", 1, "acc")]
m = pd.read_csv(f"{R}/results_main.csv")


def series(d, meth, att, eps, col):
    g = m[(m.dataset == d) & (m.method == meth)]
    if att == "fair_dp_targeted":       # worse (larger) gap of the two target groups, per run
        a = g[np.isclose(g.eps, eps) & g.attack.isin(["fair_dp_t0", "fair_dp_t1"])]
        return a.pivot_table(index="run", columns="attack", values=col).max(axis=1)
    return g[(g.attack == att) & np.isclose(g.eps, eps)].set_index("run")[col]


def corrected_t(d, ratio=RATIO):
    d = np.asarray(d, float); k = len(d); v = d.var(ddof=1)
    if v == 0:
        return np.nan, 1.0, (d.mean(), d.mean())
    se = np.sqrt((1 / k + ratio) * v)
    t = d.mean() / se
    h = stats.t.ppf(0.975, k - 1) * se
    return t, 2 * stats.t.sf(abs(t), k - 1), (d.mean() - h, d.mean() + h)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); n = len(p); adj = np.empty(n); run = 0.0
    for i, k in enumerate(o):
        run = max(run, min(1.0, (n - i) * p[k])); adj[k] = run
    return adj


def wilcoxon(x):
    x = np.asarray(x, float)
    if len(x) < 2 or np.all(x == 0):
        return 1.0
    try:
        return stats.wilcoxon(x).pvalue
    except ValueError:
        return np.nan


rows = []
for d in DS:
    block = []
    for att, eps, col in MET:
        tri = series(d, "TRIFRAME", att, eps, col)
        for b in BASE:
            base = series(d, b, att, eps, col)
            diff = (tri - base).dropna()
            _, p_c, ci = corrected_t(diff)
            p_std = stats.ttest_rel(tri.loc[diff.index], base.loc[diff.index]).pvalue
            h_std = stats.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
            w = wilcoxon(diff.groupby(diff.index.str[:2]).mean() if d == "south_german_credit" else diff)
            block.append(dict(dataset=d, attack=att, eps=eps, metric=col, baseline=b, n=len(diff),
                              diff=diff.mean(), dz=diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan,
                              p_t=p_std, ci_lo=diff.mean() - h_std, ci_hi=diff.mean() + h_std, p_t_corrected=p_c, ci_corrected_lo=ci[0], ci_corrected_hi=ci[1],
                              p_wilcoxon=w))
    B = pd.DataFrame(block)
    B["p_holm"] = holm(B.p_t.values)
    B["p_holm_corrected"] = holm(B.p_t_corrected.values)
    rows.append(B)
T = pd.concat(rows, ignore_index=True)
T.to_csv(f"{OUT}/paired_tests.csv", index=False)

hv_path = f"{R}/posthoc/hv_per_run.csv"
if os.path.exists(hv_path):
    h = pd.read_csv(hv_path)
    out = []
    for d in DS:
        piv = h[h.dataset == d].pivot(index="run", columns="method", values="hv").dropna()
        block = []
        for b in ["FAIR_ONLY", "ROBUST_ONLY", "FAIR_ADV", "FAIR_ADV_CW"]:
            diff = piv["TRIFRAME"] - piv[b]
            _, p_c, ci = corrected_t(diff)
            block.append(dict(dataset=d, baseline=b, n=len(diff), diff=diff.mean(),
                              dz=diff.mean() / diff.std(ddof=1), p_t=stats.ttest_rel(piv["TRIFRAME"], piv[b]).pvalue,
                              p_t_corrected=p_c, ci_corrected_lo=ci[0], ci_corrected_hi=ci[1]))
        B = pd.DataFrame(block)
        B["p_holm"] = holm(B.p_t.values); B["p_holm_corrected"] = holm(B.p_t_corrected.values)
        out.append(B)
    pd.concat(out, ignore_index=True).to_csv(f"{OUT}/hypervolume_tests.csv", index=False)
print("written:", OUT)
