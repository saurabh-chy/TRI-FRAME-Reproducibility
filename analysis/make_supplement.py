"""Build the supplementary material (markdown) from the full-run result files.
Usage: python make_supplement.py <output/full folder> <supplement.md> <stats folder from statistical_tests.py>
Convert to Word with: pandoc supplement.md -o supplement.docx
"""
import json, sys
import numpy as np, pandas as pd

R = sys.argv[1]; OUT = sys.argv[2]; ST = sys.argv[3]
DS = ["adult", "acs_income", "compas", "south_german_credit"]
DN = {"adult": "Adult", "acs_income": "ACS Income", "compas": "COMPAS", "south_german_credit": "South German Credit"}
MS = ["ERM", "FAIR_ONLY", "ROBUST_ONLY", "FAIR_ADV", "FAIR_ADV_CW", "TRIFRAME"]
MN = {"ERM": "ERM", "FAIR_ONLY": "Fair-Only", "ROBUST_ONLY": "Robust-Only", "FAIR_ADV": "Fair adv. (clean + adv.)",
      "FAIR_ADV_CW": "Chai & Wang style", "TRIFRAME": "TRI-FRAME", "GBDT": "GBDT"}
LONG2SHORT = {"Fair adv. training (clean + adv. loss)": "Fair adv. (clean + adv.)",
              "Fair adv. training (Chai & Wang 2023)": "Chai & Wang style"}
FEAT = {"SCHL": "educational attainment (SCHL)", "WKHP": "hours worked per week (WKHP)",
        "juv_fel_count": "juvenile felonies", "juv_misd_count": "juvenile misdemeanours",
        "juv_other_count": "juvenile other offences", "priors_count": "prior offences",
        "amount": "credit amount", "duration": "loan duration", "ALL mutable features": "all mutable features"}

m = pd.read_csv(f"{R}/results_main.csv")
out = []
P = out.append
f3 = lambda v: "–" if pd.isna(v) else f"{v:.3f}"
ms = lambda a: f"{np.mean(a):.3f} ± {np.std(a, ddof=1):.3f}" if len(a) > 1 else f"{np.mean(a):.3f}"


def table(header, rows):
    P("| " + " | ".join(header) + " |")
    P("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        P("| " + " | ".join(str(x) for x in r) + " |")
    P("")


def get(ds, meth, attack, eps, col):
    g = m[(m.dataset == ds) & (m.method == meth) & (m.attack == attack) & (np.isclose(m.eps, eps))]
    return g.sort_values("run")[col].values


def targeted(ds, meth, eps, col="dp"):
    a = m[(m.dataset == ds) & (m.method == meth) & (np.isclose(m.eps, eps)) & m.attack.isin(["fair_dp_t0", "fair_dp_t1"])]
    p = a.pivot_table(index="run", columns="attack", values=col)
    return p.max(axis=1).values


P("# Supplementary Material")
P("")
P("**Fairness Under Feasible Edits: Realistic Record Manipulation Breaks Group-Fairness Audits of Tabular Classifiers, and How to Train Against It**")
P("")
P("Saurabh Choudhary · Sachin Malhotra · Midhunchakkaravarthy Janarthanan")
P("")
P("This supplement gives the full specification of the feasible sets, the tuning grids and every configuration evaluated, complete results under all attacks and budgets, the per-feature attribution for every method, the paired tests including Wilcoxon signed-rank tests, the per-run hypervolumes, the interval-bound certificates and gradient-masking checks, per-group accuracy, training cost, and the full sensitivity and ablation results. Tables and sections are numbered S1, S2, … and refer to the main paper where relevant. Unless stated otherwise, values are means over runs (10 hold-out runs; 25 cross-validation folds for South German Credit), and ± gives the standard deviation over runs.")
P("")

# ---------------- S1 implementation details
P("## S1 Implementation details")
P("")
cfg = json.load(open(f"{R}/config.json"))["settings"]
P("**Preprocessing.** Numeric features are standardised with the mean and standard deviation of the training split of each run; categorical features are one-hot encoded with categories taken from the training split. The sensitive attribute is removed from the model inputs (except in the ablation that adds it). Integer-valued features are mapped back to raw units after every attack, rounded and clipped to the valid range of Table S1, and re-standardised.")
P("")
P(f"**Model and optimisation.** All neural models are multilayer perceptrons with hidden layers of 128 and 64 ReLU units and a single logit output, trained with Adam (learning rate 10⁻³), batch size 256, stratified on the pair (label, group), for at most {cfg['max_epochs']} epochs with early stopping (patience {cfg['patience']}) on the full validation objective, including both training attacks evaluated with a fixed seed. The parameters of the best epoch are kept.")
P("")
P(f"**Training attacks.** TRI-FRAME and the adversarial baselines use projected sign-gradient attacks with {cfg['k_train']} steps, one random start inside the feasible set and a step size of 2.5/{cfg['k_train']} of the per-feature budget. The fairness training attack uses per-record directions computed from the model's current clean probabilities, which are treated as constants within a step.")
P("")
P(f"**Evaluation attacks.** Evaluation uses {cfg['eval_steps']} steps with a step size of 2.5/{cfg['eval_steps']} of the budget and {cfg['eval_restarts']} random restarts; for each record the restart with the highest objective is kept, followed by integer projection. Budgets are multiples m ∈ {{0.25, 0.5, 1, 2, 4}} of the base budgets in Table S1. The group-targeted attack is run once for each target group, and the worse (larger) DP gap of the two is reported for each run. The combined attack maximises cross-entropy/N + β times the group-weighted probability push, with β ∈ {{0.5, 1, 2}}. Transfer attacks are crafted on a separately trained ERM network on the same split and applied to the target model.")
P("")
P(f"**Hyperparameter selection.** Each method except ERM is tuned over nine configurations (Table S3) on the first run only, using {cfg['tune_steps']}-step attacks on the validation split. The rule, fixed before the experiments, keeps the configurations whose validation accuracy is at least ERM's minus 0.01 and selects the one with the lowest validation DP gap under the group-targeted attack; if none qualifies, it selects the most accurate configuration. The selected configuration is then retrained on every run.")
P("")
P("**Hypervolume.** For every run (10 hold-out runs; 25 folds for South German Credit), every tuned configuration of every method is evaluated on the test split by its clean accuracy A and its DP gap G under the group-targeted attack at the base budget. With ERM's point (A_E, G_E) in the same run, the reference box is A ≥ A_E − 0.05 and G ≤ G_E. The hypervolume of a method's non-dominated points inside the box (accuracy to be maximised, gap to be minimised) is computed exactly by sweeping the points in order of accuracy and is divided by the box area 0.05 · G_E. Configurations more accurate than ERM extend the box upwards, which allows values above 1.")
P("")
P("**Software and hardware.** Python 3.8.8, PyTorch 2.4.1 (CPU build), NumPy 1.24.4, pandas 1.5.3, scikit-learn 1.2.2 (HistGradientBoostingClassifier with default settings for the GBDT reference), SciPy 1.10.1 and matplotlib 3.3.4, on a multi-core CPU without a GPU. Random seeds are fixed per run, and the code, configuration files and per-run results are released with the paper.")
P("")

# ---------------- S2 feasible set
P("## S2 Feasible sets")
P("")
fs = pd.read_csv(f"{R}/tables/table_s1_feasible_set.csv")
P("*Table S1. Every input feature with its type, role, base budget in raw units and in training standard deviations, and valid range in the training data. Categorical features and the sensitive attribute cannot be edited.*")
P("")
rows = []
for ds in DS:
    sub = fs[fs.Dataset == DN[ds]]
    for _, r in sub.iterrows():
        b = r["Budget (raw units, 1x)"]; s = r["Budget (SD units, 1x)"]
        rows.append([r.Dataset, r.Feature, r.Type, r.Role, "–" if pd.isna(b) else f"±{b:g}",
                     "–" if pd.isna(s) else f"{float(s):.2f}", "–" if pd.isna(r["Valid range (train)"]) else r["Valid range (train)"]])
table(["Dataset", "Feature", "Type", "Role", "Budget (raw)", "Budget (SD)", "Valid range"], rows)

# ---------------- S3 tuning
P("## S3 Tuning grids and all evaluated configurations")
P("")
P("Tables S2–S5 list every configuration evaluated during tuning, with validation accuracy, clean validation DP gap and validation DP gap under the group-targeted attack (20-step attack). The selected configuration is marked ✓; configurations chosen by the fallback rule are marked ✓\\*. Budget multipliers apply to the training attacks.")
P("")
for k, ds in enumerate(DS):
    t = pd.read_csv(f"{R}/tuning/{ds}_tuning.csv")
    sel = json.load(open(f"{R}/tuning/{ds}_selected.json"))
    erm_acc = sel["erm_val_acc"]
    P(f"*Table S{k + 2}. Tuning results on {DN[ds]} (ERM validation accuracy {erm_acc:.3f}; threshold {erm_acc - 0.01:.3f}).*")
    P("")
    rows = []
    for meth in MS:
        g = t[t.method == meth]
        for _, r in g.iterrows():
            c = json.loads(r.cfg)
            parts = []
            for key, lab in [("lam_f", "λ_f"), ("lam_r", "λ_r"), ("lam_fa", "λ_fa"), ("train_eps", "budget")]:
                if key in c:
                    parts.append(f"{lab} = {c[key]:.2g}" + ("×" if key == "train_eps" else ""))
            mark = ""
            if c == sel["selected"].get(meth, None) or (meth == "ERM"):
                mark = "✓" if (meth == "ERM" or r.val_acc >= erm_acc - 0.01) else "✓*"
            rows.append([MN[meth], ", ".join(parts) or "–", f"{r.val_acc:.3f}", f"{r.val_dp_clean:.3f}",
                         f"{r.val_dp_targeted:.3f}", mark])
    table(["Method", "Configuration", "Val. accuracy", "Val. DP (clean)", "Val. DP (group-targeted)", "Selected"], rows)

# ---------------- S4 all attacks at base budget
P("## S4 Complete results under every attack at the base budget")
P("")
P("Tables S6–S9 report, for every method, accuracy and fairness under each attack at the base budget (m = 1). Clean EO is the equalised-odds gap on clean data. DP full and DP targeted are the DP gaps under the full-population and group-targeted attacks; EO attack is the EO gap under the EO-targeted attack; Comb. β gives the DP gap under the combined attack; Transfer gives accuracy and DP gap under transfer attacks crafted on a separate ERM model.")
P("")
for k, ds in enumerate(DS):
    P(f"*Table S{k + 6}. {DN[ds]}: all attacks at the base budget (mean ± SD over runs).*")
    P("")
    rows = []
    for meth in MS + ["GBDT"]:
        if meth == "GBDT":
            rows.append([MN[meth], ms(get(ds, meth, "clean", 0, "acc")), ms(get(ds, meth, "clean", 0, "dp")),
                         ms(get(ds, meth, "clean", 0, "eo")), "–", "–", "–", "–", "–", "–",
                         ms(get(ds, meth, "transfer_pgd_acc", 1, "acc")), ms(get(ds, meth, "transfer_fair_dp", 1, "dp"))])
            continue
        rows.append([MN[meth], ms(get(ds, meth, "clean", 0, "acc")), ms(get(ds, meth, "clean", 0, "dp")),
                     ms(get(ds, meth, "clean", 0, "eo")), ms(get(ds, meth, "pgd_acc", 1, "acc")),
                     ms(get(ds, meth, "fgsm_acc", 1, "acc")), ms(get(ds, meth, "fair_dp", 1, "dp")),
                     ms(targeted(ds, meth, 1)), ms(get(ds, meth, "fair_eo", 1, "eo")),
                     " / ".join(f3(np.mean(get(ds, meth, f"combined_b{b}", 1, "dp"))) for b in ["0.5", "1.0", "2.0"]),
                     ms(get(ds, meth, "transfer_pgd_acc", 1, "acc")), ms(get(ds, meth, "transfer_fair_dp", 1, "dp"))])
    table(["Method", "Clean acc.", "Clean DP", "Clean EO", "PGD acc.", "FGSM acc.", "DP full", "DP targeted",
           "EO attack", "Comb. β = 0.5 / 1 / 2", "Transfer acc.", "Transfer DP"], rows)

# ---------------- S5 budget sweep
P("## S5 Results across budgets")
P("")
P("Tables S10–S13 give the data behind Figure 3 of the paper: the DP gap under the group-targeted attack and accuracy under the PGD accuracy attack at each budget multiplier (m = 0 is clean data).")
P("")
EPS = [0.25, 0.5, 1.0, 2.0, 4.0]
for k, ds in enumerate(DS):
    P(f"*Table S{k + 10}. {DN[ds]}: DP gap under the group-targeted attack / accuracy under PGD, by budget multiplier (means over runs).*")
    P("")
    rows = []
    for meth in MS:
        r = [MN[meth], f"{np.mean(get(ds, meth, 'clean', 0, 'dp')):.3f} / {np.mean(get(ds, meth, 'clean', 0, 'acc')):.3f}"]
        for e in EPS:
            r.append(f"{np.mean(targeted(ds, meth, e)):.3f} / {np.mean(get(ds, meth, 'pgd_acc', e, 'acc')):.3f}")
        rows.append(r)
    table(["Method", "m = 0", "m = 0.25", "m = 0.5", "m = 1", "m = 2", "m = 4"], rows)

P("*Table S14. Transfer attacks against the GBDT reference by budget multiplier: DP gap / accuracy (means over runs).*")
P("")
rows = []
for ds in DS:
    r = [DN[ds], f"{np.mean(get(ds, 'GBDT', 'clean', 0, 'dp')):.3f} / {np.mean(get(ds, 'GBDT', 'clean', 0, 'acc')):.3f}"]
    for e in EPS:
        r.append(f"{np.mean(get(ds, 'GBDT', 'transfer_fair_dp', e, 'dp')):.3f} / {np.mean(get(ds, 'GBDT', 'transfer_pgd_acc', e, 'acc')):.3f}")
    rows.append(r)
table(["Dataset", "Clean", "m = 0.25", "m = 0.5", "m = 1", "m = 2", "m = 4"], rows)

# ---------------- S6 attribution
P("## S6 Per-feature attribution for every method")
P("")
P("*Table S15. DP gap under the group-targeted attack at the base budget when only the named feature can be edited (mean over runs). The row \"all mutable features\" repeats the full attack; the first row gives the clean gap.*")
P("")
at = pd.read_csv(f"{R}/posthoc/attribution_summary.csv")
for ds in DS:
    a = at[at.dataset == ds]
    feats = ["ALL mutable features"] + sorted(f for f in a.feature.unique() if f != "ALL mutable features")
    rows = [["clean gap"] + [f3(a[a.method == mm].clean_dp.iloc[0]) for mm in MS]]
    for fe in feats:
        rows.append([FEAT.get(fe, fe)] + [f3(a[(a.method == mm) & (a.feature == fe)].attacked_dp.mean()) for mm in MS])
    P(f"**{DN[ds]}**")
    P("")
    table(["Editable feature"] + [MN[x] for x in MS], rows)

# ---------------- S7 paired tests
P("## S7 Paired tests including Wilcoxon signed-rank tests")
P("")
P("*Tables S16–S19. TRI-FRAME minus each baseline, paired by run, for the headline metrics at the base budget: mean difference with 95% confidence interval, Cohen's d_z, the standard paired t-test, the corrected resampled t-test of Nadeau and Bengio (2003) with a test-to-training ratio of 1/3 (which allows for training sets shared between runs), the Wilcoxon signed-rank test, and Holm-adjusted p-values for both t-tests (adjusted over all 45 tests of each dataset: nine metrics against five baselines, of which the headline metrics are shown). For South German Credit, whose 25 folds come from five repetitions of five-fold cross-validation, the standard t-test is not valid and is not shown; the confidence interval uses the corrected variance, and the Wilcoxon test is applied to the five repetition means (smallest attainable p = 0.0625). For accuracy, positive differences favour TRI-FRAME; for gaps, negative differences favour TRI-FRAME.*")
P("")
pt = pd.read_csv(f"{ST}/paired_tests.csv")
MET = [("clean", "acc", "Clean accuracy"), ("clean", "dp", "Clean DP"), ("pgd_acc", "acc", "Accuracy under PGD"),
       ("fair_dp_targeted", "dp", "DP, group-targeted"), ("fair_eo", "eo", "EO, EO-targeted")]
for k, ds in enumerate(DS):
    P(f"*Table S{k + 16}. {DN[ds]}.*")
    P("")
    rows = []
    cv = ds == "south_german_credit"
    sub = pt[pt.dataset == ds]
    for att, met, lab in MET:
        for _, r in sub[(sub.attack == att) & (sub.metric == met)].iterrows():
            lo, hi = (r.ci_corrected_lo, r.ci_corrected_hi) if cv else (r.ci_lo, r.ci_hi)
            rows.append([lab, MN[r.baseline], f"{r['diff']:+.3f} [{lo:+.3f}, {hi:+.3f}]", f"{r.dz:.2f}",
                         "–" if cv else f"{r.p_t:.3g}", f"{r.p_t_corrected:.3g}", f"{r.p_wilcoxon:.3g}",
                         "–" if cv else f"{r.p_holm:.3g}", f"{r.p_holm_corrected:.3g}"])
    table(["Metric", "Baseline", "Difference [95% CI]", "d_z", "p (t)", "p (corrected t)",
           "p (Wilcoxon)" + (", repeats" if cv else ""), "Holm (t)", "Holm (corrected t)"], rows)

# ---------------- S8 hypervolume
P("## S8 Hypervolume per run")
P("")
hv = pd.read_csv(f"{R}/posthoc/hv_per_run.csv")
P("*Table S20. Normalised hypervolume of each method in each of the ten hold-out runs (all nine tuned configurations per method; ERM is 0 by construction).*")
P("")
rows = []
for ds in DS[:3]:
    for meth in MS[1:]:
        v = hv[(hv.dataset == ds) & (hv.method == meth)].sort_values("run").hv.values
        rows.append([DN[ds], MN[meth]] + [f"{x:.2f}" for x in v] + [ms(v)])
table(["Dataset", "Method"] + [f"Run {i}" for i in range(1, 11)] + ["Mean ± SD"], rows)
P("*Table S21. Normalised hypervolume on South German Credit: mean over the five folds of each cross-validation repetition, and mean ± SD over all 25 folds.*")
P("")
rows = []
g = hv[hv.dataset == "south_german_credit"].copy(); g["rep"] = g.run.str[:2]
for meth in MS[1:]:
    gm = g[g.method == meth]
    rows.append([MN[meth]] + [f"{gm[gm.rep == r].hv.mean():.3f}" for r in sorted(gm.rep.unique())] + [ms(gm.hv.values)])
table(["Method"] + [f"Repetition {i}" for i in range(1, 6)] + ["All 25 folds"], rows)
ht = pd.read_csv(f"{ST}/hypervolume_tests.csv")
P("*Table S22. Hypervolume of TRI-FRAME minus each baseline, paired by run: mean difference, Cohen's d_z, standard and corrected resampled t-tests, each Holm-adjusted over the four baselines of a dataset. For South German Credit only the corrected test is valid.*")
P("")
rows = []
for ds in DS:
    cv = ds == "south_german_credit"
    for _, r in ht[ht.dataset == ds].iterrows():
        rows.append([DN[ds], MN[r.baseline], str(int(r.n)), f"{r['diff']:+.3f}", f"{r.dz:.2f}",
                     "–" if cv else f"{r.p_t:.3g}", "–" if cv else f"{r.p_holm:.3g}",
                     f"{r.p_t_corrected:.3g}", f"{r.p_holm_corrected:.3g}"])
table(["Dataset", "Baseline", "Runs", "Difference", "d_z", "p (t)", "Holm (t)", "p (corrected t)", "Holm (corrected t)"], rows)

# ---------------- S9 certificates + sanity checks
P("## S9 Gradient-masking checks and interval-bound certificates")
P("")
P("For every dataset and method we checked that accuracy under PGD decreases as the budget grows, that the 50-step PGD attack is at least as strong as a one-step attack, and that white-box attacks are at least as strong as transfer attacks in both accuracy and DP gap. We also computed interval-bound-propagation (IBP) upper bounds on the DP gap over the full-population feasible set and checked that every empirical attack stays below its bound. All checks passed. The bounds are loose: they lie well above the strongest empirical attack and are vacuous on COMPAS, so the paper makes no certified claim.")
P("")
sc = pd.read_csv(f"{R}/tables/table_sanity_checks.csv")
sc["Method"] = sc.Method.replace(LONG2SHORT)
ORD = {"Adult": 0, "ACS Income": 1, "COMPAS": 2, "South German Credit": 3}
sc = sc.sort_values("Dataset", key=lambda c: c.map(ORD), kind="stable")
P("*Table S23. Gradient-masking checks and IBP bounds at the base budget. Empirical DP is the DP gap under the full-population attack; Certified DP is the IBP upper bound.*")
P("")
rows = []
for _, r in sc.iterrows():
    rows.append([r.Dataset, r.Method, r["Acc falls with eps"], r["PGD >= FGSM"], r["White-box >= transfer (acc)"],
                 r["White-box >= transfer (DP)"], r["Empirical DP <= certified"], f"{r['Empirical DP (attack)']:.3f}",
                 f"{r['Certified DP']:.3f}"])
table(["Dataset", "Method", "Acc. falls with m", "PGD ≥ FGSM", "White-box ≥ transfer (acc.)", "White-box ≥ transfer (DP)",
       "Empirical ≤ bound", "Empirical DP", "Certified DP"], rows)

# ---------------- S10 per-group accuracy
P("## S10 Per-group accuracy")
P("")
pg = pd.read_csv(f"{R}/tables/table_s6_per_group_accuracy.csv")
pg["Method"] = pg.Method.replace(LONG2SHORT)
pg = pg.sort_values("Dataset", key=lambda c: c.map(ORD), kind="stable")
P("*Table S24. Accuracy in each group on clean data, under the PGD accuracy attack and under the group-targeted fairness attack at the base budget (mean ± SD over runs). Group a = 1 is the favoured group of each dataset.*")
P("")
table(["Dataset", "Method", "Condition", "Accuracy a = 0", "Accuracy a = 1", "Gap (a = 1 minus a = 0)"],
      [[r.Dataset, r.Method, r.Condition, r["Accuracy a=0"], r["Accuracy a=1"], f"{r['Gap (a=1 minus a=0)']:+.3f}"] for _, r in pg.iterrows()])

# ---------------- S11 cost
P("## S11 Training cost and early stopping")
P("")
cost = pd.read_csv(f"{R}/tables/table_cost.csv"); es = pd.read_csv(f"{R}/tables/table_s5_early_stopping.csv")
cost["Method"] = cost.Method.replace(LONG2SHORT); es["Method"] = es.Method.replace(LONG2SHORT)
mm_ = cost.merge(es, on=["Dataset", "Method"], how="left").sort_values("Dataset", key=lambda c: c.map(ORD), kind="stable")
P("*Table S25. Seconds per epoch on a multi-core CPU, best epoch and number of epochs run before early stopping (mean ± SD over runs).*")
P("")
def r2(x):
    try:
        a, b = x.split(" ± "); return f"{float(a):.2f} ± {float(b):.2f}"
    except Exception:
        return x
table(["Dataset", "Method", "Seconds per epoch", "Best epoch", "Epochs run"],
      [[r.Dataset, r.Method, r2(r["Seconds per epoch"]), r["Best epoch (mean ± SD)"], r["Epochs run (mean)"]] for _, r in mm_.iterrows()])

# ---------------- S12 sensitivity
P("## S12 Sensitivity to the fairness-adversarial weight")
P("")
sv = pd.read_csv(f"{R}/tables/table_sensitivity_lambda_fa.csv")
P("*Table S26. TRI-FRAME on Adult with λ_f and λ_r fixed at their selected values (0.1 and 1.0) and λ_fa varied (three runs, mean ± SD).*")
P("")
SEL = [("clean", "acc", "Clean acc."), ("clean", "dp", "Clean DP"), ("fair_dp_targeted", "dp", "DP, group-targeted"),
       ("fair_dp", "dp", "DP, full population"), ("pgd_acc", "acc", "Accuracy under PGD")]
rows = []
for lf in sorted(sv.lam_fa.unique()):
    s = sv[sv.lam_fa == lf]
    r = [f"{lf:g}"]
    for a, col, _ in SEL:
        x = s[s.attack == a].iloc[0]
        r.append(f"{x[col + '_mean']:.3f} ± {x[col + '_std']:.3f}")
    rows.append(r)
table(["λ_fa"] + [h for _, _, h in SEL], rows)

# ---------------- S13 ablations
P("## S13 Ablations")
P("")
ab = pd.read_csv(f"{R}/tables/table_ablations.csv")
P("*Table S27. Ablations on Adult (five runs, mean ± SD), as in Table 8 of the paper. The covariance row replaces the soft DP penalty by the decision-boundary covariance of Zafar et al. (2017); the unconstrained row trains against perturbations of every numeric feature, including immutable ones, without range or integer constraints; evaluation always uses the feasible set.*")
P("")
table(list(ab.columns), ab.values.tolist())

open(OUT, "w").write("\n".join(out))
