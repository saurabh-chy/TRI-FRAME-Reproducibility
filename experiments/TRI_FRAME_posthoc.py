"""TRI-FRAME post-hoc analyses on the finished full run (output/full).

Part 1, per-feature attribution (no training): for every main-run model, rerun the worst-case
  group-targeted DP attack at the base budget (1x) with only ONE mutable feature editable at a time,
  and with all of them (check against the main table). Shows which edit drives the fairness gap.
Part 2, hypervolume (methodology Section 3.10, the primary comparison): for the first 5 runs, every
  configuration in every method's tuning grid is evaluated on the TEST set: clean accuracy and the
  worst-case group-targeted DP gap at 1x (50 steps). Run 0 reuses the tuning models; runs 1-4 train the
  remaining grid models (models already trained for the main run are reused). The normalised
  hypervolume per method and run uses ERM of the same run as the reference point, and TRI-FRAME is
  compared with each method by paired t-tests with Holm correction.

It reuses the notebook's code unchanged (read from TRI_FRAME_full.py in the same folder) and the same
settings, folders and seeds. Nothing in output/full is modified; results go to output/full/posthoc/.
Resumable: finished units are recorded in posthoc/done_posthoc.json and skipped on restart.

Usage (same folder as TRI_FRAME_full.py):  python TRI_FRAME_posthoc.py [--part attribution|hv|both]
"""
import argparse, copy, hashlib, json, re, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--part", default="both", choices=["attribution", "hv", "both"])
ap.add_argument("--hv-runs", type=int, default=5)
ap.add_argument("--base", default=None, help="override BASE_DIR (normally leave unset)")
ap.add_argument("--datasets", default="adult,compas,south_german_credit,acs_income",
                help="comma-separated; default runs all four, ACS Income last")
ARGS = ap.parse_args()

# ---------------------------------------------------------------- load the notebook's definitions
src = (HERE / "TRI_FRAME_full.py").read_text(encoding="utf-8")
cells = {int(m.group(1)): body for m, body in
         zip(re.finditer(r"# ======== notebook cell (\d+) ========\n", src),
             re.split(r"# ======== notebook cell \d+ ========\n", src)[1:])}
DEF_CELLS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]      # imports .. tuning helpers; no runners
import matplotlib
matplotlib.use("Agg")
G = globals()
for i in DEF_CELLS:
    code = cells[i]
    if i == 4 and ARGS.base:
        code = code.replace("BASE_DIR = None", f"BASE_DIR = Path(r'{ARGS.base}')")
    if i == 8:   # do not rewrite output/full/config.json or start a new run log
        code = code.replace('with open(OUT / "config.json", "w") as f:', "if False:")
        code = code.replace('log(f"MODE={MODE}', 'pass  # log(f"MODE={MODE}')
    exec(compile(code, f"TRI_FRAME_full.py cell {i}", "exec"), G)

assert MODE == "full", "TRI_FRAME_full.py must have MODE = 'full'"
POST = OUT / "posthoc"
POST.mkdir(exist_ok=True)
LOG_FILE = POST / "posthoc.log"          # log() now writes here
DONE_P = POST / "done_posthoc.json"
DONE_SET = set(json.load(open(DONE_P))) if DONE_P.exists() else set()
DATASETS = [d.strip() for d in ARGS.datasets.split(",") if d.strip()]   # ACS last by default: by far the slowest
log(f"POSTHOC start | data: {DATA_DIR} | output: {POST} | part={ARGS.part}")


DATA_CACHE = {}
def get_dataset(name):          # same as the notebook's runner cell (not loaded, since it also runs the experiment)
    if name not in DATA_CACHE:
        DATA_CACHE[name] = load_dataset(name)
    return DATA_CACHE[name]


def mark(key, rows, name):
    if rows:
        path = POST / f"{name}.csv"
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)
    DONE_SET.add(key)
    json.dump(sorted(DONE_SET), open(DONE_P, "w"))


def dseed(*parts):
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def model_path(cfg, key):
    return OUT / "models" / f"{key}_{cfg_hash(cfg)}.pt"


def worst_targeted(model, x, y, a, space, steps, restarts, seed):
    """Worst of t=0 / t=1 group-targeted DP attacks at 1x; returns (dp, acc_at_worst, t)."""
    p = probs(model, x)
    best = (-1.0, None, None)
    for t in (0, 1):
        torch.manual_seed(seed + t)
        xa = pgd_attack(model, x, obj_dir(dp_directions(p, a, t)), space, 1.0, steps, restarts,
                        row_scale=(a == t).float(), round_int=ROUND_INTEGER_FEATURES)
        m = hard_metrics(probs(model, xa), y, a)
        if m["dp"] > best[0]:
            best = (m["dp"], m["acc"], t)
    return best


def load_selected(d):
    return json.load(open(OUT / "tuning" / f"{d}_selected.json"))["selected"]


# ---------------------------------------------------------------- part 1: per-feature attribution
def attribution():
    for d in DATASETS:
        try:
            df = get_dataset(d)
        except (FileNotFoundError, ImportError) as e:   # e.g. ACS file or folktables missing
            log(f"attribution: skipping {d}: {e}")
            continue
        spec, sel = DATASET_SPECS[d], load_selected(d)
        runs = make_splits(df, spec["split"], S["n_seeds"], S["sgc_repeats"])
        for k, (run_id, tr, va, te) in enumerate(runs):
            todo = [m for m in METHODS if f"attr|{d}|{run_id}|{m}" not in DONE_SET]
            if not todo:
                continue
            data = SplitData(df, tr, va, te, spec, INCLUDE_SENSITIVE_AS_FEATURE)
            base_eps = data.space.eps.clone()
            mutable = [(j, data.pre.columns[j]) for j in range(len(base_eps)) if float(base_eps[j]) > 0]
            for method in todo:
                key = f"{d}_{run_id}_{method}"
                if not model_path(sel[method], key).exists():
                    log(f"attribution: no saved model {key}, skipped")
                    mark(f"attr|{d}|{run_id}|{method}", [], "attribution")
                    continue
                model, _ = get_model(sel[method], data, 1000 + k, key)
                clean = hard_metrics(probs(model, data.Xte), data.yte, data.ate)
                rows = []
                for feat, cols in [("ALL mutable features", [j for j, _ in mutable])] + [(n, [j]) for j, n in mutable]:
                    sp = copy.copy(data.space)
                    mask = torch.zeros_like(base_eps)
                    mask[cols] = 1.0
                    sp.eps = base_eps * mask
                    dp, acc, t = worst_targeted(model, data.Xte, data.yte, data.ate, sp, S["eval_steps"],
                                                S["eval_restarts"], dseed("attr", d, run_id, method, feat))
                    rows.append(dict(dataset=d, run=run_id, method=method, feature=feat, clean_dp=clean["dp"],
                                     clean_acc=clean["acc"], attacked_dp=dp, attacked_acc=acc, worst_group=t))
                mark(f"attr|{d}|{run_id}|{method}", rows, "attribution")
                log(f"attribution {d} {run_id} {method:12s} " +
                    " ".join(f"{r['feature'][:14]}={r['attacked_dp']:.3f}" for r in rows))


# ---------------------------------------------------------------- part 2: hypervolume
def hv_points():
    for d in DATASETS:
        try:
            df = get_dataset(d)
        except (FileNotFoundError, ImportError) as e:
            log(f"hypervolume: skipping {d}: {e}")
            continue
        spec, sel = DATASET_SPECS[d], load_selected(d)
        runs = make_splits(df, spec["split"], S["n_seeds"], S["sgc_repeats"])[:ARGS.hv_runs]
        for k, (run_id, tr, va, te) in enumerate(runs):
            data = None
            for method in METHODS:
                for ci, cfg in enumerate(make_grid(method, S["tune_budget"])):
                    ukey = f"hv|{d}|{run_id}|{method}|{ci}"
                    if ukey in DONE_SET:
                        continue
                    if data is None:
                        data = SplitData(df, tr, va, te, spec, INCLUDE_SENSITIVE_AS_FEATURE)
                    if k == 0:
                        ckey, seed = f"tune_{d}_{method}_{ci}", 1000            # tuning models (run 0)
                    elif cfg_hash(cfg) == cfg_hash(sel[method]):
                        ckey, seed = f"{d}_{run_id}_{method}", 1000 + k          # main-run model
                    else:
                        ckey, seed = f"hv_{d}_{run_id}_{method}_{ci}", 1000 + k  # new
                    t0 = time.time()
                    model, info = get_model(cfg, data, seed, ckey, meta=dict(dataset=d, run=run_id, method=method))
                    acc = hard_metrics(probs(model, data.Xte), data.yte, data.ate)["acc"]
                    dp, _, _ = worst_targeted(model, data.Xte, data.yte, data.ate, data.space, S["eval_steps"], 1,
                                              dseed("hv", d, run_id, method, ci))
                    mark(ukey, [dict(dataset=d, run=run_id, method=method, cfg_id=ci, cfg=json.dumps(cfg),
                                     test_acc=acc, test_dp_targeted=dp)], "hv_points")
                    log(f"hv {d} {run_id} {method:12s} cfg{ci}: acc={acc:.4f} dp_targeted={dp:.4f} "
                        f"({time.time() - t0:.0f}s)")


def staircase_hv(points, ref_acc, ref_dp, window=0.05):
    """Normalised hypervolume (methodology 3.10): 0 = no gain over ERM, 1 = zero gap at ERM accuracy."""
    lo = ref_acc - window
    pts = sorted([(a, g) for a, g in points if a > lo and g < ref_dp], key=lambda p: -p[0])
    area, g_best = 0.0, ref_dp
    for a, g in pts:
        if g < g_best:
            area += (a - lo) * (g_best - g)
            g_best = g
    return area / (window * ref_dp) if ref_dp > 0 else float("nan")


def hv_analysis():
    path = POST / "hv_points.csv"
    if not path.exists():
        return
    P = pd.read_csv(path)
    rows = []
    for (d, run), g in P.groupby(["dataset", "run"]):
        erm = g[g.method == "ERM"]
        if erm.empty:
            continue
        ra, rg = float(erm.test_acc.iloc[0]), float(erm.test_dp_targeted.iloc[0])
        for m, gm in g.groupby("method"):
            n_expected = len(make_grid(m, S["tune_budget"]))
            rows.append(dict(dataset=d, run=run, method=m, n_configs=len(gm), complete=len(gm) == n_expected,
                             hv=staircase_hv(zip(gm.test_acc, gm.test_dp_targeted), ra, rg),
                             erm_acc=ra, erm_dp_targeted=rg))
    H = pd.DataFrame(rows)
    H.to_csv(POST / "hv_per_run.csv", index=False)
    lines = ["# Normalised hypervolume (clean accuracy vs DP under group-targeted attack, test set)", "",
             "Reference box per run: accuracy >= ERM - 0.05, DP <= ERM's attacked DP. 0 = no gain over ERM; 1 = zero gap at",
             "ERM's accuracy. Values above 1 occur when a configuration is MORE accurate than ERM on that run.", ""]
    tests = []
    for d, g in H[H.complete].groupby("dataset"):
        piv = g.pivot(index="run", columns="method", values="hv").dropna()
        lines += [f"## {DATASET_SPECS[d]['display']} ({len(piv)} complete runs)", "",
                  "| Method | Mean HV | SD |", "| --- | --- | --- |"]
        for m in [x for x in METHODS if x in piv]:
            lines.append(f"| {METHOD_LABELS[m]} | {piv[m].mean():.4f} | {piv[m].std(ddof=1):.4f} |")
        block = []
        if "TRIFRAME" in piv and len(piv) >= 2:
            for m in [x for x in METHODS if x in piv and x not in ("TRIFRAME", "ERM")]:
                diff = piv["TRIFRAME"] - piv[m]
                sd = diff.std(ddof=1)
                p = stats.ttest_rel(piv["TRIFRAME"], piv[m]).pvalue if sd > 0 else 1.0
                block.append(dict(dataset=d, baseline=m, n=len(diff), mean_diff_TRI_minus_base=diff.mean(),
                                  dz=diff.mean() / sd if sd > 0 else float("nan"), p_ttest=p))
            ps = [b["p_ttest"] for b in block]
            order = np.argsort(ps); run_max = 0.0
            for rank, idx in enumerate(order):
                run_max = max(run_max, min(1.0, (len(ps) - rank) * ps[idx]))
                block[idx]["p_holm"] = run_max
            lines += ["", "| TRI-FRAME minus | Mean diff | d_z | p (Holm) |", "| --- | --- | --- | --- |"]
            lines += [f"| {METHOD_LABELS[b['baseline']]} | {b['mean_diff_TRI_minus_base']:+.4f} | {b['dz']:.2f} | "
                      f"{b['p_holm']:.4f} |" for b in block]
        tests += block
        lines.append("")
    pd.DataFrame(tests).to_csv(POST / "hv_tests.csv", index=False)
    (POST / "hv_summary.md").write_text("\n".join(lines), encoding="utf-8")
    log("hypervolume summary written: " + str(POST / "hv_summary.md"))


def attribution_summary():
    path = POST / "attribution.csv"
    if not path.exists():
        return
    A = pd.read_csv(path)
    s = A.groupby(["dataset", "method", "feature"]).agg(clean_dp=("clean_dp", "mean"),
                                                       attacked_dp=("attacked_dp", "mean"),
                                                       sd=("attacked_dp", "std"),
                                                       attacked_acc=("attacked_acc", "mean"),
                                                       runs=("run", "nunique")).reset_index()
    s.to_csv(POST / "attribution_summary.csv", index=False)
    lines = ["# Per-feature attribution: DP gap under the worst-case group-targeted attack at 1x,",
             "# with only the named feature editable (mean over runs)", ""]
    for d, g in s.groupby("dataset"):
        feats = ["ALL mutable features"] + sorted(f for f in g.feature.unique() if f != "ALL mutable features")
        lines += [f"## {DATASET_SPECS[d]['display']}", "",
                  "| Method | Clean DP | " + " | ".join(feats) + " |", "| --- | --- | " + " | ".join("---" for _ in feats) + " |"]
        for m in [x for x in METHODS if x in set(g.method)]:
            gm = g[g.method == m].set_index("feature")
            lines.append(f"| {METHOD_LABELS[m]} | {gm.clean_dp.iloc[0]:.3f} | " +
                         " | ".join(f"{gm.attacked_dp.get(f, float('nan')):.3f}" for f in feats) + " |")
        lines.append("")
    (POST / "attribution_summary.md").write_text("\n".join(lines), encoding="utf-8")
    log("attribution summary written: " + str(POST / "attribution_summary.md"))


if ARGS.part in ("attribution", "both"):
    attribution()
    attribution_summary()
if ARGS.part in ("hv", "both"):
    hv_points()
    hv_analysis()
log("POSTHOC FINISHED")
sys.exit(0)
