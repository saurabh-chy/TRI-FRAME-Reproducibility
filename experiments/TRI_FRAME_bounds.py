"""TRI-FRAME: per-group flip rates, the exact worst-case fairness gap, and certified bounds.

Theory (Proposition 1 in the paper). Every record is edited independently inside its own feasible set
F(x_i), so the attacker's problem separates over records. For a classifier with hard predictions, let

    up_g   = share of group g that is predicted 0 but CAN be pushed to 1 by some feasible edit,
    down_g = share of group g that is predicted 1 but CAN be pushed to 0 by some feasible edit,

and r_g the clean positive rate of group g. The largest demographic-parity gap any feasible attack can
reach is then exactly

    full population:   max( r_1 - r_0 + up_1 + down_0 ,  r_0 - r_1 + up_0 + down_1 )
    group t only:      the same expression with up/down of the other group set to 0.

This script estimates up_g and down_g from both sides for every main-run model and budget:
  * lower bounds by a per-record flip attack (PGD on +-logit, the evaluation settings: 50 steps,
    5 restarts, integer projection), which gives a feasible edit that really flips the record;
  * upper bounds by interval bound propagation (IBP) over the feasible box, which certifies that a
    record CANNOT flip.
Together they bracket the exact worst-case gap: attack-based value <= exact worst case <= certified bound.
It also records robust accuracy per group (correct and not flippable), used in Corollary 2:
    worst-case gap <= clean gap + (1 - RA_1) + (1 - RA_0).

Part "main": the selected model of every method, every run, budgets 0.25x-4x.
Part "grid": every tuned configuration of every method on run 0 at 1x (the hypervolume models),
            to test the bound across all 184 configurations (46 per dataset).

Reuses the notebook code unchanged (read from TRI_FRAME_full.py in the same folder). Writes only to
output/full/bounds/. Resumable (bounds/done_bounds.json).
Usage (same folder as TRI_FRAME_full.py):  python TRI_FRAME_bounds.py [--part main|grid|both]
"""
import argparse, hashlib, json, re, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--part", default="both", choices=["main", "grid", "both"])
ap.add_argument("--base", default=None, help="override BASE_DIR (normally leave unset)")
ap.add_argument("--datasets", default="adult,compas,south_german_credit,acs_income")
ap.add_argument("--budgets", default="0.25,0.5,1,2,4")
ARGS = ap.parse_args()

# ---------------------------------------------------------------- load the notebook's definitions
src = (HERE / "TRI_FRAME_full.py").read_text(encoding="utf-8")
cells = {int(m.group(1)): body for m, body in
         zip(re.finditer(r"# ======== notebook cell (\d+) ========\n", src),
             re.split(r"# ======== notebook cell \d+ ========\n", src)[1:])}
import matplotlib
matplotlib.use("Agg")
G = globals()
for i in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]:
    code = cells[i]
    if i == 4 and ARGS.base:
        code = code.replace("BASE_DIR = None", f"BASE_DIR = Path(r'{ARGS.base}')")
    if i == 8:
        code = code.replace('with open(OUT / "config.json", "w") as f:', "if False:")
        code = code.replace('log(f"MODE={MODE}', 'pass  # log(f"MODE={MODE}')
    exec(compile(code, f"TRI_FRAME_full.py cell {i}", "exec"), G)
assert MODE == "full", "TRI_FRAME_full.py must have MODE = 'full'"

BND = OUT / "bounds"
BND.mkdir(exist_ok=True)
LOG_FILE = BND / "bounds.log"
DONE_P = BND / "done_bounds.json"
DONE_SET = set(json.load(open(DONE_P))) if DONE_P.exists() else set()
DATASETS = [d.strip() for d in ARGS.datasets.split(",") if d.strip()]
BUDGETS = [float(b) for b in ARGS.budgets.split(",")]
log(f"BOUNDS start | data: {DATA_DIR} | output: {BND} | part={ARGS.part} | budgets={BUDGETS}")

DATA_CACHE = {}
def get_dataset(name):
    if name not in DATA_CACHE:
        DATA_CACHE[name] = load_dataset(name)
    return DATA_CACHE[name]


def dseed(*parts):
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def mark(key, rows, name):
    if rows:
        path = BND / f"{name}.csv"
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)
    DONE_SET.add(key)
    json.dump(sorted(DONE_SET), open(DONE_P, "w"))


def model_file(cfg, key):
    return OUT / "models" / f"{key}_{cfg_hash(cfg)}.pt"


@torch.no_grad()
def _pred(model, x):
    return (probs(model, x) >= 0.5).float()


def flip_rates(model, x, y, a, space, eps_mult, seed):
    """Per-group shares of records that can be flipped up / down (attack lower bound and IBP upper bound)."""
    yhat = _pred(model, x)
    # attack: push each record's logit away from its current prediction
    s = torch.where(yhat > 0.5, -torch.ones_like(yhat), torch.ones_like(yhat))
    torch.manual_seed(seed)
    xa = pgd_attack(model, x, obj_dir(s), space, eps_mult, S["eval_steps"], S["eval_restarts"],
                    round_int=ROUND_INTEGER_FEATURES)
    flipped = _pred(model, xa) != yhat
    # certificate: IBP output bounds over the feasible box (a superset of the integer-feasible set)
    with torch.no_grad():
        _, lower, upper = space.bounds(x, eps_mult)
        pl, pu = ibp_prob_bounds(model, lower, upper)
    can_flip_cert = torch.where(yhat > 0.5, pl < 0.5, pu >= 0.5)      # not certified stable
    correct = yhat == y
    out = {}
    for g in (0, 1):
        m = a == g
        n = float(m.sum())
        out.update({
            f"n_{g}": int(n),
            f"pos_{g}": float(yhat[m].mean()),
            f"up_atk_{g}": float((m & (yhat < 0.5) & flipped).sum()) / n,
            f"down_atk_{g}": float((m & (yhat > 0.5) & flipped).sum()) / n,
            f"up_cert_{g}": float((m & (yhat < 0.5) & can_flip_cert).sum()) / n,
            f"down_cert_{g}": float((m & (yhat > 0.5) & can_flip_cert).sum()) / n,
            f"ra_atk_{g}": float((m & correct & ~flipped).sum()) / n,          # robust accuracy (upper estimate)
            f"ra_cert_{g}": float((m & correct & ~can_flip_cert).sum()) / n,   # certified robust accuracy
            f"acc_{g}": float((m & correct).sum()) / n,
        })
    sound = bool((flipped & ~can_flip_cert).sum() == 0)                  # an attack can never beat a certificate
    out["cert_sound"] = sound
    return out


def part_main():
    for d in DATASETS:
        try:
            df = get_dataset(d)
        except (FileNotFoundError, ImportError) as e:
            log(f"main: skipping {d}: {e}")
            continue
        spec = DATASET_SPECS[d]
        sel = json.load(open(OUT / "tuning" / f"{d}_selected.json"))["selected"]
        runs = make_splits(df, spec["split"], S["n_seeds"], S["sgc_repeats"])
        for k, (run_id, tr, va, te) in enumerate(runs):
            todo = [(m, e) for m in METHODS for e in BUDGETS if f"main|{d}|{run_id}|{m}|{e}" not in DONE_SET]
            if not todo:
                continue
            data = SplitData(df, tr, va, te, spec, INCLUDE_SENSITIVE_AS_FEATURE)
            for method in METHODS:
                es = [e for (m, e) in todo if m == method]
                if not es:
                    continue
                key = f"{d}_{run_id}_{method}"
                if not model_file(sel[method], key).exists():
                    log(f"main: no saved model {key}, skipped")
                    for e in es:
                        mark(f"main|{d}|{run_id}|{method}|{e}", [], "bounds_main")
                    continue
                model, _ = get_model(sel[method], data, 1000 + k, key)
                for e in es:
                    t0 = time.time()
                    r = flip_rates(model, data.Xte, data.yte, data.ate, data.space, e, dseed("bmain", d, run_id, method, e))
                    mark(f"main|{d}|{run_id}|{method}|{e}", [dict(dataset=d, run=run_id, method=method, eps=e, **r)],
                         "bounds_main")
                    log(f"main {d} {run_id} {method:12s} m={e}: up/down atk g1 {r['up_atk_1']:.3f}/{r['down_atk_1']:.3f} "
                        f"g0 {r['up_atk_0']:.3f}/{r['down_atk_0']:.3f} sound={r['cert_sound']} ({time.time() - t0:.0f}s)")


def part_grid():
    for d in DATASETS:
        try:
            df = get_dataset(d)
        except (FileNotFoundError, ImportError) as e:
            log(f"grid: skipping {d}: {e}")
            continue
        spec = DATASET_SPECS[d]
        run_id, tr, va, te = make_splits(df, spec["split"], S["n_seeds"], S["sgc_repeats"])[0]
        data = None
        for method in METHODS:
            for ci, cfg in enumerate(make_grid(method, S["tune_budget"])):
                ukey = f"grid|{d}|{method}|{ci}"
                if ukey in DONE_SET:
                    continue
                ckey = f"tune_{d}_{method}_{ci}"
                if not model_file(cfg, ckey).exists():
                    log(f"grid: no saved model {ckey}, skipped")
                    mark(ukey, [], "bounds_grid")
                    continue
                if data is None:
                    data = SplitData(df, tr, va, te, spec, INCLUDE_SENSITIVE_AS_FEATURE)
                model, _ = get_model(cfg, data, 1000, ckey)
                r = flip_rates(model, data.Xte, data.yte, data.ate, data.space, 1.0, dseed("bgrid", d, method, ci))
                mark(ukey, [dict(dataset=d, run=run_id, method=method, cfg_id=ci, cfg=json.dumps(cfg), eps=1.0, **r)],
                     "bounds_grid")
                log(f"grid {d} {method:12s} cfg{ci}: up_atk g1 {r['up_atk_1']:.3f} down_atk g0 {r['down_atk_0']:.3f}")


if ARGS.part in ("main", "both"):
    part_main()
if ARGS.part in ("grid", "both"):
    part_grid()
log("==== BOUNDS FINISHED ====")
