"""Analysis of the flip-rate results (output/full/bounds) for Proposition 1 and its corollaries.

Usage: python bounds_analysis.py <output/full folder> <out folder>
Writes: bounds_table.csv/.md (Table S28 of the supplement), bounds_by_budget.csv, bounds_grid_check.csv,
        figure7_bounds.png/.pdf/.eps and Fig7.tif (Figure 7 of the paper) and a short text summary (bounds_summary.txt).
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
DS = ["adult", "acs_income", "compas", "south_german_credit"]
DN = {"adult": "Adult", "acs_income": "ACS Income", "compas": "COMPAS", "south_german_credit": "South German Credit"}
MS = ["ERM", "FAIR_ONLY", "ROBUST_ONLY", "FAIR_ADV", "FAIR_ADV_CW", "TRIFRAME"]
MN = {"ERM": "ERM", "FAIR_ONLY": "Fair-Only", "ROBUST_ONLY": "Robust-Only", "FAIR_ADV": "Fair adv. (clean + adv.)",
      "FAIR_ADV_CW": "Chai & Wang style", "TRIFRAME": "TRI-FRAME"}
COL = {"ERM": "#6b6b6b", "FAIR_ONLY": "#eb6834", "ROBUST_ONLY": "#1baf7a", "FAIR_ADV": "#eda100",
       "FAIR_ADV_CW": "#e87ba4", "TRIFRAME": "#2a78d6"}
MK = {"ERM": "o", "FAIR_ONLY": "s", "ROBUST_ONLY": "^", "FAIR_ADV": "D", "FAIR_ADV_CW": "P", "TRIFRAME": "*"}
plt.rcParams.update({"font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})


def worst_cases(df, kind):
    """Exact worst-case DP gap (Proposition 1) from per-group up/down shares of the given kind (atk or cert)."""
    r1, r0 = df["pos_1"], df["pos_0"]
    u1, d1, u0, d0 = (df[f"{k}_{kind}_{g}"] for k, g in [("up", 1), ("down", 1), ("up", 0), ("down", 0)])
    full = np.maximum(r1 - r0 + u1 + d0, r0 - r1 + u0 + d1)
    t1 = np.maximum(r1 - r0 + u1, r0 - r1 + d1)          # only group 1 editable
    t0 = np.maximum(r1 - r0 + d0, r0 - r1 + u0)          # only group 0 editable
    return full, np.maximum(t1, t0)


def add_bounds(df):
    df = df.copy()
    df["clean_dp"] = (df.pos_1 - df.pos_0).abs()
    for kind in ("atk", "cert"):
        df[f"full_{kind}"], df[f"targeted_{kind}"] = worst_cases(df, kind)
    df["instab_atk"] = df.up_atk_1 + df.down_atk_1 + df.up_atk_0 + df.down_atk_0            # unweighted sum of shares
    df["cor2_bound"] = df.clean_dp + (1 - df.ra_atk_1) + (1 - df.ra_atk_0)        # Corollary 2 (attack-based estimate)
    # direction that widens the gap: up for the favoured group, down for the other
    fav1 = df.pos_1 >= df.pos_0
    df["relevant_atk"] = np.where(fav1, df.up_atk_1 + df.down_atk_0, df.up_atk_0 + df.down_atk_1)
    df["irrelevant_atk"] = np.where(fav1, df.down_atk_1 + df.up_atk_0, df.down_atk_0 + df.up_atk_1)
    return df


def measured_targeted(main):
    """Group-targeted DP of the paper's attack (results_main): worse of t=0 / t=1 per run."""
    a = main[main.attack.isin(["fair_dp_t0", "fair_dp_t1"])]
    p = a.pivot_table(index=["dataset", "run", "method", "eps"], columns="attack", values="dp")
    return p.max(axis=1).rename("paper_targeted").reset_index()


bm = add_bounds(pd.read_csv(f"{R}/bounds/bounds_main.csv"))
main = pd.read_csv(f"{R}/results_main.csv")
bm = bm.merge(measured_targeted(main), on=["dataset", "run", "method", "eps"], how="left")
bm["paper_full"] = bm.merge(main[main.attack == "fair_dp"][["dataset", "run", "method", "eps", "dp"]],
                            on=["dataset", "run", "method", "eps"], how="left")["dp"].values
bm.to_csv(f"{OUT}/bounds_per_run.csv", index=False)

# sanity: certificates are sound and the attack-based worst case is at least the paper's attack
checks = dict(cert_sound_all=bool(bm.cert_sound.all()),
              atk_ge_paper_share=float((bm.targeted_atk >= bm.paper_targeted - 1e-9).mean()),
              cert_ge_atk_share=float((bm.targeted_cert >= bm.targeted_atk - 1e-9).mean()))

# best attack found = max(paper's group-targeted attack, per-record flip attack); flip term = best - clean gap
bm["best_targeted"] = np.maximum(bm.targeted_atk, bm.paper_targeted)
bm["flip_term"] = bm.best_targeted - bm.clean_dp
checks["share_within_0.01"] = float(((bm.targeted_atk - bm.paper_targeted).abs() <= 0.01).mean())
checks["n_model_budget"] = int(len(bm))

# Table S28 (supplement): base budget, means over runs
t = bm[bm.eps == 1.0].groupby(["dataset", "method"])[
    ["clean_dp", "flip_term", "paper_targeted", "best_targeted", "targeted_cert"]].mean().reset_index()
t["dataset"] = pd.Categorical(t.dataset, DS); t["method"] = pd.Categorical(t.method, MS)
t = t.sort_values(["dataset", "method"])
t.to_csv(f"{OUT}/bounds_table.csv", index=False)
lines = ["| Dataset | Method | Clean gap | Flip term | Paper's attack | Best attack found | Certified upper bound |",
         "|---|---|---|---|---|---|---|"]
for _, r in t.iterrows():
    lines.append(f"| {DN[r.dataset]} | {MN[r.method]} | {r.clean_dp:.3f} | {r.flip_term:.3f} | {r.paper_targeted:.3f} | "
                 f"{r.best_targeted:.3f} | {r.targeted_cert:.3f} |")
open(f"{OUT}/bounds_table.md", "w").write("\n".join(lines))
bb = bm.groupby(["dataset", "eps"])[["best_targeted", "targeted_cert"]].mean().reset_index()
bb.to_csv(f"{OUT}/bounds_by_budget.csv", index=False)

# grid check across all tuned configurations (run 0, base budget)
gpath = f"{R}/bounds/bounds_grid.csv"
if os.path.exists(gpath):
    bg = add_bounds(pd.read_csv(gpath))
    hv = pd.read_csv(f"{R}/posthoc/hv_points.csv")
    bg = bg.merge(hv[["dataset", "run", "method", "cfg_id", "test_dp_targeted"]], on=["dataset", "run", "method", "cfg_id"], how="left")
    bg.to_csv(f"{OUT}/bounds_grid_check.csv", index=False)
    checks["grid_n"] = int(len(bg))
    checks["grid_corr_flip_vs_paper"] = float(np.corrcoef(bg.targeted_atk, bg.test_dp_targeted)[0, 1])

# Figure 7: (a) per-record flip attack vs the paper's attack, all models and budgets;
#           (b) attacked gap = clean gap + flip term at the base budget
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1, 1.55]})
ax = axs[0]
for m in MS:
    s = bm[bm.method == m]
    ax.scatter(s.paper_targeted, s.targeted_atk, s=8, alpha=0.6, color=COL[m], marker=MK[m], label=MN[m], linewidths=0)
lim = float(max(bm.paper_targeted.max(), bm.targeted_atk.max())) * 1.03
ax.plot([0, lim], [0, lim], color="#444", lw=0.8, ls="--")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("DP gap, group-targeted attack"); ax.set_ylabel("DP gap, per-record flip attack")
ax.set_title("(a)", loc="left", fontsize=9)
ax = axs[1]
w = 0.13; xs = np.arange(len(DS))
for j, m in enumerate(MS):
    s = t[t.method == m].set_index("dataset").reindex(DS)
    x = xs + (j - 2.5) * w
    ax.bar(x, s.clean_dp, w * 0.92, color=COL[m], edgecolor="white", linewidth=0.3)
    ax.bar(x, s.flip_term, w * 0.92, bottom=s.clean_dp, color="white", edgecolor=COL[m], hatch="////", linewidth=0.6)
ax.set_xticks(xs); ax.set_xticklabels([DN[d].replace("South German ", "South German\n").replace("ACS ", "ACS\n") for d in DS])
ax.set_ylabel("DP gap under attack (base budget)"); ax.set_title("(b)", loc="left", fontsize=9)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#888", label="clean gap"), Patch(facecolor="white", edgecolor="#888", hatch="////", label="flip term")],
          frameon=False, fontsize=7.5, loc="upper left")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=6, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.07), markerscale=1.8,
           handletextpad=0.2, columnspacing=0.8)
fig.subplots_adjust(wspace=0.28)
for ext in ("png", "pdf", "eps"):
    fig.savefig(f"{OUT}/figure7_bounds.{ext}", bbox_inches="tight", dpi=600)
fig.savefig(f"{OUT}/Fig7.tif", bbox_inches="tight", dpi=600, facecolor="white")
plt.close(fig)
try:
    from PIL import Image
    Image.open(f"{OUT}/Fig7.tif").convert("RGB").save(f"{OUT}/Fig7.tif", dpi=(600, 600), compression="tiff_lzw")
except Exception:
    pass

with open(f"{OUT}/bounds_summary.txt", "w") as f:
    for k, v in checks.items():
        f.write(f"{k}: {v}\n")
print(open(f"{OUT}/bounds_summary.txt").read())
print(open(f"{OUT}/bounds_table.md").read())
