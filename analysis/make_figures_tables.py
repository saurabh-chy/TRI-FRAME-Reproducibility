"""Build manuscript tables (markdown) and figures (PNG + PDF) from the TRI-FRAME full-run outputs.
Usage: python make_figures_tables.py <output/full folder> <out folder> [<stats folder from statistical_tests.py>]
"""
import sys, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import FancyBboxPatch
from scipy import stats

F, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
DS = ["adult", "acs_income", "compas", "south_german_credit"]
DSN = {"adult": "Adult", "acs_income": "ACS Income", "compas": "COMPAS", "south_german_credit": "South German Credit"}
M = ["ERM", "FAIR_ONLY", "ROBUST_ONLY", "FAIR_ADV", "FAIR_ADV_CW", "TRIFRAME"]
MN = {"ERM": "ERM", "FAIR_ONLY": "Fair-Only", "ROBUST_ONLY": "Robust-Only", "FAIR_ADV": "Fair adv. (clean + adv.)",
      "FAIR_ADV_CW": "Chai & Wang style", "TRIFRAME": "TRI-FRAME"}
COL = {"ERM": "#6b6b6b", "FAIR_ONLY": "#eb6834", "ROBUST_ONLY": "#1baf7a", "FAIR_ADV": "#eda100",
       "FAIR_ADV_CW": "#e87ba4", "TRIFRAME": "#2a78d6"}
MK = {"ERM": "o", "FAIR_ONLY": "s", "ROBUST_ONLY": "^", "FAIR_ADV": "D", "FAIR_ADV_CW": "P", "TRIFRAME": "*"}
LS = {"ERM": "--", "FAIR_ONLY": "-", "ROBUST_ONLY": "-", "FAIR_ADV": "-", "FAIR_ADV_CW": "-", "TRIFRAME": "-"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#888888", "axes.labelcolor": "#222222", "xtick.color": "#444444",
                     "ytick.color": "#444444", "legend.frameon": False, "savefig.dpi": 300})

R = pd.read_csv(f"{F}/results_main.csv")
FALLBACK = {("acs_income", "FAIR_ADV_CW"), ("acs_income", "TRIFRAME"), ("compas", "FAIR_ADV_CW")}
md = {}


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def worst_targeted(df):
    """Per run: worse of fair_dp_t0 / fair_dp_t1 (DP and the accuracy of that attack)."""
    t = df[df.attack.isin(["fair_dp_t0", "fair_dp_t1"])]
    idx = t.groupby(["dataset", "run", "method", "eps"]).dp.idxmax()
    return t.loc[idx]


W = worst_targeted(R)
clean = R[R.attack == "clean"]
ci = lambda x: stats.t.ppf(0.975, len(x) - 1) * np.std(x, ddof=1) / np.sqrt(len(x))

# ------------------------------------------------------------- Figure 1: ERM clean vs attacked
fig, ax = plt.subplots(figsize=(6.6, 2.9))
x = np.arange(len(DS)); w = 0.36
cl = [clean[(clean.dataset == d) & (clean.method == "ERM")].dp for d in DS]
at = [W[(W.dataset == d) & (W.method == "ERM") & (W.eps == 1.0)].dp for d in DS]
ac_c = [clean[(clean.dataset == d) & (clean.method == "ERM")].acc.mean() for d in DS]
ac_a = [W[(W.dataset == d) & (W.method == "ERM") & (W.eps == 1.0)].acc.mean() for d in DS]
b1 = ax.bar(x - w / 2 - 0.01, [v.mean() for v in cl], w, yerr=[ci(v) for v in cl], color="#9ec5f4", edgecolor="white",
            linewidth=1.5, capsize=2, error_kw=dict(elinewidth=0.8, ecolor="#444"), label="Clean test data")
b2 = ax.bar(x + w / 2 + 0.01, [v.mean() for v in at], w, yerr=[ci(v) for v in at], color="#1c5cab", edgecolor="white",
            linewidth=1.5, capsize=2, error_kw=dict(elinewidth=0.8, ecolor="#444"), hatch="///",
            label="Group-targeted feasible edits (1×)")
for i in range(len(DS)):
    ax.text(x[i] - w / 2, cl[i].mean() + ci(cl[i]) + 0.012, f"{cl[i].mean():.3f}", ha="center", fontsize=7.5, color="#333")
    ax.text(x[i] + w / 2, at[i].mean() + ci(at[i]) + 0.012, f"{at[i].mean():.3f}", ha="center", fontsize=7.5, color="#333")
ax.set_xticks(x); ax.set_xticklabels([f"{DSN[d]}\nacc. {ac_c[i]:.3f} → {ac_a[i]:.3f}" for i, d in enumerate(DS)])
ax.set_ylabel("Demographic-parity gap"); ax.set_ylim(0, 0.72); ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", fontsize=8)
save(fig, "figure1_erm_clean_vs_attacked")
md["fig1"] = {DSN[d]: (round(cl[i].mean(), 3), round(at[i].mean(), 3), round(ac_c[i], 3), round(ac_a[i], 3)) for i, d in enumerate(DS)}

# ------------------------------------------------------------- Figure 2: threat model diagram
fig, ax = plt.subplots(figsize=(7.0, 2.65)); ax.set_xlim(0, 10); ax.set_ylim(0, 4.4); ax.axis("off")
def box(x0, y0, wd, ht, txt, fc, ec="#555", fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x0, y0), wd, ht, boxstyle="round,pad=0.04,rounding_size=0.12", fc=fc, ec=ec, lw=1))
    ax.text(x0 + wd / 2, y0 + ht / 2, txt, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal")
def arrow(x0, y0, x1, y1, txt="", c="#444"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.1))
    if txt:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.13, txt, ha="center", fontsize=7, color="#333")
ax.text(0.05, 4.2, "Record of one applicant (Adult)", fontsize=8, weight="bold")
rows = [("sex", "Male", "immutable", "#eeeeee"), ("age", "38", "immutable", "#eeeeee"),
        ("occupation", "Sales", "frozen", "#eeeeee"), ("hours-per-week", "40 → 44", "± 4 h", "#cde2fb"),
        ("capital-gain", "0 → 2,000", "± $2,000", "#cde2fb"), ("capital-loss", "0", "± $200", "#cde2fb"),
        ("education-num", "9 → 10", "± 1 level", "#cde2fb")]
for i, (f, v, r, c) in enumerate(rows):
    y = 3.7 - i * 0.48
    box(0.05, y, 1.45, 0.38, f, c, fs=7); box(1.55, y, 0.95, 0.38, v, c, fs=7); box(2.55, y, 1.2, 0.38, r, c, fs=6.5)
box(4.2, 2.5, 1.85, 1.05, "Attacker edits the\nrecords of one\ngroup only (e.g.\ndata entry)", "#f9ece7", fs=7.2)
box(4.2, 0.95, 1.85, 1.1, "Feasible set F(x):\nbudget box, valid\nranges, integers", "#e5f4e4", fs=7.2)
arrow(3.8, 2.4, 4.18, 2.95)                      # record -> attacker
arrow(3.8, 1.6, 4.18, 1.5)                       # budgets -> feasible set
arrow(5.12, 2.05, 5.12, 2.53)                    # feasible set limits the attacker
ax.text(5.2, 2.29, "limits edits", ha="left", va="center", fontsize=6.5, color="#333")
box(7.0, 2.65, 1.0, 0.75, "Model\nf(x)", "#ffffff", fs=8, bold=True)
arrow(6.1, 3.02, 6.98, 3.02)
ax.text(6.54, 3.2, "edited x′", ha="center", va="center", fontsize=6.5, color="#333")
box(8.45, 3.05, 1.5, 0.95, "Audit on clean\nrecords:\nDP 0.18  ✓", "#e5f4e4", fs=7)
box(8.45, 1.55, 1.5, 0.95, "Audit on edited\nrecords:\nDP 0.37  ✗", "#fbebeb", fs=7)
arrow(8.05, 3.2, 8.42, 3.5); arrow(8.05, 2.85, 8.42, 2.05)
ax.set_ylim(0.75, 4.4)
save(fig, "figure2_threat_model")

# ------------------------------------------------------------- Figure 3: DP vs budget
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.5), sharey=False)
for ax, d in zip(axs, DS):
    for m in M:
        g = W[(W.dataset == d) & (W.method == m)]
        cg = clean[(clean.dataset == d) & (clean.method == m)]
        xs, mu, lo, hi = [0.0], [cg.dp.mean()], [cg.dp.mean() - ci(cg.dp)], [cg.dp.mean() + ci(cg.dp)]
        for e in sorted(g.eps.unique()):
            v = g[g.eps == e].dp
            xs.append(e); mu.append(v.mean()); lo.append(v.mean() - ci(v)); hi.append(v.mean() + ci(v))
        ax.plot(range(len(xs)), mu, LS[m], color=COL[m], marker=MK[m], ms=4 if m != "TRIFRAME" else 6, lw=1.4,
                label=MN[m], zorder=3 if m == "TRIFRAME" else 2)
        ax.fill_between(range(len(xs)), lo, hi, color=COL[m], alpha=0.12, lw=0)
    ax.set_xticks(range(6)); ax.set_xticklabels(["0", ".25", ".5", "1", "2", "4"])
    ax.set_title(DSN[d], fontsize=9); ax.set_xlabel("Budget multiplier m"); ax.set_ylim(bottom=0)
axs[0].set_ylabel("DP gap, group-targeted attack")
axs[0].legend(loc="upper center", bbox_to_anchor=(2.35, -0.3), ncol=6, fontsize=7, handlelength=2.2)
fig.subplots_adjust(wspace=0.32)
save(fig, "figure3_dp_vs_budget")

# ------------------------------------------------------------- Figure 4: attribution
A = pd.read_csv(f"{F}/posthoc/attribution_summary.csv")
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[5, 3, 5, 3]))
NICE = {"SCHL": "education level", "WKHP": "hours worked", "priors_count": "priors", "juv_fel_count": "juv. felonies",
        "juv_misd_count": "juv. misdem.", "juv_other_count": "juv. other", "hours-per-week": "hours/week",
        "capital-gain": "capital gain", "capital-loss": "capital loss", "education-num": "education",
        "duration": "duration", "amount": "amount", "ALL mutable features": "all features"}
md["fig4"] = {}
for ax, d in zip(axs, DS):
    g = A[(A.dataset == d) & (A.method == "ERM")].set_index("feature")
    feats = [f for f in g.index if f != "ALL mutable features"]
    feats = sorted(feats, key=lambda f: -g.loc[f, "attacked_dp"]) + ["ALL mutable features"]
    vals = [g.loc[f, "attacked_dp"] for f in feats]
    cols = ["#5598e7"] * (len(feats) - 1) + ["#1c5cab"]
    ax.bar(range(len(feats)), vals, color=cols, edgecolor="white", linewidth=1.5, width=0.72)
    ax.axhline(g["clean_dp"].iloc[0], color="#6b6b6b", ls="--", lw=1)
    ax.set_xticks(range(len(feats))); ax.set_xticklabels([NICE.get(f, f) for f in feats], rotation=55, ha="right", fontsize=7)
    ax.set_title(DSN[d], fontsize=9); ax.grid(axis="x", visible=False); ax.set_ylim(0, max(vals) * 1.15)
    md["fig4"][DSN[d]] = {NICE.get(f, f): round(v, 3) for f, v in zip(feats, vals)} | {"clean": round(g["clean_dp"].iloc[0], 3)}
axs[0].set_ylabel("ERM DP gap under attack (1×)")
axs[0].plot([], [], ls="--", color="#6b6b6b", lw=1, label="clean gap (no attack)")
fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.07), fontsize=7.5)
fig.subplots_adjust(wspace=0.45)
save(fig, "figure4_attribution")

# ------------------------------------------------------------- Figure 5: Pareto (run 0, test)
P = pd.read_csv(f"{F}/posthoc/hv_points.csv")
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.6))
for ax, d in zip(axs, DS):
    g = P[(P.dataset == d) & (P.run == P[P.dataset == d].run.iloc[0])]
    for m in M:
        gm = g[g.method == m]
        if m == "ERM":
            ax.scatter(gm.test_dp_targeted, gm.test_acc, marker="X", s=55, color="#222", zorder=5, label="ERM")
            continue
        ax.scatter(gm.test_dp_targeted, gm.test_acc, marker=MK[m], s=22 if m != "TRIFRAME" else 46, color=COL[m],
                   edgecolor="white", linewidth=0.6, alpha=0.95, label=MN[m], zorder=4 if m == "TRIFRAME" else 3)
        pts = sorted(zip(gm.test_dp_targeted, gm.test_acc))
        front, best = [], -1
        for gx, ay in pts:
            if ay > best:
                front.append((gx, ay)); best = ay
        if len(front) > 1:
            fx, fy = zip(*front); ax.step(fx, fy, where="post", color=COL[m], lw=1.1, alpha=0.9)
    ax.set_title(DSN[d], fontsize=9); ax.set_xlabel("DP gap under attack")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3)); ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
axs[0].set_ylabel("Clean test accuracy")
axs[0].legend(loc="upper center", bbox_to_anchor=(2.35, -0.3), ncol=6, fontsize=7, handletextpad=0.3)
fig.subplots_adjust(wspace=0.55)
save(fig, "figure5_pareto")

# ------------------------------------------------------------- Figure 6: lambda_fa sensitivity
S = pd.read_csv(f"{F}/tables/table_sensitivity_lambda_fa.csv")
sc = S[S.attack == "clean"].sort_values("lam_fa"); st = S[S.attack == "fair_dp_targeted"].sort_values("lam_fa")
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.4))
xl = [str(v).rstrip("0").rstrip(".") if v != 0 else "0" for v in sc.lam_fa]
axs[0].errorbar(range(len(sc)), st.dp_mean, yerr=st.dp_std, color=COL["TRIFRAME"], marker="*", ms=7, lw=1.4, capsize=2)
axs[0].set_ylabel("DP gap, group-targeted attack"); axs[0].set_ylim(bottom=0)
axs[1].errorbar(range(len(sc)), sc.acc_mean, yerr=sc.acc_std, color="#256abf", marker="o", ms=4, lw=1.4, capsize=2)
axs[1].set_ylabel("Clean accuracy")
for ax in axs:
    ax.set_xticks(range(len(sc))); ax.set_xticklabels(xl); ax.set_xlabel("λ_fa (other weights fixed)")
fig.subplots_adjust(wspace=0.35)
save(fig, "figure6_lambda_fa")
md["fig6"] = {"lam": list(sc.lam_fa), "acc": [round(v, 4) for v in sc.acc_mean], "dp_targeted": [round(v, 4) for v in st.dp_mean]}

# ------------------------------------------------------------- Table 5: main results with daggers
T = pd.read_csv(f"{F}/tables/paired_tests_holm.csv")
def sig(d, m, attack, eps, metric):
    if m == "TRIFRAME":
        return ""
    r = T[(T.dataset == d) & (T.attack == attack) & np.isclose(T.eps, eps) & (T.metric == metric) & (T.baseline == MN_TESTS[m])]
    return "†" if len(r) and bool(r["significant_holm_0.05"].iloc[0]) else ""
MN_TESTS = {"ERM": "ERM", "FAIR_ONLY": "Fair-Only", "ROBUST_ONLY": "Robust-Only",
            "FAIR_ADV": "Fair adv. training (clean + adv. loss)", "FAIR_ADV_CW": "Fair adv. training (Chai & Wang 2023)"}
pgd = R[(R.attack == "pgd_acc") & (R.eps == 1.0)]
lines = ["| Dataset | Method | Clean accuracy | Clean DP | Accuracy under PGD | DP, group-targeted attack |",
         "| --- | --- | --- | --- | --- | --- |"]
best = {}
for d in DS:
    vals = {}
    for m in M:
        c = clean[(clean.dataset == d) & (clean.method == m)]; p = pgd[(pgd.dataset == d) & (pgd.method == m)]
        t = W[(W.dataset == d) & (W.method == m) & (W.eps == 1.0)]
        vals[m] = [(c.acc.mean(), c.acc.std(ddof=1)), (c.dp.mean(), c.dp.std(ddof=1)), (p.acc.mean(), p.acc.std(ddof=1)),
                   (t.dp.mean(), t.dp.std(ddof=1))]
    bi = [max(M, key=lambda m: vals[m][0][0]), min(M, key=lambda m: vals[m][1][0]), max(M, key=lambda m: vals[m][2][0]),
          min(M, key=lambda m: vals[m][3][0])]
    for m in M:
        keys = [("clean", 0.0, "acc"), ("clean", 0.0, "dp"), ("pgd_acc", 1.0, "acc"), ("fair_dp_targeted", 1.0, "dp")]
        cells = []
        for k, ((mu, sd), (att, e, met)) in enumerate(zip(vals[m], keys)):
            s = f"{mu:.3f} ± {sd:.3f}{sig(d, m, att, e, met)}"
            cells.append(f"**{s}**" if bi[k] == m else s)
        name = MN[m] + ("*" if (d, m) in FALLBACK else "")
        lines.append(f"| {DSN[d] if m == 'ERM' else ''} | {name} | " + " | ".join(cells) + " |")
md["table5"] = "\n".join(lines)

# ------------------------------------------------------------- Table 6: hypervolume
H = pd.read_csv(f"{F}/posthoc/hv_per_run.csv"); HT = pd.read_csv(f"{F}/posthoc/hv_tests.csv")
HS = pd.read_csv(sys.argv[3] + "/hypervolume_tests.csv") if len(sys.argv) > 3 else None
lines = ["| Method | " + " | ".join(DSN[d] for d in DS) + " |", "| --- | " + " | ".join("---" for _ in DS) + " |"]
hvm = {(d, m): H[(H.dataset == d) & (H.method == m)].hv for d in DS for m in M}
for m in M[1:]:
    cells = []
    for d in DS:
        best_m = max(M[1:], key=lambda mm: hvm[(d, mm)].mean())
        s = f"{hvm[(d, m)].mean():.3f} ± {hvm[(d, m)].std(ddof=1):.3f}"
        cells.append(f"**{s}**" if best_m == m else s)
    lines.append(f"| {MN[m]} | " + " | ".join(cells) + " |")
def _p(x):
    return "p < 0.001" if x < 0.001 else f"p = {x:.3f}"
def _cell(d, base):
    if HS is not None:      # Holm-adjusted standard; corrected (South German Credit: corrected only)
        r = HS[(HS.dataset == d) & (HS.baseline == base)].iloc[0]
        if d == "south_german_credit":
            return f"{r['diff']:+.3f} ({_p(r.p_holm_corrected)})"
        return f"{r['diff']:+.3f} ({_p(r.p_holm)}; {r.p_holm_corrected:.3f})"
    r = HT[(HT.dataset == d) & (HT.baseline == base)].iloc[0]
    return f"{r.mean_diff_TRI_minus_base:+.3f} ({_p(r.p_holm)})"
lines.append("| TRI-FRAME − Fair adv. (clean + adv.) | " + " | ".join(_cell(d, "FAIR_ADV") for d in DS) + " |")
lines.append("| TRI-FRAME − Chai & Wang style | " + " | ".join(_cell(d, "FAIR_ADV_CW") for d in DS) + " |")
md["table6"] = "\n".join(lines)

# ------------------------------------------------------------- Table 7: held-out
def mean_of(attack, eps, metric, d, m):
    g = R[(R.dataset == d) & (R.method == m) & (R.attack == attack) & np.isclose(R.eps, eps)]
    return g[metric].mean()
lines = ["| Dataset | Method | EO gap, EO-targeted attack | DP gap, combined attack (β = 1) | DP gap at 4× budget | DP gap, transfer attack |",
         "| --- | --- | --- | --- | --- | --- |"]
for d in DS:
    vals = {m: [mean_of("fair_eo", 1.0, "eo", d, m), mean_of("combined_b1.0", 1.0, "dp", d, m),
                W[(W.dataset == d) & (W.method == m) & (W.eps == 4.0)].dp.mean(), mean_of("transfer_fair_dp", 1.0, "dp", d, m)] for m in M}
    for m in M:
        cells = []
        for k in range(4):
            s = f"{vals[m][k]:.3f}"
            cells.append(f"**{s}**" if min(M, key=lambda mm: vals[mm][k]) == m else s)
        lines.append(f"| {DSN[d] if m == 'ERM' else ''} | {MN[m]} | " + " | ".join(cells) + " |")
md["table7"] = "\n".join(lines)

# ------------------------------------------------------------- Table 4: datasets, Table 8: ablations (from tables)
md["table4_src"] = open(f"{F}/tables/table1_datasets.md").read()
md["table8_src"] = open(f"{F}/tables/table_ablations.md").read()
md["cost"] = open(f"{F}/tables/table_cost.md").read()
md["gbdt"] = {DSN[d]: (round(clean[(clean.dataset == d) & (clean.method == "GBDT")].dp.mean(), 3),
                       round(mean_of("transfer_fair_dp", 1.0, "dp", d, "GBDT"), 3),
                       round(clean[(clean.dataset == d) & (clean.method == "GBDT")].acc.mean(), 3)) for d in DS}
md["faironly"] = {DSN[d]: (round(clean[(clean.dataset == d) & (clean.method == "FAIR_ONLY")].dp.mean(), 3),
                           round(W[(W.dataset == d) & (W.method == "FAIR_ONLY") & (W.eps == 1.0)].dp.mean(), 3)) for d in DS}
md["full_pop"] = {DSN[d]: round(mean_of("fair_dp", 1.0, "dp", d, "ERM"), 3) for d in DS}
json.dump({k: v for k, v in md.items() if not k.startswith("table")}, open(f"{OUT}/numbers.json", "w"), indent=1, default=str)
for k in ["table5", "table6", "table7"]:
    open(f"{OUT}/{k}.md", "w").write(md[k])
print(json.dumps({k: v for k, v in md.items() if k in ("fig1", "fig4", "fig6", "gbdt", "faironly", "full_pop")}, indent=1, default=str))
print(md["table6"]); print(md["cost"]); print(md["table4_src"])
