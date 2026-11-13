"""TRI-FRAME v2 pipeline, MODE = "full", exported from TRI_FRAME_v2_pipeline.ipynb.
Identical code to the notebook except: MODE fixed to "full", matplotlib draws to files only
(no windows), and the script exits with code 0 only after every cell has run.
Run it from the SAME folder as the notebook so it finds the same data and output folders.
"""
import matplotlib
matplotlib.use('Agg')

# ======== notebook cell 2 ========
import os, sys, json, time, math, zipfile, fnmatch, itertools, warnings, platform, datetime, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

# ======== notebook cell 4 ========
MODE = "full"          # "smoke" (quick check) | "pilot" (Adult only, decision study) | "full" (all experiments)

# Folder layout: TRI_FRAME/download (datasets) and TRI_FRAME/output (results).
# None = auto-detect: looks for a "download" folder in the notebook's folder and its parents.
BASE_DIR = None         # e.g. Path(r"C:\Users\you\Desktop\TRI_FRAME")

ALLOW_ACS_DOWNLOAD = True        # if no ACS files are in download/, let folktables fetch them from the Census site
ACS_YEAR = "2018"
ACS_STATES = ["CA"]              # training state(s) for ACS Income
ACS_MAX_ROWS = 100_000           # subsample ACS for CPU training (None = use all rows)
ACS_SHIFT_STATES = ["TX", "NY", "FL"]   # target states for the cross-state experiment (full mode)

INCLUDE_SENSITIVE_AS_FEATURE = False   # paper default: the sensitive attribute is NOT a model input (ablation flips it)
ROUND_INTEGER_FEATURES = True          # evaluation attacks round integer-valued features (hours, counts, months)
SAVE_MODELS = True                     # cache trained models in output/<MODE>/models (enables fast resume)
DEVICE = "cpu"
NUM_THREADS = os.cpu_count() or 1

# Model and optimisation (Section 6.3)
HIDDEN = (128, 64)
LR = 1e-3
BATCH_SIZE = 256

# Hyperparameter grids (Section 6.4). Each tuned method gets TUNE_BUDGET configurations.
LAMBDA_LEVELS = [0.1, 0.3, 1.0]               # levels for lambda_f, lambda_r, lambda_fa in joint grids
FAIR_ONLY_RANGE = (0.03, 10.0)                # geometric range for Fair-Only lambda_f
ROBUST_ONLY_RANGE = (0.1, 3.0)                # geometric range for Robust-Only lambda_r
TRAIN_EPS_LEVELS = [0.5, 1.0, 2.0]            # training-attack budget multipliers tuned for Robust-Only and Chai & Wang
ACC_TOLERANCE = 0.01                          # selection: within 1 percentage point of ERM validation accuracy

# Pilot decision thresholds (pilot study table in Section 6)
PILOT_MIN_DP_INCREASE_ABS = 0.02              # attack must raise ERM's DP gap by at least 2 points ...
PILOT_MIN_DP_INCREASE_REL = 0.20              # ... and by at least 20% of the clean gap

# Ablation: budget for the "unconstrained" training attack, in standard deviations, applied to ALL numeric features
UNCONSTRAINED_EPS_STD = 0.1

MODE_SETTINGS = {
    "smoke": dict(datasets=["adult", "acs_income", "compas", "south_german_credit"], n_seeds=1, sgc_repeats=1,
                  subsample=2500, max_epochs=3, patience=2, k_train=3,
                  eval_eps=[0.0, 1.0, 2.0], eval_steps=5, eval_restarts=1, tune_budget=2, tune_steps=3,
                  combined_betas=[1.0], run_ablations=True, ablation_seeds=1, run_sensitivity=True, sens_seeds=1,
                  run_certified=True, run_gbdt=True, run_acs_shift=False),
    "pilot": dict(datasets=["adult"], n_seeds=3, sgc_repeats=1,
                  subsample=None, max_epochs=30, patience=5, k_train=5,
                  eval_eps=[0.0, 0.5, 1.0, 2.0], eval_steps=20, eval_restarts=1, tune_budget=4, tune_steps=10,
                  combined_betas=[1.0], run_ablations=False, ablation_seeds=0, run_sensitivity=False, sens_seeds=0,
                  run_certified=True, run_gbdt=True, run_acs_shift=False),
    "full": dict(datasets=["adult", "acs_income", "compas", "south_german_credit"], n_seeds=10, sgc_repeats=5,
                 subsample=None, max_epochs=50, patience=5, k_train=5,
                 eval_eps=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0], eval_steps=50, eval_restarts=5, tune_budget=9, tune_steps=20,
                 combined_betas=[0.5, 1.0, 2.0], run_ablations=True, ablation_seeds=5, run_sensitivity=True, sens_seeds=3,
                 run_certified=True, run_gbdt=True, run_acs_shift=True),
}
S = MODE_SETTINGS[MODE]

# ======== notebook cell 6 ========
DATASET_SPECS = {
    "adult": dict(
        display="Adult",
        numeric=["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"],
        categorical=["workclass", "marital-status", "occupation", "relationship", "race", "native-country"],
        integer=["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"],
        budget={"hours-per-week": 4, "capital-gain": 2000, "capital-loss": 200, "education-num": 1},
        sensitive_name="sex (1 = male)", split="holdout"),
    "acs_income": dict(
        display="ACS Income",
        numeric=["AGEP", "SCHL", "WKHP"],
        categorical=["COW", "MAR", "OCCP_GRP", "POBP_GRP", "RELP", "RAC1P"],
        integer=["AGEP", "SCHL", "WKHP"],
        budget={"WKHP": 4, "SCHL": 1},
        sensitive_name="SEX (1 = male)", split="holdout"),
    "compas": dict(
        display="COMPAS",
        numeric=["age", "priors_count", "juv_fel_count", "juv_misd_count", "juv_other_count"],
        categorical=["sex", "c_charge_degree"],
        integer=["age", "priors_count", "juv_fel_count", "juv_misd_count", "juv_other_count"],
        budget={"priors_count": 1, "juv_fel_count": 1, "juv_misd_count": 1, "juv_other_count": 1},
        sensitive_name="race (1 = Caucasian, 0 = African-American)", split="holdout"),
    "south_german_credit": dict(
        display="South German Credit",
        numeric=["duration", "amount", "installment_rate", "present_residence", "number_credits"],
        categorical=["status", "credit_history", "purpose", "savings", "employment_duration", "personal_status_sex",
                     "other_debtors", "property", "other_installment_plans", "housing", "job", "people_liable",
                     "telephone", "foreign_worker"],
        integer=["duration", "amount", "installment_rate", "present_residence", "number_credits"],
        budget={"duration": 6, "amount": 500},
        sensitive_name="age (1 = over 25, 0 = 25 or under)", split="cv"),
}

METHODS = ["ERM", "FAIR_ONLY", "ROBUST_ONLY", "FAIR_ADV", "FAIR_ADV_CW", "TRIFRAME"]
METHOD_LABELS = {"ERM": "ERM", "FAIR_ONLY": "Fair-Only", "ROBUST_ONLY": "Robust-Only",
                 "FAIR_ADV": "Fair adv. training (clean + adv. loss)",
                 "FAIR_ADV_CW": "Fair adv. training (Chai & Wang 2023)", "TRIFRAME": "TRI-FRAME",
                 "GBDT": "GBDT (reference)"}
COLORS = {"ERM": "#555555", "FAIR_ONLY": "#1f77b4", "ROBUST_ONLY": "#ff7f0e", "FAIR_ADV": "#2ca02c",
          "FAIR_ADV_CW": "#9467bd", "TRIFRAME": "#d62728"}
MARKERS = {"ERM": "o", "FAIR_ONLY": "s", "ROBUST_ONLY": "^", "FAIR_ADV": "D", "FAIR_ADV_CW": "P", "TRIFRAME": "*"}

# ======== notebook cell 8 ========
def detect_base_dir():
    if BASE_DIR is not None:
        return Path(BASE_DIR)
    here = Path.cwd().resolve()
    for cand in [here, *here.parents]:
        for sub in cand.iterdir() if cand.exists() else []:
            if sub.is_dir() and sub.name.lower() in ("download", "downloads"):
                return cand
    raise FileNotFoundError("Could not find a 'download' folder. Set BASE_DIR in Section 1 to your TRI_FRAME folder.")

BASE = detect_base_dir()
DATA_DIR = next(p for p in BASE.iterdir() if p.is_dir() and p.name.lower() in ("download", "downloads"))
OUT = BASE / "output" / MODE
for sub in ["tables", "figures", "tuning", "models", "logs", "curves", "meta"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)

LOG_FILE = OUT / "logs" / f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"

def log(*msg):
    line = f"[{datetime.datetime.now():%H:%M:%S}] " + " ".join(str(m) for m in msg)
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

torch.set_num_threads(NUM_THREADS)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

with open(OUT / "config.json", "w") as f:
    json.dump(dict(MODE=MODE, settings=S, specs=DATASET_SPECS, ACS_YEAR=ACS_YEAR, ACS_STATES=ACS_STATES,
                   ACS_MAX_ROWS=ACS_MAX_ROWS, INCLUDE_SENSITIVE_AS_FEATURE=INCLUDE_SENSITIVE_AS_FEATURE,
                   ROUND_INTEGER_FEATURES=ROUND_INTEGER_FEATURES, HIDDEN=HIDDEN, LR=LR, BATCH_SIZE=BATCH_SIZE,
                   LAMBDA_LEVELS=LAMBDA_LEVELS, ACC_TOLERANCE=ACC_TOLERANCE,
                   versions=dict(python=platform.python_version(), torch=torch.__version__,
                                 numpy=np.__version__, pandas=pd.__version__)), f, indent=2)
log(f"MODE={MODE} | data: {DATA_DIR} | output: {OUT} | threads={NUM_THREADS}")

# ======== notebook cell 10 ========
def extract_zips(data_dir):
    for z in data_dir.rglob("*.zip"):
        if "_extracted" in z.parts:
            continue
        target = data_dir / "_extracted" / z.stem
        if not target.exists():
            try:
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(target)
                log(f"Extracted {z.name}")
            except zipfile.BadZipFile:
                log(f"Skipping unreadable zip {z.name}")

def find_files(patterns, data_dir=None):
    data_dir = data_dir or DATA_DIR
    hits = [p for p in data_dir.rglob("*") if p.is_file()
            and any(fnmatch.fnmatch(p.name.lower(), pat) for pat in patterns)]
    # prefer files the user placed directly over copies extracted from zips
    return sorted(hits, key=lambda p: ("_extracted" in p.parts, len(p.parts), str(p)))

extract_zips(DATA_DIR)

ADULT_COLS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
              "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
ADULT_SYNONYMS = {"educational-num": "education-num", "gender": "sex", "class": "income"}

def load_adult():
    data = find_files(["adult.data"])
    test = find_files(["adult.test"])
    if data:
        files = [data[0]] + ([test[0]] if test else [])
    else:
        files = [f for f in find_files(["adult*.csv"]) if "names" not in f.name.lower()][:1]
    if not files:
        raise FileNotFoundError("Adult: put adult.data and adult.test (or adult.csv) in download/")
    frames = []
    for f in files:
        first = open(f, encoding="utf-8", errors="ignore").readline().lower()
        if first.startswith("age"):     # CSV with a header row (e.g. Kaggle copies)
            df = pd.read_csv(f, skipinitialspace=True, na_values="?")
            df.columns = [ADULT_SYNONYMS.get(c.strip().lower().replace(".", "-").replace("_", "-"),
                                             c.strip().lower().replace(".", "-").replace("_", "-")) for c in df.columns]
        else:
            df = pd.read_csv(f, header=None, names=ADULT_COLS, skipinitialspace=True, na_values="?", comment="|")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().replace({"?": np.nan, "nan": np.nan})
    n0 = len(df)
    df = df.dropna(subset=[c for c in ADULT_COLS if c in df.columns]).copy()
    df["income"] = df["income"].str.replace(".", "", regex=False)
    df["y"] = df["income"].str.contains(">50K").astype(int)
    df["a"] = (df["sex"] == "Male").astype(int)
    log(f"Adult: {n0} rows read from {[f.name for f in files]}, {len(df)} after dropping missing values")
    return df

STATE_CODES = {'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06', 'CO': '08', 'CT': '09', 'DE': '10',
               'FL': '12', 'GA': '13', 'HI': '15', 'ID': '16', 'IL': '17', 'IN': '18', 'IA': '19', 'KS': '20',
               'KY': '21', 'LA': '22', 'ME': '23', 'MD': '24', 'MA': '25', 'MI': '26', 'MN': '27', 'MS': '28',
               'MO': '29', 'MT': '30', 'NE': '31', 'NV': '32', 'NH': '33', 'NJ': '34', 'NM': '35', 'NY': '36',
               'NC': '37', 'ND': '38', 'OH': '39', 'OK': '40', 'OR': '41', 'PA': '42', 'RI': '44', 'SC': '45',
               'SD': '46', 'TN': '47', 'TX': '48', 'UT': '49', 'VT': '50', 'VA': '51', 'WA': '53', 'WV': '54',
               'WI': '55', 'WY': '56', 'PR': '72'}
ACS_RAW = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P"]

def _acs_features(df):
    """folktables ACSIncome task: adult filter, label PINCP > 50,000, grouped high-cardinality codes."""
    if "PINCP" in df.columns and df["PINCP"].max() > 1:
        for col, cond in [("AGEP", lambda s: s > 16), ("PINCP", lambda s: s > 100), ("WKHP", lambda s: s > 0)]:
            df = df[cond(df[col])]
        if "PWGTP" in df.columns:
            df = df[df["PWGTP"] >= 1]
        df = df.assign(y=(df["PINCP"] > 50000).astype(int))
    else:                                   # already-labelled copies (e.g. OpenML ACSIncome)
        lab = [c for c in df.columns if c.upper() in ("PINCP", "TARGET", "LABEL", "CLASS")][0]
        df = df.assign(y=df[lab].astype(str).str.lower().isin(["1", "true", "1.0", ">50k"]).astype(int))
    df = df.copy()
    for c in ["COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "RAC1P"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1)
    df["OCCP_GRP"] = (df["OCCP"] // 100).astype(int).astype(str)
    df["POBP_GRP"] = np.where(df["POBP"] < 100, "US", (df["POBP"] // 100 * 100).astype(int).astype(str))
    for c in ["COW", "MAR", "RELP", "RAC1P"]:
        df[c] = df[c].astype(int).astype(str)
    df["a"] = (pd.to_numeric(df["SEX"]) == 1).astype(int)
    return df

def load_acs(states=None):
    states = states or ACS_STATES
    codes = {STATE_CODES[s] for s in states}
    pums = find_files(["psam_p*.csv"])
    chosen = [f for f in pums if f.stem[-2:] in codes]
    if chosen:
        use = ACS_RAW + ["PINCP", "PWGTP", "ST"]
        df = pd.concat([pd.read_csv(f, usecols=lambda c: c in use) for f in chosen], ignore_index=True)
        src = [f.name for f in chosen]
    else:
        pre = [f for f in find_files(["*acs*income*.csv", "*acs*income*.parquet", "*acs*income*.arff"])]
        if pre:
            f = pre[0]
            if f.suffix == ".parquet":
                df = pd.read_parquet(f)
            elif f.suffix == ".arff":
                from scipy.io import arff
                raw, _ = arff.loadarff(f)
                df = pd.DataFrame(raw)
                for c in df.columns:
                    if df[c].dtype == object:
                        df[c] = df[c].str.decode("utf-8")
            else:
                df = pd.read_csv(f)
            if "ST" in df.columns:
                df = df[pd.to_numeric(df["ST"]).astype(int).astype(str).str.zfill(2).isin(codes)]
            src = [f.name]
        elif ALLOW_ACS_DOWNLOAD:
            from folktables import ACSDataSource
            log(f"ACS: no local files found, downloading {ACS_YEAR} {states} via folktables ...")
            ds = ACSDataSource(survey_year=ACS_YEAR, horizon="1-Year", survey="person", root_dir=str(DATA_DIR / "acs"))
            df = ds.get_data(states=list(states), download=True)
            src = ["folktables download"]
        else:
            raise FileNotFoundError("ACS Income: no psam_p??.csv or ACSIncome file in download/, and ALLOW_ACS_DOWNLOAD=False")
    df = _acs_features(df)
    if ACS_MAX_ROWS and len(df) > ACS_MAX_ROWS:
        df = df.sample(ACS_MAX_ROWS, random_state=0)
    log(f"ACS Income ({','.join(states)}): {len(df)} rows from {src}")
    return df.reset_index(drop=True)

def load_compas():
    files = find_files(["compas-scores-two-years.csv"])
    if not files:
        raise FileNotFoundError("COMPAS: put compas-scores-two-years.csv in download/")
    df = pd.read_csv(files[0])
    n0 = len(df)
    # ProPublica filtering
    df = df[(df["days_b_screening_arrest"] <= 30) & (df["days_b_screening_arrest"] >= -30)
            & (df["is_recid"] != -1) & (df["c_charge_degree"] != "O") & (df["score_text"] != "N/A")]
    df = df[df["race"].isin(["African-American", "Caucasian"])].copy()
    df["y"] = df["two_year_recid"].astype(int)
    df["a"] = (df["race"] == "Caucasian").astype(int)
    log(f"COMPAS: {n0} rows, {len(df)} after ProPublica filtering and Black/White restriction")
    return df.reset_index(drop=True)

SGC_GERMAN_TO_EN = {"laufkont": "status", "laufzeit": "duration", "moral": "credit_history", "verw": "purpose",
                    "hoehe": "amount", "sparkont": "savings", "beszeit": "employment_duration", "rate": "installment_rate",
                    "famges": "personal_status_sex", "buerge": "other_debtors", "wohnzeit": "present_residence",
                    "verm": "property", "alter": "age", "weitkred": "other_installment_plans", "wohn": "housing",
                    "bishkred": "number_credits", "beruf": "job", "pers": "people_liable", "telef": "telephone",
                    "gastarb": "foreign_worker", "kredit": "credit_risk"}

def load_sgc():
    files = find_files(["southgermancredit*.asc", "*south*german*.csv", "southgermancredit*.csv"])
    if not files:
        raise FileNotFoundError("South German Credit: put SouthGermanCredit.asc (UCI update) in download/")
    f = files[0]
    df = pd.read_csv(f, sep=r"\s+" if f.suffix == ".asc" else ",")
    df = df.rename(columns=lambda c: SGC_GERMAN_TO_EN.get(c.strip().lower(), c.strip().lower()))
    df["y"] = df["credit_risk"].astype(int)          # 1 = good credit
    df["a"] = (df["age"] > 25).astype(int)           # sensitive attribute: age over 25
    for c in DATASET_SPECS["south_german_credit"]["categorical"]:
        df[c] = df[c].astype(str)
    log(f"South German Credit: {len(df)} rows from {f.name}; group sizes a=0: {(df.a==0).sum()}, a=1: {(df.a==1).sum()}")
    return df

LOADERS = {"adult": load_adult, "acs_income": load_acs, "compas": load_compas, "south_german_credit": load_sgc}

def load_dataset(name):
    df = LOADERS[name]()
    if S["subsample"] and len(df) > S["subsample"]:
        df = df.sample(S["subsample"], random_state=0).reset_index(drop=True)
    return df

# ======== notebook cell 12 ========
class Preprocessor:
    def __init__(self, spec, include_sensitive=False):
        self.spec, self.include_sensitive = spec, include_sensitive

    def fit(self, df):
        self.num = list(self.spec["numeric"]) + (["sensitive_attr"] if self.include_sensitive else [])
        df = self._with_sensitive(df)
        X = df[self.num].astype(float)
        self.mean, self.std = X.mean().values, X.std().replace(0, 1).fillna(1).values
        self.min, self.max = X.min().values, X.max().values
        self.cats = {c: sorted(df[c].astype(str).unique()) for c in self.spec["categorical"]}
        self.columns = self.num + [f"{c}={v}" for c in self.cats for v in self.cats[c]]
        return self

    def _with_sensitive(self, df):
        return df.assign(sensitive_attr=df["a"]) if self.include_sensitive else df

    def transform(self, df):
        df = self._with_sensitive(df)
        Xn = (df[self.num].astype(float).values - self.mean) / self.std
        blocks = [Xn]
        for c, vals in self.cats.items():
            col = df[c].astype(str).values
            blocks.append(np.stack([(col == v) for v in vals], axis=1).astype(float))
        return np.concatenate(blocks, axis=1).astype(np.float32)

    def attack_space(self, unconstrained=False):
        d, k = len(self.columns), len(self.num)
        eps, lo, hi = np.zeros(d), np.zeros(d), np.ones(d)
        mean, std = np.zeros(d), np.ones(d)
        mean[:k], std[:k] = self.mean, self.std
        lo[:k], hi[:k] = (self.min - self.mean) / self.std, (self.max - self.mean) / self.std
        for j, c in enumerate(self.num):
            if unconstrained:
                eps[j] = UNCONSTRAINED_EPS_STD
            elif c in self.spec["budget"]:
                eps[j] = self.spec["budget"][c] / self.std[j]
        if unconstrained:
            lo[:k], hi[:k] = -1e6, 1e6
        int_idx = [j for j, c in enumerate(self.num) if c in self.spec["integer"]]
        return AttackSpace(eps, lo, hi, int_idx, mean, std)


class AttackSpace:
    """Feasible set F(x): per-feature budgets eps_j (0 = immutable), valid ranges, integer projection."""
    def __init__(self, eps, lo, hi, int_idx, mean, std):
        t = lambda v: torch.tensor(v, dtype=torch.float32, device=DEVICE)
        self.eps, self.lo, self.hi, self.mean, self.std = t(eps), t(lo), t(hi), t(mean), t(std)
        self.int_idx = torch.tensor(int_idx, dtype=torch.long, device=DEVICE)

    def bounds(self, x, eps_mult, row_scale=None):
        eps = self.eps * eps_mult
        eps = eps.expand_as(x) if row_scale is None else eps.unsqueeze(0) * row_scale.unsqueeze(1)
        lower = torch.maximum(x - eps, torch.minimum(self.lo, x))   # never move an already out-of-range value further
        upper = torch.minimum(x + eps, torch.maximum(self.hi, x))
        return eps, lower, upper

    def round_int(self, x):
        if self.int_idx.numel() == 0:
            return x
        x = x.clone()
        c = self.int_idx
        raw = x[:, c] * self.std[c] + self.mean[c]
        x[:, c] = (torch.round(raw) - self.mean[c]) / self.std[c]
        return x


def make_splits(df, split_kind, n_seeds, sgc_repeats):
    """Returns list of (run_id, train_idx, val_idx, test_idx). Holdout: 60/20/20 per seed, stratified on (y, a).
    CV (South German Credit): repeated stratified 5-fold; validation = 25% of each training fold (60/20/20 overall)."""
    strat = (df["y"].astype(str) + "_" + df["a"].astype(str)).values
    idx = np.arange(len(df))
    runs = []
    if split_kind == "holdout":
        for seed in range(n_seeds):
            tr, te = train_test_split(idx, test_size=0.2, stratify=strat, random_state=seed)
            tr, va = train_test_split(tr, test_size=0.25, stratify=strat[tr], random_state=seed)
            runs.append((f"s{seed}", tr, va, te))
    else:
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=sgc_repeats, random_state=0)
        for k, (tr, te) in enumerate(rskf.split(idx, strat)):
            tr, va = train_test_split(tr, test_size=0.25, stratify=strat[tr], random_state=k)
            runs.append((f"r{k // 5}f{k % 5}", tr, va, te))
    return runs


class SplitData:
    def __init__(self, df, tr, va, te, spec, include_sensitive=False):
        self.pre = Preprocessor(spec, include_sensitive).fit(df.iloc[tr])
        t = lambda arr: torch.tensor(arr, device=DEVICE)
        for name, ix in [("tr", tr), ("va", va), ("te", te)]:
            part = df.iloc[ix]
            setattr(self, f"X{name}", t(self.pre.transform(part)))
            setattr(self, f"y{name}", t(part["y"].values.astype(np.float32)))
            setattr(self, f"a{name}", t(part["a"].values.astype(np.float32)))
        self.space = self.pre.attack_space()
        self.d = self.Xtr.shape[1]

# ======== notebook cell 14 ========
def pgd_attack(model, x, objective, space, eps_mult, steps, restarts=1, alpha_frac=None,
               rand_init=True, row_scale=None, round_int=False, chunk=8192):
    """Maximise objective(logits, start, end) (a per-record vector) over the feasible set F(x)."""
    if eps_mult == 0 or steps == 0:
        return x.clone()
    alpha_frac = alpha_frac if alpha_frac is not None else 2.5 / steps
    out = torch.empty_like(x)
    was_training = model.training
    model.eval()
    for s in range(0, len(x), chunk):
        e = min(s + chunk, len(x))
        xc = x[s:e]
        rs = None if row_scale is None else row_scale[s:e]
        eps, lower, upper = space.bounds(xc, eps_mult, rs)
        step = alpha_frac * eps
        with torch.no_grad():
            best, best_obj = xc.clone(), objective(model(xc), s, e)
        for _ in range(restarts):
            xa = xc.clone()
            if rand_init:
                xa = torch.max(torch.min(xa + (torch.rand_like(xa) * 2 - 1) * eps, upper), lower)
            for _ in range(steps):
                xa = xa.detach().requires_grad_(True)
                with torch.enable_grad():
                    obj = objective(model(xa), s, e)
                    g, = torch.autograd.grad(obj.sum(), xa)
                xa = torch.max(torch.min(xa.detach() + step * g.sign(), upper), lower)
            if round_int:
                xa = torch.max(torch.min(space.round_int(xa), upper), lower)
            with torch.no_grad():
                o = objective(model(xa), s, e)
                better = o > best_obj
                best[better], best_obj = xa[better], torch.where(better, o, best_obj)
        out[s:e] = best
    model.train(was_training)
    return out.detach()


def _sign(v):
    return 1.0 if float(v) >= 0 else -1.0

def dp_directions(p, a, target=None):
    s = _sign(p[a == 1].mean() - p[a == 0].mean()) if (a == 1).any() and (a == 0).any() else 1.0
    d = torch.where(a == 1, torch.full_like(a, s), torch.full_like(a, -s))
    return d if target is None else d * (a == target).float()

def eo_directions(p, a, y):
    d = torch.zeros_like(a)
    for yy in (0.0, 1.0):
        m = y == yy
        if (m & (a == 1)).any() and (m & (a == 0)).any():
            s = _sign(p[m & (a == 1)].mean() - p[m & (a == 0)].mean())
            d[m] = torch.where(a[m] == 1, torch.full_like(a[m], s), torch.full_like(a[m], -s))
    return d

def obj_ce(y):
    return lambda z, s, e: F.binary_cross_entropy_with_logits(z, y[s:e], reduction="none")

def obj_dir(d):
    return lambda z, s, e: d[s:e] * z

def obj_combined(y, a, d, beta):
    n = len(y); n1 = max(float((a == 1).sum()), 1.0); n0 = max(float((a == 0).sum()), 1.0)
    w = torch.where(a == 1, torch.full_like(a, 1 / n1), torch.full_like(a, 1 / n0))
    return lambda z, s, e: (F.binary_cross_entropy_with_logits(z, y[s:e], reduction="none") / n
                            + beta * d[s:e] * torch.sigmoid(z) * w[s:e])

# ======== notebook cell 16 ========
class MLP(nn.Module):
    def __init__(self, d, hidden=HIDDEN):
        super().__init__()
        layers, prev = [], d
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def soft_dp(p, a):
    if not ((a == 1).any() and (a == 0).any()):
        return p.sum() * 0
    return (p[a == 1].mean() - p[a == 0].mean()).abs()

def soft_eo(p, a, y):
    tot = p.sum() * 0
    for yy in (0.0, 1.0):
        m = y == yy
        tot = tot + soft_dp(p[m], a[m])
    return tot

def cov_penalty(p, a):
    return ((a - a.mean()) * (p - p.mean())).mean().abs()

def fairness_term(p, a, y, kind):
    """Fairness surrogate: soft DP gap (default), covariance (v1, ablation) or soft EO gap (ablation)."""
    if kind == "dp":
        return soft_dp(p, a)
    if kind == "cov":
        return cov_penalty(p, a)
    if kind == "eo":
        return soft_eo(p, a, y)
    raise ValueError(kind)

DEFAULT_CFG = dict(lam_f=0.0, lam_r=0.0, lam_fa=0.0, surrogate="dp", train_attack="pgd", unconstrained=False,
                   train_eps=1.0, adv_only=False)

def composite_loss(model, x, y, a, cfg, space, k_train, grad=True):
    """TRI-FRAME objective (Section 4.1). Returns (total, parts)."""
    cfg = {**DEFAULT_CFG, **cfg}
    z = model(x)
    p = torch.sigmoid(z)
    steps, alpha, rinit = (1, 1.0, False) if cfg["train_attack"] == "fgsm" else (k_train, None, True)
    te = cfg["train_eps"]
    if cfg["adv_only"]:      # Chai & Wang: train on accuracy-adversarial samples only
        xa = pgd_attack(model, x, obj_ce(y), space, te, steps, 1, alpha, rinit)
        parts = {"task": F.binary_cross_entropy_with_logits(model(xa), y)}
    else:
        parts = {"task": F.binary_cross_entropy_with_logits(z, y)}
    if cfg["lam_f"] > 0:
        parts["fair"] = fairness_term(p, a, y, cfg["surrogate"])
    if cfg["lam_r"] > 0 and not cfg["adv_only"]:
        xa = pgd_attack(model, x, obj_ce(y), space, te, steps, 1, alpha, rinit)
        parts["rob"] = F.binary_cross_entropy_with_logits(model(xa), y)
    if cfg["lam_fa"] > 0:
        d = eo_directions(p.detach(), a, y) if cfg["surrogate"] == "eo" else dp_directions(p.detach(), a)
        xf = pgd_attack(model, x, obj_dir(d), space, te, steps, 1, alpha, rinit)
        parts["fair_adv"] = fairness_term(torch.sigmoid(model(xf)), a, y, cfg["surrogate"])
    total = parts["task"] + cfg["lam_f"] * parts.get("fair", 0) + cfg["lam_r"] * parts.get("rob", 0) \
        + cfg["lam_fa"] * parts.get("fair_adv", 0)
    return total, parts


def stratified_batches(y, a, batch_size, rng):
    """Shuffle so every mini-batch contains both groups and both labels in roughly population proportions."""
    cell = (y.cpu().numpy() * 2 + a.cpu().numpy()).astype(int)
    key = np.empty(len(cell))
    for c in np.unique(cell):
        m = np.where(cell == c)[0]
        key[m] = (rng.permutation(len(m)) + rng.random(len(m))) / len(m)
    order = np.argsort(key)
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


def train_model(cfg, data, seed, train_space=None):
    """Algorithm 1 with early stopping on the validation objective. Returns (model, info)."""
    cfg = {**DEFAULT_CFG, **cfg}
    set_seed(seed)
    rng = np.random.default_rng(seed)
    space = train_space or (data.pre.attack_space(unconstrained=True) if cfg["unconstrained"] else data.space)
    model = MLP(data.d).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best_val, best_state, bad, times, ep, history = float("inf"), None, 0, [], 0, []
    for ep in range(1, S["max_epochs"] + 1):
        t0 = time.time()
        model.train()
        batch_losses = []
        for idx in stratified_batches(data.ytr, data.atr, BATCH_SIZE, rng):
            idx = torch.as_tensor(idx, device=DEVICE)
            loss, _ = composite_loss(model, data.Xtr[idx], data.ytr[idx], data.atr[idx], cfg, space, S["k_train"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            batch_losses.append(float(loss.detach()))
        times.append(time.time() - t0)
        model.eval()
        val, _ = composite_loss(model, data.Xva, data.yva, data.ava, cfg, space, S["k_train"])
        val = float(val.detach())
        vm = hard_metrics(probs(model, data.Xva), data.yva, data.ava)
        history.append(dict(epoch=ep, train_objective=float(np.mean(batch_losses)), val_objective=val,
                            val_acc=vm["acc"], val_dp=vm["dp"], sec=times[-1]))
        if val < best_val - 1e-5:
            best_val, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= S["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    best_epoch = min(history, key=lambda h: h["val_objective"])["epoch"]
    return model, dict(epochs=ep, best_epoch=best_epoch, sec_per_epoch=float(np.mean(times)),
                       best_val_objective=best_val, history=history)


def cfg_hash(cfg):
    return hashlib.md5(json.dumps({**DEFAULT_CFG, **cfg}, sort_keys=True).encode()).hexdigest()[:8]

def get_model(cfg, data, seed, cache_key, meta=None):
    """Train, or load from output/<MODE>/models if SAVE_MODELS and already trained (resume).
    The cache key includes a hash of the configuration, so changing a grid never reuses a stale model.
    Per-epoch training curves are written to output/<MODE>/curves/."""
    cache_key = f"{cache_key}_{cfg_hash(cfg)}"
    path = OUT / "models" / f"{cache_key}.pt"
    if SAVE_MODELS and path.exists():
        blob = torch.load(path, map_location=DEVICE, weights_only=False)
        model = MLP(data.d).to(DEVICE)
        model.load_state_dict(blob["state"])
        model.eval()
        return model, blob["info"]
    model, info = train_model(cfg, data, seed)
    pd.DataFrame(info["history"]).assign(key=cache_key, cfg=json.dumps(cfg), **(meta or {})).to_csv(
        OUT / "curves" / f"{cache_key}.csv", index=False)
    if SAVE_MODELS:
        torch.save({"state": model.state_dict(), "info": info, "cfg": cfg}, path)
    return model, info

# ======== notebook cell 18 ========
@torch.no_grad()
def ibp_prob_bounds(model, lower, upper):
    l, u = lower, upper
    for layer in model.net:
        if isinstance(layer, nn.Linear):
            mid, rad = (u + l) / 2, (u - l) / 2
            c = mid @ layer.weight.T + layer.bias
            r = rad @ layer.weight.abs().T
            l, u = c - r, c + r
        else:
            l, u = F.relu(l), F.relu(u)
    return torch.sigmoid(l).squeeze(-1), torch.sigmoid(u).squeeze(-1)

@torch.no_grad()
def certified_dp(model, x, a, space, eps_mult=1.0):
    _, lower, upper = space.bounds(x, eps_mult)
    pl, pu = ibp_prob_bounds(model, lower, upper)
    g1, g0 = a == 1, a == 0
    soft = max(float(pu[g1].mean() - pl[g0].mean()), float(pu[g0].mean() - pl[g1].mean()))
    # hard labels: a record can be pushed to 1 if pu >= 0.5, and is stuck at 1 only if pl >= 0.5
    up1, dn0 = (pu[g1] >= 0.5).float().mean(), (pl[g0] >= 0.5).float().mean()
    up0, dn1 = (pu[g0] >= 0.5).float().mean(), (pl[g1] >= 0.5).float().mean()
    hard = max(float(up1 - dn0), float(up0 - dn1))
    return soft, hard

# ======== notebook cell 20 ========
def hard_metrics(p, y, a):
    yhat = (p >= 0.5).float()
    out = {"acc": float((yhat == y).float().mean())}
    rates = {}
    for g in (0, 1):
        m = a == g
        rates[g] = dict(pos=float(yhat[m].mean()) if m.any() else np.nan,
                        tpr=float(yhat[m & (y == 1)].mean()) if (m & (y == 1)).any() else np.nan,
                        fpr=float(yhat[m & (y == 0)].mean()) if (m & (y == 0)).any() else np.nan,
                        acc=float((yhat[m] == y[m]).float().mean()) if m.any() else np.nan)
    tpr = float(yhat[y == 1].mean()) if (y == 1).any() else np.nan
    tnr = float(1 - yhat[y == 0].mean()) if (y == 0).any() else np.nan
    out.update(bal_acc=(tpr + tnr) / 2, dp=abs(rates[1]["pos"] - rates[0]["pos"]),
               tpr_gap=abs(rates[1]["tpr"] - rates[0]["tpr"]), fpr_gap=abs(rates[1]["fpr"] - rates[0]["fpr"]),
               acc_g0=rates[0]["acc"], acc_g1=rates[1]["acc"], pos_g0=rates[0]["pos"], pos_g1=rates[1]["pos"],
               soft_dp=float(soft_dp(p, a)))
    out["eo"] = max(out["tpr_gap"], out["fpr_gap"])
    return out

@torch.no_grad()
def probs(model, x):
    if isinstance(model, HistGradientBoostingClassifier):
        return torch.tensor(model.predict_proba(x.cpu().numpy())[:, 1], dtype=torch.float32, device=DEVICE)
    return torch.cat([torch.sigmoid(model(x[i:i + 16384])) for i in range(0, len(x), 16384)])


def craft_attacks(model, x, y, a, space, eps_mult, steps, restarts, kinds, betas=()):
    """Adversarial inputs crafted against `model` (white-box). Returns {attack_name: x_adv}."""
    p = probs(model, x)
    R = ROUND_INTEGER_FEATURES
    adv = {}
    if "acc" in kinds:
        adv["pgd_acc"] = pgd_attack(model, x, obj_ce(y), space, eps_mult, steps, restarts, round_int=R)
    if "dp" in kinds:
        adv["fair_dp"] = pgd_attack(model, x, obj_dir(dp_directions(p, a)), space, eps_mult, steps, restarts, round_int=R)
    if "targeted" in kinds:
        for t in (0, 1):
            adv[f"fair_dp_t{t}"] = pgd_attack(model, x, obj_dir(dp_directions(p, a, t)), space, eps_mult, steps,
                                              restarts, row_scale=(a == t).float(), round_int=R)
    if "eo" in kinds:
        adv["fair_eo"] = pgd_attack(model, x, obj_dir(eo_directions(p, a, y)), space, eps_mult, steps, restarts, round_int=R)
    if "fgsm" in kinds:
        adv["fgsm_acc"] = pgd_attack(model, x, obj_ce(y), space, eps_mult, 1, 1, alpha_frac=1.0, rand_init=False, round_int=R)
    for b in betas:
        adv[f"combined_b{b}"] = pgd_attack(model, x, obj_combined(y, a, dp_directions(p, a), b), space, eps_mult,
                                           steps, restarts, round_int=R)
    return adv


def evaluate(model, data, dataset, run_id, method, source=None, eval_eps=None, full=True):
    """All evaluation attacks of Table in Section 6.6 for one trained model. Returns a list of result rows."""
    x, y, a, space = data.Xte, data.yte, data.ate, data.space
    eval_eps = eval_eps if eval_eps is not None else S["eval_eps"]
    steps, restarts = S["eval_steps"], S["eval_restarts"]
    is_mlp = not isinstance(model, HistGradientBoostingClassifier)
    rows = []
    def add(attack, eps_mult, x_eval, **extra):
        rows.append(dict(dataset=dataset, run=run_id, method=method, attack=attack, eps=eps_mult,
                         **hard_metrics(probs(model, x_eval), y, a), **extra))
    add("clean", 0.0, x)
    for em in [e for e in eval_eps if e > 0]:
        if is_mlp:
            kinds = ["acc", "dp", "targeted", "eo"] + (["fgsm"] if em == 1.0 and full else [])
            betas = S["combined_betas"] if (em == 1.0 and full) else ()
            for name, xa in craft_attacks(model, x, y, a, space, em, steps, restarts, kinds, betas).items():
                add(name, em, xa)
        if source is not None:     # transfer attacks crafted on a separately trained ERM model
            for name, xa in craft_attacks(source, x, y, a, space, em, steps, restarts, ["acc", "dp"]).items():
                add("transfer_" + name, em, xa)
    if is_mlp and full and S["run_certified"] and 1.0 in eval_eps:
        soft, hard = certified_dp(model, x, a, space, 1.0)
        rows.append(dict(dataset=dataset, run=run_id, method=method, attack="certified", eps=1.0,
                         cert_soft_dp=soft, cert_dp=hard))
    return rows

# ======== notebook cell 22 ========
def make_grid(method, budget, seed=0):
    rng = np.random.default_rng(seed)
    L = LAMBDA_LEVELS
    if method == "ERM":
        return [dict()]
    if method == "FAIR_ONLY":
        return [dict(lam_f=float(v)) for v in np.geomspace(*FAIR_ONLY_RANGE, budget)]
    if method == "ROBUST_ONLY":       # lambda_r and the training budget (Section 6.4)
        combos = [dict(lam_r=float(r), train_eps=te) for r, te in
                  itertools.product(np.geomspace(*ROBUST_ONLY_RANGE, 3), TRAIN_EPS_LEVELS)]
    elif method == "FAIR_ADV_CW":     # Chai & Wang: fairness weight and training perturbation level
        combos = [dict(lam_f=f, train_eps=te, adv_only=True) for f, te in itertools.product(L, TRAIN_EPS_LEVELS)]
    elif method == "FAIR_ADV":
        combos = [dict(lam_f=f, lam_r=r) for f, r in itertools.product(L, L)]
    elif method == "TRIFRAME":
        combos = [dict(lam_f=f, lam_r=r, lam_fa=fa) for f, r, fa in itertools.product(L, L, L)]
    else:
        raise ValueError(method)
    if len(combos) > budget:
        combos = [combos[i] for i in sorted(rng.choice(len(combos), budget, replace=False))]
    return combos


def targeted_dp(model, x, y, a, space, steps, eps_mult=1.0):
    p = probs(model, x)
    worst = 0.0
    for t in (0, 1):
        xa = pgd_attack(model, x, obj_dir(dp_directions(p, a, t)), space, eps_mult, steps, 1,
                        row_scale=(a == t).float(), round_int=ROUND_INTEGER_FEATURES)
        worst = max(worst, hard_metrics(probs(model, xa), y, a)["dp"])
    return worst


def tune(dataset, df, runs):
    sel_path = OUT / "tuning" / f"{dataset}_selected.json"
    if sel_path.exists():
        sel = json.load(open(sel_path))["selected"]
        if all(m in sel for m in METHODS):
            return sel
    spec = DATASET_SPECS[dataset]
    run_id, tr, va, te = runs[0]
    data = SplitData(df, tr, va, te, spec, INCLUDE_SENSITIVE_AS_FEATURE)
    tab_path = OUT / "tuning" / f"{dataset}_tuning.csv"
    old = pd.read_csv(tab_path) if tab_path.exists() else pd.DataFrame(columns=["method"])
    rows = old.to_dict("records")
    for method in [m for m in METHODS if m not in set(old["method"])]:
        for ci, cfg in enumerate(make_grid(method, S["tune_budget"])):
            model, info = get_model(cfg, data, 1000, f"tune_{dataset}_{method}_{ci}",
                                    meta=dict(dataset=dataset, run="tuning", method=method))
            val = hard_metrics(probs(model, data.Xva), data.yva, data.ava)
            test = hard_metrics(probs(model, data.Xte), data.yte, data.ate)
            info = {k: v for k, v in info.items() if k != "history"}
            rows.append(dict(method=method, cfg_id=ci, cfg=json.dumps(cfg), **info,
                             val_acc=val["acc"], val_dp_clean=val["dp"],
                             val_dp_targeted=targeted_dp(model, data.Xva, data.yva, data.ava, data.space, S["tune_steps"]),
                             # test values are stored ONLY for the Pareto figure; they are never used for selection
                             test_acc=test["acc"],
                             test_dp_targeted=targeted_dp(model, data.Xte, data.yte, data.ate, data.space, S["tune_steps"])))
            log(f"  tune {dataset} {method} cfg{ci} {cfg}: val_acc={rows[-1]['val_acc']:.4f} "
                f"val_dp_targeted={rows[-1]['val_dp_targeted']:.4f} ({info['epochs']} ep)")
    tab = pd.DataFrame(rows)
    tab.to_csv(tab_path, index=False)
    erm_acc = tab.loc[tab.method == "ERM", "val_acc"].iloc[0]
    selected = {"ERM": {}}
    for method in METHODS[1:]:
        sub = tab[tab.method == method]
        ok = sub[sub.val_acc >= erm_acc - ACC_TOLERANCE]
        pick = (ok.sort_values(["val_dp_targeted", "val_acc"], ascending=[True, False]).iloc[0] if len(ok)
                else sub.sort_values("val_acc", ascending=False).iloc[0])
        selected[method] = json.loads(pick.cfg)
    json.dump(dict(selected=selected, erm_val_acc=erm_acc, rule="lowest val DP under group-targeted attack "
                   f"among configs with val acc >= ERM - {ACC_TOLERANCE}", budget=S["tune_budget"]),
              open(sel_path, "w"), indent=2)
    log(f"Selected for {dataset}: {selected}")
    return selected

# ======== notebook cell 24 ========
RESULT_COLS = ["experiment", "variant", "dataset", "run", "method", "attack", "eps", "acc", "bal_acc", "dp", "eo",
               "tpr_gap", "fpr_gap", "acc_g0", "acc_g1", "pos_g0", "pos_g1", "soft_dp", "cert_soft_dp", "cert_dp",
               "epochs", "sec_per_epoch"]
DONE_PATH = OUT / "done.json"
DONE = set(json.load(open(DONE_PATH))) if DONE_PATH.exists() else set()

def save_rows(rows, name, key):
    path = OUT / f"results_{name}.csv"
    pd.DataFrame(rows).reindex(columns=RESULT_COLS).to_csv(path, mode="a", header=not path.exists(), index=False)
    DONE.add(key)
    json.dump(sorted(DONE), open(DONE_PATH, "w"))

def tag(rows, experiment, variant="", info=None):
    for r in rows:
        r.update(experiment=experiment, variant=variant)
        if info:
            r.update(epochs=info["epochs"], sec_per_epoch=info["sec_per_epoch"])
    return rows

def write_dataset_meta(dataset, df, runs):
    """Table 1 (dataset characteristics) and supplement Table S1 (feasible perturbation set), from split 1."""
    spec = DATASET_SPECS[dataset]
    run_id, tr, va, te = runs[0]
    pre = Preprocessor(spec, INCLUDE_SENSITIVE_AS_FEATURE).fit(df.iloc[tr])
    char = {"Dataset": spec["display"], "Records": len(df), "Raw features": len(spec["numeric"]) + len(spec["categorical"]),
            "Model inputs (encoded)": len(pre.columns), "Positive rate": round(df.y.mean(), 4),
            "Sensitive attribute": spec["sensitive_name"], "Share a=1": round(df.a.mean(), 4),
            "Positive rate a=0": round(df.y[df.a == 0].mean(), 4), "Positive rate a=1": round(df.y[df.a == 1].mean(), 4),
            "Split": "60/20/20 holdout" if spec["split"] == "holdout" else "repeated stratified 5-fold (val = 25% of train)",
            "Runs": len(runs), "Train / val / test (run 1)": f"{len(tr)} / {len(va)} / {len(te)}"}
    feas = []
    for j, c in enumerate(pre.num):
        b = spec["budget"].get(c)
        feas.append({"Dataset": spec["display"], "Feature": c,
                     "Type": "numeric, integer" if c in spec["integer"] else "numeric",
                     "Role": "mutable" if b else "immutable", "Budget (raw units, 1x)": b if b else "",
                     "Budget (SD units, 1x)": round(b / pre.std[j], 3) if b else "",
                     "Valid range (train)": f"[{pre.min[j]:g}, {pre.max[j]:g}]"})
    for c, vals in pre.cats.items():
        feas.append({"Dataset": spec["display"], "Feature": c, "Type": f"categorical ({len(vals)} levels)",
                     "Role": "frozen", "Budget (raw units, 1x)": "", "Budget (SD units, 1x)": "", "Valid range (train)": ""})
    feas.append({"Dataset": spec["display"], "Feature": "sensitive attribute",
                 "Type": spec["sensitive_name"],
                 "Role": "model input (immutable)" if INCLUDE_SENSITIVE_AS_FEATURE else "not a model input",
                 "Budget (raw units, 1x)": "", "Budget (SD units, 1x)": "", "Valid range (train)": ""})
    json.dump(dict(characteristics=char, feasible=feas), open(OUT / "meta" / f"{dataset}.json", "w"), indent=2, default=str)

DATA_CACHE = {}
def get_dataset(name):
    if name not in DATA_CACHE:
        DATA_CACHE[name] = load_dataset(name)
    return DATA_CACHE[name]

def run_main():
    for dataset in S["datasets"]:
        try:
            df = get_dataset(dataset)
        except FileNotFoundError as e:
            log(f"SKIPPING {dataset}: {e}")
            continue
        spec = DATASET_SPECS[dataset]
        runs = make_splits(df, spec["split"], S["n_seeds"], S["sgc_repeats"])
        log(f"=== {spec['display']}: {len(df)} rows, {len(runs)} runs, positive rate {df.y.mean():.3f}, "
            f"group a=1 share {df.a.mean():.3f}")
        write_dataset_meta(dataset, df, runs)
        selected = tune(dataset, df, runs)
        for k, (run_id, tr, va, te) in enumerate(runs):
            todo = [m for m in METHODS + (["GBDT"] if S["run_gbdt"] else []) if f"main|{dataset}|{run_id}|{m}" not in DONE]
            if not todo:
                continue
            data = SplitData(df, tr, va, te, spec, INCLUDE_SENSITIVE_AS_FEATURE)
            source, _ = get_model({}, data, 5000 + k, f"{dataset}_{run_id}_ERMsource",
                                  meta=dict(dataset=dataset, run=run_id, method="ERM_source"))
            for method in [m for m in todo if m != "GBDT"]:
                t0 = time.time()
                model, info = get_model(selected[method], data, 1000 + k, f"{dataset}_{run_id}_{method}",
                                        meta=dict(dataset=dataset, run=run_id, method=method))
                rows = evaluate(model, data, dataset, run_id, method, source=source)
                save_rows(tag(rows, "main", info=info), "main", f"main|{dataset}|{run_id}|{method}")
                clean = rows[0]
                log(f"{dataset} {run_id} {method:12s} acc={clean['acc']:.4f} dp={clean['dp']:.4f} "
                    f"({info['epochs']} ep, {info['sec_per_epoch']:.1f}s/ep, total {time.time() - t0:.0f}s)")
            if "GBDT" in todo:
                gb = HistGradientBoostingClassifier(random_state=k).fit(data.Xtr.cpu().numpy(), data.ytr.cpu().numpy())
                rows = evaluate(gb, data, dataset, run_id, "GBDT", source=source, full=False)
                save_rows(tag(rows, "main"), "main", f"main|{dataset}|{run_id}|GBDT")

run_main()

# ======== notebook cell 26 ========
def ablation_dataset():
    for d in ["adult"] + S["datasets"]:
        try:
            return d, get_dataset(d)
        except FileNotFoundError:
            continue
    return None, None

def selected_cfg(dataset):
    return json.load(open(OUT / "tuning" / f"{dataset}_selected.json"))["selected"]["TRIFRAME"]

ABLATIONS = {
    "TRI-FRAME (soft DP, PGD, constrained)": (dict(), False),
    "Covariance surrogate (v1)": (dict(surrogate="cov"), False),
    "EO surrogate": (dict(surrogate="eo"), False),
    "FGSM training attack": (dict(train_attack="fgsm"), False),
    "Unconstrained training attack": (dict(unconstrained=True), False),
    "Sensitive attribute as input": (dict(), True),
}

def run_ablations():
    dataset, df = ablation_dataset()
    if dataset is None or not (OUT / "tuning" / f"{dataset}_selected.json").exists():
        return
    base = selected_cfg(dataset)
    runs = make_splits(df, DATASET_SPECS[dataset]["split"], max(S["ablation_seeds"], 1), S["sgc_repeats"])[:S["ablation_seeds"]]
    for k, (run_id, tr, va, te) in enumerate(runs):
        for name, (extra, incl) in ABLATIONS.items():
            key = f"ablation|{dataset}|{run_id}|{name}"
            if key in DONE:
                continue
            data = SplitData(df, tr, va, te, DATASET_SPECS[dataset], incl)
            slug = "".join(ch for ch in name if ch.isalnum())[:30]
            model, info = get_model({**base, **extra}, data, 1000 + k, f"abl_{dataset}_{run_id}_{slug}")
            rows = evaluate(model, data, dataset, run_id, "TRIFRAME", eval_eps=[0.0, 1.0], full=True)
            save_rows(tag(rows, "ablation", name, info), "ablation", key)
            log(f"ablation {dataset} {run_id} {name}: acc={rows[0]['acc']:.4f}")

def run_sensitivity():
    dataset, df = ablation_dataset()
    if dataset is None or not (OUT / "tuning" / f"{dataset}_selected.json").exists():
        return
    base = selected_cfg(dataset)
    runs = make_splits(df, DATASET_SPECS[dataset]["split"], max(S["sens_seeds"], 1), S["sgc_repeats"])[:S["sens_seeds"]]
    for k, (run_id, tr, va, te) in enumerate(runs):
        data = SplitData(df, tr, va, te, DATASET_SPECS[dataset], INCLUDE_SENSITIVE_AS_FEATURE)
        for lam_fa in [0.0, 0.1, 0.3, 1.0, 3.0]:
            key = f"sensitivity|{dataset}|{run_id}|{lam_fa}"
            if key in DONE:
                continue
            model, info = get_model({**base, "lam_fa": lam_fa}, data, 1000 + k, f"sens_{dataset}_{run_id}_{lam_fa}")
            rows = evaluate(model, data, dataset, run_id, "TRIFRAME", eval_eps=[0.0, 1.0], full=False)
            save_rows(tag(rows, "sensitivity", f"lam_fa={lam_fa}", info), "sensitivity", key)

class _TargetData:
    def __init__(self, source_data, df):
        t = lambda arr: torch.tensor(arr, device=DEVICE)
        self.Xte = t(source_data.pre.transform(df))
        self.yte, self.ate = t(df.y.values.astype(np.float32)), t(df.a.values.astype(np.float32))
        self.space = source_data.space

def run_acs_shift():
    if "acs_income" not in S["datasets"] or not (OUT / "tuning" / "acs_income_selected.json").exists():
        return
    df = get_dataset("acs_income")
    run_id, tr, va, te = make_splits(df, "holdout", 1, 1)[0]
    data = SplitData(df, tr, va, te, DATASET_SPECS["acs_income"], INCLUDE_SENSITIVE_AS_FEATURE)
    for state in ACS_SHIFT_STATES:
        key = f"shift|{state}"
        if key in DONE:
            continue
        try:
            tdf = load_acs([state])
        except Exception as e:
            log(f"ACS shift: skipping {state}: {e}")
            continue
        target = _TargetData(data, tdf)
        rows = []
        for method in METHODS:
            model, _ = get_model(json.load(open(OUT / "tuning" / "acs_income_selected.json"))["selected"][method],
                                 data, 1000, f"acs_income_{run_id}_{method}")
            rows += evaluate(model, target, "acs_income", f"{ACS_STATES[0]}->{state}", method, eval_eps=[0.0, 1.0], full=False)
        save_rows(tag(rows, "shift", state), "shift", key)
        log(f"ACS shift {ACS_STATES[0]} -> {state} done")

if S["run_ablations"]:
    run_ablations()
if S["run_sensitivity"]:
    run_sensitivity()
if S["run_acs_shift"]:
    run_acs_shift()

# ======== notebook cell 28 ========
def read_results(name):
    path = OUT / f"results_{name}.csv"
    if not path.exists():
        return pd.DataFrame(columns=RESULT_COLS)
    df = pd.read_csv(path)
    return add_targeted_worst(df)

def add_targeted_worst(df):
    """Group-targeted attack, worst case over the attacked group t (headline attack, Section 3.6)."""
    t = df[df.attack.isin(["fair_dp_t0", "fair_dp_t1"])]
    if t.empty:
        return df
    keys = ["experiment", "variant", "dataset", "run", "method", "eps"]
    worst = t.loc[t.groupby(keys, dropna=False)["dp"].idxmax()].copy()
    worst["attack"] = "fair_dp_targeted"
    return pd.concat([df, worst], ignore_index=True)

def summarize(df, keys=("dataset", "method", "attack", "eps")):
    metrics = ["acc", "bal_acc", "dp", "eo", "soft_dp", "acc_g0", "acc_g1", "cert_soft_dp", "cert_dp", "sec_per_epoch", "epochs"]
    g = df.groupby(list(keys), dropna=False)[metrics]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std")).join(g.count()[["acc"]].rename(columns={"acc": "n"}))
    for m in metrics:
        out[f"{m}_ci95"] = stats.t.ppf(0.975, np.maximum(out["n"] - 1, 1)) * out[f"{m}_std"] / np.sqrt(out["n"])
    return out.reset_index()

def to_markdown(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)

def save_table(df, name):
    df.to_csv(OUT / "tables" / f"{name}.csv", index=False)
    (OUT / "tables" / f"{name}.md").write_text(to_markdown(df), encoding="utf-8")

def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj

HEADLINE = [("clean", 0.0, "acc", "higher"), ("clean", 0.0, "dp", "lower"), ("clean", 0.0, "eo", "lower"),
            ("pgd_acc", 1.0, "acc", "higher"), ("fair_dp", 1.0, "dp", "lower"), ("fair_dp_targeted", 1.0, "dp", "lower"),
            ("fair_eo", 1.0, "eo", "lower"), ("combined_b1.0", 1.0, "dp", "lower"), ("combined_b1.0", 1.0, "acc", "higher")]

def paired_tests(raw):
    out = []
    for dataset in raw.dataset.unique():
        block = []
        for att, eps, metric, better in HEADLINE:
            sub = raw[(raw.dataset == dataset) & (raw.attack == att) & (raw.eps == eps)]
            if sub.empty:
                continue
            pv = sub.pivot_table(index="run", columns="method", values=metric)
            if "TRIFRAME" not in pv:
                continue
            for base in [m for m in METHODS if m != "TRIFRAME" and m in pv]:
                diff = (pv["TRIFRAME"] - pv[base]).dropna()
                n = len(diff)
                if n < 2:
                    continue
                sd = diff.std(ddof=1)
                t_p = stats.ttest_rel(pv.loc[diff.index, "TRIFRAME"], pv.loc[diff.index, base]).pvalue if sd > 0 else 1.0
                w_p = stats.wilcoxon(diff).pvalue if (n >= 5 and (diff != 0).any()) else np.nan
                half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
                favours = (diff.mean() > 0) == (better == "higher")
                block.append(dict(dataset=dataset, attack=att, eps=eps, metric=metric, better=better,
                                  baseline=METHOD_LABELS[base], n=n, mean_diff_TRI_minus_base=diff.mean(),
                                  ci95_low=diff.mean() - half, ci95_high=diff.mean() + half,
                                  cohens_dz=diff.mean() / sd if sd > 0 else np.nan,
                                  p_ttest=t_p, p_wilcoxon=w_p, favours_TRIFRAME=bool(favours)))
        if block:
            b = pd.DataFrame(block)
            b["p_holm"] = holm(b["p_ttest"].fillna(1.0))
            b["significant_holm_0.05"] = b["p_holm"] < 0.05
            out.append(b)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

def fmt(m, s, digits=4):
    return "" if pd.isna(m) else (f"{m:.{digits}f} ± {s:.{digits}f}" if not pd.isna(s) else f"{m:.{digits}f}")

def cell(summ, dataset, method, attack, eps, metric):
    r = summ[(summ.dataset == dataset) & (summ.method == method) & (summ.attack == attack) & (np.isclose(summ.eps, eps))]
    return fmt(r[f"{metric}_mean"].iloc[0], r[f"{metric}_std"].iloc[0]) if len(r) else ""

raw = read_results("main")
if raw.empty:
    log("No main results yet.")
else:
    summ = summarize(raw)
    save_table(summ, "summary_all_attacks")
    tests = paired_tests(raw)
    if not tests.empty:
        save_table(tests, "paired_tests_holm")
    sig = lambda d, m, att, met: ("†" if not tests.empty and ((tests.dataset == d) & (tests.baseline == METHOD_LABELS[m])
                                  & (tests.attack == att) & (tests.metric == met) & tests["significant_holm_0.05"]).any() else "")
    max_eps = max(S["eval_eps"])
    t2, t2b, timing = [], [], []
    for d in raw.dataset.unique():
        for m in METHODS + ["GBDT"]:
            if m not in raw[raw.dataset == d].method.unique():
                continue
            name = DATASET_SPECS[d]["display"]
            if m != "GBDT":
                t2.append({"Dataset": name, "Method": METHOD_LABELS[m],
                           "Clean acc": cell(summ, d, m, "clean", 0, "acc") + sig(d, m, "clean", "acc"),
                           "Clean DP": cell(summ, d, m, "clean", 0, "dp") + sig(d, m, "clean", "dp"),
                           "Clean EO": cell(summ, d, m, "clean", 0, "eo") + sig(d, m, "clean", "eo"),
                           "Robust acc (PGD)": cell(summ, d, m, "pgd_acc", 1, "acc") + sig(d, m, "pgd_acc", "acc"),
                           "DP under fairness attack": cell(summ, d, m, "fair_dp", 1, "dp") + sig(d, m, "fair_dp", "dp"),
                           "EO under fairness attack": cell(summ, d, m, "fair_dp", 1, "eo"),
                           "DP under group-targeted attack": cell(summ, d, m, "fair_dp_targeted", 1, "dp")
                           + sig(d, m, "fair_dp_targeted", "dp")})
                timing.append({"Dataset": name, "Method": METHOD_LABELS[m],
                               "Seconds per epoch": cell(summ, d, m, "clean", 0, "sec_per_epoch"),
                               "Epochs (early stopping)": cell(summ, d, m, "clean", 0, "epochs")})
            t2b.append({"Dataset": name, "Method": METHOD_LABELS[m],
                        "DP group-targeted t=0": cell(summ, d, m, "fair_dp_t0", 1, "dp"),
                        "DP group-targeted t=1": cell(summ, d, m, "fair_dp_t1", 1, "dp"),
                        "EO under EO-targeted attack": cell(summ, d, m, "fair_eo", 1, "eo") + sig(d, m, "fair_eo", "eo"),
                        "Acc, combined (beta=1)": cell(summ, d, m, "combined_b1.0", 1, "acc"),
                        "DP, combined (beta=1)": cell(summ, d, m, "combined_b1.0", 1, "dp") + sig(d, m, "combined_b1.0", "dp"),
                        f"DP at {max_eps:g}x budget": cell(summ, d, m, "fair_dp", max_eps, "dp"),
                        "Acc under transfer attack": cell(summ, d, m, "transfer_pgd_acc", 1, "acc"),
                        "DP under transfer fairness attack": cell(summ, d, m, "transfer_fair_dp", 1, "dp")})
    save_table(pd.DataFrame(t2), "table2_main_results")
    save_table(pd.DataFrame(t2b), "table2b_heldout_attacks")
    save_table(pd.DataFrame(timing), "table_cost")
    sel = [{"Dataset": DATASET_SPECS[f.name.replace("_selected.json", "")]["display"], "Method": METHOD_LABELS[m], "Selected": json.dumps(c)}
           for f in sorted((OUT / "tuning").glob("*_selected.json")) for m, c in json.load(open(f))["selected"].items()]
    save_table(pd.DataFrame(sel), "table_selected_hyperparameters")
    print("\nTable 2 (main results, base budget; † = significant vs TRI-FRAME after Holm correction)")
    print(to_markdown(pd.DataFrame(t2)))
    print("\nTable 2b (held-out attacks)")
    print(to_markdown(pd.DataFrame(t2b)))

# ======== notebook cell 30 ========
TOL = 0.005
def sanity_checks(summ):
    rows = []
    for d in summ.dataset.unique():
        for m in METHODS:
            s = summ[(summ.dataset == d) & (summ.method == m)]
            if s.empty:
                continue
            get = lambda att, e, met: s[(s.attack == att) & np.isclose(s.eps, e)][f"{met}_mean"]
            curve = [get("clean", 0, "acc").iloc[0]] + [get("pgd_acc", e, "acc").iloc[0] for e in S["eval_eps"] if e > 0
                                                         and len(get("pgd_acc", e, "acc"))]
            mono = all(b <= a + TOL for a, b in zip(curve, curve[1:]))
            pgd, fgsm = get("pgd_acc", 1, "acc"), get("fgsm_acc", 1, "acc")
            wb, tr = get("pgd_acc", 1, "acc"), get("transfer_pgd_acc", 1, "acc")
            wbf, trf = get("fair_dp", 1, "dp"), get("transfer_fair_dp", 1, "dp")
            emp_soft, cert_soft = get("fair_dp", 1, "soft_dp"), get("certified", 1, "cert_soft_dp")
            emp_hard, cert_hard = get("fair_dp", 1, "dp"), get("certified", 1, "cert_dp")
            ok = lambda cond, *xs: ("n/a" if any(len(x) == 0 for x in xs) else ("pass" if cond() else "FAIL"))
            rows.append({"Dataset": DATASET_SPECS[d]["display"], "Method": METHOD_LABELS[m],
                         "Acc falls with eps": "pass" if mono else "FAIL",
                         "PGD >= FGSM": ok(lambda: pgd.iloc[0] <= fgsm.iloc[0] + TOL, pgd, fgsm),
                         "White-box >= transfer (acc)": ok(lambda: wb.iloc[0] <= tr.iloc[0] + TOL, wb, tr),
                         "White-box >= transfer (DP)": ok(lambda: wbf.iloc[0] >= trf.iloc[0] - TOL, wbf, trf),
                         "Empirical soft DP <= certified": ok(lambda: emp_soft.iloc[0] <= cert_soft.iloc[0] + 1e-6, emp_soft, cert_soft),
                         "Empirical DP <= certified": ok(lambda: emp_hard.iloc[0] <= cert_hard.iloc[0] + 1e-6, emp_hard, cert_hard),
                         "Empirical DP (attack)": round(float(emp_hard.iloc[0]), 4) if len(emp_hard) else np.nan,
                         "Certified DP": round(float(cert_hard.iloc[0]), 4) if len(cert_hard) else np.nan})
    return pd.DataFrame(rows)

if not raw.empty:
    checks = sanity_checks(summ)
    save_table(checks, "table_sanity_checks")
    print(to_markdown(checks))

# ======== notebook cell 32 ========
metas = [json.load(open(f)) for f in sorted((OUT / "meta").glob("*.json"))]
if metas:
    t1 = pd.DataFrame([m["characteristics"] for m in metas])
    s1 = pd.DataFrame([row for m in metas for row in m["feasible"]])
    save_table(t1, "table1_datasets")
    save_table(s1, "table_s1_feasible_set")
    print("Table 1\n" + to_markdown(t1))
    print("\nTable S1 (mutable features only)\n" + to_markdown(s1[s1.Role == "mutable"]))

if not raw.empty:
    s6 = []
    for d in raw.dataset.unique():
        for m in METHODS:
            for att, e, label in [("clean", 0.0, "clean"), ("pgd_acc", 1.0, "PGD accuracy attack"),
                                  ("fair_dp_targeted", 1.0, "group-targeted fairness attack")]:
                r = summ[(summ.dataset == d) & (summ.method == m) & (summ.attack == att) & np.isclose(summ.eps, e)]
                if r.empty:
                    continue
                r = r.iloc[0]
                s6.append({"Dataset": DATASET_SPECS[d]["display"], "Method": METHOD_LABELS[m], "Condition": label,
                           "Accuracy a=0": fmt(r.acc_g0_mean, r.acc_g0_std), "Accuracy a=1": fmt(r.acc_g1_mean, r.acc_g1_std),
                           "Gap (a=1 minus a=0)": f"{r.acc_g1_mean - r.acc_g0_mean:+.4f}"})
    save_table(pd.DataFrame(s6), "table_s6_per_group_accuracy")

# ======== notebook cell 34 ========
curve_files = [f for f in (OUT / "curves").glob("*.csv")]
curves = pd.concat([pd.read_csv(f) for f in curve_files], ignore_index=True) if curve_files else pd.DataFrame()
if not curves.empty and "dataset" in curves:
    main_curves = curves[(curves.run != "tuning") & curves.method.isin(METHODS)]
    ds = list(main_curves.dataset.unique())
    fig, axes = plt.subplots(2, len(ds), figsize=(4.2 * len(ds), 6.2), squeeze=False)
    ep_rows = []
    for j, d in enumerate(ds):
        for m in METHODS:
            c = main_curves[(main_curves.dataset == d) & (main_curves.method == m)]
            if c.empty:
                continue
            g = c.groupby("epoch")
            axes[0][j].plot(g.val_acc.mean().index, g.val_acc.mean().values, color=COLORS[m], marker=MARKERS[m],
                            ms=3, label=METHOD_LABELS[m])
            axes[1][j].plot(g.val_objective.mean().index, g.val_objective.mean().values, color=COLORS[m],
                            marker=MARKERS[m], ms=3)
            per_run = pd.DataFrame({"best": c.loc[c.groupby("key").val_objective.idxmin()].set_index("key").epoch,
                                    "ran": c.groupby("key").epoch.max()})
            ep_rows.append({"Dataset": DATASET_SPECS[d]["display"], "Method": METHOD_LABELS[m], "Runs": len(per_run),
                            "Best epoch (mean ± SD)": fmt(per_run.best.mean(), per_run.best.std(), 1),
                            "Epochs run (mean)": round(per_run.ran.mean(), 1), "Max epochs": S["max_epochs"]})
        axes[0][j].set_title(DATASET_SPECS[d]["display"])
        axes[0][j].set_ylabel("Validation accuracy (clean)")
        axes[1][j].set_ylabel("Validation objective")
        axes[1][j].set_xlabel("Epoch")
        for ax in (axes[0][j], axes[1][j]):
            ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_s5_training_curves.png", dpi=200)
    plt.close('all')
    save_table(pd.DataFrame(ep_rows), "table_s5_early_stopping")
    print(to_markdown(pd.DataFrame(ep_rows)))

# ======== notebook cell 36 ========
def curve_plot(summ, attack, metric, ylabel, fname, include_clean=True):
    ds = list(summ.dataset.unique())
    fig, axes = plt.subplots(1, len(ds), figsize=(4.2 * len(ds), 3.4), squeeze=False)
    for ax, d in zip(axes[0], ds):
        for m in METHODS:
            s = summ[(summ.dataset == d) & (summ.method == m) & (summ.attack == attack)].sort_values("eps")
            if include_clean:
                s = pd.concat([summ[(summ.dataset == d) & (summ.method == m) & (summ.attack == "clean")], s])
            if s.empty:
                continue
            ax.plot(s.eps, s[f"{metric}_mean"], marker=MARKERS[m], color=COLORS[m], label=METHOD_LABELS[m], lw=1.5, ms=5)
            ax.fill_between(s.eps, s[f"{metric}_mean"] - s[f"{metric}_ci95"].fillna(0),
                            s[f"{metric}_mean"] + s[f"{metric}_ci95"].fillna(0), color=COLORS[m], alpha=0.12)
        ax.set_title(DATASET_SPECS[d]["display"])
        ax.set_xlabel("Budget multiplier (x base budget)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / fname, dpi=200)
    plt.close('all')

def hypervolume(acc, dp, ref_acc, ref_dp):
    """Area dominated by points (maximise acc, minimise dp) relative to reference (ref_acc, ref_dp)."""
    pts = sorted([(a, d) for a, d in zip(acc, dp) if a > ref_acc and d < ref_dp], key=lambda t: -t[0])
    hv, best_dp = 0.0, ref_dp
    for a, d in pts:
        if d < best_dp:
            hv += (a - ref_acc) * (best_dp - d)
            best_dp = d
    return hv

def pareto_plot(dataset, split="val"):
    f = OUT / "tuning" / f"{dataset}_tuning.csv"
    if not f.exists():
        return
    t = pd.read_csv(f)
    acc, dp = t[f"{split}_acc"], t[f"{split}_dp_targeted"]
    ref_acc, ref_dp = acc.min() - 0.005, dp.max() + 0.005
    fig, ax = plt.subplots(figsize=(5, 4))
    for m in METHODS:
        s = t[t.method == m]
        hv = hypervolume(s[f"{split}_acc"], s[f"{split}_dp_targeted"], ref_acc, ref_dp)
        ax.scatter(s[f"{split}_dp_targeted"], s[f"{split}_acc"], color=COLORS[m], marker=MARKERS[m],
                   label=f"{METHOD_LABELS[m]} (HV={hv:.4f})", s=40)
    ax.set_xlabel("DP gap under group-targeted attack")
    ax.set_ylabel("Clean accuracy")
    ax.set_title(f"{DATASET_SPECS[dataset]['display']}: tuning configurations ({split})")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / f"fig2_pareto_{dataset}_{split}.png", dpi=200)
    plt.close('all')

if not raw.empty:
    curve_plot(summ, "fair_dp_targeted", "dp", "DP gap (group-targeted attack)", "fig1_dp_targeted_vs_eps.png")
    curve_plot(summ, "fair_dp", "dp", "DP gap (full-population attack)", "fig1b_dp_full_vs_eps.png")
    curve_plot(summ, "pgd_acc", "acc", "Accuracy under PGD", "fig_robust_acc_vs_eps.png")
    for d in raw.dataset.unique():
        pareto_plot(d, "val")

# ======== notebook cell 38 ========
abl = read_results("ablation")
if not abl.empty:
    sa = summarize(abl, ("variant", "attack", "eps"))
    rows = []
    for v in ABLATIONS:
        g = lambda att, met: fmt(*sa[(sa.variant == v) & (sa.attack == att)][[f"{met}_mean", f"{met}_std"]].iloc[0]) \
            if len(sa[(sa.variant == v) & (sa.attack == att)]) else ""
        rows.append({"Variant": v, "Clean acc": g("clean", "acc"), "Clean DP": g("clean", "dp"),
                     "Robust acc (PGD)": g("pgd_acc", "acc"), "DP, group-targeted": g("fair_dp_targeted", "dp"),
                     "EO, EO-targeted": g("fair_eo", "eo")})
    save_table(pd.DataFrame(rows), "table_ablations")
    print(to_markdown(pd.DataFrame(rows)))

sens = read_results("sensitivity")
if not sens.empty:
    ss = summarize(sens, ("variant", "attack", "eps"))
    ss["lam_fa"] = ss.variant.str.replace("lam_fa=", "").astype(float)
    fig, ax1 = plt.subplots(figsize=(5, 3.5))
    a_ = ss[ss.attack == "fair_dp_targeted"].sort_values("lam_fa")
    c_ = ss[ss.attack == "clean"].sort_values("lam_fa")
    ax1.errorbar(a_.lam_fa, a_.dp_mean, yerr=a_.dp_ci95, marker="o", color="#d62728", label="DP, group-targeted")
    ax1.set_xscale("symlog", linthresh=0.1)
    ax1.set_xlabel("lambda_fa")
    ax1.set_ylabel("DP gap under attack", color="#d62728")
    ax2 = ax1.twinx()
    ax2.errorbar(c_.lam_fa, c_.acc_mean, yerr=c_.acc_ci95, marker="s", color="#1f77b4")
    ax2.set_ylabel("Clean accuracy", color="#1f77b4")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_sensitivity_lambda_fa.png", dpi=200)
    plt.close('all')
    save_table(ss[["lam_fa", "attack", "acc_mean", "acc_std", "dp_mean", "dp_std", "n"]].sort_values(["attack", "lam_fa"]),
               "table_sensitivity_lambda_fa")

shift = read_results("shift")
if not shift.empty:
    sh = summarize(shift, ("variant", "method", "attack", "eps"))
    save_table(sh[["variant", "method", "attack", "acc_mean", "dp_mean", "eo_mean"]], "table_acs_cross_state")

# ======== notebook cell 40 ========
def pilot_decision(raw, dataset="adult"):
    r = raw[raw.dataset == dataset]
    if r.empty:
        return "Adult results not available; run the pilot first."
    lines = [f"# Pilot decision ({DATASET_SPECS[dataset]['display']}, {r.run.nunique()} runs, mode = {MODE})", ""]
    erm = r[r.method == "ERM"]
    clean = erm[erm.attack == "clean"].dp.mean()
    lines.append(f"ERM clean DP gap: {clean:.4f}")
    passes1 = False
    for e in sorted(x for x in S["eval_eps"] if x > 0):
        att = erm[(erm.attack == "fair_dp_targeted") & np.isclose(erm.eps, e)].dp.mean()
        full = erm[(erm.attack == "fair_dp") & np.isclose(erm.eps, e)].dp.mean()
        inc = att - clean
        lines.append(f"- budget {e:g}x: DP under group-targeted attack {att:.4f} (+{inc:.4f}), full-population {full:.4f}")
        if e == 1.0:
            passes1 = inc >= PILOT_MIN_DP_INCREASE_ABS and inc >= PILOT_MIN_DP_INCREASE_REL * max(clean, 1e-9)
    lines += ["", f"**Check 1 (attack is meaningful at base budget): {'YES' if passes1 else 'NO'}**", ""]
    t = r[(r.attack == "fair_dp_targeted") & np.isclose(r.eps, 1.0)].pivot_table(index="run", columns="method", values="dp")
    passes2, need = False, None
    fa = [m for m in ("FAIR_ADV", "FAIR_ADV_CW") if m in t.columns]
    if "TRIFRAME" in t.columns and fa:
        base = min(fa, key=lambda m: t[m].mean())             # compare against the STRONGER fair-adv baseline
        for m in fa:
            lines.append(f"- {METHOD_LABELS[m]}: DP under group-targeted attack {t[m].mean():.4f}")
        diff = (t[base] - t["TRIFRAME"]).dropna()              # positive = TRI-FRAME has the smaller gap
        md, sd = diff.mean(), diff.std(ddof=1) if len(diff) > 1 else float("nan")
        passes2 = bool(md > 0 and (np.isnan(sd) or md > sd))
        need = int(math.ceil(((1.96 + 0.84) * sd / md) ** 2)) if (md > 0 and sd > 0) else None
        lines.append(f"TRI-FRAME {t['TRIFRAME'].mean():.4f} vs strongest baseline ({METHOD_LABELS[base]}) "
                     f"{t[base].mean():.4f}; mean paired improvement {md:.4f}, seed SD {sd:.4f}")
        lines.append(f"Estimated runs needed to detect this effect (paired, alpha 0.05, power 0.8): {need}")
    lines += ["", f"**Check 2 (TRI-FRAME beats fair adversarial training beyond seed noise): {'YES' if passes2 else 'NO'}**", ""]
    if passes1 and passes2:
        verdict = "Method paper: run MODE = 'full' with the complete template."
    elif passes1:
        verdict = ("Empirical / threat-model paper: keep TRI-FRAME as one evaluated defense, not the headline. "
                   "Still run MODE = 'full' for the evidence.")
    else:
        verdict = ("Reframe around RQ1 (realistic constraints limit fairness attacks), or widen the threat model "
                   "(budgets in DATASET_SPECS) with a written justification, then re-run the pilot.")
    lines.append(f"**Decision:** {verdict}")
    text = "\n".join(lines)
    (OUT / "pilot_decision.md").write_text(text, encoding="utf-8")
    return text

if not raw.empty:
    print(pilot_decision(raw))

# ======== notebook cell 42 ========
for p in sorted(OUT.rglob("*")):
    if p.is_file() and "models" not in p.parts:
        print(p.relative_to(BASE))

log("ALL CELLS FINISHED")
import sys as _s
_s.exit(0)
