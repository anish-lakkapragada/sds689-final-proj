"""
Analyze the 500 runs in grokking-llc project. 
Time: October 27th, 2025
"""

# %% 
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle 

from dotenv import load_dotenv

load_dotenv()

from wandb import Api
from tqdm import tqdm 


def _safe_get(row, key, default=np.nan):
    v = row.get(key, default)
    return v if v is not None else default

def fetch_timeseries_for_run(api: Api, run_full_path: str):
    run = api.run(run_full_path)

    steps, train_loss, train_acc, val_loss, val_acc = [], [], [], [], []
    for row in run.scan_history(keys=["step", "_step", "train_loss", "train_acc", "val_loss", "val_acc"]):
        step_val = _safe_get(row, "step", np.nan)
        if (isinstance(step_val, float) and math.isnan(step_val)) or step_val is None:
            step_val = _safe_get(row, "_step", np.nan)

        steps.append(step_val)
        train_loss.append(_safe_get(row, "train_loss"))
        train_acc.append(_safe_get(row, "train_acc"))
        val_loss.append(_safe_get(row, "val_loss"))
        val_acc.append(_safe_get(row, "val_acc"))

    steps = np.array(steps, dtype=float)
    order = np.argsort(steps) if steps.size else np.array([], dtype=int)

    return {
        "step":       steps[order] if steps.size else np.array([], dtype=float),
        "train_loss": np.array(train_loss, dtype=float)[order] if steps.size else np.array([], dtype=float),
        "train_acc":  np.array(train_acc,  dtype=float)[order] if steps.size else np.array([], dtype=float),
        "val_loss":   np.array(val_loss,   dtype=float)[order] if steps.size else np.array([], dtype=float),
        "val_acc":    np.array(val_acc,    dtype=float)[order] if steps.size else np.array([], dtype=float),
    }

def fetch_summary_for_run(api: Api, run_full_path: str):
    run = api.run(run_full_path)
    s = run.summary
    cfg = dict(run.config)

    def g(k):
        return s[k] if k in s and s[k] is not None else np.nan

    ft = g("first_train_idx")
    fv = g("first_val_idx")
    time_to_grok = (fv - ft) if (not np.isnan(ft) and not np.isnan(fv)) else np.nan

    pre_mean = g("llc_mean_pre_grok")
    post_mean = g("llc_mean_post_grok")
    llc_jump = (post_mean - pre_mean) if (not np.isnan(pre_mean) and not np.isnan(post_mean)) else np.nan

    return {
        "run_name": run.name,
        "first_train_idx": float(ft) if not np.isnan(ft) else np.nan,
        "first_val_idx": float(fv) if not np.isnan(fv) else np.nan,
        "time_to_grok": float(time_to_grok) if not np.isnan(time_to_grok) else np.nan,
        "llc_mean_pre_grok": float(pre_mean) if not np.isnan(pre_mean) else np.nan,
        "llc_mean_post_grok": float(post_mean) if not np.isnan(post_mean) else np.nan,
        "llc_std_pre_grok": float(g("llc_std_pre_grok")) if not np.isnan(g("llc_std_pre_grok")) else np.nan,
        "llc_std_post_grok": float(g("llc_std_post_grok")) if not np.isnan(g("llc_std_post_grok")) else np.nan,
        "llc_jump": float(llc_jump) if not np.isnan(llc_jump) else np.nan,
        "config": cfg,
        "run_id": run.id,
        "entity": run.entity,
        "project": run.project,
    }

def load_all_runs(project_path: str, state_filter=("finished", "crashed", "failed", "running")):
    """
    project_path = "entity/project"  (e.g., "anish/grokking-llc")
    Returns:
      curves_by_run: { run_name -> {step, train_loss, train_acc, val_loss, val_acc} }
      summary_table: dict of np arrays (aligned across runs)
      configs_by_run: { run_name -> config dict }
      ids_by_run: { run_name -> "<entity>/<project>/<run_id>" }
    """
    api = Api()
    runs = api.runs(project_path, filters={"state": {"$in": list(state_filter)}})

    curves_by_run = {}
    summaries = []
    configs_by_run = {}
    ids_by_run = {}

    for run in tqdm(runs):
        full = f"{run.entity}/{run.project}/{run.id}"

        ts = fetch_timeseries_for_run(api, full)
        curves_by_run[run.name] = ts
        ids_by_run[run.name] = full

        sm = fetch_summary_for_run(api, full)
        summaries.append(sm)
        configs_by_run[run.name] = sm["config"]

    def col(key, dtype=float):
        vals = []
        for sm in summaries:
            v = sm[key]
            if dtype is object:
                vals.append(v)
            else:
                vals.append(np.nan if (not isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))) else v)
        return np.array(vals, dtype=dtype)

    summary_table = {
        "run_name":           col("run_name", dtype=object),
        "first_train_idx":    col("first_train_idx"),
        "first_val_idx":      col("first_val_idx"),
        "time_to_grok":       col("time_to_grok"),
        "llc_mean_pre_grok":  col("llc_mean_pre_grok"),
        "llc_mean_post_grok": col("llc_mean_post_grok"),
        "llc_std_pre_grok":   col("llc_std_pre_grok"),
        "llc_std_post_grok":  col("llc_std_post_grok"),
        "llc_jump":           col("llc_jump"),
    }

    return curves_by_run, summary_table, configs_by_run, ids_by_run

# %% 

def plot_acc(curves_by_run, run_name):
    c = curves_by_run[run_name]
    plt.figure()
    plt.plot(c["step"], c["train_acc"], label="train_acc")
    plt.plot(c["step"], c["val_acc"], label="val_acc")
    plt.title(f"{run_name} • accuracy vs step")
    plt.xlabel("step")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_loss(curves_by_run, run_name):
    c = curves_by_run[run_name]
    plt.figure()
    plt.plot(c["step"], c["train_loss"], label="train_loss")
    plt.plot(c["step"], c["val_loss"], label="val_loss")
    plt.title(f"{run_name} • loss vs step")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.grid(True)
    plt.legend()
    plt.show()

# %%%
def plot_llc_vs_time(summary_table):
    t = summary_table["time_to_grok"]
    j = summary_table["llc_jump"]
    mask = ~np.isnan(t) & ~np.isnan(j) & (t > 0)
    if not np.any(mask):
        print("No runs with both grokking and LLC info.")
        return np.array([]), np.array([])
    x = np.log(t[mask])
    y = j[mask]
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel("log(time_to_grok)")
    plt.ylabel("LLC jump (post - pre)")
    plt.title("LLC change vs grokking time")
    plt.grid(True)
    plt.show()
    return x, y

with open("data/wandb-500-runs.pkl", "rb") as f: 
    info = pickle.load(f)
    curves_by_run, summary_table, configs_by_run, ids_by_run = info["curves_by_run"], info["summary_table"], info["configs_by_run"], info["ids_by_run"]
    # mask = ~np.isnan(summary_table["time_to_grok"])
    # summary_table["time_to_grok"] = summary_table["time_to_grok"][mask]
    # summary_table["llc_jump"] = summary_table["llc_jump"][mask]

# plot_llc_vs_time(summary_table)
# %%

from sklearn.linear_model import LinearRegression

def plot_free_energy_quantities(curves_by_run, summary_table, free_energy_quantity="free_energy"): 
    N = 0.4 * 53 ** 2

    x_list = []  
    y_list = []

    # Build aligned (x, y) pairs per valid run
    for idx, run in enumerate(summary_table['run_name']):
        try:
            llc_jump = float(summary_table["llc_jump"][idx])
            t_grok   = float(summary_table["time_to_grok"][idx])
            pre_idx  = summary_table['first_train_idx'][idx]
            post_idx = summary_table['first_val_idx'][idx]

            # validity checks
            if np.isnan(llc_jump) or np.isnan(t_grok) or t_grok <= 0:
                continue
            if np.isnan(pre_idx) or np.isnan(post_idx):
                continue

            pre_idx  = int(pre_idx)
            post_idx = int(post_idx)


            # pull the training-loss curve and compute ΔL_n
            L_n_over_time = curves_by_run[run]['train_loss']
            if pre_idx < 0 or post_idx < 0 or pre_idx >= len(L_n_over_time) or post_idx >= len(L_n_over_time):
                continue

            delta_L_n = L_n_over_time[post_idx] - L_n_over_time[pre_idx]

            if free_energy_quantity == "llc_jump": 
                y_list.append(llc_jump)
            elif free_energy_quantity == "loss_jump": 
                y_list.append(delta_L_n)
            elif free_energy_quantity == "free_energy": 
                delta_F_ij = np.log(N) * llc_jump + N * delta_L_n
                y_list.append(delta_F_ij)      # y = ΔF_{i->j}

            x_list.append(np.log(t_grok))  # x = log r_{i->j}
    
        except Exception:
            # skip any run with unexpected structure
            continue

    x = np.array(x_list, dtype=float)
    y = np.array(y_list, dtype=float)

    if x.size < 2:
        print("Not enough valid runs with grokking time and LLC jump to fit a model.")
        return
    
    X = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)

    a = float(model.coef_[0])
    b = float(model.intercept_)
    r2 = float(model.score(X, y))

    xs = np.linspace(x.min(), x.max(), 200)
    ys = model.predict(xs.reshape(-1, 1))

    latex_label = {
        "free_energy": "\\Delta F_{{i \\to j}}", 
        "llc_jump": "\\Delta \lambda_{{i \\to j}}", 
        "loss_jump": "\\Delta L_n"
    }[free_energy_quantity]

    plt.figure(figsize=(6.8, 4.6))
    plt.scatter(x, y, s=26, alpha=0.75, label="runs")
    plt.plot(xs, ys, linewidth=2.2,
             label=fr"fit: ${latex_label} = {a:.3g}\,\log r_{{i \to j}} {b:.3g}$")

    plt.xlabel(r"$\log(r_{i \to j})$")
    plt.ylabel(fr"${latex_label}$")
    plt.title(fr"${latex_label}$ vs. $\log r_{{i \to j}}$  ($R^2 = {r2:.3f}$)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    
plot_free_energy_quantities(curves_by_run, summary_table, free_energy_quantity="free_energy")
plot_free_energy_quantities(curves_by_run, summary_table, free_energy_quantity="llc_jump")
plot_free_energy_quantities(curves_by_run, summary_table, free_energy_quantity="loss_jump")
# %%
