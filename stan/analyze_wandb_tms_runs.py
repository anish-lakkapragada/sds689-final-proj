"""
Analyze transitions in the tms-study runs.

For each run, we:
  - reconstruct epoch-wise curves for train/loss, train/acc, llc/mean, llc/std
  - detect steep downward transitions in train/loss
  - define "states" as the segments between steep transitions
  - build Transition objects describing each state

We also aggregate across runs:
  - time_to_transition           (epoch gap between consecutive states)
  - delta_train_loss_between    (change in mean train/loss between states)
  - delta_llc_mean_between      (change in mean llc/mean between states)
"""

# %%
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from wandb import Api
from tqdm import tqdm

load_dotenv()


# ------------------------ W&B loading helpers ------------------------ #

def _safe_get(row, key, default=np.nan):
    v = row.get(key, default)
    return v if v is not None else default


def fetch_timeseries_for_run(api: Api, run_full_path: str):
    """
    For a given run, build epoch-aligned timeseries:

    Returns dict with keys:
      "epoch", "train_loss", "train_acc", "llc_mean", "llc_std"
    """
    run = api.run(run_full_path)

    # bucket rows by epoch, because metrics/LLC are logged in separate rows
    buckets = {}  # epoch -> {train_loss: [...], train_acc: [...], llc_mean: [...], llc_std: [...]}

    for row in run.scan_history(keys=["epoch", "train/loss", "train/acc", "llc/mean", "llc/std"]):
        epoch_val = _safe_get(row, "epoch", np.nan)
        if (isinstance(epoch_val, float) and math.isnan(epoch_val)) or epoch_val is None:
            continue

        e = int(epoch_val)
        if e not in buckets:
            buckets[e] = {
                "train_loss": [],
                "train_acc": [],
                "llc_mean": [],
                "llc_std": [],
            }

        for wb_key, local_key in [
            ("train/loss", "train_loss"),
            ("train/acc",  "train_acc"),
            ("llc/mean",   "llc_mean"),
            ("llc/std",    "llc_std"),
        ]:
            v = _safe_get(row, wb_key, np.nan)
            if not (isinstance(v, float) and math.isnan(v)):
                buckets[e][local_key].append(float(v))

    if not buckets:
        return {
            "epoch":      np.array([], dtype=float),
            "train_loss": np.array([], dtype=float),
            "train_acc":  np.array([], dtype=float),
            "llc_mean":   np.array([], dtype=float),
            "llc_std":    np.array([], dtype=float),
        }

    epochs_sorted = np.array(sorted(buckets.keys()), dtype=float)

    def agg(key):
        vals = []
        for e in sorted(buckets.keys()):
            arr = np.array(buckets[e][key], dtype=float)
            if arr.size == 0:
                vals.append(np.nan)
            else:
                vals.append(float(arr.mean()))
        return np.array(vals, dtype=float)

    return {
        "epoch":      epochs_sorted,
        "train_loss": agg("train_loss"),
        "train_acc":  agg("train_acc"),
        "llc_mean":   agg("llc_mean"),
        "llc_std":    agg("llc_std"),
    }


def load_all_runs_tms(project_path: str, state_filter=("finished", "crashed", "failed", "running")):
    """
    project_path = "entity/project"  (e.g., "anish/tms-study")

    Returns:
      curves_by_run: { run_name -> {epoch, train_loss, train_acc, llc_mean, llc_std} }
      ids_by_run:   { run_name -> "<entity>/<project>/<run_id>" }
    """
    api = Api()
    runs = api.runs(project_path, filters={"state": {"$in": list(state_filter)}})

    curves_by_run: Dict[str, dict] = {}
    ids_by_run: Dict[str, str] = {}

    for run in tqdm(runs):
        full = f"{run.entity}/{run.project}/{run.id}"

        ts = fetch_timeseries_for_run(api, full)
        curves_by_run[run.name] = ts
        ids_by_run[run.name] = full

    return curves_by_run, ids_by_run


# ------------------------ Transition detection ------------------------ #

@dataclass
class Transition:
    """One state between steep transitions in train loss."""
    num_steps: int            # number of epochs in this state (end_epoch - start_epoch)
    delta_train_loss: float   # train_loss(end) - train_loss(start)
    mean_llc_std: float       # average llc/std in this state
    mean_llc_mean: float      # average llc/mean in this state
    mean_train_loss: float    # average train/loss in this state
    mean_train_acc: float     # average train/acc in this state


def _nanmean_or_nan(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0 or np.all(np.isnan(x)):
        return float("nan")
    return float(np.nanmean(x))


def _build_transition(
    epochs: np.ndarray,
    losses: np.ndarray,
    acc: np.ndarray,
    llc_mean: np.ndarray,
    llc_std: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> Transition:
    s = slice(start_idx, end_idx + 1)

    e_seg = epochs[s]
    loss_seg = losses[s]
    acc_seg = acc[s]
    llc_mean_seg = llc_mean[s]
    llc_std_seg = llc_std[s]

    start_epoch = float(e_seg[0])
    end_epoch = float(e_seg[-1])

    num_steps = int(end_epoch - start_epoch)
    delta_train_loss = float(loss_seg[-1] - loss_seg[0])

    return Transition(
        num_steps=num_steps,
        delta_train_loss=delta_train_loss,
        mean_llc_std=_nanmean_or_nan(llc_std_seg),
        mean_llc_mean=_nanmean_or_nan(llc_mean_seg),
        mean_train_loss=_nanmean_or_nan(loss_seg),
        mean_train_acc=_nanmean_or_nan(acc_seg),
    )

def detect_transitions_for_curve(
    curve: dict,
    drop_frac: float = 0.10,
    window: int = 10,
) -> Tuple[List[Transition], List[int]]:
    """
    Given a single run's curve dict, detect steep *downward* transitions and return:
      - transitions: list of Transition objects (states between steep drops)
      - start_epochs: integer epoch for the start of each state

    Heuristic:
      - compute the total downward movement loss[0] - min(loss)
      - define a minimum "significant" drop: min_drop = drop_frac * total_drop
      - slide a window over the curve and compare mean loss BEFORE vs AFTER:
            drop_score(i) = mean(loss[i-window : i]) - mean(loss[i : i+window])
        (positive = downward transition around i)
      - find contiguous regions where drop_score >= min_drop and, in each region,
        keep the index with the largest drop_score as the boundary.
      - fall back to the old per-step diff heuristic if the curve is too short.
    """
    epochs = np.asarray(curve["epoch"], dtype=float)
    losses = np.asarray(curve["train_loss"], dtype=float)
    acc = np.asarray(curve["train_acc"], dtype=float)
    llc_mean = np.asarray(curve["llc_mean"], dtype=float)
    llc_std = np.asarray(curve["llc_std"], dtype=float)

    # keep only points where epoch AND loss are finite
    mask = (~np.isnan(epochs)) & (~np.isnan(losses))
    if not np.any(mask):
        return [], []

    epochs = epochs[mask]
    losses = losses[mask]
    acc = acc[mask]
    llc_mean = llc_mean[mask]
    llc_std = llc_std[mask]

    # sort by epoch (W&B can be a bit messy)
    order = np.argsort(epochs)
    epochs = epochs[order]
    losses = losses[order]
    acc = acc[order]
    llc_mean = llc_mean[order]
    llc_std = llc_std[order]

    n = len(epochs)
    if n < 2:
        tr = _build_transition(epochs, losses, acc, llc_mean, llc_std, 0, n - 1)
        return [tr], [int(epochs[0])]

    # total downward movement over the run
    total_drop = losses[0] - np.nanmin(losses)
    if (not np.isfinite(total_drop)) or total_drop <= 0:
        # no net drop: treat as single state
        tr = _build_transition(epochs, losses, acc, llc_mean, llc_std, 0, n - 1)
        return [tr], [int(epochs[0])]

    min_drop = drop_frac * total_drop

    # ---------------- windowed drop scores ---------------- #
    win = max(int(window), 5)  # you can tune this; 20–50 works well for long runs
    boundaries: List[int] = []

    if 2 * win >= n:
        # Too few points for windowing; fall back to per-step diffs
        diffs = losses[:-1] - losses[1:]      # positive = loss decreases
        steep_indices = np.where(diffs >= min_drop)[0] + 1
        if steep_indices.size == 0:
            tr = _build_transition(epochs, losses, acc, llc_mean, llc_std, 0, n - 1)
            return [tr], [int(epochs[0])]
        boundaries = [int(i) for i in steep_indices]
    else:
        # drop_score(i) = mean(loss before window) - mean(loss after window)
        # positive & large => strong downward transition around i
        drop_scores = np.zeros(n, dtype=float)
        for i in range(win, n - win):
            before = float(np.nanmean(losses[i - win : i]))
            after = float(np.nanmean(losses[i : i + win]))
            drop_scores[i] = before - after   # only downward changes are positive

        mask_drop = drop_scores >= min_drop

        # group contiguous regions where drop_scores >= min_drop
        # and pick the index with the largest drop_score in each region
        candidate_idxs: List[int] = []
        i = 0
        while i < n:
            if not mask_drop[i]:
                i += 1
                continue
            j = i
            while j < n and mask_drop[j]:
                j += 1
            region = slice(i, j)
            rel = int(np.nanargmax(drop_scores[region]))
            best_idx = i + rel
            candidate_idxs.append(best_idx)
            i = j

        if not candidate_idxs:
            tr = _build_transition(epochs, losses, acc, llc_mean, llc_std, 0, n - 1)
            return [tr], [int(epochs[0])]

        boundaries = candidate_idxs

    # ensure we include first & last point and everything is ordered
    boundaries = sorted(set(b for b in boundaries if 0 < b < n - 1))
    boundaries = [0] + boundaries + [n - 1]

    transitions: List[Transition] = []
    start_epochs: List[int] = []

    for start_idx, end_idx in zip(boundaries[:-1], boundaries[1:]):
        if end_idx <= start_idx:
            continue
        tr = _build_transition(epochs, losses, acc, llc_mean, llc_std, start_idx, end_idx)
        transitions.append(tr)
        start_epochs.append(int(epochs[start_idx]))

    if not transitions:
        tr = _build_transition(epochs, losses, acc, llc_mean, llc_std, 0, n - 1)
        return [tr], [int(epochs[0])]

    return transitions, start_epochs
    
def compute_all_transitions(
    curves_by_run: Dict[str, dict],
    ids_by_run: Dict[str, str],
    drop_frac: float = 0.10,
):
    """
    Loop over all runs and compute:
      - transitions_by_run_id: {run_id -> [Transition, ...]}
      - time_to_transition:           np.array of epoch gaps between consecutive states
      - delta_train_loss_between:     np.array of Δ(mean train_loss) between consecutive states
      - delta_llc_mean_between:       np.array of Δ(mean llc_mean) between consecutive states
    """
    transitions_by_run_id: Dict[str, List[Transition]] = {}
    time_to_transition: List[float] = []
    delta_train_loss_between: List[float] = []
    delta_llc_mean_between: List[float] = []

    for run_name, curve in curves_by_run.items():
        transitions, start_epochs = detect_transitions_for_curve(curve, drop_frac=drop_frac)
        if not transitions:
            continue

        run_id = ids_by_run.get(run_name, run_name)
        transitions_by_run_id[run_id] = transitions

        # build arrays over consecutive states
        for i in range(len(transitions) - 1):
            dt = start_epochs[i + 1] - start_epochs[i]
            time_to_transition.append(float(dt))

            d_loss = transitions[i + 1].mean_train_loss - transitions[i].mean_train_loss
            d_llc = transitions[i + 1].mean_llc_mean - transitions[i].mean_llc_mean

            delta_train_loss_between.append(float(d_loss))
            delta_llc_mean_between.append(float(d_llc))

    return (
        transitions_by_run_id,
        np.array(time_to_transition, dtype=float),
        np.array(delta_train_loss_between, dtype=float),
        np.array(delta_llc_mean_between, dtype=float),
    )


# ------------------------ Visualization helpers ------------------------ #
# %% 
def plot_transition_histograms(
    time_to_transition: np.ndarray,
    delta_train_loss_between: np.ndarray,
    delta_llc_mean_between: np.ndarray,
):
    """Quick 1D histograms of the consecutive-transition stats."""
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))

    axs[0].hist(time_to_transition, bins=30)
    axs[0].set_title("Time between transitions")
    axs[0].set_xlabel("Δ epoch")
    axs[0].set_ylabel("# of transitions")
    axs[0].grid(True, alpha=0.3)

    axs[1].hist(delta_train_loss_between, bins=30)
    axs[1].set_title("Δ mean train loss")
    axs[1].set_xlabel("Δ mean train/loss (next - prev)")
    axs[1].grid(True, alpha=0.3)

    axs[2].hist(delta_llc_mean_between, bins=30)
    axs[2].set_title("Δ mean LLC")
    axs[2].set_xlabel("Δ mean llc/mean (next - prev)")
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_transition_scatter(
    time_to_transition: np.ndarray,
    delta_train_loss_between: np.ndarray,
    delta_llc_mean_between: np.ndarray,
):
    """Scatter plots: how Δloss and ΔLLC relate to time between transitions."""
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))

    axs[0].scatter(np.log(time_to_transition), delta_train_loss_between, alpha=0.7, s=20)
    axs[0].set_xlabel("Log Time between transitions (Δ epoch)")
    axs[0].set_ylabel("Δ mean train/loss")
    axs[0].set_title("Δ mean train/loss vs time between transitions")
    axs[0].grid(True, alpha=0.3)

    axs[1].scatter(np.log(time_to_transition), delta_llc_mean_between, alpha=0.7, s=20)
    axs[1].set_xlabel("Log Time between transitions (Δ epoch)")
    axs[1].set_ylabel("Δ mean llc/mean")
    axs[1].set_title("Δ mean llc/mean vs time between transitions")
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_free_energy_vs_transition_time(
    time_to_transition: np.ndarray,
    delta_train_loss_between: np.ndarray,
    delta_llc_mean_between: np.ndarray,
    N: float = 0.4 * 53 ** 2,
):
    """
    Plot Δ free energy between consecutive transitions vs log(time between transitions).

    ΔF = log(N) * Δ(llc/mean) + N * Δ(train/loss)

    Also overlays linear and quadratic fits using sklearn.linear_model.LinearRegression.
    """
    from sklearn.linear_model import LinearRegression

    mask = (
        (time_to_transition > 0)
        & np.isfinite(time_to_transition)
        & np.isfinite(delta_train_loss_between)
        & np.isfinite(delta_llc_mean_between)
    )
    
    if not np.any(mask):
        print("No valid transitions to plot.")
        return

    x = np.log(time_to_transition[mask])
    y = np.log(N) * delta_llc_mean_between[mask] + N * delta_train_loss_between[mask]

    if x.size < 2:
        print("Fewer than 2 valid transitions; plotting scatter only.")
        plt.figure(figsize=(6.8, 4.6))
        plt.scatter(x, y, s=26, alpha=0.8, label="transitions")
        plt.xlabel(r"$\log(\Delta \text{epoch})$")
        plt.ylabel(r"$\Delta F$")
        plt.title(r"$\Delta F$ vs. $\log(\Delta \text{epoch})$")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        return

    # linear fit: y = a1 x + b1
    X_lin = x.reshape(-1, 1)
    lin_model = LinearRegression()
    lin_model.fit(X_lin, y)
    a1 = float(lin_model.coef_[0])
    b1 = float(lin_model.intercept_)

    # quadratic fit: y = a2 x^2 + b2 x + c2
    quad_model = None
    have_quad = x.size >= 3
    if have_quad:
        X_quad = np.stack([x ** 2, x], axis=1)
        quad_model = LinearRegression()
        quad_model.fit(X_quad, y)
        a2 = float(quad_model.coef_[0])
        b2 = float(quad_model.coef_[1])
        c2 = float(quad_model.intercept_)

    xs = np.linspace(x.min(), x.max(), 200)
    y_lin = lin_model.predict(xs.reshape(-1, 1))
    if have_quad:
        Xs_quad = np.stack([xs ** 2, xs], axis=1)
        y_quad = quad_model.predict(Xs_quad)

    plt.figure(figsize=(6.8, 4.6))
    plt.scatter(x, y, s=26, alpha=0.75, label="transitions")
    plt.plot(xs, y_lin, linewidth=2.0,
             label=fr"linear fit: $y = {a1:.3g} x {b1:+.3g}$")
    if have_quad:
        plt.plot(xs, y_quad, linewidth=2.0, linestyle="--",
                 label=fr"quadratic fit: $y = {a2:.3g} x^2 {b2:+.3g} x {c2:+.3g}$")

    plt.xlabel(r"$\log(\Delta \text{epoch})$")
    plt.ylabel(r"$\Delta F$")
    plt.title(r"$\Delta F$ vs. $\log(\Delta \text{epoch})$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# %%

PROJECT_PATH = "anishlk/tms-study"  # "entity/project" for your tms runs

curves_by_run, ids_by_run = load_all_runs_tms(PROJECT_PATH)

(
    transitions_by_run_id,
    time_to_transition,
    delta_train_loss_between,
    delta_llc_mean_between,
) = compute_all_transitions(curves_by_run, ids_by_run, drop_frac=0.05)

# %% 
plot_transition_histograms(time_to_transition, delta_train_loss_between, delta_llc_mean_between)
plot_transition_scatter(time_to_transition, delta_train_loss_between, delta_llc_mean_between)
plot_free_energy_vs_transition_time(time_to_transition, delta_train_loss_between, delta_llc_mean_between)
# %%
