# %%
import os
from typing import Callable, Optional, Union
from dataclasses import dataclass
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

import wandb
from devinterp.optim.sgld import SGLD
from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from devinterp.utils import evaluate_mse, default_nbeta

# --------------------- your model & data classes (unchanged) ---------------------
class ToyAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        tied: bool = True,
        final_bias: bool = False,
        hidden_bias: bool = False,
        nonlinearity: Callable = F.relu,
        unit_weights: bool = False,
        standard_magnitude: bool = False,
        initial_bias: Optional[torch.Tensor] = None,
        initial_embed: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nonlinearity = nonlinearity
        self.tied = tied
        self.final_bias = final_bias
        self.unit_weights = unit_weights
        self.standard_magnitude = standard_magnitude

        self.embedding = nn.Linear(self.input_dim, self.hidden_dim, bias=hidden_bias)
        if initial_embed is not None:
            self.embedding.weight.data = initial_embed

        self.unembedding = nn.Linear(self.hidden_dim, self.input_dim, bias=final_bias)
        if initial_bias is not None:
            self.unembedding.bias.data = initial_bias

        if self.standard_magnitude:
            avg_norm = torch.norm(self.embedding.weight.data, p=2, dim=0).mean()
            self.embedding.weight.data = (
                F.normalize(self.embedding.weight.data, p=2, dim=0) * avg_norm
            )

        if self.unit_weights:
            self.embedding.weight.data = F.normalize(
                self.embedding.weight.data, p=2, dim=0
            )

        if tied:
            self.unembedding.weight = torch.nn.Parameter(
                self.embedding.weight.transpose(0, 1)
            )

    def forward(self, x: torch.Tensor):
        if self.unit_weights:
            self.embedding.weight.data = F.normalize(
                self.embedding.weight.data, p=2, dim=0
            )
        if self.standard_magnitude:
            avg_norm = torch.norm(self.embedding.weight.data, p=2, dim=0).mean()
            self.embedding.weight.data = (
                F.normalize(self.embedding.weight.data, p=2, dim=0) * avg_norm
            )
        if self.tied:
            self.unembedding.weight.data = self.embedding.weight.data.transpose(0, 1)

        x = self.embedding(x)
        x = self.unembedding(x)
        x = self.nonlinearity(x)
        return x


class SyntheticDataset(Dataset):
    def __init__(self, num_samples: int, num_features: int, sparsity: Union[float, int]):
        self.num_samples = num_samples
        self.num_features = num_features
        self.sparsity = sparsity
        self.data = self.generate_data()

    def generate_values(self):
        raise NotImplementedError

    def generate_mask(self):
        if isinstance(self.sparsity, float):
            return torch.bernoulli(
                torch.ones((self.num_samples, self.num_features)) * (1 - self.sparsity)
            )
        elif isinstance(self.sparsity, int):
            mask = torch.zeros((self.num_samples, self.num_features))
            for i in range(self.num_samples):
                idx = torch.randperm(self.num_features)[: self.sparsity]
                mask[i, idx] = 1
            return mask
        else:
            raise ValueError(f"Sparsity must be float or int, got {type(self.sparsity)}.")

    def generate_data(self):
        return self.generate_mask() * self.generate_values()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


class SyntheticBinaryValued(SyntheticDataset):
    def generate_values(self):
        return 1.0

# ------------------------------------------------------------------------------
def generate_2d_kgon_vertices(k, rot=0.0, pad_to=None, force_length=0.9):
    theta = np.linspace(0, 2 * np.pi, k, endpoint=False) + rot
    x = np.cos(theta); y = np.sin(theta)
    result = np.vstack((x, y))
    if pad_to is not None and k < pad_to:
        result = np.hstack([result, np.zeros((2, pad_to - k))])
    return result * force_length

def generate_init_param(
    m, n, init_kgon, prior_std=1.0, no_bias=True, init_zerobias=True,
    seed=0, force_negb=False, noise=0.01
):
    np.random.seed(seed)
    if init_kgon is None or m != 2:
        init_W = np.random.normal(size=(m, n)) * prior_std
    else:
        assert init_kgon <= n
        rand_angle = np.random.uniform(0, 2 * np.pi, size=(1,))
        noiseW = np.random.normal(size=(m, n)) * noise
        init_W = generate_2d_kgon_vertices(init_kgon, rot=rand_angle, pad_to=n) + noiseW

    if no_bias:
        return {"W": init_W}
    init_b = np.random.normal(size=(n, 1)) * prior_std
    if force_negb:
        init_b = -np.abs(init_b)
    if init_zerobias:
        init_b = init_b * 0
    return {"W": init_W, "b": init_b}

# %% 
@dataclass
class LLCEstimatorParams:
    lr: float = 5e-5         # SGLD step size (ε)
    gamma: float = 1.0        # localization strength
    num_draws: int = 1000      # steps per chain (moderate)
    num_chains: int = 10       # moderate # of chains
    batch_size: Optional[int] = None  # LLC loader batch size (set below)
    nbeta: Optional[float] = None     # inverse temperature; default via default_nbeta


def _compute_llc_for_autoencoder(
    model: nn.Module,
    dataset: SyntheticDataset,
    llc_params: LLCEstimatorParams,
    device: torch.device,
):
    model_copy = deepcopy(model).to(device).eval()

    # Use a moderate batch size for SGLD/LLC (not full-batch to avoid overkill)
    bs = llc_params.batch_size or min(len(dataset), 512)
    x = dataset.data
    loader = DataLoader(TensorDataset(x, x), batch_size=bs, shuffle=True)

    nbeta_val = llc_params.nbeta if llc_params.nbeta is not None else default_nbeta(len(dataset))

    summary = estimate_learning_coeff_with_summary(
        model_copy,
        loader=loader,
        evaluate=evaluate_mse,
        sampling_method=SGLD,
        optimizer_kwargs=dict(
            lr=llc_params.lr,
            nbeta=nbeta_val,
            localization=llc_params.gamma,
        ),
        num_chains=llc_params.num_chains,
        num_draws=llc_params.num_draws,
        device=str(device),
        online=False,
    )
    return float(summary["llc/mean"]), float(summary["llc/std"])

def create_and_train(
    m: int,
    n: int,
    num_samples: int,
    batch_size: int = 100,
    num_epochs: int = 20000,
    lr: float = 0.01,
    device: Optional[torch.device] = None,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    init_kgon: Optional[int] = 2,
    init_zerobias: bool = False,
    prior_std: float = 10.0,
    seed: int = 1,
    # logging/checkpoints
    wandb_project: str = "tms-study",
    run_name: Optional[str] = None,
    metric_linear_n: int = 100,   # train loss/acc at these epochs (linear in epochs)
    llc_linear_n: int = 100,      # LLC at these epochs (linear in epochs)
    llc_log_n: int = 100,         # LLC also at these epochs (log in epochs)
    checkpoint_root: str = "./checkpoints",
    save_artifact: bool = True,
    llc_params: LLCEstimatorParams = LLCEstimatorParams(),
    start_log_epoch: int = 100,   # <-- NEW: don't log anything before this epoch
):
    # pick device
    if device is None:
        device = torch.device(
            "cuda:0" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    torch.manual_seed(seed); np.random.seed(seed)

    # model
    model = ToyAutoencoder(m, n, final_bias=True).to(device)
    init_weights = generate_init_param(
        n, m, init_kgon, no_bias=False, init_zerobias=init_zerobias,
        prior_std=prior_std, seed=seed
    )
    model.embedding.weight.data = torch.from_numpy(init_weights["W"]).float().to(device)
    if "b" in init_weights:
        model.unembedding.bias.data = torch.from_numpy(
            init_weights["b"].flatten()
        ).float().to(device)

    # data
    dataset = SyntheticBinaryValued(num_samples, m, 1)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # optim
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # epoch schedules
    def lin_epochs(T, k):
        k = min(k, T)
        return np.linspace(1, T, num=k, dtype=int).tolist()

    def log_epochs(T, k):
        k = min(k, T)
        xs = np.unique(np.logspace(0, np.log10(T), num=k, dtype=int))
        xs = xs[(xs >= 1) & (xs <= T)]
        return xs.tolist()

    metric_epochs_all = set(lin_epochs(num_epochs, metric_linear_n))
    llc_epochs_all = set(lin_epochs(num_epochs, llc_linear_n)).union(set(log_epochs(num_epochs, llc_log_n)))

    # filter: only log at/after start_log_epoch
    metric_epochs = {e for e in metric_epochs_all if e >= start_log_epoch}
    llc_epochs    = {e for e in llc_epochs_all    if e >= start_log_epoch}

    # nbeta default
    if llc_params.nbeta is None:
        llc_params.nbeta = default_nbeta(num_samples)
    if llc_params.batch_size is None:
        llc_params.batch_size = min(num_samples, 512)

    # wandb
    run = wandb.init(
        project=wandb_project,
        name=run_name,
        config=dict(
            input_dim=m, hidden_dim=n, num_samples=num_samples, batch_size=batch_size,
            num_epochs=num_epochs, lr=lr, momentum=momentum, weight_decay=weight_decay,
            init_kgon=init_kgon, init_zerobias=init_zerobias, prior_std=prior_std,
            seed=seed, device=str(device),
            metric_linear_n=metric_linear_n, llc_linear_n=llc_linear_n, llc_log_n=llc_log_n,
            llc_num_draws=llc_params.num_draws, llc_nbeta=llc_params.nbeta,
            llc_gamma=llc_params.gamma, llc_lr=llc_params.lr,
            start_log_epoch=start_log_epoch,
        ),
        reinit=True,
    )
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("llc/*",   step_metric="epoch")

    if run_name is None:
        run_name = wandb.run.name

    # checkpoints dir
    run_ckpt_dir = os.path.join(checkpoint_root, "tms-study", run_name)
    os.makedirs(run_ckpt_dir, exist_ok=True)

    def evaluate_metrics():
        model.eval()
        with torch.no_grad():
            # full-batch eval for speed (fits easily)
            x = dataset.data.to(device)
            out = model(x)
            loss = criterion(out, x).item()
            acc = (out.round() == x).float().mean().item()
        model.train()
        return loss, acc

    def save_checkpoint(epoch: int):
        path = os.path.join(run_ckpt_dir, f"epoch_{epoch}.pt")
        payload = {
            "state_dict": model.state_dict(),
            "model_kwargs": dict(input_dim=m, hidden_dim=n, tied=True, final_bias=True),
            "epoch": epoch,
            "seed": seed,
        }
        torch.save(payload, path)
        return path

    ckpt_paths = {}
    model.train()
    for epoch in tqdm(range(1, num_epochs + 1), desc="Training by epoch"):
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch)
            loss.backward()
            optimizer.step()

        # metrics: only if epoch in schedule AND >= start_log_epoch
        if epoch in metric_epochs:
            train_loss, train_acc = evaluate_metrics()
            wandb.log({"epoch": epoch, "train/loss": train_loss, "train/acc": train_acc}, step=epoch)

        # LLC: only if epoch in schedule AND >= start_log_epoch
        if epoch in llc_epochs:
            ckpt_path = save_checkpoint(epoch)
            ckpt_paths[epoch] = ckpt_path
            llc_mean, llc_std = _compute_llc_for_autoencoder(model, dataset, llc_params, device)
            wandb.log({"epoch": epoch, "llc/mean": llc_mean, "llc/std": llc_std}, step=epoch)

    # optional: aggregate checkpoints as artifact
    if ckpt_paths and save_artifact:
        art = wandb.Artifact(name=f"{run_name}-checkpoints", type="model")
        for p in ckpt_paths.values():
            art.add_file(p)
        wandb.log_artifact(art)

    wandb.finish()
    return ckpt_paths

# ------------------------ driver (unchanged except params) ----------------------
if __name__ == "__main__":
    NUM_FEATURES = 8
    NUM_HIDDEN_UNITS = 2
    NUM_SAMPLES = 1000
    NUM_EPOCHS = 4500
    INIT_KGON = 2
    N_RUNS = 500

    for run_idx in tqdm(range(N_RUNS)):
        seed = run_idx + 1
        run_name = f"run-{seed:03d}"
        create_and_train(
            m=NUM_FEATURES,
            n=NUM_HIDDEN_UNITS,
            num_samples=NUM_SAMPLES,
            batch_size=100,
            lr=0.01,
            num_epochs=NUM_EPOCHS,
            init_kgon=INIT_KGON,
            init_zerobias=False,
            seed=seed,
            wandb_project="tms-study-run-2",
            run_name=run_name,
            metric_linear_n=100,   # metrics logged only for epochs >= start_log_epoch
            llc_linear_n=100,      # LLC at 100 linear epochs (>= start_log_epoch)
            llc_log_n=100,         # LLC at 100 log epochs (>= start_log_epoch)
            checkpoint_root="./checkpoints",
            save_artifact=True,
            # moderate (not overkill) LLC config
            llc_params=LLCEstimatorParams(
                lr=5e-4, gamma=1.0, num_draws=500, num_chains=10, batch_size=512
            ),
            start_log_epoch=100,   # <-- gating for all WANDB logs
        )

# %%
