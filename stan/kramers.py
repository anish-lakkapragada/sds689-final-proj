# %% 
import os
from dotenv import load_dotenv
load_dotenv()

import wandb

import random
from copy import deepcopy
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np

from devinterp.optim.sgld import SGLD
from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from devinterp.utils import evaluate_ce

mps_works = torch.backends.mps.is_available() and torch.backends.mps.is_built()
if torch.cuda.is_available(): DEVICE = "cuda"
elif mps_works: DEVICE = "mps"
else: DEVICE = "cpu"

@dataclass
class ExperimentParams:
    p: int = 53
    n_batches: int = 25000
    n_save_model_checkpoints: int = 250
    lr: float = 0.005
    label_noise : float = 0.0
    batch_size: int = 128
    hidden_size: int = 48
    embed_dim: int = 12
    train_frac: float = 0.4
    random_seed: int = 0
    device: str = DEVICE
    weight_decay: float = 0.0002


class MLP(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.embedding = nn.Embedding(params.p, params.embed_dim)
        self.linear1r = nn.Linear(params.embed_dim, params.hidden_size, bias=True)
        self.linear1l = nn.Linear(params.embed_dim, params.hidden_size, bias=True)
        self.linear2 = nn.Linear(params.hidden_size, params.p, bias=False)
        self.act = nn.GELU()
        self.vocab_size = params.p

    def forward(self, x):
        x1 = self.embedding(x[..., 0])
        x2 = self.embedding(x[..., 1])
        x1 = self.linear1l(x1)
        x2 = self.linear1r(x2)
        x = x1 + x2
        x = self.act(x)
        x = self.linear2(x)
        return x


def test(model, dataset, device):
    n_correct = 0
    total_loss = 0
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in dataset:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            total_loss += loss.item()

            pred = torch.argmax(out)
            if pred == y:
                n_correct += 1
    return n_correct / len(dataset), total_loss / len(dataset)


def train(train_dataset, test_dataset, params, verbose=True):
    all_models = []
    model = MLP(params).to(params.device)

    optimizer = torch.optim.Adam(
        model.parameters(), weight_decay=params.weight_decay, lr=params.lr
    )
    loss_fn = torch.nn.CrossEntropyLoss()


    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)

    checkpoint_every = None
    if params.n_save_model_checkpoints > 0:
        checkpoint_every = params.n_batches // params.n_save_model_checkpoints
    print_every = checkpoint_every

    if wandb.run is not None:
        wandb.watch(model, log="gradients", log_freq=checkpoint_every or 100)

    loss_data = []
    if verbose:
        pbar = tqdm(total=params.n_batches, desc="Training")

    for i in range(params.n_batches):
        batch = next(iter(train_loader))
        X, Y = batch
        X, Y = X.to(params.device), Y.to(params.device)

        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, Y)
        loss.backward()
        optimizer.step()

        if checkpoint_every and (i + 1) % checkpoint_every == 0:
            all_models += [deepcopy(model)]
            val_acc, val_loss = test(model, test_dataset, params.device)
            train_acc, train_loss = test(model, train_dataset, params.device)

            loss_data.append(
                {
                    "batch": i + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )

            if wandb.run is not None:
                wandb.log(
                    {
                        "step": i + 1,
                        "train_loss": float(train_loss),
                        "train_acc": float(train_acc),
                        "val_loss": float(val_loss),
                        "val_acc": float(val_acc),
                    },
                    step=i + 1,
                )

            if verbose:
                pbar.set_postfix(
                    {
                        "train_loss": f"{train_loss:.4f}",
                        "train_acc": f"{train_acc:.4f}",
                        "val_loss": f"{val_loss:.4f}",
                        "val_acc": f"{val_acc:.4f}",
                    }
                )
                pbar.update(print_every)
    if verbose:
        pbar.close()
    df = pd.DataFrame(loss_data)
    train_acc, train_loss = test(model, train_dataset, params.device)
    val_acc, val_loss = test(model, test_dataset, params.device)
    if verbose:
        print(f"Final Train Acc: {val_acc:.4f} | Final Train Loss: {val_loss:.4f}")
        print(f"Final Val Acc: {val_acc:.4f} | Final Val Loss: {val_loss:.4f}")
    return all_models, df


def deterministic_shuffle(lst, seed):
    rng = random.Random(seed)   # local RNG, does NOT affect global random.*
    lst = list(lst)
    rng.shuffle(lst)
    return lst

def get_all_pairs(p):
    pairs = []
    for i in range(p):
        for j in range(p):
            pairs.append((i, j))
    return set(pairs)


def make_dataset(p, label_noise=0.0):
    data = []
    pairs = get_all_pairs(p)
    for a, b in pairs:
        data.append((torch.tensor([a, b]), torch.tensor((a + b) % p)))

    if label_noise > 0.0: 
        n_noisy = int(label_noise * len(data))
        noisy_indices = random.sample(range(len(data)), n_noisy)
        for idx in noisy_indices:
            a, b = data[idx][0]
            noisy_label = random.randint(0, p - 1)
            while noisy_label == (a.item() + b.item()) % p:
                noisy_label = random.randint(0, p - 1)
            data[idx] = (torch.tensor([a, b]), torch.tensor(noisy_label))
    return data



def train_test_split(dataset, train_split_proportion, seed):
    l = len(dataset)
    train_len = int(train_split_proportion * l)
    idx = list(range(l))
    idx = deterministic_shuffle(idx, seed)
    train_idx = idx[:train_len]
    test_idx = idx[train_len:]
    return [dataset[i] for i in train_idx], [dataset[i] for i in test_idx]


@dataclass
class LLCEstimatorParams:
  lr = 3e-3
  gamma = 5
  nbeta = 2.0
  num_draws = 500

def get_llc_info(model_checkpoint):
  llc_estimation = estimate_learning_coeff_with_summary(
      model_checkpoint,
      loader=DataLoader(train_data, batch_size=params.batch_size, shuffle=True),
      evaluate=evaluate_ce,
      sampling_method=SGLD,
      optimizer_kwargs=dict(lr=LLCEstimatorParams.lr,
                            nbeta=LLCEstimatorParams.nbeta,
                            localization=LLCEstimatorParams.gamma),
      num_chains=3,
      num_draws=LLCEstimatorParams.num_draws,
      device=DEVICE,
      online=False,
  )

  llc_point_estimate = llc_estimation["llc/mean"]
  return llc_point_estimate, llc_estimation["llc/std"]

N_models = 500
start = 54

grok_times_and_energies = []

for model_run in range(start, N_models):
  params = ExperimentParams()
  torch.manual_seed(model_run)

  run = wandb.init(
      project="grokking-llc",
      name=f"run-{model_run}",
      config=vars(params),
      reinit=True,
  )

  dataset = make_dataset(params.p, label_noise=params.label_noise)
  train_data, test_data = train_test_split(dataset, params.train_frac, model_run)

  all_checkpointed_models, history_df = train(
      train_dataset=train_data, test_dataset=test_data, params=params, verbose=False
  )

  train_mask = history_df['train_acc'].ge(1.0 - 1e-12)
  val_mask   = history_df['val_acc'].ge(0.95)

  first_train_idx = train_mask.idxmax() if train_mask.any() else None
  first_val_idx  = val_mask.idxmax()   if val_mask.any()   else None

  print(f"first train: {first_train_idx}, first_val: {first_val_idx}")
  print(first_train_idx, first_val_idx)

  if (not first_train_idx or not first_val_idx) or first_val_idx - first_train_idx <= 20: 
      wandb.finish()
      continue

  llc_mean_post_grok, llc_std_post_grok = get_llc_info(
      all_checkpointed_models[first_val_idx]
  )

  llc_mean_pre_grok, llc_std_pre_grok = get_llc_info(
      all_checkpointed_models[first_train_idx]
  )

  wandb.log({
      "llc_mean_post_grok": float(llc_mean_post_grok),
      "llc_std_post_grok": float(llc_std_post_grok),
      "llc_mean_pre_grok": float(llc_mean_pre_grok),
      "llc_std_pre_grok": float(llc_std_pre_grok),
      "first_train_idx": int(first_train_idx),
      "first_val_idx": int(first_val_idx),
  })

  wandb.finish()
# %%
