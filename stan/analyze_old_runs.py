# %%
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

@dataclass
class ExperimentParams:
    p: int = 53
    n_batches: int = 25000
    n_save_model_checkpoints: int = 100
    lr: float = 0.005
    label_noise : float = 0.0
    batch_size: int = 128
    hidden_size: int = 48
    embed_dim: int = 12
    train_frac: float = 0.4
    # the shown grokking / llc curve behavior is robust to change of seed from my experiments, but not all seeds show grokking withying the first 100 checkpoints, NB!
    random_seed: int = 0
    device: str = "DEVICE"
    weight_decay: float = 0.0002

with open("iid-runs-300.pkl", "rb") as f:
    info = pickle.load(f)

info = np.array(info)

time_to_grok = np.array(info[:, 2] - info[:, 1], dtype=np.float64)
free_energy_diff = np.array(info[:, 4] - info[:, 3], dtype=np.float64)

x = np.log(time_to_grok)
X = x.reshape(-1, 1)
y = free_energy_diff

lin = LinearRegression()
lin.fit(X, y)
y_pred_lin = lin.predict(X)
r2_lin = r2_score(y, y_pred_lin)

x_plot = np.linspace(x.min(), x.max(), 400).reshape(-1, 1)
y_plot_lin = lin.predict(x_plot)

quad = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("lin", LinearRegression())
])
quad.fit(X, y)
y_pred_quad = quad.predict(X)
r2_quad = r2_score(y, y_pred_quad)

y_plot_quad = quad.predict(x_plot)

print("Linear model: y = a * log_time + b")
print(f"  a (slope) = {lin.coef_[0]:.6f}, b (intercept) = {lin.intercept_:.6f}")
coef_quad = quad.named_steps["lin"].coef_
intercept_quad = quad.named_steps["lin"].intercept_
a1, a2 = coef_quad[0], coef_quad[1]
print("Quadratic model: y = a1 * log_time + a2 * log_time^2 + b")
print(f"  a1 = {a1:.6f}, a2 = {a2:.6f}, b = {intercept_quad:.6f}")


plt.figure(figsize=(7.5, 5.0))
plt.scatter(x, y, s=16, alpha=0.75, label="data")
plt.plot(x_plot.ravel(), y_plot_lin, linewidth=2,
         label=f"Linear fit  (R² = {r2_lin:.3f})")
plt.plot(x_plot.ravel(), y_plot_quad, linewidth=2,
         label=f"Quadratic fit (R² = {r2_quad:.3f})")

plt.xlabel(r"$\log$(time to grok)")
plt.ylabel(r"$\Delta F$")
plt.title(r"Free Energy Difference vs. Time To Grok (Log) for 60 identical Modulo Models ($p = 53$)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
# %%
with open("label-noise-runs-500.pkl", "rb") as f:
    info = pickle.load(f)

info = np.array(info)
label_noises = np.array([exp.label_noise for exp in info[:, 0]], dtype=np.float32)
time_to_grok = np.array(info[:, 2] - info[:, 1], dtype=np.float64)
free_energy_diff = np.array(info[:, 4] - info[:, 3], dtype=np.float64)
# %%
