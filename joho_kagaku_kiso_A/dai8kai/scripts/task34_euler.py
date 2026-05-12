import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "generated"
IMG = ROOT / "img"
GEN.mkdir(exist_ok=True)
IMG.mkdir(exist_ok=True)


def exact1(t):
    c1 = (5 + 3 * math.sqrt(5)) / 10
    c2 = (5 - 3 * math.sqrt(5)) / 10
    l1 = (-3 + math.sqrt(5)) / 2
    l2 = (-3 - math.sqrt(5)) / 2
    return c1 * math.exp(l1 * t) + c2 * math.exp(l2 * t)


def exact2(t):
    return (1000 / 999) * math.exp(-t) - (1 / 999) * math.exp(-1000 * t)


def euler(A, h, T=3.0):
    n = int(round(T / h))
    y = np.array([1.0, 0.0], dtype=float)
    xs = [1.0]
    ts = [0.0]
    for k in range(n):
        y = y + h * (A @ y)
        t = (k + 1) * h
        ts.append(t)
        xs.append(float(y[0]))
    return np.array(ts), np.array(xs)


def max_error(ts, xs, exact):
    ex = np.array([exact(float(t)) for t in ts])
    return float(np.max(np.abs(xs - ex)))


A1 = np.array([[0.0, 1.0], [-1.0, -3.0]], dtype=float)
A2 = np.array([[0.0, 1.0], [-1000.0, -1001.0]], dtype=float)

h_list = [0.5, 0.2, 0.1, 0.05, 0.01, 0.002, 0.001]

rows = []
for h in h_list:
    t1, x1 = euler(A1, h)
    e1 = max_error(t1, x1, exact1)

    t2, x2 = euler(A2, h)
    e2 = max_error(t2, x2, exact2)

    rows.append((h, e1, e2))

with (GEN / "task34_table_rows.tex").open("w", encoding="utf-8") as f:
    for h, e1, e2 in rows:
        f.write(f"{h:g} & {e1:.6e} & {e2:.6e} \\\\\n")

# plot
fig, axes = plt.subplots(2, 1, figsize=(8.0, 9.6))

# system1
te = np.linspace(0, 3, 1200)
axes[0].plot(te, [exact1(float(t)) for t in te], label="Exact", linewidth=2.2)
for h in [0.5, 0.1, 0.01]:
    t, x = euler(A1, h)
    axes[0].plot(t, x, marker="o", markersize=3, linewidth=1.0, label=f"Euler h={h:g}")
axes[0].set_title("System (1): stable for wider h")
axes[0].set_xlabel("t")
axes[0].set_ylabel("x(t)")
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=8)

# system2
axes[1].plot(te, [exact2(float(t)) for t in te], label="Exact", linewidth=2.2)
for h in [0.01, 0.002, 0.001]:
    t, x = euler(A2, h)
    axes[1].plot(t, x, marker="o", markersize=2.5, linewidth=1.0, label=f"Euler h={h:g}")
axes[1].set_ylim(-0.2, 1.2)
axes[1].set_title("System (2): strict step-size limit")
axes[1].set_xlabel("t")
axes[1].set_ylabel("x(t)")
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=8)

fig.tight_layout()
fig.savefig(IMG / "task34_compare.png", dpi=180)
print("generated:", GEN / "task34_table_rows.tex", IMG / "task34_compare.png")
