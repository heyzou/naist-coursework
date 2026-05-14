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


def euler(A, h, T):
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
    if not np.all(np.isfinite(xs)):
        return float("inf")
    ex = np.array([exact(float(t)) for t in ts])
    if not np.all(np.isfinite(ex)):
        return float("inf")
    return float(np.max(np.abs(xs - ex)))


A1 = np.array([[0.0, 1.0], [-1.0, -3.0]], dtype=float)
A2 = np.array([[0.0, 1.0], [-1000.0, -1001.0]], dtype=float)

T1 = 10.0
T2 = 0.1
h_list1 = [0.8, 0.5, 0.2, 0.1]
h_list2 = [0.0021, 0.002, 0.0019]

rows1 = []
for h in h_list1:
    t1, x1 = euler(A1, h, T1)
    e1 = max_error(t1, x1, exact1)
    rows1.append((h, e1))

rows2 = []
for h in h_list2:
    t2, x2 = euler(A2, h, T2)
    e2 = max_error(t2, x2, exact2)
    rows2.append((h, e2))

with (GEN / "task34_table_sys1_rows.tex").open("w", encoding="utf-8") as f:
    for h, e1 in rows1:
        f.write(f"{h:g} & {e1:.6e} \\\\\n")

with (GEN / "task34_table_sys2_rows.tex").open("w", encoding="utf-8") as f:
    for h, e2 in rows2:
        f.write(f"{h:g} & {e2:.6e} \\\\\n")

colors1 = plt.cm.tab10(np.linspace(0.0, 1.0, len(h_list1)))
colors2 = plt.cm.tab10(np.linspace(0.0, 1.0, len(h_list2)))

# system1 plot
te1 = np.linspace(0, T1, 1500)
te2 = np.linspace(0, T2, 1500)
fig1, ax1 = plt.subplots(figsize=(8.8, 4.6))
ax1.plot(te1, [exact1(float(t)) for t in te1], label="Exact", linewidth=2.6, color="red")
for i, h in enumerate(h_list1):
    t, x = euler(A1, h, T1)
    n = len(t)
    markevery = max(1, n // 25)
    marker = "o" if h >= 0.05 else None
    ax1.plot(
        t,
        x,
        linewidth=1.2,
        linestyle="-",
        color=colors1[i],
        marker=marker,
        markersize=2.8,
        markevery=markevery,
        alpha=0.95,
        label=f"Euler h={h:g}",
    )
ax1.set_title("")
ax1.set_xlabel("t")
ax1.set_ylabel("x(t)")
ax1.set_xlim(0.0, T1)
ax1.set_ylim(-1.0, 1.2)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=8, ncol=3)
fig1.tight_layout()
fig1.savefig(IMG / "task34_system1.png", dpi=180)
plt.close(fig1)

# system2 plot
fig2, ax2 = plt.subplots(figsize=(8.8, 4.6))
ax2.plot(te2, [exact2(float(t)) for t in te2], label="Exact", linewidth=2.6, color="red")
for i, h in enumerate(h_list2):
    t, x = euler(A2, h, T2)
    n = len(t)
    markevery = max(1, n // 25)
    marker = "o" if h >= 0.05 else None
    ax2.plot(
        t,
        x,
        linewidth=1.2,
        linestyle="-",
        color=colors2[i],
        marker=marker,
        markersize=2.8,
        markevery=markevery,
        alpha=0.95,
        label=f"Euler h={h:g}",
    )
ax2.set_xlim(0.0, T2)
ax2.set_ylim(0.5, 1.1)
ax2.set_title("")
ax2.set_xlabel("t")
ax2.set_ylabel("x(t)")
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8, ncol=3)
fig2.tight_layout()
fig2.savefig(IMG / "task34_system2.png", dpi=180)
plt.close(fig2)

print(
    "generated:",
    GEN / "task34_table_sys1_rows.tex",
    GEN / "task34_table_sys2_rows.tex",
    IMG / "task34_system1.png",
    IMG / "task34_system2.png",
)
