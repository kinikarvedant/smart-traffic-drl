# notebooks/plot_results.py - figures for the README.
# Run from the repo root after training and eval:  python notebooks/plot_results.py
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)
BLUE, GREY, INK, MUTED = "#2a78d6", "#a3a29c", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8, "axes.axisbelow": True,
})

# --- training curve -------------------------------------------------------
mon = pd.read_csv("results/ppo_v2_logs/monitor.csv", skiprows=1)
mon["steps"] = mon["l"].cumsum()
summary = pd.read_csv("results/eval_summary.csv").query("demand == 'regular'").set_index("controller")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, col, title in [(axes[0], "r", "Episode reward (negative pressure)"),
                       (axes[1], "mean_wait", "Mean waiting time per vehicle (s)")]:
    ax.plot(mon["steps"], mon[col], "o", ms=3, color=BLUE, alpha=0.3, label="episode")
    ax.plot(mon["steps"], mon[col].rolling(10, min_periods=1).mean(), color=BLUE, lw=2, label="10-episode mean")
    ax.set_title(title, loc="left", color=INK)
    ax.set_xlabel("Training steps (one step = one 5 s decision)")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
for name in summary.index.drop("PPO"):
    y = summary.loc[name, "mean_wait_mean"]
    axes[1].axhline(y, color=GREY, lw=1, ls="--")
    axes[1].annotate(f"{name} (eval)", (mon["steps"].iloc[-1], y), xytext=(0, 3),
                     textcoords="offset points", ha="right", va="bottom", fontsize=8, color=MUTED)
axes[0].legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "training_curve.png", dpi=150)
print("Saved", OUT / "training_curve.png")

# --- PPO vs fixed-time ----------------------------------------------------
# Throughput is left to the table: at this demand every controller clears the traffic,
# so it differs by a few vehicles at most.
ep = pd.read_csv("results/eval_episodes.csv")
order = list(dict.fromkeys(ep["controller"]))
demands = {"regular": "Regular arrivals (training demand)", "random": "Random arrivals, same mean (never trained on)"}
panels = [("mean_wait", "Mean wait per vehicle (s)"), ("mean_queue", "Mean queue (vehicles)"),
          ("max_queue", "Max queue (vehicles)")]
fig, axes = plt.subplots(2, 3, figsize=(10, 7), sharey="col")
for row, (demand, label) in zip(axes, demands.items()):
    d = ep[ep["demand"] == demand]
    for ax, (col, title) in zip(row, panels):
        stats = d.groupby("controller")[col].agg(["mean", "std"]).loc[order]
        x = range(len(order))
        ax.bar(x, stats["mean"], yerr=stats["std"], color=[BLUE if c == "PPO" else GREY for c in order],
               width=0.65, capsize=4, error_kw={"ecolor": INK, "elinewidth": 1})
        for i, c in enumerate(order):
            seeds = d.loc[d["controller"] == c, col]
            ax.plot([i] * len(seeds), seeds, "o", ms=3.5, color=INK, alpha=0.55, zorder=3)
            ax.annotate(f"{stats.loc[c, 'mean']:.2f}" if col != "max_queue" else f"{stats.loc[c, 'mean']:.1f}",
                        (i, max(stats.loc[c, "mean"] + stats.loc[c, "std"], seeds.max())), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=8, color=INK)
        ax.set_xticks(list(x), [c.replace("Fixed-time ", "Fixed\n") for c in order], fontsize=8)
        ax.set_title(title, loc="left", color=INK, fontsize=10)
    row[0].annotate(label, (0, 1.16), xycoords="axes fraction", fontsize=11, fontweight="bold",
                    color=INK).set_in_layout(False)  # keep tight_layout from widening column 1
fig.suptitle(f"PPO vs fixed-time control, {ep['seed'].nunique()} seeds each "
             "(bar = mean, line = ±1 std, dots = individual seeds; lower is better)",
             x=0.01, ha="left", color=MUTED, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=4)  # room for the headings
fig.savefig(OUT / "ppo_vs_fixed_time.png", dpi=150)
print("Saved", OUT / "ppo_vs_fixed_time.png")
