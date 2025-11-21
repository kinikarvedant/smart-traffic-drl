# notebooks/plot_results.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

Path("results").mkdir(parents=True, exist_ok=True)

rl = pd.read_csv("results/eval_metrics.csv")
base = pd.read_csv("results/baseline_metrics.csv")

plt.figure(figsize=(6,4))
plt.bar(rl["run"].astype(str), rl["total_wait"])
plt.title("RL: total wait per run")
plt.ylabel("Total wait")
plt.xlabel("Run")
plt.savefig("results/rl_wait.png")
print("Saved results/rl_wait.png")

plt.figure(figsize=(6,4))
plt.plot(base["green"], base["avg_wait"], marker="o")
plt.title("Baseline avg wait vs green time")
plt.xlabel("Green time (s)")
plt.ylabel("Average waiting time")
plt.savefig("results/baseline_wait.png")
print("Saved results/baseline_wait.png")
