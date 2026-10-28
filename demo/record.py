# demo/record.py - record the evaluation episodes behind the browser replay in demo/index.html.
# Run from the repo root:  python demo/record.py
#
# Re-runs PPO and the best fixed-time plan (10 s greens) on seeds 1-5 under both demand patterns,
# through the same env and policies as agents/eval.py, and saves one trace per episode:
# the signal phase, halted / departed / arrived vehicles and every vehicle's position, each second.
# Stops if any episode's metrics differ from its row in results/eval_episodes.csv.
import json
import os
import sys

import pandas as pd
import traci
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.eval import MODEL, SCENARIOS, SEEDS  # noqa: E402
from baselines.fixed_time import FixedTimePolicy  # noqa: E402
from envs.sumo_env import SumoTrafficEnv  # noqa: E402

OUT = os.path.join("demo", "runs")
CONTROLLERS = {"ppo": "PPO", "fixed10": "Fixed-time 10 s"}
ROUTES = "NESW"  # flowN starts at the north end and drives south, and so on (sumo_scenario/simple.rou.xml)
B36 = "0123456789abcdefghijklmnopqrstuvwxyz"


def record(policy, cfg, seed):
    env = SumoTrafficEnv(cfg)
    obs, info = env.reset(seed=seed)
    phase, halted, departed, arrived, veh = [], [], [], [], {}
    sim_step = env._sim_step

    def step_and_record():
        before = env.queue_sum, env.departed, env.arrived
        reward = sim_step()
        frame = env.seconds - 1
        halted.append(env.queue_sum - before[0])
        departed.append(env.departed - before[1])
        arrived.append(env.arrived - before[2])
        phase.append(traci.trafficlight.getPhase(env.tls_id))
        for v in traci.vehicle.getIDList():
            r = ROUTES.index(v.split(".")[0][-1])
            x, y = traci.vehicle.getPosition(v)
            rec = veh.setdefault(v, [r, frame, []])
            assert rec[1] + len(rec[2]) == frame, v  # on the network every second since it entered
            rec[2].append(round(y if r in (0, 2) else x))  # the coordinate that changes along its route
        return reward

    env._sim_step = step_and_record
    done = False
    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
    env.close()
    enc = lambda xs: "".join(B36[x] for x in xs)
    return info, {"seconds": len(phase), "phase": enc(phase), "halted": enc(halted), "departed": enc(departed),
                  "arrived": enc(arrived), "vehicles": list(veh.values())}


def main():
    os.makedirs(OUT, exist_ok=True)
    expected = pd.read_csv(os.path.join("results", "eval_episodes.csv"))
    policies = {"ppo": PPO.load(MODEL, device="cpu"), "fixed10": FixedTimePolicy(green=10)}
    index = []
    for demand, cfg in SCENARIOS.items():
        for key, name in CONTROLLERS.items():
            for seed in SEEDS:
                info, trace = record(policies[key], cfg, seed)
                row = expected[(expected.demand == demand) & (expected.controller == name) & (expected.seed == seed)].iloc[0]
                for m in ("mean_wait", "mean_queue", "max_queue", "throughput", "departed", "switches"):
                    assert abs(info[m] - row[m]) < 1e-9, f"{demand} {name} seed {seed}: {m} {info[m]} != {row[m]} in eval_episodes.csv"
                # the page's counters, from the trace alone (index.html computes them the same way)
                dec = lambda s: [B36.index(c) for c in s]
                halted, ph = dec(trace["halted"]), dec(trace["phase"])
                assert abs(sum(halted) / sum(dec(trace["departed"])) - info["mean_wait"]) < 1e-9
                assert sum(dec(trace["arrived"])) == info["throughput"] and max(halted) == info["max_queue"]
                assert sum(p % 2 == 1 and (i == 0 or ph[i - 1] % 2 == 0) for i, p in enumerate(ph)) == info["switches"]  # a yellow starts each switch
                f = f"{demand}-{key}-{seed}.json"
                with open(os.path.join(OUT, f), "w", encoding="utf-8") as fh:
                    json.dump({"demand": demand, "controller": name, "seed": seed, **trace}, fh, separators=(",", ":"))
                index.append({"file": f, "demand": demand, "controller": name, "seed": seed,
                              **{m: info[m] for m in ("mean_wait", "mean_queue", "max_queue", "throughput", "departed", "switches")}})
                print(f, round(info["mean_wait"], 3), "matches eval_episodes.csv")
    summary = pd.read_csv(os.path.join("results", "eval_summary.csv"))
    for demand in SCENARIOS:
        for name in CONTROLLERS.values():
            got = [r["mean_wait"] for r in index if r["demand"] == demand and r["controller"] == name]
            want = summary[(summary.demand == demand) & (summary.controller == name)].mean_wait_mean.iloc[0]
            assert abs(sum(got) / len(got) - want) < 1e-9, (demand, name)
    with open(os.path.join(OUT, "index.json"), "w", encoding="utf-8") as fh:
        json.dump({"source": "results/eval_episodes.csv", "episodes": index}, fh, indent=1)
    print("ok: every recorded episode reproduces results/eval_episodes.csv, and the seed means match results/eval_summary.csv")


if __name__ == "__main__":
    main()
