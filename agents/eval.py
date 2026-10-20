# agents/eval.py - run PPO and fixed-time control over the same seeds, same env, same metrics.
# Run from the repo root:  python -m agents.eval
import os
import sys
import pandas as pd
from stable_baselines3 import PPO

from envs.sumo_env import SumoTrafficEnv, METRICS
from baselines.fixed_time import FixedTimePolicy

SCENARIOS = {
    "regular": os.path.join("sumo_scenario", "simple.sumocfg"),  # training demand: evenly spaced arrivals
    "random": os.path.join("sumo_scenario", "random.sumocfg"),   # never trained on: random arrivals, same mean
}
MODEL = os.path.join("results", "ppo_traffic_v2.zip")
SEEDS = [1, 2, 3, 4, 5]  # training uses seed 0, then draws from 1000 upwards, so these are unseen


def run_episode(policy, seed, cfg=SCENARIOS["regular"], gui=False):
    env = SumoTrafficEnv(cfg, gui=gui)
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
    env.close()
    return info


def record_gif(policy, path, start=300, seconds=90):
    """Animated GIF of `policy` in sumo-gui: one frame per simulated second, played at 5x."""
    import tempfile
    import time
    import traci
    from PIL import Image

    env = SumoTrafficEnv(SCENARIOS["regular"], gui=True)
    obs, _ = env.reset(seed=SEEDS[0])
    traci.gui.setSchema("View #0", "real world")
    traci.gui.setOffset("View #0", 250, 250)  # junction centre
    traci.gui.setZoom("View #0", 500)
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        sim_step = env._sim_step

        def capture_then_step():
            if start <= traci.simulation.getTime() < start + seconds:
                files.append(os.path.join(tmp, f"{len(files):04d}.png"))
                # written during the next step; passing a size, or no pause, crashed sumo-gui 1.25 here
                traci.gui.screenshot("View #0", files[-1])
                time.sleep(0.05)
            return sim_step()

        env._sim_step = capture_then_step
        while traci.simulation.getTime() <= start + seconds:
            obs, *_ = env.step(int(policy.predict(obs, deterministic=True)[0]))
        env.close()
        frames = []
        for f in files:
            with Image.open(f) as im:
                w, h = im.size
                side = min(w, h)
                square = im.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2))
                frames.append(square.resize((400, 400)).convert("P", palette=Image.ADAPTIVE, colors=32))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=200, loop=0)
    print("Saved", path)


def main():
    controllers = {
        "PPO": PPO.load(MODEL, device="cpu"),
        "Fixed-time 30 s": FixedTimePolicy(green=30),  # main baseline: 30 s green, 3 s yellow, 66 s cycle
        "Fixed-time 10 s": FixedTimePolicy(green=10),  # shortest green the env allows (min_green); short cycles suit light demand
        "Fixed-time 42 s": FixedTimePolicy(green=42),  # the plan netconvert wrote into simple.net.xml
    }
    rows = []
    for demand, cfg in SCENARIOS.items():
        for name, policy in controllers.items():
            for seed in SEEDS:
                info = run_episode(policy, seed, cfg)
                rows.append({"demand": demand, "controller": name, "seed": seed, **info})
                print(rows[-1])

    episodes = pd.DataFrame(rows)
    episodes.to_csv(os.path.join("results", "eval_episodes.csv"), index=False)
    summary = episodes.groupby(["demand", "controller"], sort=False)[list(METRICS) + ["departed", "switches"]].agg(["mean", "std", "min", "max"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.insert(0, "seeds", len(SEEDS))
    summary.to_csv(os.path.join("results", "eval_summary.csv"))
    print(summary.round(2).T)


if __name__ == "__main__":
    if "--gif" in sys.argv:  # python -m agents.eval --gif  (opens sumo-gui)
        os.makedirs(os.path.join("results", "figures"), exist_ok=True)
        record_gif(PPO.load(MODEL, device="cpu"), os.path.join("results", "figures", "ppo_demo.gif"))
    else:
        main()
