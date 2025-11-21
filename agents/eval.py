# agents/eval.py (auto-loader version)
import os, time, csv, glob
from stable_baselines3 import PPO
from envs.sumo_env import SumoTrafficEnv

CFG = os.path.join("sumo_scenario", "simple.sumocfg")
OUT_CSV = os.path.join("results", "eval_metrics.csv")

def find_latest_model():
    candidates = sorted(glob.glob("results/ppo_traffic*.zip") + glob.glob("results/ppo_traffic_checkpoint_*.zip"), key=os.path.getmtime)
    if not candidates:
        raise FileNotFoundError("No model files found in results/. Expected names like 'ppo_traffic.zip' or 'ppo_traffic_checkpoint_*.zip'")
    return candidates[-1]

def run_one(env, model, render=True):
    obs, info = env.reset()
    done = False
    total_wait = 0.0
    steps = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        steps += 1
        try:
            for lane in env.lanes:
                q, _ = env._get_lane_queue_and_wait(lane)
                total_wait += q
        except Exception:
            pass
        done = bool(terminated or truncated)
        if render:
            time.sleep(0.01)
    return {"total_wait": float(total_wait), "steps": steps}

def main(runs=3, render=True):
    model_path = find_latest_model()
    print("Loading model:", model_path)
    env = SumoTrafficEnv(sumo_cfg=CFG, gui=render)
    model = PPO.load(model_path, env=env)
    rows = []
    for i in range(runs):
        m = run_one(env, model, render=render)
        print(f"Run {i+1}: {m}")
        rows.append({"run": i+1, **m})
    env.close()
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "total_wait", "steps"])
        writer.writeheader()
        writer.writerows(rows)
    print("Saved eval metrics to", OUT_CSV)

if __name__ == "__main__":
    main(runs=3, render=True)
