# test_sumo.py - smoke test: SUMO starts and the env runs the signal phases correctly.
# Run from the repo root:  python test_sumo.py
import traci
from envs.sumo_env import SumoTrafficEnv
from baselines.fixed_time import FixedTimePolicy

env = SumoTrafficEnv("sumo_scenario/simple.sumocfg")
obs, info = env.reset(seed=1)
print("SUMO connected. obs:", obs)

# record the signal phase shown during every simulated second
phases = []
sim_step = env._sim_step
env._sim_step = lambda: (phases.append(traci.trafficlight.getPhase(env.tls_id)), sim_step())[1]

# asking for EW straight away is refused until NS has had min_green (10 s)
for _ in range(2):
    env.step(1)
assert phases == [0] * 10, phases
# third request is accepted: 3 s NS yellow, then EW green for the 5 s decision step
env.step(1)
assert phases[10:] == [1] * 3 + [2] * 5, phases[10:]
assert traci.simulation.getTime() == 18

# fixed-time 30 s: phases cycle 0 -> 1 -> 2 -> 3 with 30 s greens and 3 s yellows
env.reset(seed=1)
phases.clear()
policy = FixedTimePolicy(30)
for _ in range(30):
    obs, _, _, _, info = env.step(policy.predict(obs)[0])
runs = []
for p in phases:
    if runs and runs[-1][0] == p:
        runs[-1][1] += 1
    else:
        runs.append([p, 1])
assert runs[:5] == [[0, 30], [1, 3], [2, 30], [3, 3], [0, 30]], runs
print("phase sequence OK:", runs[:5], "| metrics after", len(phases), "s:", info)
env.close()
