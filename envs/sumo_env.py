# envs/sumo_env.py
import os
import sys
import shutil
import numpy as np
import gymnasium as gym

# ensure SUMO tools are available
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.append(tools)
else:
    possible = r"C:\Program Files (x86)\Eclipse\Sumo"
    if os.path.isdir(possible):
        tools = os.path.join(possible, "tools")
        if tools not in sys.path:
            sys.path.append(tools)

import traci

# simple.net.xml phases: 0 NS green, 1 NS yellow, 2 EW green, 3 EW yellow.
GREEN_PHASES = (0, 2)
HOLD = 1e6  # phase duration long enough that SUMO never auto-advances the static programme
N_LANES = 4
# obs = [vehicles per approach lane x4, waiting time per lane (s) x4, current green (0/1), seconds in this green]
OBS_SCALE = np.array([10.0] * N_LANES + [100.0] * N_LANES + [1.0, 60.0], dtype=np.float32)
METRICS = ("mean_wait", "mean_queue", "max_queue", "throughput")


class SumoTrafficEnv(gym.Env):
    """
    One signalised 4-way junction in SUMO, Gymnasium API.

    Every `delta_time` seconds the agent picks which approach pair gets green:
    0 = north-south, 1 = east-west. Switching inserts the yellow phase itself
    (`yellow_time` s) and is refused until the current green has run `min_green` s.

    Reward per second: minus the traffic pressure (vehicles on each incoming lane
    minus the mean on its outgoing lanes, floored at 0), minus `switching_penalty`
    per switch. The episode is truncated at `horizon` seconds of simulated time.

    `info` carries the evaluation metrics for the episode so far (see METRICS).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, sumo_cfg, gui=False, delta_time=5, yellow_time=3, min_green=10,
                 switching_penalty=1.0, horizon=3600):
        super().__init__()
        self.sumo_cfg = os.path.abspath(sumo_cfg)
        self.sumo_binary = "sumo-gui" if gui else "sumo"
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.switching_penalty = switching_penalty
        self.horizon = horizon
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(0, np.inf, shape=(2 * N_LANES + 2,), dtype=np.float32)
        self.tls_id = None

    def _find_sumo_executable(self):
        if shutil.which(self.sumo_binary):
            return self.sumo_binary
        for root in (r"C:\Program Files (x86)\Eclipse\Sumo", r"C:\Program Files\Sumo"):
            exe = os.path.join(root, "bin", self.sumo_binary + ".exe")
            if os.path.exists(exe):
                return exe
        return self.sumo_binary

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # explicit seed -> reproducible run; otherwise draw one so training episodes differ
        sumo_seed = seed if seed is not None else int(self.np_random.integers(1000, 2**31 - 1))
        self.close()
        traci.start([self._find_sumo_executable(), "-c", self.sumo_cfg, "--seed", str(sumo_seed),
                     "--no-warnings", "true", "--no-step-log", "true", "--verbose", "false",
                     "--time-to-teleport", "-1"]  # no teleports: every stopped second counts
                    + (["--start", "--quit-on-end"] if self.sumo_binary == "sumo-gui" else []))

        self.tls_id = traci.trafficlight.getIDList()[0]
        links = [link for linkset in traci.trafficlight.getControlledLinks(self.tls_id) for link in linkset]
        self.lanes = list(dict.fromkeys(inc for inc, _, _ in links))
        assert len(self.lanes) == N_LANES, f"expected {N_LANES} approach lanes, got {self.lanes}"
        self.in_to_out = {lane: [out for inc, out, _ in links if inc == lane] for lane in self.lanes}
        self.all_lanes = list(dict.fromkeys(self.lanes + [out for _, out, _ in links]))
        self.dt = traci.simulation.getDeltaT()

        self.green, self.elapsed = 0, 0.0
        self._set_phase(GREEN_PHASES[0])
        self.queue_sum = self.queue_max = self.seconds = self.departed = self.arrived = self.switches = 0
        return self._get_obs(), self._info()

    def _set_phase(self, phase):
        traci.trafficlight.setPhase(self.tls_id, phase)
        traci.trafficlight.setPhaseDuration(self.tls_id, HOLD)

    def _sim_step(self):
        """Advance one simulation step, update episode metrics, return the pressure reward."""
        traci.simulationStep()
        halted = sum(traci.lane.getLastStepHaltingNumber(l) for l in self.lanes)
        self.queue_sum += halted
        self.queue_max = max(self.queue_max, halted)
        self.seconds += 1
        self.departed += traci.simulation.getDepartedNumber()
        self.arrived += traci.simulation.getArrivedNumber()

        count = {l: traci.lane.getLastStepVehicleNumber(l) for l in self.all_lanes}
        pressure = sum(max(count[i] - np.mean([count[o] for o in self.in_to_out[i]]), 0.0) for i in self.lanes)
        return -float(pressure)

    def _run(self, seconds):
        reward = 0.0
        for _ in range(int(round(seconds / self.dt))):
            if traci.simulation.getTime() >= self.horizon:
                break
            reward += self._sim_step()
        return reward

    def _get_obs(self):
        queues = [traci.lane.getLastStepVehicleNumber(l) for l in self.lanes]
        waits = [traci.lane.getWaitingTime(l) for l in self.lanes]
        raw = np.array(queues + waits + [self.green, self.elapsed], dtype=np.float32)
        return raw / OBS_SCALE

    def _info(self):
        return {
            # seconds stopped (speed < 0.1 m/s) on the approaches, per vehicle that entered
            "mean_wait": self.queue_sum * self.dt / max(1, self.departed),
            "mean_queue": self.queue_sum / max(1, self.seconds),  # halted vehicles on approaches
            "max_queue": self.queue_max,
            "throughput": self.arrived,  # vehicles that finished their trip
            "departed": self.departed,
            "switches": self.switches,
        }

    def step(self, action):
        want = int(action)
        reward = 0.0
        if want != self.green and self.elapsed >= self.min_green:
            self._set_phase(GREEN_PHASES[self.green] + 1)  # yellow for the current green
            reward += self._run(self.yellow_time)
            self.green, self.elapsed = want, 0.0
            self._set_phase(GREEN_PHASES[want])
            reward -= self.switching_penalty
            self.switches += 1
        reward += self._run(self.delta_time)
        self.elapsed += self.delta_time

        truncated = traci.simulation.getTime() >= self.horizon
        terminated = not truncated and traci.simulation.getMinExpectedNumber() == 0
        return self._get_obs(), reward, terminated, truncated, self._info()

    def render(self, mode="human"):
        pass  # run with gui=True to watch in sumo-gui

    def close(self):
        try:
            traci.close()
        except Exception:
            pass
