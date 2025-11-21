# envs/sumo_env.py
import os
import sys
import shutil
import random
import time
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

class SumoTrafficEnv(gym.Env):
    """
    SUMO environment with Traffic-Pressure reward, switching penalty, and min-phase duration.
    Gymnasium compatible.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, sumo_cfg, gui=False, sim_step=1, action_repeat=5,
                 min_green=5, switching_penalty=1.0):
        super().__init__()
        self.sumo_cfg = os.path.abspath(sumo_cfg)
        self.gui = gui
        self.sim_step = sim_step
        self.action_repeat = action_repeat
        self.min_green = min_green
        self.switching_penalty = switching_penalty

        self.sumo_binary = "sumo-gui" if gui else "sumo"
        self.tls_id = None
        self.lanes = []
        self.in_to_out = {}    # mapping incoming lane -> list of outgoing lanes
        self._phase_elapsed = 0
        self._current_phase = 0
        self._last_action = None
        self._last_seed = None

        # placeholders - set in reset()
        self.observation_space = gym.spaces.Box(low=0, high=1e9, shape=(10,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)

    def _find_sumo_executable(self):
        if shutil.which(self.sumo_binary):
            return self.sumo_binary
        candidates = [
            r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe",
            r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe",
            r"C:\Program Files\Sumo\bin\sumo.exe",
            r"C:\Program Files\Sumo\bin\sumo-gui.exe",
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return self.sumo_binary

    def _build_cmd(self, seed=None):
        exe = self._find_sumo_executable()
        cmd = [exe, "-c", self.sumo_cfg, "--step-length", str(self.sim_step), "--no-warnings", "true"]
        if seed is not None:
            cmd += ["--seed", str(int(seed))]
        return cmd

    def _start_sumo(self, seed=None):
        cmd = self._build_cmd(seed=seed)
        print("Starting SUMO with command:", " ".join(cmd))
        traci.start(cmd)

    def reset(self, *, seed=None, options=None):
        # seed handling
        if seed is not None:
            if isinstance(seed, (list, tuple)):
                seed = int(seed[0])
            random.seed(int(seed))
            np.random.seed(int(seed))
            self._last_seed = int(seed)
        else:
            self._last_seed = None

        # close previous connection
        try:
            traci.close()
        except Exception:
            pass

        # start SUMO
        self._start_sumo(seed=self._last_seed)

        # warmup
        for _ in range(5):
            traci.simulationStep()

        # detect TLS and controlled links
        tls = traci.trafficlight.getIDList()
        self.tls_id = tls[0] if tls else None

        if self.tls_id:
            controlled = traci.trafficlight.getControlledLinks(self.tls_id)
            incoming = []
            # build incoming list and mapping to outgoing lanes
            mapping = {}
            for linkset in controlled:
                for link in linkset:
                    inc_lane = link[0]
                    out_lane = link[1]
                    incoming.append(inc_lane)
                    mapping.setdefault(inc_lane, []).append(out_lane)
            # unique incoming lanes
            incoming_unique = []
            for l in incoming:
                if l not in incoming_unique:
                    incoming_unique.append(l)
            self.lanes = incoming_unique
            self.in_to_out = mapping
        else:
            # fallback to first edges
            try:
                edges = traci.edge.getIDList()
                self.lanes = [e + "_0" for e in edges[:4]]
                # fallback mapping: same edge -> no outgoing
                self.in_to_out = {ln: [] for ln in self.lanes}
            except Exception:
                self.lanes = []
                self.in_to_out = {}

        # define obs/action spaces
        n_lanes = max(4, len(self.lanes))
        obs_len = n_lanes + n_lanes + 2  # queues + waits + phase + elapsed
        self.observation_space = gym.spaces.Box(low=0, high=1e9, shape=(obs_len,), dtype=np.float32)

        if self.tls_id:
            try:
                nph = traci.trafficlight.getPhaseNumber(self.tls_id)
                self.action_space = gym.spaces.Discrete(max(2, nph))
            except Exception:
                self.action_space = gym.spaces.Discrete(2)
        else:
            self.action_space = gym.spaces.Discrete(2)

        # init internal trackers
        try:
            self._current_phase = traci.trafficlight.getPhase(self.tls_id) if self.tls_id else 0
        except Exception:
            self._current_phase = 0
        self._phase_elapsed = 0
        self._last_action = self._current_phase

        obs = self._get_obs()
        # Gymnasium-compatible reset returns (obs, info)
        return obs, {}

    def _get_lane_queue_and_wait(self, lane):
        try:
            q = traci.lane.getLastStepVehicleNumber(lane)
            w = traci.lane.getWaitingTime(lane)
        except Exception:
            # fallback to edge
            edge = lane.split("_")[0]
            try:
                q = traci.edge.getLastStepVehicleNumber(edge)
                w = traci.edge.getLastStepHaltingNumber(edge)
            except Exception:
                q, w = 0, 0
        return float(q), float(w)

    def _get_obs(self):
        queues = []
        waits = []
        for i in range(max(4, len(self.lanes))):
            if i < len(self.lanes):
                lane = self.lanes[i]
                q, w = self._get_lane_queue_and_wait(lane)
                queues.append(q)
                waits.append(w)
            else:
                queues.append(0.0)
                waits.append(0.0)
        return np.array(queues + waits + [float(self._current_phase), float(self._phase_elapsed)], dtype=np.float32)

    def _compute_pressure_reward(self):
        # pressure = incoming_queue - outgoing_queue
        total = 0.0
        for inc in self.lanes:
            inc_q, _ = self._get_lane_queue_and_wait(inc)
            out_q = 0.0
            outs = self.in_to_out.get(inc, [])
            if outs:
                for out in outs:
                    oq, _ = self._get_lane_queue_and_wait(out)
                    out_q += oq
                out_q = out_q / len(outs)
            # pressure per approach
            pressure = max(inc_q - out_q, 0.0)
            total += pressure
        # return negative pressure (we minimize)
        return -float(total)

    def step(self, action):
        # action handling: enforce min green
        # if action is not allowed due to min_green, ignore change
        intended_action = int(action)
        changed = (intended_action != self._current_phase)

        if changed and self._phase_elapsed < self.min_green:
            # ignore switch, keep current
            intended_action = self._current_phase
            changed = False

        # apply if TLS exists
        if self.tls_id:
            try:
                traci.trafficlight.setPhase(self.tls_id, intended_action)
                self._current_phase = intended_action
                if changed:
                    self._phase_elapsed = 0
            except Exception:
                pass

        reward = 0.0
        done = False
        info = {}

        # advance simulation
        for _ in range(self.action_repeat):
            traci.simulationStep()
            # accumulate pressure reward
            reward += self._compute_pressure_reward()
            self._phase_elapsed += self.sim_step

        # switching penalty
        if self._last_action is not None and intended_action != self._last_action:
            reward -= float(self.switching_penalty)
        self._last_action = intended_action

        # termination check
        if traci.simulation.getMinExpectedNumber() == 0:
            done = True

        obs = self._get_obs()
        # return Gymnasium-style (obs, reward, terminated, truncated, info)
        terminated = bool(done)
        truncated = False
        return obs, float(reward), terminated, truncated, info

    def render(self, mode="human"):
        pass

    def close(self):
        try:
            traci.close()
        except Exception:
            pass
