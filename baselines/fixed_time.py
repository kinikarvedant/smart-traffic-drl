# baselines/fixed_time.py
from envs.sumo_env import OBS_SCALE


class FixedTimePolicy:
    """
    Fixed-time signal plan run through the same env as PPO: each direction gets
    `green` seconds of green, then the env's yellow (3 s), then the other direction.
    green=30 gives a 66 s cycle. Same predict() signature as an SB3 model.
    """

    def __init__(self, green=30):
        self.green = green

    def predict(self, obs, deterministic=True):
        current = int(round(float(obs[-2] * OBS_SCALE[-2])))
        elapsed = round(float(obs[-1] * OBS_SCALE[-1]))
        return (1 - current if elapsed >= self.green else current), None
