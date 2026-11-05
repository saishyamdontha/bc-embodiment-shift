"""Scan candidate target robots with the source expert and write the chosen one into common.py.
Choice: the candidate where source-expert success is closest to 0.45 (a clear but learnable gap)."""
import pathlib
import re

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

CANDIDATES = {
    "gravity=-6.0": dict(gravity=-6.0),
    "gravity=-8.0": dict(gravity=-8.0),
    "gravity=-11.5": dict(gravity=-11.5),
    "enable_wind=True, wind_power=5.0, turbulence_power=0.5":
        dict(enable_wind=True, wind_power=5.0, turbulence_power=0.5),
    "enable_wind=True, wind_power=10.0, turbulence_power=1.0":
        dict(enable_wind=True, wind_power=10.0, turbulence_power=1.0),
    "enable_wind=True, wind_power=15.0, turbulence_power=1.5":
        dict(enable_wind=True, wind_power=15.0, turbulence_power=1.5),
}
model = PPO.load("models/expert_source", device="cpu")
results = {}
for name, kw in CANDIDATES.items():
    env, rets = gym.make("LunarLander-v3", **kw), []
    for i in range(30):
        obs, _ = env.reset(seed=20000 + i)
        done, R = False, 0.0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            R += r
            done = term or trunc
        rets.append(R)
    r = np.array(rets)
    results[name] = (float(np.mean(r >= 200)), float(r.mean()))
    print(f"{name:<58} success {results[name][0]:.2f}   mean return {results[name][1]:7.1f}")

chosen = min(results, key=lambda k: abs(results[k][0] - 0.45))
p = pathlib.Path("common.py")
s = p.read_text()
s, n = re.subn(r'^    "target": dict\(.*\),$', f'    "target": dict({chosen}),', s, flags=re.M)
assert n == 1, "target line not found in common.py"
p.write_text(s)
print(f"\nchosen target: {chosen}  (source expert success {results[chosen][0]:.2f})")
