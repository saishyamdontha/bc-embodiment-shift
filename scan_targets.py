"""How much does the source expert degrade on each candidate target robot?"""
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

CANDIDATES = {
    "gravity_-6": dict(gravity=-6.0),
    "gravity_-8": dict(gravity=-8.0),
    "gravity_-11.5": dict(gravity=-11.5),
    "wind_5_turb_0.5": dict(enable_wind=True, wind_power=5.0, turbulence_power=0.5),
    "wind_10_turb_1.0": dict(enable_wind=True, wind_power=10.0, turbulence_power=1.0),
    "wind_15_turb_1.5": dict(enable_wind=True, wind_power=15.0, turbulence_power=1.5),
}
model = PPO.load("models/expert_source", device="cpu")
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
    print(f"{name:<18} source expert: mean return {r.mean():7.1f}   success {np.mean(r >= 200):.2f}")
