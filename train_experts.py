"""Train a PPO expert for one domain; it is used only to generate demonstrations."""
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from common import DOMAINS, make

ap = argparse.ArgumentParser()
ap.add_argument("--domain", choices=list(DOMAINS), required=True)
ap.add_argument("--steps", type=int, default=1_000_000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--init", default=None, help="start from an existing model (warm start)")
a = ap.parse_args()

env = make_vec_env(lambda: make(a.domain), n_envs=16, seed=a.seed)
if a.init:
    model = PPO.load(a.init, env=env, device="cpu")
else:
    model = PPO("MlpPolicy", env, n_steps=1024, batch_size=64, gae_lambda=0.98, gamma=0.999,
                n_epochs=4, ent_coef=0.01, seed=a.seed, device="cpu", verbose=0)
model.learn(total_timesteps=a.steps)
model.save(f"models/expert_{a.domain}")

for d in DOMAINS:
    mean, std = evaluate_policy(model, Monitor(make(d)), n_eval_episodes=50, deterministic=True)
    print(f"expert_{a.domain} on {d}: mean return {mean:.1f} +/- {std:.1f}")
