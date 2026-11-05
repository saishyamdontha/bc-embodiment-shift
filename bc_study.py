"""How many target-robot demonstrations does behaviour cloning need after a dynamics change,
and do source-robot demonstrations help or hurt?"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO

from common import SUCCESS_RETURN, make

torch.set_num_threads(4)


def rollouts(model, domain, n, seed0):
    env, obs_eps, act_eps, rets = make(domain), [], [], []
    for i in range(n):
        obs, _ = env.reset(seed=seed0 + i)
        done, R, O, A = False, 0.0, [], []
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            O.append(obs)
            A.append(int(a))
            obs, r, term, trunc, _ = env.step(a)
            R += r
            done = term or trunc
        obs_eps.append(np.array(O, dtype=np.float32))
        act_eps.append(np.array(A, dtype=np.int64))
        rets.append(R)
    return obs_eps, act_eps, np.array(rets)


def train_bc(X, y, w, seed, steps):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(8, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 4))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt, yt, wt = torch.tensor(X), torch.tensor(y), torch.tensor(w, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        b = torch.randint(0, len(Xt), (256,), generator=g)
        loss = nn.functional.cross_entropy(net(Xt[b]), yt[b], reduction="none")
        loss = (loss * wt[b]).sum() / wt[b].sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def evaluate(net, domain, n, seed0):
    env, rets = make(domain), []
    for i in range(n):
        obs, _ = env.reset(seed=seed0 + i)
        done, R = False, 0.0
        while not done:
            with torch.no_grad():
                a = int(net(torch.tensor(obs, dtype=torch.float32)).argmax())
            obs, r, term, trunc, _ = env.step(a)
            R += r
            done = term or trunc
        rets.append(R)
    r = np.array(rets)
    return float((r >= SUCCESS_RETURN).mean()), float(r.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-demos", type=int, default=50)
    ap.add_argument("--target-counts", default="1,2,5,10,25,50")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--steps", type=int, default=4000)
    a = ap.parse_args()
    counts = [int(x) for x in a.target_counts.split(",")]
    out = Path("results")

    src_exp = PPO.load("models/expert_source", device="cpu")
    tgt_exp = PPO.load("models/expert_target", device="cpu")

    # Demonstration seeds: source 0+, target 5000+; evaluation seeds 10000+ (never seen in training)
    S_obs, S_act, S_ret = rollouts(src_exp, "source", a.source_demos, 0)
    T_obs, T_act, T_ret = rollouts(tgt_exp, "target", max(counts), 5000)
    print(f"source demos: {len(S_obs)} episodes, {sum(map(len, S_obs))} steps, "
          f"demonstrator success {np.mean(S_ret >= SUCCESS_RETURN):.2f}")
    print(f"target demos: {len(T_obs)} episodes, {sum(map(len, T_obs))} steps, "
          f"demonstrator success {np.mean(T_ret >= SUCCESS_RETURN):.2f}")

    ref = {}
    for name, exp in (("source_demonstrator", src_exp), ("target_demonstrator", tgt_exp)):
        _, _, r = rollouts(exp, "target", a.eval_episodes, 10000)
        ref[name] = (float(np.mean(r >= SUCCESS_RETURN)), float(r.mean()))
        print(f"reference {name} on target robot: success {ref[name][0]:.2f}, mean return {ref[name][1]:.1f}")

    Xs, ys = np.concatenate(S_obs), np.concatenate(S_act)
    rows = []

    def run(condition, m, X, y, w, seed):
        net = train_bc(X, y, w, seed, a.steps)
        succ, ret = evaluate(net, "target", a.eval_episodes, 10000)
        rows.append(dict(condition=condition, target_demos=m, seed=seed,
                         train_samples=len(X), success=succ, mean_return=ret))
        print(f"  {condition:<24} m={m:>2} seed={seed}  success {succ:.2f}  return {ret:7.1f}", flush=True)

    for seed in range(a.seeds):
        run("source_only", 0, Xs, ys, np.ones(len(Xs)), seed)
        for m in counts:
            Xt, yt = np.concatenate(T_obs[:m]), np.concatenate(T_act[:m])
            run("target_only", m, Xt, yt, np.ones(len(Xt)), seed)
            Xm, ym = np.concatenate([Xs, Xt]), np.concatenate([ys, yt])
            run("mixed", m, Xm, ym, np.ones(len(Xm)), seed)
            # Upweight target samples so both robots carry equal total weight
            wm = np.concatenate([np.ones(len(Xs)), np.full(len(Xt), len(Xs) / len(Xt))])
            run("mixed_target_upweighted", m, Xm, ym, wm, seed)

    with open(out / "bc_runs.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print("\n=== success rate on the TARGET robot (mean over seeds, min-max) ===")
    print(f"{'condition':<26}" + "".join(f"{f'm={m}':>14}" for m in [0] + counts))
    summary = ["condition,target_demos,success_mean,success_min,success_max,return_mean"]
    for cond in ("source_only", "target_only", "mixed", "mixed_target_upweighted"):
        line = f"{cond:<26}"
        for m in [0] + counts:
            s = [r["success"] for r in rows if r["condition"] == cond and r["target_demos"] == m]
            if not s:
                line += f"{'-':>14}"
                continue
            rt = [r["mean_return"] for r in rows if r["condition"] == cond and r["target_demos"] == m]
            line += f"{f'{np.mean(s):.2f} ({min(s):.2f}-{max(s):.2f})':>14}"
            summary.append(f"{cond},{m},{np.mean(s):.4f},{min(s):.4f},{max(s):.4f},{np.mean(rt):.2f}")
        print(line)
    for name, (s, r) in ref.items():
        summary.append(f"reference_{name},-,{s:.4f},{s:.4f},{s:.4f},{r:.2f}")
    (out / "bc_summary.csv").write_text("\n".join(summary) + "\n")
    print("\nwrote results/bc_runs.csv and results/bc_summary.csv")


if __name__ == "__main__":
    main()
