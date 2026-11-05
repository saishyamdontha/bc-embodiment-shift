# bc-embodiment-shift

**When a robot's dynamics change, how many new demonstrations does behaviour cloning need, and do
the old robot's demonstrations help or hurt?**

A small, simulated study on LunarLander (Gymnasium). The *source* robot has default dynamics; the
*target* robot flies in strong wind and turbulence (wind power 15, turbulence 1.5). A PPO policy trained
for each robot serves as the demonstrator; behaviour cloning (a 2-layer MLP in PyTorch) learns from
their demonstrations. Every condition is evaluated on the **target** robot over 50 held-out episodes and
3 seeds. Success = episode return of at least 200 (a landing).

## Key findings

- **Five target demonstrations were enough.** Cloning 5 target episodes reached 0.69 success, about the
  target demonstrator's own 0.66, against 0.35 when cloning 50 episodes from the source robot.
- **The old robot's demonstrations hurt when target data was scarce, and never clearly helped.** Mixing 50
  source episodes into training gave 0.31 against 0.42 with 1 target episode, and 0.59 against 0.69 with 5.
  With 25 or more target episodes, mixed and target-only training performed about the same.
- **Upweighting target samples did not fix it.** Giving both robots equal total weight was the worst option
  with 1 or 2 target episodes (0.28 and 0.51), most likely from overfitting a few heavily weighted samples.

## Results (success on the target robot, mean over 3 seeds)

| Training data | 1 | 2 | 5 | 10 | 25 | 50 target episodes |
|---|---|---|---|---|---|---|
| Target only | 0.42 | 0.61 | 0.69 | 0.73 | 0.65 | 0.69 |
| Source (50) + target | 0.31 | 0.65 | 0.59 | 0.65 | 0.69 | 0.69 |
| Source (50) + target, target upweighted | 0.28 | 0.51 | 0.60 | 0.63 | 0.67 | 0.69 |

| Reference on the target robot | Success | Mean return |
|---|---|---|
| Behaviour cloning on source demonstrations only | 0.35 | 57 |
| Source demonstrator (PPO) | 0.40 | 65 |
| Target demonstrator (PPO, warm-started from the source one) | 0.66 | 174 |

Per-run results: `results/bc_runs.csv`; summary: `results/bc_summary.csv`; full log: `results/bc_study_run.txt`.

## Method

- **Demonstrators:** PPO (stable-baselines3), 1M steps for the source robot. The target demonstrator
  started from the source policy and trained 1.5M more steps on the target robot. It lands in about 2 of 3
  episodes, so it is a ceiling for behaviour cloning, not a perfect expert.
- **Choosing the target robot:** six candidate shifts (gravity and wind settings) were scanned, and the one
  where the source demonstrator's success was closest to 45% was used (`results/target_scan.txt`).
- **Demonstrations:** 50 source episodes and up to 50 target episodes, deterministic demonstrator actions.
  Demonstration seeds and evaluation seeds never overlap.
- **Behaviour cloning:** MLP 8-256-256-4, cross-entropy, 4,000 steps of batch 256, the same for every
  condition. "Upweighted" scales target samples so both robots carry equal total weight.

## Limitations

- A toy, simulated task with discrete actions; not a vision-language-action policy or a real robot.
- One pair of robots and one demonstrator per robot. The target demonstrator is imperfect (0.66).
- 3 seeds x 50 evaluation episodes: differences smaller than about 0.07 are within run-to-run noise.
- Why mixing hurts is not tested directly; conflicting actions for similar states are the likely cause.

## Reproduce

~~~bash
python3 -m venv .venv && source .venv/bin/activate
pip install swig && pip install "gymnasium[box2d]" stable-baselines3 torch numpy
python3 train_experts.py --domain source
python3 select_target.py
python3 train_experts.py --domain target --init models/expert_source --steps 1500000
python3 bc_study.py
~~~
