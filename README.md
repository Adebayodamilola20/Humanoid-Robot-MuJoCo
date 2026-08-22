# humanoid-rl

Teach a MuJoCo humanoid to walk with deep reinforcement learning — one CLI, reproducible runs, and a policy you can actually watch.

```bash
python -m humanoid_rl train          # learn a gait
python -m humanoid_rl play           # watch it in a 3D window
```

---

## What this is

A small, honest reinforcement-learning project built on **MuJoCo**, **Gymnasium** and **Stable-Baselines3**. The agent starts as a ragdoll that collapses instantly and, given enough steps, works out how to stay upright and move forward.

Four things it does properly, which most tutorial versions skip:

- **Observation normalisation travels with the policy.** Statistics are saved next to every checkpoint, so a policy that scored well during training scores the same when you play it back.
- **Every run is self-describing.** `config.json` records the exact settings, so `play`, `record` and `eval` need no flags — they read them.
- **Ctrl-C keeps your work.** Interrupting training saves the policy instead of throwing away hours of compute.
- **Tuned defaults.** PPO and SAC ship with hyperparameters adapted from RL Baselines3 Zoo, not library defaults that never learn to walk.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Optional musculoskeletal environments (MyoSuite):

```bash
pip install -e '.[myo]'
```

## Use

| Command | What it does |
| --- | --- |
| `train` | Learn a policy, writing checkpoints and TensorBoard logs |
| `play` | Open a MuJoCo window and watch a policy |
| `record` | Render episodes to `.mp4` / `.gif` — no display needed |
| `eval` | Score a policy over N episodes, headless and fast |
| `ls` | List saved runs |

```bash
# Train PPO on Humanoid-v5 with the tuned defaults
python -m humanoid_rl train

# Fewer steps for a quick smoke test
python -m humanoid_rl train --steps 200_000 --n-envs 4

# SAC instead — far fewer steps to a gait, slower per step
python -m humanoid_rl train --algo sac --steps 2e6

# Pick up where you left off
python -m humanoid_rl train --resume latest --steps 5e6

# Override any hyperparameter without touching the code
python -m humanoid_rl train --set learning_rate=1e-4 --set batch_size=512

# Watch, record, score
python -m humanoid_rl play --episodes 3
python -m humanoid_rl record --out walk.mp4
python -m humanoid_rl eval --episodes 20
```

Every command defaults to the most recent run. Target an older one with `--run <name>`, and choose which checkpoint with `--model best|final|last`.

## Anatomy of a run

```
runs/20260821-142530-humanoid-v5-ppo/
├── config.json          exact settings used
├── best_model.zip       highest evaluation reward so far
├── final_model.zip      last policy — written even on Ctrl-C
├── vecnormalize.pkl     observation statistics for that policy
├── checkpoints/         periodic snapshots
├── monitor/             per-episode reward CSVs
└── tb/                  TensorBoard event files
```

```bash
tensorboard --logdir runs/latest/tb
```

Watch `rollout/ep_rew_mean` climb. On `Humanoid-v5` a reward around 1000 means the humanoid stays upright for a while; 5000+ is a real walking gait.

## Project layout

```
src/humanoid_rl/
├── cli.py        argument parsing and command dispatch
├── config.py     tuned per-algorithm defaults; serialised into each run
├── envs.py       environment construction shared by training and playback
├── runs.py       run directories, checkpoint discovery, `latest` resolution
├── train.py      model construction, callbacks, the learning loop
├── rollout.py    play / record / eval
└── console.py    terminal output
```

## Choosing an algorithm

| | PPO | SAC |
| --- | --- | --- |
| Steps to a decent gait | ~10M | ~2M |
| Parallel envs | 8 (scales well) | 1 |
| Wall-clock per step | fast | slower (gradient step per env step) |
| Stability | very stable | sensitive to seed |

PPO is the default because it parallelises across CPU cores and rarely diverges. Reach for SAC when you are short on sample budget rather than on time.

## Troubleshooting

**`NSWindow should only be instantiated on the main thread!` on macOS.** You ran `play` under `mjpython`. Use plain `python` instead. Gymnasium does not use MuJoCo's `launch_passive` viewer — it opens its own GLFW window, and GLFW must create that window on the process's main thread. `mjpython` runs your script on a *secondary* thread (it reserves the main one for MuJoCo's Cocoa event loop), so it is the one interpreter that cannot work here. The CLI now detects this and says so.

**No window appears at all.** It often opens *behind* your terminal — check Mission Control or the Dock. Over SSH or in a headless session there is no display to draw on; use `record` and watch the file.

**Training seems stuck near zero reward.** Humanoid is genuinely hard — a flat curve for the first few hundred thousand steps is normal. Check TensorBoard rather than the terminal, and give PPO at least 2M steps before judging it.

**`Unknown environment id`.** MyoSuite environments (`myoLegWalk-v0` and friends) need the `[myo]` extra installed.

**Playback looks worse than training.** Almost always mismatched normalisation statistics. Use the CLI rather than loading checkpoints by hand; it pairs each policy with the stats saved alongside it.

## License

MIT © adebayostephen
