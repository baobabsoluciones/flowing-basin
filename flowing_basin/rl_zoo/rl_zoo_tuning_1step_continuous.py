"""
Hyperparameter tuning with RL Zoo with one-step continuous actions

Algorithms to tune:
- SAC
- A2C
- PPO

Environments in which to tune these algorithms:
- The real (G0) and simplified (G1) environments.
- The one-step continuous action (A1).
- Only the normal observation (O231), normal reward (R1), and normal training configuration (T1),
since they should not influence on the performance of the hyperparameters.
- With and without normalization (SB3 VecNormalize wrapper).
"""

import os
from itertools import product

NUM_TRIALS = 100
NUM_TIMESTEPS_PER_TRIAL = 49_500  # Half of training T0
NUM_PARALLEL_JOBS = -1  # This will set it to the system's CPU count
EVAL_EPISODES = 10  # Same as training T0

# Possible values: --algo {a2c,ddpg,dqn,ppo,sac,td3,ars,qrdqn,tqc,trpo,ppo_lstm}
algorithms = ["sac", "a2c", "ppo"]
generals = ["G0", "G1"]
normalizations = [True, False]
action = "A1"
observation = "O231"
reward = "R1"
training = "T0"

for algorithm, general, normalization in product(algorithms, generals, normalizations):

    # Skip optimizations that have already finished
    if (normalization and algorithm in {"a2c", "ppo"}) or (not normalization and algorithm == "sac"):
        print(f"Skipped the tuning of {algorithm} in {general} with normalization {normalization}.")
        continue

    env_id = f"{action}{general}{observation}{reward}{training}"
    if normalization:
        env_id += "normalize"

    hyperparams_filename = f"{algorithm}"
    if normalization:
        hyperparams_filename += "_normalize"

    print(f"Tuning {algorithm} in environment {env_id}...")
    instruction = (
        f"python rl_zoo3/train.py --algo {algorithm} --env RLEnvironment-{env_id}"
        f" -n {NUM_TIMESTEPS_PER_TRIAL} -optimize --n-trials {NUM_TRIALS} --eval-episodes {EVAL_EPISODES}"
        f" --n-jobs {NUM_PARALLEL_JOBS}"
        f" --conf-file default_hyperparams/{hyperparams_filename}.yml"
        f" --continuous-actions"
    )
    os.system(instruction)
