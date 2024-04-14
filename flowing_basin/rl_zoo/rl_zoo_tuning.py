"""
Hyperparameter tuning with RL Zoo

Relevant documentation:
Custom environments: https://rl-baselines3-zoo.readthedocs.io/en/master/guide/custom_env.html
Hyperparameter tuning: https://rl-baselines3-zoo.readthedocs.io/en/master/guide/tuning.html

This script uses the forked RL Zoo library contained within this repository.

The fixed values for hyperparameters that will not be tuned
are read from the default_hyperparams directory,
which were set to match StableBaselines3 default values.

Algorithms to tune:
- SAC
- A2C
- PPO

Environments in which to tune these algorithms:
- The real (G0) and simplified (G1) environments.
- The normal action (A1), the 99-block action (A113) and, for A2C/PPO, the discrete actions
(A31, A32, A33 and their 99-block variants, A313, A323, A333).
- Only the normal observation (O231), normal reward (R1), and normal training configuration (T1),
since they should not influence on the performance of the hyperparameters.
"""

import os
from itertools import product

NUM_TRIALS = 100
NUM_TIMESTEPS_PER_TRIAL = 49_500  # Half of training T1
NUM_PARALLEL_JOBS = 1
EVAL_EPISODES = 10  # Same as training T1
CONTINUOUS_ACTIONS = {"A1", "A113"}

# Possible values: --algo {a2c,ddpg,dqn,ppo,sac,td3,ars,qrdqn,tqc,trpo,ppo_lstm}
algorithms = ["sac", "a2c", "ppo"]
generals = ["G0", "G1"]
observation = "O231"
reward = "R1"
training = "T0"

for algorithm, general in product(algorithms, generals):

    if algorithm == "sac":
        actions = ["A1", "A113"]
    else:
        actions = ["A1", "A113", "A31", "A32", "A33", "A313", "A323", "A333"]

    for action in actions:

        env_id = f"{action}{general}{observation}{reward}{training}"
        instruction = (
            f"python rl_zoo3/train.py --algo {algorithm} --env RLEnvironment-{env_id}"
            f" -n {NUM_TIMESTEPS_PER_TRIAL} -optimize --n-trials {NUM_TRIALS} --eval-episodes {EVAL_EPISODES}"
            f" --n-jobs {NUM_PARALLEL_JOBS}"
            f" --conf-file default_hyperparams/{algorithm}.yml"
        )
        if action in CONTINUOUS_ACTIONS:
            instruction += " --continuous-actions"
        os.system(instruction)
