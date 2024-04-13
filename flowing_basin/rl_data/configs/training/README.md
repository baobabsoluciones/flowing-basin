- `T0XYZ`: training a SAC agent.
- `T1XYZ`: same as T0, but saving the best agent on evaluation rewards (EvalCallback)
and not rollout rewards (CheckpointCallback).
- `T2XYZ`: same as T1, but with a more aggressive learning rate (1e-3 instead of 3e-4).
- `T3XYZ`: same as T1, but with a smaller replay buffer size (100_000 instead of 1_000_000).
This is necessary to avoid an ArrayMemoryError with very large observation arrays,
such as rl-A113G0O2R1T.* with observations of shape (1736,).
- `T4XYZ`: this combines T2 and T3 (more aggressive learning rate and smaller replay buffer).

In all of these, `X` can be nothing, `0`, `1`, `2`, `3` or `4`:
- If `X` is nothing or `0`, the training is done for 99000 timesteps.
- If `X` is `1`, the training is done for 495 timesteps.
- If `X` is `2`, the training is done for 990 timesteps.
- If `X` is `3`, the training is done for 15 timesteps
(this is for action types with big block sizes, which are extremely slow to train).
- If `X` is `4`, the training is done for 198000 timesteps.

Note one episode (day) is equivalent to `99 // num_actions_block` timesteps.

In all of these, `Y` can be nothing, `0`, `1`, `2`, `3` or `4`:
- If `Y` is nothing or `0`, the actor and critic networks have \[256, 256] neurons.
- If `Y` is `1`, the actor and critic networks have \[512, 512] neurons.
- If `Y` is `2`, the actor and critic networks have \[512, 256] neurons.
- If `Y` is `3`, the actor and critic networks have \[`N`, 256] neurons,
where `N` equals the size of the observation space (i.e., the number of actor input neurons).
- If `Y` is `4`, the actor network has \[512, 512] neurons and the critic \[256, 256].
- If `Y` is `5`, the actor network has \[512, 256] neurons and the critic \[256, 256].
- If `Y` is `6`, the actor network has \[`N`, 256] neurons and the critic \[256, 256],
where `N` equals the size of the observation space (i.e., the number of actor input neurons).
- If `Y` is `7`, the actor network has \[1536, 1536] neurons and the critic \[1024, 1024].

In all of these, `Z` can be nothing, `0`, `1`, or `2`:
- If `Z` is nothing or `0`, the RL algorithm used is SAC.
- If `Z` is `1`, the RL algorithm used is A2C.
- If `Z` is `2`, the RL algorithm used is PPO.
