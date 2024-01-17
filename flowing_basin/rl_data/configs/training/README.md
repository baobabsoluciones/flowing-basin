- `T0X`: training a SAC agent with Nx256x256xM networks.
- `T1X`: same as T0, but saving the best agent on evaluation rewards (EvalCallback)
and not rollout rewards (CheckpointCallback).
- `T2X`: same as T1, but with a more aggressive learning rate (1e-3 instead of 3e-4).
- `T3X`: same as T1, but with a smaller replay buffer size (100_000 instead of 1_000_000).
This is necessary to avoid an ArrayMemoryError with very large observation arrays,
such as rl-A113G0O2R1T.* with observations of shape (1736,).
- `T4X`: this combines T2 and T3 (more aggressive learning rate and smaller replay buffer).

In all of these, `X` can be nothing, `1` or `2`:
- If `X` is nothing, the training is done for 99000 timesteps.
- If `X` is `1`, the training is done for 495 timesteps.
- If `X` is `2`, the training is done for 990 timesteps.
- If `X` is `3`, the training is done for 15 timesteps
(this is for action types with big block sizes, which are extremely slow to train).

Note one episode (day) is equivalent to `99 // num_actions_block` timesteps.
