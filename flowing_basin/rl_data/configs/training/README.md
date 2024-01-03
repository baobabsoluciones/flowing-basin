- `T0X`: training a SAC agent with Nx256x256xM networks.
- `T1X`: same as T0, but saving the best agent on evaluation rewards (EvalCallback)
and not rollout rewards (CheckpointCallback).
- `T2X`: same as T1, but with a more aggressive learning rate (1e-3 instead of 3e-4).

In all of these, `X` can be nothing, `1` or `2`:
- If `X` is nothing, the training is done for 1000 episodes.
- If `X` is `1`, the training is done for 5 episodes.
- If `X` is `2`, the training is done for 10 episodes.