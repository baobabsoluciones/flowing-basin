- `R0`: income minus startup or limit zone penalties, divided by the episode's maximum price.
- `R1`: like R0, but with a penalty for not fulfilling the flow smoothing parameter.
- `R21`: like R1, but subtracting rl-greedy's average reward: `reward = rew_agent - max(0., avg_rew_greedy)`.
- `R22X`: like R21, but using a ratio in addition:
`reward = (rew_agent - max(0., avg_rew_greedy)) / max(1., avg_rew_greedy)`.
- `R23X`, like R1, but using rl-greedy's average reward to estimate MILP's average reward
and use it as a reference.
- `R24`: like R23, but also estimating rl-random's average reward.

In these, `X` can be nothing, `1`, `2` or `3`:
- If `X` is nothing, then rl-greedy's average reward is computed at the beginning
of the episode and the value is kept unchanged in subsequent periods.
- If `X` is `1`, then rl-greedy's average reward is recomputed on every period
by running rl-greedy for the rest of the episode.
- If `X` is `2`, then rl-greedy's average reward is recomputed on every period
by running rl-greedy for the next `max_sight // 2` PERIODS.
- If `X` is `3`, then rl-greedy's average reward is recomputed on every period
by running rl-greedy for the next `max_sight` PERIODS.
