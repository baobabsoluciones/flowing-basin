- `R0`: income minus startup or limit zone penalties, divided by the episode's maximum price.
- `R1`: like R0, but with a penalty for not fulfilling the flow smoothing parameter.
- `R21`: like R1, but subtracting rl-greedy's average reward: `reward = rew_agent - max(0, avg_rew_greedy)`.
- `R22`: like R21, but using a ratio in addition:
`reward = (rew_agent - avg_rew_greedy) / avg_rew_greedy if avg_rew_greedy > 0 else rew_agent`.