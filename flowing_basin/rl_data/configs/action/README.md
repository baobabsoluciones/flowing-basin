- `A0`: the agent gives relative variations of flow.
- `A1`: the agent gives the exiting flows directly.
- `A110`: the agent gives the exiting flows for the next 9 (99 // 11) timesteps.
- `A111`: the agent gives the exiting flows for the next 20 (~ 99 // 5) timesteps.
- `A112`: the agent gives the exiting flows for the next 33 (99 // 3) timesteps.
- `A113`: the agent gives the exiting flows for the next 99 (99 // 1) timesteps.
- `A21`: the agent adjusts the actions of the solution (initially rl-greedy's) until it no longer improves.
- `A22`: like A21, but the initial greedy actions are taken with a greeediness of 80% instead of 100%
(which has similar performance and avoids clipping all positive adjustments).