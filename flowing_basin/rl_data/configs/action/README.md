Single-step continuous actions:
- `A0`: the agent gives relative variations of flow.
- `A1`: the agent gives the exiting flows directly.

Multi-step continuous actions:
- `A110`: the agent gives the exiting flows for the next 9 (99 // 11) timesteps.
- `A111`: the agent gives the exiting flows for the next 20 (~ 99 // 5) timesteps.
- `A112`: the agent gives the exiting flows for the next 33 (99 // 3) timesteps.
- `A113`: the agent gives the exiting flows for the next 99 (99 // 1) timesteps.

Adjustments actions:
- `A21`: the agent adjusts the actions of the solution (initially rl-greedy's) until it no longer improves.
- `A22`: like A21, but the initial greedy actions are taken with a greeediness of 80% instead of 100%
(which has similar performance and avoids clipping all positive adjustments).
- `A23`: like A22, but Gaussian noise with 0.15 standard deviation is added to the initial greedy actions.
- `A24`: like A21, but the initial actions are completely random (no greediness).
- `A25`: like A21, with these differences:
  - The agent is allowed to worsen the solution, and the episode always ends after 10 adjustments.
  - The agent is given a bonus of +100 when its solution finally exceeds rl-greedy's solution.
  - The agent is given a penalty of -100 and the episode terminates without exceeding the initial rl-greedy solution.
  - The reward is the _relative_ difference (in %) between the current and previous solutions
    (e.g., the reward is +20 when the solution is improved by 20%).
- `A26`: like A25, but using the absolute difference instead of the relative difference.

Single-step discrete actions:
- `A31`: the agent can only choose among the given optimal flows.
- `A32`: the agent chooses the flow from the given number of discretization levels for each dam.
- `A33`: the agent chooses the turbine count and the flow level within this number of turbines.
- `A313`: like A31, but making the choice for the next 99 (99 // 1) timesteps.
- `A323`: like A32, but making the choice for the next 99 (99 // 1) timesteps.
- `A333`: like A33, but making the choice for the next 99 (99 // 1) timesteps.