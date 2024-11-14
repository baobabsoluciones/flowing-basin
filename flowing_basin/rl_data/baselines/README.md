This folder contains the solutions of the following solvers:

- **MILP**.
- **PSO** with random initialization, and PSO with RBO initialization ("PSO-RBO").
- **Greedy** approach ("rl-greedy" or "RLgreedy").
- A random agent ("rl-random" or "RLrandom").
- A heuristic solver.

The name of each folder corresponds to the configuration of the
simulation model used (explained in `flowing_basin/rl_data/configs/general/README.md`):

- `G0`: 2 dams; flow smoothing of 2; startup and limit zone penalty of 50€.
- `G01`: 2 dams; maximum flow variation of 20%; startup and limit zone penalty of 50€.
- `G1`: 2 dams; no flow smoothing, startup or limit zone penalty.
- `G2`: 6 dams; flow smoothing of 2; startup and limit zone penalty of 50€.
- `G21`: 6 dams; maximum flow variation of 20%; startup and limit zone penalty of 50€.
- `G3`: 6 dams; no flow smoothing, startup or limit zone penalty.
