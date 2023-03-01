from flowing_basin.core import Instance
from flowing_basin.solvers import PSOFlows
import numpy as np


instance = Instance.from_json("../data/input_example1.json")
paths_power_models = {
    "dam1": "../ml_models/model_E1.sav",
    "dam2": "../ml_models/model_E2.sav",
}
pso = PSOFlows(
    instance=instance,
    paths_power_models=paths_power_models,
    num_particles=3,
)

# Test particle and flows equivalence ---- #
decisionsABC = np.array(
        [[[6.79, 8, 1], [6.58, 7, 1]], [[7.49, 9, 1], [6.73, 8.5, 1]]]
    )
padding = np.array([[[0, 0, 0], [0, 0, 0]] for _ in range(instance.get_num_time_steps() - decisionsABC.shape[0])])
decisionsABC_all_periods = np.concatenate([decisionsABC, padding])
print("flows:", decisionsABC_all_periods)
swarm = pso.input_to_swarm(decisionsABC_all_periods)
print("relvars -> swarm:", swarm)
print("swarm -> flows:", pso.swarm_to_input(swarm))
print("first particle:", swarm[0])
print("first particle -> flows:", pso.particle_to_flows(swarm[0]))

# Test objective function ---- #
print(pso.objective_function(swarm))
print(pso.river_basin.get_state())
