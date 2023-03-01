from flowing_basin.core import Instance
from flowing_basin.solvers import PSOFlowVariations
import numpy as np


instance = Instance.from_json("../data/input_example1.json")
paths_power_models = {
    "dam1": "../ml_models/model_E1.sav",
    "dam2": "../ml_models/model_E2.sav",
}
pso = PSOFlowVariations(
    instance=instance,
    paths_power_models=paths_power_models,
    num_particles=3,
)

# Test instance and river basin are correctly saved ---- #
# print(pso.instance.data)
# print(pso.solution.data)
# print(pso.river_basin.get_state())
# print(pso.num_particles)
# print(pso.num_dimensions)

# Test particle and relvars equivalence ---- #
decisionsNVABC = np.array(
    [
        [
            [0.5, 0.75, 1],
            [0.5, 0.75, 1],
        ],
        [
            [0.25, 0.5, 1],
            [0.25, 0.5, 1],
        ],
    ]
)
padding = np.array([[[0, 0, 0], [0, 0, 0]] for _ in range(instance.get_num_time_steps() - decisionsNVABC.shape[0])])
decisionsNVABC_all_periods = np.concatenate([decisionsNVABC, padding])
print("relvars:", decisionsNVABC_all_periods)
swarm = pso.input_to_swarm(decisionsNVABC_all_periods)
print("relvars -> swarm:", swarm)
print("swarm -> relvars:", pso.swarm_to_input(swarm))
print("first particle:", swarm[0])
print("first particle -> relvars:", pso.particle_to_relvar(swarm[0]))

# Test objective function ---- #
print(pso.objective_function(swarm))
