from flowing_basin.core import Instance
from flowing_basin.solvers import PSO


instance = Instance.from_json("../data/input_example1.json")
paths_power_models = {
    "dam1": "../ml_models/model_E1.sav",
    "dam2": "../ml_models/model_E2.sav",
}
pso = PSO(
    instance=instance,
    paths_power_models=paths_power_models,
    num_particles=10,
)

print(pso.instance.data)
print(pso.solution.data)
print(pso.river_basin.get_state())
print(pso.num_particles)
print(pso.num_dimensions)
