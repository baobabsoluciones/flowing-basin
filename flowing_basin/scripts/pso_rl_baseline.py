from flowing_basin.core import Instance
from flowing_basin.solvers import PSO, PSOConfiguration
from flowing_basin.solvers.rl import ReinforcementLearning
from itertools import product
import os

TIME_LIMIT_MINUTES = 0.05
GENERAL = ["G0", "G1"]
# EXAMPLES = [f"Percentile{percentile:02}" for percentile in range(0, 110, 10)]
EXAMPLES = ["Percentile00"]

for general_config, example in product(GENERAL, EXAMPLES):

    path_instance = f"../instances/instances_base/instance{example}.json"
    instance = Instance.from_json(path_instance)

    general_config_name = "k=2_PowerPenalties" if general_config == "G0" else "k=0_NoPowerPenalties"
    sol_filename = f"instance{example}_PSO_{general_config_name}.json"
    path_dir = os.path.join(ReinforcementLearning.baselines_folder, general_config)
    path_sol = os.path.join(path_dir, sol_filename)

    general_config_folder, general_config_class = ReinforcementLearning.configs_info['G']
    general_config_path = os.path.join(general_config_folder, general_config + ".json")
    general_config_dict = general_config_class.from_json(general_config_path).to_dict()
    config = PSOConfiguration.from_dict(
        dict(
            **general_config_dict,
            max_time=TIME_LIMIT_MINUTES * 60,
            num_particles=200,
            use_relvars=False,
            max_relvar=0.5,
            bounds_handling="periodic",
            topology="star",
            cognitive_coefficient=2.905405139888455,
            social_coefficient=0.4232260541405988,
            inertia_weight=0.4424113459034113,
        )
    )

    pso = PSO(config=config, instance=instance)
    pso.solve()

    # Check solution
    sol_inconsistencies = pso.solution.check()
    if sol_inconsistencies:
        raise Exception(f"There are inconsistencies in the given solution: {sol_inconsistencies}")
    print("Optimal solution:", pso.solution.data)

    # Save solution
    pso.solution.data["instance_name"] = example
    pso.solution.to_json(path_sol)
    print("Saved solution in ", path_sol)
