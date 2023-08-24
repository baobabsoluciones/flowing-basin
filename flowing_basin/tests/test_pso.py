from flowing_basin.core import Instance, Solution
from flowing_basin.solvers import PSOConfiguration, PSO
from datetime import datetime
import os

NEW_SOLUTION = False
EXAMPLE = 3
NUM_DAMS = 2
NUM_DAYS = 1
K_PARAMETER = 2
USE_RELVARS = True

# Instance we want to solve
instance = Instance.from_json(f"../data/input_example{EXAMPLE}_{NUM_DAMS}dams_{NUM_DAYS}days.json")

# PSO object to find the solution
config = PSOConfiguration(
    volume_shortage_penalty=3,
    volume_exceedance_bonus=0.1,
    startups_penalty=50,
    limit_zones_penalty=0,
    volume_objectives={
        "dam1": 59627.42324,
        "dam2": 31010.43613642857,
        "dam3_dam2copy": 31010.43613642857,
        "dam4_dam2copy": 31010.43613642857,
        "dam5_dam1copy": 59627.42324,
        "dam6_dam1copy": 59627.42324,
        "dam7_dam2copy": 31010.43613642857,
        "dam8_dam1copy": 59627.42324,
    },
    use_relvars=USE_RELVARS if NEW_SOLUTION else False,
    max_relvar=1,
    flow_smoothing=K_PARAMETER if NEW_SOLUTION else 0,
    mode="linear",
    num_particles=200,
    num_iterations=100_000,
    timeout=8*60,
    cognitive_coefficient=2.905405139888455,
    social_coefficient=0.4232260541405988,
    inertia_weight=0.4424113459034113
)
pso = PSO(
    instance=instance,
    config=config,
)

# Solution ---- #

path_parent = "../data"
if NEW_SOLUTION:
    # Optimal solution found by PSO
    dir_name = f"output_instance{EXAMPLE}_PSO_{NUM_DAMS}dams_{NUM_DAYS}days_{datetime.now().strftime('%Y-%m-%d %H.%M')}" \
               f"_mode={pso.config.mode}_k={pso.config.flow_smoothing}{'_no_relvars' if not config.use_relvars else ''}"
    status = pso.solve()
    print("status:", status)
    print("solver info:", pso.solver_info)
else:
    # Given solution
    dir_name = "RL_model_2023-08-02 18.46_sol_example1"
    pso.solution = Solution.from_json("../data/RL_model_2023-08-02 18.46_sol_example1.json")

sol_inconsistencies = pso.solution.check()
if sol_inconsistencies:
    raise Exception(f"There are inconsistencies in the given solution: {sol_inconsistencies}")
print(pso.solution.check())
print("optimal solution:", pso.solution.data)
pso.river_basin.deep_update(pso.solution.to_flows(), is_relvars=False)
print("optimal solution's objective function values:", pso.objective_function_values_env())
print("optimal solution's full objective function value:", pso.objective_function_env())
print("optimal solution's full objective function value (cornflow method):", pso.get_objective())
print(pso.river_basin.history.to_string())
pso.save_solution_info(path_parent=path_parent, dir_name=dir_name)
pso.river_basin.history.to_excel(os.path.join(path_parent, dir_name, "history.xlsx"))

# Search for best PSO parameters ---- #
# options_search = {"c1": [0.1, 5], "c2": [0.1, 5], "w": [0.3, 0.9], "k": [1, 2], "p": 1}
# # We need to put "k": [1, 2], "p": 1 for the method to work,
# # even though "k" and "p" are not used in GlobalBestPSO
# print(pso.search_best_options(options_search, num_iters_selection=30, num_iters_each_test=20))
# Result: 'c1': 2.905405139888455, 'c2': 0.4232260541405988, 'w': 0.4424113459034113 | cost=-10114.100850959443

# Study configuration ---- #
# print(pso.study_configuration(options))
