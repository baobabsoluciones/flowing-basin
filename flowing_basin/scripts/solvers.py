from flowing_basin.core import Instance
from flowing_basin.solvers import LPModel, LPConfiguration, PSO, PSOConfiguration, PsoRbo, PsoRboConfiguration
import matplotlib.pyplot as plt
from itertools import product
import os
import warnings

SOLVERS = ['MILP']
INSTANCES = ['Percentile75']
NUMS_DAMS = [2, 4]
POWER_PENALTIES = [True, False]
# 3 * 2 * 4 * 2 * 15min = 48 * 15min = 12h

PLOT_SOL = False
SAVE_SOLUTION = True
TIME_LIMIT_MINUTES = 15

for solver, example, num_dams, power_penalty in product(SOLVERS, INSTANCES, NUMS_DAMS, POWER_PENALTIES):

    print(
        f" --- Using solver {solver} to solve instance {example} with {num_dams} dams"
        f"{' (no power penalty)' if not power_penalty else ''}"
        f"... --- "
    )

    # Paths
    path_instance = f"../instances/instances_big/instance{example}_{num_dams}dams_1days.json"
    path_sol = {
        'MILP': f"../solutions/test_milp/instance{example}_MILP_{num_dams}dams_1days_VolExceed{'_NoPowerPenalty' if not power_penalty else ''}.json",
        'PSO': f"../solutions/test_pso/instance{example}_PSO_{num_dams}dams_1days_VolExceed{'_NoPowerPenalty' if not power_penalty else ''}.json",
        'PSO-RBO': f"../solutions/test_pso_rbo_boundaries/instance{example}_PSO-RBO_{num_dams}dams_1days_v=False_b=intermediate_VolExceed{'_NoPowerPenalty' if not power_penalty else ''}.json"
    }[solver]
    if os.path.exists(path_sol):
        print(f"Solution '{path_sol}' already exists. Skipping to next iteration...")
        continue

    # Instance
    instance = Instance.from_json(path_instance)

    # Configuration
    config_milp_pso_psorbo = dict(
        volume_shortage_penalty=3,
        volume_exceedance_bonus=0.035,
        startups_penalty=50 if power_penalty else 0,
        limit_zones_penalty=50 if power_penalty else 0,
        volume_objectives={
            dam_id: (
                 instance.get_min_vol_of_dam(dam_id) + (
                    instance.get_max_vol_of_dam(dam_id) - instance.get_min_vol_of_dam(dam_id)
                 ) / 2
            ) for dam_id in instance.get_ids_of_dams()
        },
        max_time=TIME_LIMIT_MINUTES * 60,
        flow_smoothing=2,
    )
    config_pso_psorbo = dict(
        max_relvar=1,
        mode="linear",
        num_particles=200,
        cognitive_coefficient=2.905405139888455,
        social_coefficient=0.4232260541405988,
        inertia_weight=0.4424113459034113
    )
    config = {
        'MILP': lambda: LPConfiguration(
            **config_milp_pso_psorbo,
            MIPGap=0.01,
        ),
        'PSO': lambda: PSOConfiguration(
            **config_milp_pso_psorbo,
            **config_pso_psorbo,
            use_relvars=True,
            bounds_handling='periodic',
        ),
        'PSO-RBO': lambda: PsoRboConfiguration(
            **config_milp_pso_psorbo,
            **config_pso_psorbo,
            use_relvars=False,
            bounds_handling='intermediate',
            random_biased_flows=True,
            prob_below_half=0.15,
            random_biased_sorting=True,
            common_ratio=0.6,
            fraction_rbo_init=0.5,
        )
    }[solver]()

    # Solver object and solution
    solver_object = {
        'MILP': lambda: LPModel(config=config, instance=instance),
        'PSO': lambda: PSO(config=config, instance=instance),
        'PSO-RBO': lambda: PsoRbo(config=config, instance=instance)
    }[solver]()
    solver_object.solve()

    # Check solution
    sol_inconsistencies = solver_object.solution.check()
    if sol_inconsistencies:
        raise Exception(f"There are inconsistencies in the given solution: {sol_inconsistencies}")
    if not solver_object.solution.complies_with_flow_smoothing(
        flow_smoothing=2,
        initial_flows={
            dam_id: instance.get_initial_lags_of_channel(dam_id)[0]
            for dam_id in instance.get_ids_of_dams()
        }
    ):
        warnings.warn("The solution does not comply with the flow smoothing parameter.")
    print("Optimal solution:", solver_object.solution.data)

    # Save solution
    if SAVE_SOLUTION:
        solver_object.solution.to_json(path_sol)
        print(f"Created file '{path_sol}'.")

    # Plot simple solution graph for each dam
    if PLOT_SOL:
        for dam_id in instance.get_ids_of_dams():
            fig, ax = plt.subplots()
            solver_object.solution.plot_solution_for_dam(dam_id, ax)
            plt.show()
    print()
