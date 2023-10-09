from flowing_basin.core import Instance
from flowing_basin.solvers import LPModel, LPConfiguration
from datetime import datetime
import matplotlib.pyplot as plt
from itertools import product

INSTANCES = ['Percentile25', 'Percentile75']
NUMS_DAMS = [2, 6]

PLOT_SOL = False
SAVE_SOLUTION = True
TIME_LIMIT_MINUTES = 15

for example, num_dams in product(INSTANCES, NUMS_DAMS):

    path_instance = f"../instances/instances_big/instance{example}_{num_dams}dams_1days.json"
    path_sol = f"../solutions/test_milp/instance{example}_MILP_{num_dams}dams_1days.json"

    instance = Instance.from_json(path_instance)
    config = LPConfiguration(
        volume_shortage_penalty=3,
        volume_exceedance_bonus=0,
        startups_penalty=50,
        limit_zones_penalty=50,
        volume_objectives={
            dam_id: instance.get_historical_final_vol_of_dam(dam_id) for dam_id in instance.get_ids_of_dams()
        },
        MIPGap=0.01,
        time_limit_seconds=TIME_LIMIT_MINUTES * 60,
        flow_smoothing=2,
    )

    lp = LPModel(config=config, instance=instance)
    lp.LPModel_print()
    lp.solve()

    # Check solution
    sol_inconsistencies = lp.solution.check()
    if sol_inconsistencies:
        raise Exception(f"There are inconsistencies in the given solution: {sol_inconsistencies}")
    print("Optimal solution:", lp.solution.data)

    # Save solution
    if SAVE_SOLUTION:
        lp.solution.to_json(path_sol)

    # Plot simple solution graph for each dam
    if PLOT_SOL:
        for dam_id in instance.get_ids_of_dams():
            fig, ax = plt.subplots()
            lp.solution.plot_solution_for_dam(dam_id, ax)
            plt.show()
