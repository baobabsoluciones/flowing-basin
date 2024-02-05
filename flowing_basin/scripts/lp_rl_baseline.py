from flowing_basin.core import Instance
from flowing_basin.solvers import LPModel, LPConfiguration

TIME_LIMIT_MINUTES = 15
EXAMPLES = [f"Percentile{percentile:02}" for percentile in range(0, 110, 10)]

for example in EXAMPLES:

    path_instance = f"../instances/instances_base/instance{example}.json"
    path_sol = f"../solutions/rl_baselines/instance{example}_LPmodel_k=0_NoPowerPenalties.json"

    instance = Instance.from_json(path_instance)
    config = LPConfiguration(
        volume_shortage_penalty=0,
        volume_exceedance_bonus=0,
        startups_penalty=0.,
        limit_zones_penalty=0.,
        volume_objectives={
            dam_id: instance.get_historical_final_vol_of_dam(dam_id) for dam_id in instance.get_ids_of_dams()
        },
        MIPGap=0.01,
        max_time=TIME_LIMIT_MINUTES * 60,
        flow_smoothing=0,
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
    lp.solution.data["instance_name"] = example
    lp.solution.to_json(path_sol)
    print("Saved solution in ", path_sol)
