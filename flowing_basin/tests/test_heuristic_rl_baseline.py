from flowing_basin.core import Instance
from flowing_basin.solvers import HeuristicConfiguration, Heuristic

TIME_LIMIT_MINUTES = 15
EXAMPLES = [f"Percentile{percentile:02}" for percentile in range(0, 110, 10)]

for example in EXAMPLES:

    path_instance = f"../instances/instances_base/instance{example}.json"
    path_sol = f"../solutions/rl_baselines/instance{example}_Heuristic_k=2_PowerPenalties.json"

    instance = Instance.from_json(path_instance)
    config = HeuristicConfiguration(
        mode="linear",
        volume_shortage_penalty=0,
        volume_exceedance_bonus=0,
        startups_penalty=50.,
        limit_zones_penalty=50.,
        volume_objectives={
            dam_id: instance.get_historical_final_vol_of_dam(dam_id) for dam_id in instance.get_ids_of_dams()
        },
        flow_smoothing=2,
    )

    # Putting greedy=True is redundant, since random bias is False by default, but just to make sure...
    heuristic = Heuristic(config=config, instance=instance, greedy=True)
    heuristic.solve()

    # Check solution
    sol_inconsistencies = heuristic.solution.check()
    if sol_inconsistencies:
        raise Exception(f"There are inconsistencies in the given solution: {sol_inconsistencies}")
    print("Optimal solution:", heuristic.solution.data)

    # Save solution
    heuristic.solution.to_json(path_sol)
    print("Saved solution in ", path_sol)
