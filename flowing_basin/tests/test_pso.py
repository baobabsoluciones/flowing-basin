from flowing_basin.core import Instance
from flowing_basin.solvers import PSOConfiguration, PSO
from itertools import product

SAVE_SOLUTION = True
INSTANCES = ['1', '3', 'Percentile25', 'Percentile75']
NUMS_DAMS = [2, 6]
# INSTANCES = ['1']
# NUMS_DAMS = [2]
K_PARAMETER = 2
USE_RELVARS = True
TIME_LIMIT_MINUTES = 15

for example, num_dams in product(INSTANCES, NUMS_DAMS):

    path_instance = f"../instances/instances_big/instance{example}_{num_dams}dams_1days.json"
    path_sol = f"../solutions/test_pso/instance{example}_PSO_{num_dams}dams_1days.json"

    instance = Instance.from_json(path_instance)
    config = PSOConfiguration(
        volume_shortage_penalty=3,
        volume_exceedance_bonus=0,
        startups_penalty=50,
        limit_zones_penalty=50,
        volume_objectives={
            dam_id: instance.get_historical_final_vol_of_dam(dam_id) for dam_id in instance.get_ids_of_dams()
        },
        use_relvars=USE_RELVARS,
        max_relvar=1,
        flow_smoothing=K_PARAMETER,
        mode="linear",
        num_particles=200,
        max_time=TIME_LIMIT_MINUTES * 60,
        cognitive_coefficient=2.905405139888455,
        social_coefficient=0.4232260541405988,
        inertia_weight=0.4424113459034113
    )

    pso = PSO(instance=instance, config=config)

    # Find solution
    status = pso.solve()
    print("Status:", status)

    # Check solution
    sol_inconsistencies = pso.solution.check()
    if sol_inconsistencies:
        raise Exception(f"There are inconsistencies in the given solution: {sol_inconsistencies}")
    assert pso.solution.complies_with_flow_smoothing(
        flow_smoothing=2,
        initial_flows={
            dam_id: instance.get_initial_lags_of_channel(dam_id)[0]
            for dam_id in instance.get_ids_of_dams()
        }
    ), "The solution does not comply with the flow smoothing parameter"
    print("Optimal solution:", pso.solution.data)

    # Save solution
    if SAVE_SOLUTION:
        pso.solution.to_json(path_sol)
        print(f"Created file '{path_sol}'.")
