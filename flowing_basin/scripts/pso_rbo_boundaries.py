from flowing_basin.core import Instance
from flowing_basin.solvers import PsoRboConfiguration, PsoRbo
from itertools import product


# Values of discrete parameters to consider
USE_RELVARS = [True, False]
BOUNDARIES = ['periodic', 'nearest', 'intermediate', 'shrink', 'reflective']
FRACTION_RBO_INIT = 0.5
TIME_LIMIT_MINS = 15

# Instances to solve
INSTANCES = ['Percentile25', 'Percentile75']
NUMS_DAMS = [2, 6]

# Solve every instance with every configuration
# This will take 2 * 2 * 5 * 2 * 1/4h = 10h to execute
for relvar, boundary, example, num_dams in product(USE_RELVARS, BOUNDARIES, INSTANCES, NUMS_DAMS):

    path_instance = f"../instances/instances_big/instance{example}_{num_dams}dams_1days.json"
    path_sol = f"../solutions/test_pso_rbo_boundaries/instance{example}_PSO-RBO_{num_dams}dams_1days" \
               f"_v={relvar}_b={boundary}.json"
    print(f"---- Solving instance {example} with {num_dams} dams using v={relvar} and b={boundary} ----")

    instance = Instance.from_json(path_instance)
    config = PsoRboConfiguration(
        volume_shortage_penalty=3,
        volume_exceedance_bonus=0,
        startups_penalty=50,
        limit_zones_penalty=50,
        volume_objectives={
            dam_id: instance.get_historical_final_vol_of_dam(dam_id) for dam_id in instance.get_ids_of_dams()
        },
        num_particles=200,
        cognitive_coefficient=2.905405139888455,
        social_coefficient=0.4232260541405988,
        inertia_weight=0.4424113459034113,
        use_relvars=relvar,
        max_relvar=1,
        bounds_handling=boundary,
        topology='star',
        random_biased_flows=True,
        prob_below_half=0.15,
        random_biased_sorting=True,
        common_ratio=0.6,
        fraction_rbo_init=FRACTION_RBO_INIT,
        max_time=TIME_LIMIT_MINS * 60,
        flow_smoothing=2,
        mode="linear",
    )

    pso_rbo = PsoRbo(instance=instance, config=config)

    # Find solution
    status = pso_rbo.solve()
    print("Status:", status)

    # Check solution
    sol_inconsistencies = pso_rbo.solution.check()
    if sol_inconsistencies:
        raise Exception(f"There are inconsistencies in the given solution: {sol_inconsistencies}")
    print("Optimal solution:", pso_rbo.solution.data)

    # Save solution
    pso_rbo.solution.to_json(path_sol)
    print(f"Created file {path_sol}.")

