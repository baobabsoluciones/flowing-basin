from flowing_basin.core import Instance
from flowing_basin.solvers import PsoRboConfiguration, PsoRbo
from flowing_basin.tools import RiverBasin
from datetime import datetime
import numpy as np
import random


def test_relvars_from_flows():

    river_basin = RiverBasin(instance=instance, mode="linear", num_scenarios=config.num_particles)
    flows = np.array(
        [
            [
                [
                    random.uniform(0, instance.get_max_flow_of_channel(dam_id))
                    for _ in range(config.num_particles)
                ]
                for dam_id in instance.get_ids_of_dams()
            ]
            for _ in range(instance.get_largest_impact_horizon())
        ]
    )  # Array of shape num_time_steps x num_dams x num_scenarios
    river_basin.deep_update_flows(flows, fast_mode=True)
    clipped_flows1 = river_basin.all_past_clipped_flows
    # print(clipped_flows1)
    relvars = pso_rbo.relvars_from_flows(clipped_flows1)
    river_basin.reset()
    river_basin.deep_update_relvars(relvars, fast_mode=True)
    clipped_flows2 = river_basin.all_past_clipped_flows
    # print(clipped_flows2)
    assert (clipped_flows1 == clipped_flows2).all()  # noqa


def test_generate_rbo_flows():

    flows = pso_rbo.generate_rbo_flows(10)
    obj_fun_values1 = - pso_rbo.pso.calculate_cost(pso_rbo.pso.reshape_as_swarm(flows), is_relvars=False)
    obj_fun_mean, obj_fun_max, obj_fun_min = np.mean(obj_fun_values1), np.max(obj_fun_values1), np.min(obj_fun_values1)
    print("Mean:", obj_fun_mean, "Max:", obj_fun_max, "Min:", obj_fun_min)
    obj_fun_values2 = - pso_rbo.pso.calculate_cost(pso_rbo.pso.reshape_as_swarm(
        pso_rbo.relvars_from_flows(flows)
    ), is_relvars=True)
    assert (np.abs(obj_fun_values1 - obj_fun_values2) < 1e-4).all()  # noqa


SAVE_SOLUTION = False
EXAMPLE = 3
NUM_DAMS = 2
K_PARAMETER = 2
USE_RELVARS = True
RANDOM_BIASED_FLOWS = True
PROB_BELOW_HALF = 0.15
RANDOM_BIASED_SORTING = True
COMMON_RATIO = 0.6
FRAC_RBO_INIT = 0.5
TIME_LIMIT_MINUTES = 0.4

path_instance = f"../instances/instances_big/instance{EXAMPLE}_{NUM_DAMS}dams_1days.json"
path_sol = f"../solutions/instance{EXAMPLE}_PSO-RBO_{NUM_DAMS}dams_1days" \
           f"_time{datetime.now().strftime('%Y-%m-%d_%H-%M')}.json"

instance = Instance.from_json(path_instance)
config = PsoRboConfiguration(
    volume_shortage_penalty=3,
    volume_exceedance_bonus=0,
    startups_penalty=50,
    limit_zones_penalty=0,
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
    inertia_weight=0.4424113459034113,
    random_biased_flows=RANDOM_BIASED_FLOWS,
    prob_below_half=PROB_BELOW_HALF,
    random_biased_sorting=RANDOM_BIASED_SORTING,
    common_ratio=COMMON_RATIO,
    fraction_rbo_init=FRAC_RBO_INIT,
)
print(config)

pso_rbo = PsoRbo(instance=instance, config=config)
print(pso_rbo.pso.config)

# Test `relvars_from_flows` and `generate_rbo_flows` methods (slow)
# test_relvars_from_flows()
# test_generate_initial_solutions()

# Find solution
status = pso_rbo.solve()
print("Status:", status)

# Check solution
sol_inconsistencies = pso_rbo.solution.check()
if sol_inconsistencies:
    raise Exception(f"There are inconsistencies in the given solution: {sol_inconsistencies}")
print("Optimal solution:", pso_rbo.solution.data)

# Save solution
if SAVE_SOLUTION:
    pso_rbo.solution.to_json(path_sol)
