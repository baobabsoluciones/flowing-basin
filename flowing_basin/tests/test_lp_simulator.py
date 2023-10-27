from flowing_basin.core import Instance, Solution
from flowing_basin.solvers import PSO, PSOConfiguration, LPConfiguration, LPModel


def get_lp_sol_and_obj_fun(config: LPConfiguration, instance: Instance) -> tuple[Solution, float]:

    lp = LPModel(config=config, instance=instance)
    lp.solve()

    sol = lp.solution
    print("LP solution:", sol.data)

    obj_fun = lp.get_objective()
    for dam_id in instance.get_ids_of_dams():
        print(f"LP details for {dam_id}:", sol.get_objective_details(dam_id))

    return sol, obj_fun


def get_pso_obj_fun(config: PSOConfiguration, instance: Instance, solution: Solution) -> float:

    pso = PSO(instance=instance, config=config, solution=solution)

    # Details from simulator
    pso.river_basin.deep_update_flows(solution.get_flows_array())
    print(pso.river_basin.history.to_string())
    for dam_id in instance.get_ids_of_dams():
        print(f"PSO details for {dam_id}:")
        print(f"\t{dam_id}'s income from energy: {pso.river_basin.history[f'{dam_id}_income'].sum()}")
        print(f"\t{dam_id}'s number of startups: {pso.river_basin.history[f'{dam_id}_startups'].sum()}")
        print(f"\t{dam_id}'s number of limit zones: {pso.river_basin.history[f'{dam_id}_limits'].sum()}")
        print("\t", pso.env_objective_function_details(dam_id))

    return pso.get_objective()


if __name__ == '__main__':

    # Configuration
    volume_shortage_penalty = 3
    volume_exceedance_bonus = 0
    startups_penalty = 50
    limit_zones_penalty = 50
    flow_smoothing = 2
    lp_config = LPConfiguration(
        volume_shortage_penalty=volume_shortage_penalty,
        volume_exceedance_bonus=volume_exceedance_bonus,
        startups_penalty=startups_penalty,
        limit_zones_penalty=limit_zones_penalty,
        volume_objectives=dict(),  # Set below
        MIPGap=0.01,
        time_limit_seconds=-1,  # Set below
        flow_smoothing=flow_smoothing
    )
    pso_config = PSOConfiguration(
        volume_shortage_penalty=volume_shortage_penalty,
        volume_exceedance_bonus=volume_exceedance_bonus,
        startups_penalty=startups_penalty,
        limit_zones_penalty=limit_zones_penalty,
        volume_objectives=dict(),  # Set below
        mode="linear",
        use_relvars=False,
        num_particles=1,
        max_iterations=0,
        max_time=0,
        cognitive_coefficient=2.905405139888455,
        social_coefficient=0.4232260541405988,
        inertia_weight=0.4424113459034113,
        flow_smoothing=flow_smoothing
    )

    # Instances
    nums_dams = [2, 6]  # [i for i in range(1, 9)]
    # examples = [f'Percentile{i*10:02d}' for i in range(11)]  # ['1', '3'] + [f'Percentile{i*10:02d}' for i in range(11)]
    examples = ['Percentile25', 'Percentile75']
    for num_dams in nums_dams:
        for example in examples:

            instance = Instance.from_json(f"../instances/instances_big/instance{example}_{num_dams}dams_1days.json")
            print(f"Solving instance {example} with {num_dams} dams...")

            lp_config.volume_objectives = pso_config.volume_objectives = {
                dam_id: instance.get_historical_final_vol_of_dam(dam_id) for dam_id in instance.get_ids_of_dams()
            }
            lp_config.max_time = num_dams * 30

            lp_sol, lp_obj = get_lp_sol_and_obj_fun(lp_config, instance)
            pso_obj = get_pso_obj_fun(pso_config, instance, lp_sol)
            print(f"LP's obj is {lp_obj} and PSO's is {pso_obj}.")
            if abs(lp_obj - pso_obj) > 10:
                raise Exception(f"LP and PSO do not give the same obj fun for instance {example} with {num_dams} dams.")

    # Results (is LP equivalent to simulator?) -
    # 1dam instance1 instance3 instancePercentile00 .. instancePercentile100 YES
    # 2dams instance1 instancePercentile00 NO
    # 3dams ...
    # ...

    # ISSUE Nº 1
    # The issue is that, when startup_flow = shutdown_flow (1.428571429 m3/s in dam1 and 2.423809524 m3/s in dam2),
    # and the turbined_flow happens to equal this flow, the LP model considers the number of dams that is
    # most convenient to it (sometimes it is 0, and sometimes it is 1; it is not consistent)
    # I do not think this is a big problem, since the PSO-RBO can find a similar solution w/ almost the same value
    # by taking 1.41 m3/s when 0 power groups are convenient or 1.43 m3/s when 1 power group is convenient

    # ISSUE Nº 2
    # In instances with many dams, the LP model does not give correct estimations in the beginning
    # This is ssignaled by the gap going up and down, instead of decreasing steadily
    # This is fixed by giving more computational time to the LP model in these instances
