from flowing_basin.core import Instance, Solution
from flowing_basin.solvers import PSO, PSOConfiguration, LPConfiguration, LPModel


def get_lp_sol_and_obj_fun(config: LPConfiguration, instance: Instance) -> tuple[Solution, float]:

    lp = LPModel(config=config, instance=instance)
    lp.solve()

    sol = lp.solution
    obj_fun = lp.get_objective()

    return sol, obj_fun


def get_pso_obj_fun(config: PSOConfiguration, instance: Instance, solution: Solution) -> float:

    pso = PSO(instance=instance, config=config, solution=solution)

    pso.river_basin.deep_update_flows(solution.get_exiting_flows_array())
    print(pso.river_basin.history.to_string())
    for dam_id in instance.get_ids_of_dams():
        print(f"{dam_id}'s income from energy: {pso.river_basin.history[f'{dam_id}_income'].sum()}")
        print(f"{dam_id}'s number of startups: {pso.river_basin.history[f'{dam_id}_startups'].sum()}")
        print(f"{dam_id}'s number of limit zones: {pso.river_basin.history[f'{dam_id}_limits'].sum()}")
    print(pso.objective_function_values_env())

    return pso.get_objective()


if __name__ == '__main__':

    # Configuration
    volume_shortage_penalty = 3
    volume_exceedance_bonus = 0
    startups_penalty = 50
    limit_zones_penalty = 50
    volume_objectives = {
        "dam1": 59627.42324,
        "dam2": 31010.43613642857,
        "dam3_dam2copy": 31010.43613642857,
        "dam4_dam2copy": 31010.43613642857,
        "dam5_dam1copy": 59627.42324,
        "dam6_dam1copy": 59627.42324,
        "dam7_dam2copy": 31010.43613642857,
        "dam8_dam1copy": 59627.42324,
    }
    lp_config = LPConfiguration(
        volume_shortage_penalty=volume_shortage_penalty,
        volume_exceedance_bonus=volume_exceedance_bonus,
        startups_penalty=startups_penalty,
        limit_zones_penalty=limit_zones_penalty,
        volume_objectives=volume_objectives,
        MIPGap=0.01,
        time_limit_seconds=-1,  # Set below
        flow_smoothing=2
    )
    pso_config = PSOConfiguration(
        volume_shortage_penalty=volume_shortage_penalty,
        volume_exceedance_bonus=volume_exceedance_bonus,
        startups_penalty=startups_penalty,
        limit_zones_penalty=limit_zones_penalty,
        volume_objectives=volume_objectives,
        mode="linear",
        use_relvars=False,
        num_particles=1,
        max_iterations=0,
        max_time=0,
        cognitive_coefficient=2.905405139888455,
        social_coefficient=0.4232260541405988,
        inertia_weight=0.4424113459034113
    )

    # Instances
    nums_dams = [i for i in range(2, 9)]  # [i for i in range(1, 9)]
    examples = ['3', '1']  # ['1', '3'] + [f'_intermediate{i}' for i in range(11)]
    for num_dams in nums_dams:
        for example in examples:
            lp_config.time_limit_seconds = num_dams * 10
            instance = Instance.from_json(f"../instances/instances_big/instance{example}_{num_dams}dams_1days.json")
            print(f"Solving instance {example} with {num_dams} dams...")
            lp_sol, lp_obj = get_lp_sol_and_obj_fun(lp_config, instance)
            pso_obj = get_pso_obj_fun(pso_config, instance, lp_sol)
            print(f"LP's obj is {lp_obj} and PSO's is {pso_obj}.")
            if abs(lp_obj - pso_obj) > 10:
                raise Exception(f"LP and PSO do not give the same obj fun for instance {example} with {num_dams} dams.")

    # Results (is LP equivalent to simulator?) -
    # 1dam instance1 NO
    # 1dam instance3 instance_intermediate0 .. instance_intermediate10 YES
    # 2dams instance1 NO
    # 2dams instance3 instance_intermediate0 .. instance_intermediate10 YES
    # 3dams ...
    # ...
