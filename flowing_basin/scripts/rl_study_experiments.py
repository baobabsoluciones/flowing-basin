"""
RL Study Experiments

This script allows analyzing several trained agent(s).
"""

from flowing_basin.solvers.rl import ReinforcementLearning
from itertools import product


def get_experiment11_agents(generals: list[str] = None, rewards: list[str] = None) -> list[str]:

    if generals is None:
        generals = ['G0', 'G1']
    if rewards is None:
        rewards = ["R1", "R22"]

    actions = ["A1"]
    observations = ["O231"]
    trainings = [f"T{norm_digit}00{algo_digit}" for norm_digit in ["1", "5", "6"] for algo_digit in ["0", "1", "2"]]
    agents = []
    for action, general, observation, reward, training in product(actions, generals, observations, rewards, trainings):
        if training == "T1000":
            training = "T1"
        agent = f"rl-{action}{general}{observation}{reward}{training}"
        agents.append(agent)
    print(f"Returning {len(agents)} agent identifiers:", agents)
    return agents


def get_experiment12_agents(general_configs: list[str]) -> list[str]:
    experiment_agents = [
        "rl-A31G2O231R1T1402",
        "rl-A31G3O231R1T1402",
        "rl-A1G2O231R22T14",
        "rl-A1G3O231R22T14",
        "rl-A1G2O2R1T14",
        "rl-A1G3O2R1T14",
        "rl-A23G2O3R1T74",
        "rl-A23G2O3R1T748",
        "rl-A23G3O3R1T74",
        "rl-A23G3O3R1T748",
        "rl-A113G3O2R22T342",
        "rl-A113G3O2R22T749",
        "rl-A113G3O232R22T74",
        "rl-A113G3O232R22T748"
    ]
    experiment_agents = [
        agent for agent in experiment_agents if any(agent.find(gen_config) != -1 for gen_config in general_configs)
    ]
    return experiment_agents


def get_experiment12cont_agents(general_configs: list[str]) -> list[str]:
    experiment_agents = [
        "rl-A21G2O3R1T74",
        "rl-A21G2O3R1T748",
        "rl-A21G3O3R1T74",
        "rl-A21G3O3R1T748",
    ]
    experiment_agents = [
        agent for agent in experiment_agents if any(agent.find(gen_config) != -1 for gen_config in general_configs)
    ]
    return experiment_agents


def get_experiment13_agents(general_configs: list[str]) -> list[str]:
    num_replications = 3
    normal_2dams = [f"rl-A1G0O2R1T1-{i}" for i in range(num_replications)]
    normal_6dams = [f"rl-A1G2O2R1T14-{i}" for i in range(num_replications)]
    adjustments_2dams = [f"rl-A21G0O3R1T3-{i}" for i in range(num_replications)]
    adjustments_new_2dams = [f"rl-A25G0O3R1T3-{i}" for i in range(num_replications)]
    adjustments_6dams = [f"rl-A21G2O3R1T74-{i}" for i in range(num_replications)]
    adjustments_new_6dams = [f"rl-A25G2O3R1T74-{i}" for i in range(num_replications)]
    discrete_2dams = [f"rl-A31G0O231R1T1002-{i}" for i in range(num_replications)]
    discrete_6dams = [f"rl-A31G2O231R1T1402-{i}" for i in range(num_replications)]
    adjusted_2dams = [f"rl-A1G0O231R22T1-{i}" for i in range(num_replications)]
    adjusted_6dams = [f"rl-A1G2O231R22T14-{i}" for i in range(num_replications)]
    all_agents = [
        *normal_2dams, *normal_6dams, *adjustments_2dams, *adjustments_new_2dams, *adjustments_6dams,
        *adjustments_new_6dams, *discrete_2dams, *discrete_6dams, *adjusted_2dams,*adjusted_6dams
    ]
    all_agents = [agent[:-2] if agent.endswith("-0") else agent for agent in all_agents]
    experiment_agents = [
        agent for agent in all_agents if any(agent.find(gen_config) != -1 for gen_config in general_configs)
    ]
    return experiment_agents


def main():

    # general = "G0"
    # general = "G2"
    # all_agents = get_experiment13_agents([general])
    # print(f"Giving {len(all_agents)} agents for {general}:", sorted(all_agents))
    # agents = ReinforcementLearning.get_all_agents(all_agents)
    # print(f"Found {len(agents)} agents in folder:", sorted(agents))
    # regex = agents
    regex = [
        "rl-A1G0O2R1T1-2",
        "rl-A31G0O231R1T1002-1",
        "rl-A21G0O3R1T3-2",
        # "rl-A1G2O2R1T14-2",
        # "rl-A31G2O231R1T1402-2",
        # "rl-A21G2O3R1T74-1"
    ]
    colors = {
        "rl-A1G0O2R1T1-2": "green",
        "rl-A31G0O231R1T1002-1": "blue",
        "rl-A21G0O3R1T3-2": "red",
        "rl-A1G2O2R1T14-2": "green",
        "rl-A31G2O231R1T1402-2": "blue",
        "rl-A21G2O3R1T74-1": "red"
    }
    names = {
        "rl-A1G0O2R1T1-2": "RL (Continuous, SAC)",
        "rl-A31G0O231R1T1002-1": "RL (Discrete, PPO)",
        "rl-A21G0O3R1T3-2": "RL (Adjustments, SAC)",
        "rl-A1G2O2R1T14-2": "RL (Continuous, SAC)",
        "rl-A31G2O231R1T1402-2": "RL (Discrete, PPO)",
        "rl-A21G2O3R1T74-1": "RL (Adjustments, SAC)",
        "rl-greedy": "Greedy",
        "rl-random": "Random"
    }

    # training_times = ReinforcementLearning.get_training_times(agents)
    # total_time = sum(training_times)
    # print("Total time (min):", total_time)
    # ReinforcementLearning.histogram_training_times(regex, hours=True, filename='reports/rl_hist_training_times', filter_timesteps=99_000)
    ReinforcementLearning.barchart_training_times(
        regex, hours=True, names_mapping=names, colors_mapping=colors,
        title="Two reservoirs", filename="reports/rl_times_2_reservoirs"
    )
    # ReinforcementLearning.print_training_times(regex, hours=True, csv_filepath="reports/training_times_experiments1-9.csv")
    # print("Average training time:", ReinforcementLearning.get_avg_training_time(regex))

    # ReinforcementLearning.print_max_avg_incomes(
    #     regex, permutation='GOART', baselines=["MILP", "rl-greedy"],
    #     csv_filepath=f"reports/results_{general}_experiment13.csv"
    # )
    # ReinforcementLearning.print_avg_constraint_violations(regex, permutation='GOART')
    # ReinforcementLearning.barchart_instances_incomes(regex, baselines=["Heuristic", "MILP", "rl-greedy", "rl-random"])
    ReinforcementLearning.plot_all_training_curves(
        regex, baselines=["MILP", "rl-greedy", "rl-random"], names_mapping=names, colors_mapping=colors,
        title="Two reservoirs", filename="reports/rl_best_agents_2_reservoirs"
    )
    # ReinforcementLearning.print_spaces(regex)


if __name__ == "__main__":

    main()

