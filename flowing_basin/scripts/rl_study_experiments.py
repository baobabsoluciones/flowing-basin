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


if __name__ == "__main__":

    general = 'G3'
    # regex = f"rl-A3.{general}"
    # agents = ReinforcementLearning.get_all_agents(regex)
    agents = get_experiment12_agents([general])
    print(f"Found {len(agents)} agents in folder:", agents)
    regex = agents
    # ReinforcementLearning.barchart_training_times(regex, hours=True)
    # ReinforcementLearning.print_training_times(regex, hours=True, csv_filepath="reports/training_times_experiments1-9.csv")
    # ReinforcementLearning.print_max_avg_incomes(
    #     regex, permutation='GOART', baselines=["MILP", "rl-greedy"],
    #     csv_filepath=f"reports/results_{general}_experiment12.csv"
    # )
    ReinforcementLearning.plot_all_training_curves(regex, baselines=["MILP", "rl-greedy", "rl-random"])

    # ReinforcementLearning.print_spaces(regex)
    # ReinforcementLearning.print_training_times(regex)
    # print("Average training time:", ReinforcementLearning.get_avg_training_time(regex))
    # ReinforcementLearning.barchart_instances_incomes(regex, baselines=["Heuristic", "MILP", "rl-greedy", "rl-random"])
