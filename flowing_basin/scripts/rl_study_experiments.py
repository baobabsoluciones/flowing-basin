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


if __name__ == "__main__":

    # general = 'G1'
    regex = get_experiment11_agents()
    # agents = ReinforcementLearning.get_all_agents(regex)
    # print(f"Found {len(agents)} agents in folder:", agents)
    ReinforcementLearning.barchart_training_times(regex, hours=False)
    # ReinforcementLearning.print_training_times(regex, hours=True, csv_filepath="reports/training_times_experiments1-9.csv")
    # ReinforcementLearning.print_max_avg_incomes(
    #     regex, permutation='GOART', baselines=["MILP", "rl-greedy", "rl-random"],
    #     csv_filepath=f"reports/results_{general}_experiment11.csv"
    # )
    # ReinforcementLearning.plot_all_training_curves(regex, baselines=["MILP", "rl-greedy", "rl-random"])

    # ReinforcementLearning.print_spaces(regex)
    # ReinforcementLearning.print_training_times(regex)
    # print("Average training time:", ReinforcementLearning.get_avg_training_time(regex))
    # ReinforcementLearning.barchart_instances_incomes(regex, baselines=["rl-greedy"])
