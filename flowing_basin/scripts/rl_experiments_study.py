"""
RL Study Experiments

This script allows analyzing several trained agent(s).
"""

from flowing_basin.solvers.rl import ReinforcementLearning
import json


def get_experiment_agents(experiment_config_path: str, general_config: str = None) -> list[str]:
    with open(experiment_config_path, 'r') as f:
        experiment_data = json.load(f)
    agents = experiment_data["agents"]
    if general_config is not None:
        agents = [agent for agent in agents if general_config in agent]
    return agents


def main():

    # general = "G0"
    # general = "G2"
    # all_agents = get_experiment_agents("experiments/experiment13.json", general)
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

