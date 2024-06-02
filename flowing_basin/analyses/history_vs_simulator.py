from flowing_basin.core import Instance
from flowing_basin.tools import RiverBasin
from flowing_basin.solvers.common import CONSTANTS_PATH, get_episode_length, confidence_interval, lighten_color
from flowing_basin.solvers.rl import RLEnvironment
from cornflow_client.core.tools import load_json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter


NUM_DAMS = 2
NUM_REPLICATIONS = 500
MAIN_COLOR = 'red'
SECOND_COLOR = 'black'
SHOW_INDIVIDUAL_INSTANCES = False
VERBOSE = 0
LOAD_DATA = True
DATA_PATH = 'simulation_vs_history/all_relative_differences.npy'
FIG_PATH = 'simulation_vs_history/relative_differences.eps'


historial_data_path = "../data/history/historical_data_reliable_only.pickle"
constants = Instance.from_dict(load_json(CONSTANTS_PATH.format(num_dams=NUM_DAMS)))
length_episodes = get_episode_length(constants=constants)
historical_data = pd.read_pickle(historial_data_path)
timestep_hours = constants.get_time_step_seconds() / 3600
if VERBOSE > 0:
    print("Historical data:")
    print(historical_data.head().to_string())

historical_data["incomes"] = sum(
    historical_data[f"{dam_id}_power"] * timestep_hours * historical_data["price"]
    for dam_id in constants.get_ids_of_dams()
)
if VERBOSE > 0:
    print("Incomes:")
    print(historical_data["incomes"])


def get_all_relative_differences():

    all_relative_differences = []
    for replication in range(NUM_REPLICATIONS):

        instance = RLEnvironment.create_instance(
            length_episodes=length_episodes, constants=constants, historical_data=historical_data
        )
        inconsistencies = instance.check()
        if inconsistencies:
            print(
                f"Replication {replication}. The instance was SKIPPED due to "
                f"inconsistent data from missing values (since we are using only the reliable data)."
            )
            continue

        start = instance.get_start_decisions_datetime()
        initial_row = historical_data.index[historical_data["datetime"] == start].tolist()[0]
        last_row = initial_row + instance.get_largest_impact_horizon() - 1
        print(f"Replication {replication}. Instance initial row: {initial_row}; last row: {last_row}")
        if VERBOSE > 0:
            print("Instance historical data:")
            print(historical_data.loc[initial_row:last_row].to_string())

        real_outflows = np.array(
            [
                [[dam1_flow], [dam2_flow]]
                for dam1_flow, dam2_flow in zip(
                    historical_data["dam1_flow"].loc[initial_row:last_row],
                    historical_data["dam2_flow"].loc[initial_row:last_row],
                )
            ]
        )
        if VERBOSE > 0:
            print("Instance original outflows:", real_outflows.shape)
            print(real_outflows)

        real_acc_incomes = historical_data["incomes"].loc[initial_row:last_row].cumsum()
        real_acc_incomes = real_acc_incomes.to_numpy()
        if VERBOSE > 0:
            print("Instance real accumulated incomes:")
            print(real_acc_incomes)

        simulated_acc_incomes = []
        river_basin = RiverBasin(instance=instance, num_scenarios=1, do_history_updates=False, mode="linear")
        for flow in real_outflows:
            river_basin.update(flow)
            simulated_acc_incomes.append(river_basin.get_acc_income().item())
        simulated_acc_incomes = np.array(simulated_acc_incomes)
        if VERBOSE > 0:
            print("Instance simulated accumulated incomes:")
            print(simulated_acc_incomes)

        relative_difference = (simulated_acc_incomes - real_acc_incomes) / real_acc_incomes
        if VERBOSE > 0:
            print("Instance difference (%):")
            print(relative_difference)

        all_relative_differences.append(relative_difference)

    all_relative_differences = np.array(all_relative_differences)
    return all_relative_differences


def percent_format(x, _):
    return f'{int(x * 100)}%'


def main():

    if not LOAD_DATA:
        all_relative_diffs = get_all_relative_differences()
        np.save(DATA_PATH, all_relative_diffs)
    else:
        all_relative_diffs = np.load(DATA_PATH)

    means = np.mean(all_relative_diffs, axis=0)
    lower_bounds, upper_bounds = confidence_interval(np.transpose(all_relative_diffs))

    print("Starting plots...")
    plt.figure(figsize=(10, 6))
    plt.axhline(y=0, color='black', linewidth=0.75)
    time_steps = np.arange(len(means))

    if SHOW_INDIVIDUAL_INSTANCES:
        label_added = False
        for relative_diff in all_relative_diffs:
            plot_kwagrs = dict()
            if not label_added:
                plot_kwagrs.update(label='Relative Difference in Each Day')
                label_added = True
            plt.plot(
                time_steps, relative_diff, color=lighten_color(SECOND_COLOR), **plot_kwagrs
            )

    plt.plot(time_steps, means, label='Mean Relative Difference', color=MAIN_COLOR, linewidth=2.5)
    plt.fill_between(
        time_steps, lower_bounds, upper_bounds, color=lighten_color(MAIN_COLOR, amount=0.25),
        label='95% Confidence Interval'
    )

    plt.xlabel('Day Periods (0 to 99)')
    plt.ylabel('Difference in Accumulated Income')
    plt.title('Difference Between Real and Predicted Accumulated Income')
    plt.legend()
    plt.grid(True)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_format))

    plt.savefig(FIG_PATH)
    plt.show()


if __name__ == "__main__":

    main()
