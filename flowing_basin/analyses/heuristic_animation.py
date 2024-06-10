"""
heuristic_animation.py
This script creates an animation of the heuristic solving a single dam.
"""

from flowing_basin.core import Instance, Solution
from flowing_basin.tools import RiverBasin
from flowing_basin.solvers import Baseline
from flowing_basin.solvers.heuristic import HeuristicSingleDam
from math import log
from matplotlib import pyplot as plt
import os
import numpy as np

INSTANCE = "Percentile80"
CONFIG = 'G8'
FILENAME_HEURISTIC = f"heuristic_animation/heuristic_{INSTANCE}_{CONFIG}"
FILENAME_RAND = f"heuristic_animation/random_{INSTANCE}_{CONFIG}"
FPS = 5

baseline = Baseline(solver="Heuristic", general_config=CONFIG)
config = baseline.config
num_dams = baseline.num_dams
instance = Instance.from_name(INSTANCE, num_dams=num_dams)
if instance.get_num_dams() > 1:
    raise ValueError(f"Please specify a configuration with only one dam.")
dam_id = instance.get_ids_of_dams()[0]


def create_heuristic_animation():

    """Create a .gif file showing the heuristic solving the instance step-by-step."""

    # Solve the given instance with the heuristic, recording every partial solution
    heuristic = HeuristicSingleDam(
        instance=instance, config=config, dam_id=dam_id, flow_contribution=instance.get_all_incoming_flows(),
        bias_weight=log(config.prob_below_half) / log(0.5), record_solutions=True
    )
    heuristic.solve()
    print(f"Finished solving.")

    # Save solution plots
    images = []
    for i, sol in enumerate(heuristic.solutions):
        fig, ax = plt.subplots()
        sol.plot_solution_for_dam(dam_id=dam_id, ax=ax)
        # plt.show()
        repeat = 10 if i == len(heuristic.solutions) - 1 else 1
        for j in range(repeat):
            image = f"{FILENAME_HEURISTIC}_{i}_{j}.png"
            plt.savefig(image)
            print(f"Saved {image}.")
            images.append(image)
        plt.close()

    # Create the animation using ImageMagick
    animation = FILENAME_HEURISTIC + '.gif'
    os.system('convert -delay {} +dither +remap -layers Optimize {} "{}"'.
              format(100 // FPS, ' '.join(['"' + img + '"' for img in images]), animation))
    print(f"Crated animation {animation}.")
    for img in images:
        if os.path.exists(img):
            os.remove(img)
            print(f"Deleted {img}.")


def create_random_figure():

    """Crate a .png file with the solution of a random agent"""

    flows = np.random.rand(
        instance.get_largest_impact_horizon(), instance.get_num_dams(), 1
    ) * instance.get_max_flow_of_channel(dam_id)
    river_basin = RiverBasin(
        instance=instance, flow_smoothing=config.flow_smoothing, max_relvar=config.max_relvar, do_history_updates=False,
        mode="linear"
    )
    river_basin.deep_update_flows(flows)

    start_datetime, end_datetime, _, _, _, solution_datetime = instance.get_instance_current_datetimes()
    sol_dict = dict(
        instance_datetimes=dict(
            start=start_datetime,
            end_decisions=end_datetime
        ),
        instance_name=instance.get_instance_name(),
        solution_datetime=solution_datetime,
        solver="Heuristic",
        configuration=config.to_dict(),
        objective_function=river_basin.get_objective_function_value(config=config),
        dams=[
            dict(
                id=dam_id,
                flows=river_basin.all_past_clipped_flows[:, instance.get_order_of_dam(dam_id) - 1, :].squeeze().tolist(),
                power=river_basin.all_past_powers[dam_id].squeeze().tolist(),
                volume=river_basin.all_past_volumes[dam_id].squeeze().tolist(),
                objective_function_details=river_basin.get_objective_function_details(dam_id, config=config)
            )
        ],
        price=instance.get_all_prices(),
    )
    sol = Solution.from_dict(sol_dict)

    fig, ax = plt.subplots()
    sol.plot_solution_for_dam(dam_id=dam_id, ax=ax)
    plt.savefig(FILENAME_RAND + '.png')
    plt.show()


if __name__ == "__main__":

    # create_heuristic_animation()
    create_random_figure()
