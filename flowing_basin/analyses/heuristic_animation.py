"""
heuristic_animation.py
This script creates an animation of the heuristic solving a single dam.
"""

from flowing_basin.core import Instance
from flowing_basin.solvers import Baseline
from flowing_basin.solvers.heuristic import HeuristicSingleDam
from math import log
from matplotlib import pyplot as plt
import os

INSTANCE = "Percentile80"
CONFIG = 'G8'
FILENAME = f"heuristic_animation/heuristic_{INSTANCE}_{CONFIG}"
FPS = 5

if __name__ == "__main__":
    baseline = Baseline(solver="Heuristic", general_config=CONFIG)
    config = baseline.config
    num_dams = baseline.num_dams
    instance = Instance.from_name(INSTANCE, num_dams=num_dams)
    dam_id = instance.get_ids_of_dams()[0]
    heuristic = HeuristicSingleDam(
        instance=instance, config=config, dam_id=dam_id, flow_contribution=instance.get_all_incoming_flows(),
        bias_weight=log(config.prob_below_half) / log(0.5), record_solutions=True
    )
    heuristic.solve()
    print(f"Finished solving.")
    images = []
    for i, sol in enumerate(heuristic.solutions):
        fig, ax = plt.subplots()
        sol.plot_solution_for_dam(dam_id=dam_id, ax=ax)
        # plt.show()
        repeat = 10 if i == len(heuristic.solutions) - 1 else 1
        for j in range(repeat):
            image = f"{FILENAME}_{i}_{j}.png"
            plt.savefig(image)
            print(f"Saved {image}.")
            images.append(image)
        plt.close()

    # Create the animation using ImageMagick
    animation = FILENAME + '.gif'
    os.system('convert -delay {} +dither +remap -layers Optimize {} "{}"'.
              format(100 // FPS, ' '.join(['"' + img + '"' for img in images]), animation))
    print(f"Crated animation {animation}.")
    for img in images:
        if os.path.exists(img):
            os.remove(img)
            print(f"Deleted {img}.")
