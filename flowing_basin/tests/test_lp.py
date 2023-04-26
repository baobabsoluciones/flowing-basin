from flowing_basin.core import Instance
from flowing_basin.solvers import LPModel, LPConfiguration

config = LPConfiguration(
    volume_objectives={
        "dam1": 59627.42324,
        "dam2": 31010.43613642857
        },
    volume_shortage_penalty=3,
    volume_exceedance_bonus=0.1,
    limit_zones_penalty=50,
    step_min=4,
    MIPGap=0.04
)
instance = Instance.from_json("../data/input_example1.json")
lp = LPModel(config=config, instance=instance)
lp.LPModel_print()

