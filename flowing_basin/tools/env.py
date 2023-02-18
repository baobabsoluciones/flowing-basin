from flowing_basin.core import Instance
from flowing_basin.tools import RiverBasin


class Environment:

    def __init__(self, instance: Instance, paths_power_models: dict[str, str]):

        self.river_basin = RiverBasin(instance=instance, paths_power_models=paths_power_models)

        self.limits = {
            "incoming_flow_max": instance.get_max_incoming_flow(),
            "price_max": max(instance.get_price(0, num_steps=instance.get_total_num_time_steps()))
        }
        for dam_id in instance.get_ids_of_dams():
            self.limits[dam_id] = {
                "vol_max": instance.get_max_vol_of_dam(dam_id),
                "vol_min": instance.get_min_vol_of_dam(dam_id),
                "flow_max": instance.get_max_flow_of_channel(dam_id),
                "unregulated_flow_max": instance.get_max_unregulated_flow_of_dam(dam_id)
            }
