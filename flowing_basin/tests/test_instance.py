from flowing_basin.core import Instance

# "../" means "go one step up"; in this case, to the flowing-basin directory
instance = Instance.from_json("../data/input.json")

dam = "dam1"
print("data property:", instance.data)
print("dictionary:", instance.to_dict())
print("IDs of dams:", instance.get_ids_of_dams())
print("initial volume:", instance.get_initial_vol_of_dam(dam))
print("min volume:", instance.get_min_vol_of_dam(dam))
print("max volume:", instance.get_max_vol_of_dam(dam))
print("unregulated flow:", instance.get_unregulated_flow_of_dam(dam))
print("initial lags:", instance.get_initial_lags_of_channel(dam))
print("relevant lags:", instance.get_relevant_lags_of_dam(dam))
print("points", instance.get_max_flow_points_of_channel(dam))
print("incoming flow:", instance.get_incoming_flow(0))
