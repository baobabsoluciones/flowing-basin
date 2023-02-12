from flowing_basin.core import Instance

# "../" means "go one step up"; in this case, to the flowing-basin directory
instance = Instance.from_json("../data/input_example2.json")

# Make sure data follows schema
schema_errors = instance.check_schema()
if schema_errors:
    raise Exception(f"Data does not follow schema. Errors: {schema_errors}")

# Make sure there are no inconsistencies
inconsistencies = instance.check_inconsistencies()
if inconsistencies:
    raise Exception(f"There are inconsistencies in the data: {inconsistencies}")

# Print general info
time = 0
print("data property:", instance.data)
print("dictionary:", instance.to_dict())
print("IDs of dams:", instance.get_ids_of_dams())
print("incoming flow:", instance.get_incoming_flow(time))
print("price:", instance.get_price(time))

# Print dam info
for dam in instance.get_ids_of_dams():
    print("-----", dam, "-----")
    print("initial volume:", instance.get_initial_vol_of_dam(dam))
    print("min volume:", instance.get_min_vol_of_dam(dam))
    print("max volume:", instance.get_max_vol_of_dam(dam))
    print("initial lags:", instance.get_initial_lags_of_channel(dam))
    print("relevant lags:", instance.get_relevant_lags_of_dam(dam))
    print("max flow:", instance.get_max_flow_of_channel(dam))
    print("flow limit observations:", instance.get_flow_limit_obs_for_channel(dam))
    print("unregulated flow:", instance.get_unregulated_flow_of_dam(time, dam))

# Plot channel limit flow points
# from matplotlib import pyplot as plt
# limit_flow_points = instance.get_flow_limit_obs_for_channel("dam2")
# plt.plot(limit_flow_points["observed_vols"], limit_flow_points["observed_flows"])
# plt.show()
