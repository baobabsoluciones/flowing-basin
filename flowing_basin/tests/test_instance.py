from flowing_basin.core import Instance
import pandas as pd

EXAMPLES = ['1', '3']
NUMS_DAMS = [i for i in range(1, 9)]
EPSILON = 0.01  # Rounding error tolerance in asserts

for example in EXAMPLES:
    for num_dams in NUMS_DAMS:

        print(f" ---- INSTANCE {example} WITH {num_dams} DAMS ---- ")

        # "../" means "go one step up"; in this case, to the flowing-basin directory
        instance = Instance.from_json(f"../instances/instances_big/instance{example}_{num_dams}dams_1days.json")

        # Make sure data follows schema and has no inconsistencies
        inconsistencies = instance.check()
        if inconsistencies:
            raise Exception(f"There are inconsistencies in the data: {inconsistencies}")

        # Print general info
        time = 0
        print("data property:", instance.data)
        print("dictionary:", instance.to_dict())
        start, end = instance.get_start_end_datetimes()
        print("start of decisions:", start)
        print("end of decisions:", end)
        print("start of information:", instance.get_start_information_datetime())
        print("start of information offset:", instance.get_start_information_offset())
        print("decision horizon:", instance.get_decision_horizon())
        print("impact horizon:", instance.get_largest_impact_horizon())
        print("information horizon:", instance.get_information_horizon())
        print("IDs of dams:", instance.get_ids_of_dams())
        print("incoming flow:", instance.get_incoming_flow(time))
        print("incoming flows (next 12 steps):", instance.get_incoming_flow(time, num_steps=12))
        print("maximum incoming flow:", instance.get_max_incoming_flow())
        print("price:", instance.get_price(time))
        print("prices (next 12 steps):", instance.get_price(time, num_steps=12))

        # Average inflow
        calculated_avg_inflow = instance.calculate_total_avg_inflow()
        print("total avg inflow:", calculated_avg_inflow)
        date = instance.get_start_end_datetimes()[0].date()
        daily_inflow_data = pd.read_pickle("../data/history/historical_data_daily_avg_inflow.pickle")
        avg_inflow = daily_inflow_data.loc[date, 'total_avg_inflow']
        print("total avg inflow (as stored in data):", avg_inflow)
        if instance.get_num_dams() > 1:
            assert abs(calculated_avg_inflow - avg_inflow) < EPSILON
            # If the number of dams is 1, the total avg inflow will NOT be "correct",
            # as this instance will not have the unregulated flows of dam2

        # Print dam info
        for dam in instance.get_ids_of_dams():
            print(dam)
            print("order:", instance.get_order_of_dam(dam))
            print("initial volume:", instance.get_initial_vol_of_dam(dam))
            print("final volume:", instance.get_historical_final_vol_of_dam(dam))
            print("min volume:", instance.get_min_vol_of_dam(dam))
            print("max volume:", instance.get_max_vol_of_dam(dam))
            print("initial lags:", instance.get_initial_lags_of_channel(dam))
            print("relevant lags:", instance.get_relevant_lags_of_dam(dam))
            print("verification lags:", instance.get_verification_lags_of_dam(dam))
            print("max flow:", instance.get_max_flow_of_channel(dam))
            print("flow limit observations:", instance.get_flow_limit_obs_for_channel(dam))
            print("turbined flow observations:", instance.get_turbined_flow_obs_for_power_group(dam))
            print("startup flows:", instance.get_startup_flows_of_power_group(dam))
            print("shutdown flows:", instance.get_shutdown_flows_of_power_group(dam))
            print("unregulated flow:", instance.get_unregulated_flow_of_dam(time, dam))
            print("unregulated flows (next 12 steps):", instance.get_unregulated_flow_of_dam(time, dam, num_steps=12))
            print("maximum unregulated flow:", instance.get_max_unregulated_flow_of_dam(dam))
            print()

        # Plot channel limit flow points
        # from matplotlib import pyplot as plt
        # limit_flow_points = instance.get_flow_limit_obs_for_channel("dam2")
        # plt.plot(limit_flow_points["observed_vols"], limit_flow_points["observed_flows"])
        # plt.show()
