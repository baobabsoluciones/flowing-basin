from flowing_basin.core import Instance
from flowing_basin.tools import RiverBasin
import numpy as np


def check_volume(
    old_state: dict,
    state: dict,
    time_step: float,
    min_vol_dam1: float = 0,
    max_vol_dam1: float = None,
    min_vol_dam2: float = 0,
    max_vol_dam2: float = None,
):

    """
    Check volumes are consistent throughout states
    """

    # Calculate dam1 volume
    dam1_vol = np.clip(
        old_state["dam1"]["vol"]
        + (
            old_state["next_incoming_flow"]
            + old_state["dam1"]["next_unregulated_flow"]
            - state["dam1"]["lags"][:, 0]
        )
        * time_step,
        min_vol_dam1,
        max_vol_dam1,
    )

    # Calculate dam2 volume
    dam2_vol = np.clip(
        old_state["dam2"]["vol"]
        + (
            state["dam1"]["turbined_flow"]
            + old_state["dam2"]["next_unregulated_flow"]
            - state["dam2"]["lags"][:, 0]
        )
        * time_step,
        min_vol_dam2,
        max_vol_dam2,
    )

    # Assert calculations are the same as the state values
    for (
            state_dam1_vol_scenario,
            dam1_vol_scenario,
            state_dam2_vol_scenario,
            dam2_vol_scanario,
    ) in zip(state["dam1"]["vol"], dam1_vol, state["dam2"]["vol"], dam2_vol):
        assert round(state_dam1_vol_scenario, 4) == round(
            dam1_vol_scenario, 4
        ), f"Volume of dam1 should be {dam1_vol_scenario} but it is {state_dam1_vol_scenario}"
        assert round(state_dam2_vol_scenario, 4) == round(
            dam2_vol_scanario, 4
        ), f"Volume of dam2 should be {dam2_vol_scanario} but it is {state_dam2_vol_scenario}"


def check_income(
    income: float | np.ndarray,
    old_state: dict,
    state: dict,
    time_step: float,
):

    """
    Check the income has the correct value
    """

    income_calc = (
        old_state["next_price"]
        * (state["dam1"]["power"] + state["dam2"]["power"])
        * time_step
        / 3600
    )
    for (income_calc_scenario, income_scenario) in zip(income_calc, income):
        assert round(income_calc_scenario, 4) == round(
            income_scenario, 4
        ), f"Income should be {income_calc_scenario} but it is {income_scenario}"


def test_river_basin(
    river_basin: RiverBasin,
    decisions: np.ndarray,
    check_incomes: bool = True,
    check_volumes: bool = True,
):

    state = river_basin.get_state()
    print("initial state:", state)
    old_state = state

    for i, flows in enumerate(decisions):

        print(f">>>> decision {i}")
        river_basin.update(flows)
        income = river_basin.get_income()
        print(f"income={income}")
        state = river_basin.get_state()
        print(f"state after decision {i}:", state)
        print(
            f"income obtained after decision {i} (EUR): "
            f"price (EUR/MWh) * power (MW) * time_step (hours) = "
            f"{old_state['next_price']} * ({state['dam1']['power']} + {state['dam2']['power']}) * {river_basin.instance.get_time_step_seconds() / 3600} = "
            f"{income}"
        )

        if check_incomes:
            check_income(
                old_state=old_state,
                state=state,
                time_step=river_basin.instance.get_time_step_seconds(),
                income=income,
            )

        if check_volumes:
            check_volume(
                old_state=old_state,
                state=state,
                time_step=river_basin.instance.get_time_step_seconds(),
                min_vol_dam1=river_basin.instance.get_min_vol_of_dam("dam1"),
                max_vol_dam1=river_basin.instance.get_max_vol_of_dam("dam1"),
                min_vol_dam2=river_basin.instance.get_min_vol_of_dam("dam2"),
                max_vol_dam2=river_basin.instance.get_max_vol_of_dam("dam2"),
            )

        old_state = state


if __name__ == "__main__":

    instance = Instance.from_json("../data/input_example1.json")
    river_basin = RiverBasin(instance=instance, mode="linear")

    print("---- SCENARIO A ----")
    decisionsA = np.array([[[6.79], [6.58]], [[7.49], [6.73]], [[7.49], [6.73]], [[7.49], [6.73]], [[7.49], [6.73]]])
    test_river_basin(river_basin, decisions=decisionsA)
    print("--- history")
    print(river_basin.history.to_string())

    # print("---- SCENARIO A, WITH DEEP UPDATE ----")
    # river_basin.reset(num_scenarios=1)
    # river_basin.deep_update_flows(decisionsA)
    # print(">>>> deep update")
    # print(f"state after deep update: {river_basin.get_state()}")
    # print(f"accumulated income: {river_basin.accumulated_income}")
    # print("--- history")
    # print(river_basin.history.to_string())

    print("---- SCENARIO A, WITH DEEP UPDATE THROUGH ALL PERIODS ----")
    river_basin.reset(num_scenarios=1)
    paddingA = np.array([[[0], [0]] for _ in range(instance.get_largest_impact_horizon() - decisionsA.shape[0])])
    decisionsA_all_periods = np.concatenate([decisionsA, paddingA])
    print(f"number of time steps: {instance.get_largest_impact_horizon()}")
    river_basin.deep_update_flows(decisionsA_all_periods)
    print(">>>> deep update")
    print(f"state after deep update: {river_basin.get_state()}")
    print(f"accumulated income: {river_basin.get_acc_income()}")
    print("--- history")
    print(river_basin.history.to_string())
    print(f"{river_basin.get_final_volume_of_dams()=}")
    print(f"{river_basin.get_acc_num_startups()=}")
    print(f"{river_basin.get_acc_num_times_in_limit()=}")

    print("---- SCENARIO B ----")
    river_basin.reset(num_scenarios=1)
    decisionsB = np.array([[[8], [7]], [[9], [8.5]]])
    test_river_basin(
        river_basin,
        decisions=decisionsB,
    )

    print("---- SCENARIOS A and B ----")
    river_basin.reset(num_scenarios=2)
    decisionsAB = np.array([[[6.79, 8], [6.58, 7]], [[7.49, 9], [6.73, 8.5]]])
    test_river_basin(
        river_basin,
        decisions=decisionsAB,
    )

    # print("---- SCENARIOS A and B, WITH DEEP UPDATE ----")
    # river_basin.reset(num_scenarios=2)
    # river_basin.deep_update_flows(decisionsAB)
    # print(">>>> deep update")
    # print(f"state after deep update: {river_basin.get_state()}")
    # print(f"accumulated income: {river_basin.accumulated_income}")
    #
    # print("---- SCENARIOS A and B, WITH DEEP UPDATE THROUGH ALL PERIODS ----")
    # padding = np.array([[[0, 0], [0, 0]] for _ in range(instance.get_largest_impact_horizon() - decisionsAB.shape[0])])
    # decisionsAB_all_periods = np.concatenate([decisionsAB, padding])
    # river_basin.reset(num_scenarios=2)
    # river_basin.deep_update_flows(decisionsAB_all_periods)
    # print(">>>> deep update")
    # print(f"state after deep update: {river_basin.get_state()}")
    # print(f"accumulated income: {river_basin.accumulated_income}")
    #
    # print("---- SCENARIOS A, B and C, WITH DEEP UPDATE ----")
    # decisionsABC = np.array(
    #     [[[6.79, 8, 1], [6.58, 7, 1]], [[7.49, 9, 1], [6.73, 8.5, 1]]]
    # )
    # river_basin.reset(num_scenarios=3)
    # river_basin.deep_update_flows(decisionsABC)
    # print(">>>> deep update")
    # print(f"state after deep update: {river_basin.get_state()}")
    # print(f"accumulated income: {river_basin.accumulated_income}")
    #
    # print("---- SCENARIOS A, B and C, WITH DEEP UPDATE THROUGH ALL PERIODS ----")
    # padding = np.array([[[0, 0, 0], [0, 0, 0]] for _ in range(instance.get_largest_impact_horizon() - decisionsABC.shape[0])])
    # decisionsABC_all_periods = np.concatenate([decisionsABC, padding])
    # river_basin.reset(num_scenarios=3)
    # river_basin.deep_update_flows(decisionsABC_all_periods)
    # print(">>>> deep update")
    # print(f"state after deep update: {river_basin.get_state()}")
    # print(f"accumulated income: {river_basin.accumulated_income}")
    #
    # print("---- SCENARIO VA, DEEP UPDATE WITH VARIATIONS ----")
    # river_basin.reset(flow_smoothing=1, num_scenarios=1)
    # decisionsVA = np.array([
    #     [[0.5], [0.5]],
    #     [[-0.25], [-0.25]],
    #     [[-0.25], [-0.25]],
    # ])
    # river_basin.deep_update_relvars(decisionsVA)
    # print(f"accumulated income: {river_basin.accumulated_income}")
    # equivalent_flows = river_basin.all_past_clipped_flows
    # print(f"equivalent flows: {equivalent_flows}")
    # print(river_basin.history.to_string())
    #
    # print("---- SCENARIO VA, EQUIVALENT NORMAL DEEP UPDATE ----")
    # river_basin.reset(num_scenarios=1)
    # decisionsNVA = equivalent_flows
    # river_basin.deep_update_flows(decisionsNVA)
    # print(f"accumulated income: {river_basin.accumulated_income}")
    # print(river_basin.history.to_string())
    #
    # print("---- SCENARIO VB, DEEP UPDATE WITH VARIATIONS ----")
    # river_basin.reset(num_scenarios=1)
    # decisionsVB = np.array([
    #     [[-0.25], [-0.25]],
    #     [[0.5], [0.5]],
    #     [[0.5], [0.5]],
    # ])
    # river_basin.deep_update_relvars(decisionsVB)
    # print(f"accumulated income: {river_basin.accumulated_income}")
    # print(f"equivalent flows: {river_basin.all_past_clipped_flows}")
    # print(river_basin.history.to_string())
    #
    # print("---- SCENARIOS VA, VB and VC, DEEP UPDATE WITH VARIATIONS ----")
    # river_basin.reset(num_scenarios=3)
    # decisionsVABC = np.array(
    #     [
    #         [
    #             [0.5, -0.25, 0.25],
    #             [0.5, -0.25, 0.5],
    #         ],
    #         [
    #             [-0.25, 0.5, 0.5],
    #             [-0.25, 0.5, 0.3],
    #         ],
    #         [
    #             [-0.25, 0.5, 0.5],
    #             [-0.25, 0.5, 0.3],
    #         ],
    #     ]
    # )
    # river_basin.deep_update_relvars(decisionsVABC)
    # print(f"accumulated income: {river_basin.accumulated_income}")
    # equivalent_flows = river_basin.all_past_clipped_flows
    # print(f"equivalent flows: {equivalent_flows}")
    #
    # print("---- SCENARIOS VA, VB and VC, EQUIVALENT NORMAL DEEP UPDATE ----")
    # river_basin.reset(num_scenarios=3)
    # decisionsNVABC = equivalent_flows
    # river_basin.deep_update_flows(decisionsNVABC)
    # print(f"accumulated income: {river_basin.accumulated_income}")
    #
    # print(
    #     "---- SCENARIOS VA, VB and VC, DEEP UPDATE WITH VARIATIONS THROUGH ALL PERIODS ----"
    # )
    # river_basin.reset(num_scenarios=3)
    # padding = np.array(
    #     [
    #         [[0, 0, 0], [0, 0, 0]]
    #         for _ in range(instance.get_largest_impact_horizon() - decisionsVABC.shape[0])
    #     ]
    # )
    # decisionsVABC_all_periods = np.concatenate([decisionsVABC, padding])
    # print(decisionsVABC_all_periods)
    # river_basin.deep_update_relvars(decisionsVABC_all_periods)
    # print(f"state after deep update: {river_basin.get_state()}")
    # print(f"accumulated income: {river_basin.accumulated_income}")
    # equivalent_flows = river_basin.all_past_clipped_flows
    # print(f"equivalent flows: {equivalent_flows}")
    #
    # print(
    #     "---- SCENARIOS VA, VB and VC, EQUIVALENT NORMAL DEEP UPDATE THROUGH ALL PERIODS ----"
    # )
    # river_basin.reset(num_scenarios=3)
    # decisionsNVABC_all_periods = equivalent_flows
    # river_basin.deep_update_flows(decisionsNVABC_all_periods)
    # print(f"state after deep update: {river_basin.get_state()}")
    # print(f"accumulated income: {river_basin.accumulated_income}")
