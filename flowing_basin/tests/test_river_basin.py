from flowing_basin.core import Instance
from flowing_basin.tools import RiverBasin
import numpy as np
import pandas as pd


def check_volume(
    old_state: dict,
    state: dict,
    time_step: float,
    num_scenarios: int = 1,
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
    # print(
    #     "dam1 calc in TEST",
    #     old_state["dam1"]["vol"],
    #     old_state["next_incoming_flow"],
    #     old_state["dam1"]["next_unregulated_flow"],
    #     state["dam1"]["lags"][:, 0],
    #     dam1_vol,
    # )
    if num_scenarios == 1:
        dam1_vol = dam1_vol.item()

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
    # print(
    #     "dam2 calc in TEST",
    #     old_state["dam2"]["vol"],
    #     state["dam1"]["turbined_flow"],
    #     old_state["dam2"]["next_unregulated_flow"],
    #     state["dam2"]["lags"][:, 0],
    #     dam2_vol,
    # )
    if num_scenarios == 1:
        dam2_vol = dam2_vol.item()

    # Assert calculations are the same as the state values
    if num_scenarios == 1:
        assert round(state["dam1"]["vol"], 4) == round(
            dam1_vol, 4
        ), f"Volume of dam1 should be {dam1_vol} but it is {state['dam1']['vol']}"
        assert round(state["dam2"]["vol"], 4) == round(
            dam2_vol, 4
        ), f"Volume of dam2 should be {dam2_vol} but it is {state['dam2']['vol']}"
    else:

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
    num_scenarios: int = 1,
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
    if num_scenarios == 1:
        assert round(income_calc, 4) == round(
            income, 4
        ), f"Income should be {income_calc} but it is {income}"
    else:
        for (income_calc_scenario, income_scenario) in zip(income_calc, income):
            assert round(income_calc_scenario, 4) == round(
                income_scenario, 4
            ), f"Income should be {income_calc_scenario} but it is {income_scenario}"


def test_river_basin(
    river_basin: RiverBasin,
    decisions: list[list[float]] | np.ndarray,
    check_incomes: bool = True,
    check_volumes: bool = True,
):

    state = river_basin.get_state()
    print("initial state:", state)
    old_state = state

    for i, flows in enumerate(decisions):

        print(f">>>> decision {i}")
        income = river_basin.update(flows)
        print(f"income={income}")
        state = river_basin.get_state()
        print(f"state after decision {i}:", state)
        print(
            f"income obtained after decision {i} (EUR): "
            f"price (EUR/MWh) * power (MW) * time_step (hours) = "
            f"{old_state['next_price']} * ({state['dam1']['power']} + {state['dam2']['power']}) * {river_basin.instance.get_time_step() / 3600} = "
            f"{income}"
        )

        if check_incomes:
            check_income(
                old_state=old_state,
                state=state,
                time_step=river_basin.instance.get_time_step(),
                num_scenarios=river_basin.num_scenarios,
                income=income,
            )

        if check_volumes:
            check_volume(
                old_state=old_state,
                state=state,
                time_step=river_basin.instance.get_time_step(),
                num_scenarios=river_basin.num_scenarios,
                min_vol_dam1=river_basin.instance.get_min_vol_of_dam("dam1"),
                max_vol_dam1=river_basin.instance.get_max_vol_of_dam("dam1"),
                min_vol_dam2=river_basin.instance.get_min_vol_of_dam("dam2"),
                max_vol_dam2=river_basin.instance.get_max_vol_of_dam("dam2"),
            )

        old_state = state


if __name__ == "__main__":

    instance = Instance.from_json("../data/input_example1.json")
    paths_power_models = {
        "dam1": "../ml_models/model_E1.sav",
        "dam2": "../ml_models/model_E2.sav",
    }
    river_basin = RiverBasin(
        instance=instance,
        paths_power_models=paths_power_models
    )

    # print("---- SCENARIO A ----")
    # river_basin.reset(num_scenarios=1)
    # decisionsA = [[6.79, 6.58], [7.49, 6.73], [7.49, 6.73], [7.49, 6.73], [7.49, 6.73]]
    # test_river_basin(river_basin, decisions=decisionsA)
    # print("--- log")
    # print(river_basin.log)
    # print("--- history")
    # pd.set_option('display.max_columns', None)
    # print(river_basin.history)
    #
    # print("---- SCENARIO A, WITH DEEP UPDATE ----")
    # river_basin.reset(num_scenarios=1)
    # income = river_basin.deep_update_flows(decisionsA)
    # print(">>>> deep update")
    # print(f"state after deep update: {river_basin.get_state()}")
    # print(f"accumulated income: {income}")
    #
    # print("---- SCENARIO A, WITH DEEP UPDATE THROUGH ALL PERIODS ----")
    # river_basin.reset(num_scenarios=1)
    # decisions_all_periods = [[0, 0] for _ in range(instance.get_num_time_steps())]
    # print(f"number of time steps: {instance.get_num_time_steps()}")
    # income = river_basin.deep_update_flows(decisions_all_periods)
    # print(">>>> deep update")
    # print(f"state after deep update: {river_basin.get_state()}")
    # print(f"accumulated income: {income}")
    #
    # print("---- SCENARIO B ----")
    # river_basin.reset(num_scenarios=1)
    # decisionsB = [[8, 7], [9, 8.5]]
    # test_river_basin(
    #     river_basin,
    #     decisions=decisionsB,
    # )
    #
    # print("---- SCENARIOS A and B ----")
    # river_basin.reset(num_scenarios=2)
    # decisionsAB = np.array([[[6.79, 8], [6.58, 7]], [[7.49, 9], [6.73, 8.5]]])
    # test_river_basin(
    #     river_basin,
    #     decisions=decisionsAB,
    # )
    #
    # print("---- SCENARIOS A and B, WITH DEEP UPDATE ----")
    # river_basin.reset(num_scenarios=2)
    # income = river_basin.deep_update_flows(decisionsAB)
    # print(">>>> deep update")
    # print(f"state after deep update: {river_basin.get_state()}")
    # print(f"accumulated income: {income}")
    #
    # print("---- SCENARIOS A and B, WITH DEEP UPDATE THROUGH ALL PERIODS ----")
    # padding = np.array([[[0, 0], [0, 0]] for _ in range(instance.get_num_time_steps() - decisionsAB.shape[0])])
    # decisionsAB_all_periods = np.concatenate([decisionsAB, padding])
    # river_basin.reset(num_scenarios=2)
    # income = river_basin.deep_update_flows(decisionsAB)
    # print(">>>> deep update")
    # print(f"state after deep update: {river_basin.get_state()}")
    # print(f"accumulated income: {income}")

    print("---- SCENARIOS A, B and C, WITH DEEP UPDATE ----")
    decisionsABC = np.array(
        [[[6.79, 8, 1], [6.58, 7, 1]], [[7.49, 9, 1], [6.73, 8.5, 1]]]
    )
    river_basin.reset(num_scenarios=3)
    income = river_basin.deep_update_flows(decisionsABC)
    print(">>>> deep update")
    print(f"state after deep update: {river_basin.get_state()}")
    print(f"accumulated income: {income}")

    print("---- SCENARIOS A, B and C, WITH DEEP UPDATE THROUGH ALL PERIODS ----")
    padding = np.array([[[0, 0, 0], [0, 0, 0]] for _ in range(instance.get_num_time_steps() - decisionsABC.shape[0])])
    decisionsABC_all_periods = np.concatenate([decisionsABC, padding])
    river_basin.reset(num_scenarios=3)
    income = river_basin.deep_update_flows(decisionsABC_all_periods)
    print(">>>> deep update")
    print(f"state after deep update: {river_basin.get_state()}")
    print(f"accumulated income: {income}")

    # print("---- SCENARIO VA, DEEP UPDATE WITH VARIATIONS ----")
    # river_basin.reset(num_scenarios=1)
    # decisionsVA = [
    #     [0.5, 0.5],
    #     [0.25, 0.25],
    # ]
    # income, equivalent_flows = river_basin.deep_update_relvars(decisionsVA, return_equivalent_flows=True)
    # print(f"accumulated income: {income}")
    # print(f"equivalent flows: {equivalent_flows}")
    # print(river_basin.log)
    #
    # print("---- SCENARIO VA, EQUIVALENT NORMAL DEEP UPDATE ----")
    # river_basin.reset(num_scenarios=1)
    # decisionsNVA = equivalent_flows
    # income = river_basin.deep_update_flows(decisionsNVA)
    # print(f"accumulated income: {income}")
    # # print(river_basin1.log)

    # print("---- SCENARIOS VA, VB and VC, DEEP UPDATE WITH VARIATIONS ----")
    # river_basin.reset(num_scenarios=3)
    # decisionsVABC = np.array(
    #     [
    #         [
    #             [0.5, 0.75, 1],
    #             [0.5, 0.75, 1],
    #         ],
    #         [
    #             [0.25, 0.5, 1],
    #             [0.25, 0.5, 1],
    #         ],
    #     ]
    # )
    # income, equivalent_flows = river_basin.deep_update_relvars(decisionsVABC, return_equivalent_flows=True)
    # print(f"accumulated income: {income}")
    # print(f"equivalent flows: {equivalent_flows}")
    #
    # print("---- SCENARIOS VA, VB and VC, EQUIVALENT NORMAL DEEP UPDATE ----")
    # river_basin.reset(num_scenarios=3)
    # decisionsNVABC = np.array(
    #     [
    #         [
    #             [13.60645663, 17.14395663, 20.68145663],
    #             [12.17849932, 14.99599932, 17.81349932],
    #         ],
    #         [
    #             [17.14395663, 21.225, 28.3],
    #             [9.19225623, 12.00975623, 17.64475623],
    #         ],
    #     ]
    # )
    # income = river_basin.deep_update_flows(decisionsNVABC)
    # print(f"accumulated income: {income}")
    #
    # print("---- SCENARIOS VA, VB and VC, DEEP UPDATE WITH VARIATIONS THROUGH ALL PERIODS ----")
    # river_basin.reset(num_scenarios=3)
    # padding = np.array([[[0, 0, 0], [0, 0, 0]] for _ in range(instance.get_num_time_steps() - decisionsVABC.shape[0])])
    # decisionsVABC_all_periods = np.concatenate([decisionsVABC, padding])
    # print(decisionsVABC_all_periods)
    # income, equivalent_flows = river_basin.deep_update_relvars(decisionsVABC_all_periods, return_equivalent_flows=True)
    # print(f"state after deep update: {river_basin.get_state()}")
    # print(f"accumulated income: {income}")
    # print(f"equivalent flows: {equivalent_flows}")
    #
    # print("---- SCENARIOS VA, VB and VC, EQUIVALENT NORMAL DEEP UPDATE THROUGH ALL PERIODS ----")
    # river_basin.reset(num_scenarios=3)
    # decisionsNVABC_all_periods = equivalent_flows
    # income = river_basin.deep_update_flows(decisionsNVABC_all_periods)
    # print(f"state after deep update: {river_basin.get_state()}")
    # print(f"accumulated income: {income}")
