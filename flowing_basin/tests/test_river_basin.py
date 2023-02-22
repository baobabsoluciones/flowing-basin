from flowing_basin.core import Instance
from flowing_basin.tools import RiverBasin
import numpy as np


def test_river_basin(
    river_basin: RiverBasin,
    decisions: list[list[float]] | np.ndarray,
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
        income_calc = old_state['next_price'] * (state['dam1']['power'] + state['dam2']['power']) * river_basin.instance.get_time_step() / 3600
        print(
            f"income obtained after decision {i} (EUR): "
            f"price (EUR/MWh) * power (MW) * time_step (hours) = "
            f"{old_state['next_price']} * ({state['dam1']['power']} + {state['dam2']['power']}) * {river_basin.instance.get_time_step() / 3600} = "
            f"{income_calc}"
        )

        if river_basin.num_scenarios == 1:
            assert round(income_calc, 4) == round(
                income, 4
            ), f"Income after decision {i} should be {income_calc} but it is {income}"
        else:
            for (
                    income_calc_scenario,
                    income_scenario
            ) in zip(
                income_calc, income
            ):
                assert round(income_calc_scenario, 4) == round(
                    income_scenario, 4
                ), f"Income after decision {i} should be {income_calc_scenario} but it is {income_scenario}"

        if check_volumes:

            # Check volumes are consistent throughout states
            dam1_vol = np.clip(
                old_state["dam1"]["vol"]
                + (
                    old_state["next_incoming_flow"]
                    + old_state["dam1"]["next_unregulated_flow"]
                    - state["dam1"]["lags"][:, 0]
                )
                * river_basin.instance.get_time_step(),
                river_basin.instance.get_min_vol_of_dam("dam1"),
                river_basin.instance.get_max_vol_of_dam("dam1"),
            )
            # print(
            #     "dam1 calc in TEST",
            #     old_state["dam1"]["vol"],
            #     old_state["next_incoming_flow"],
            #     old_state["dam1"]["next_unregulated_flow"],
            #     state["dam1"]["lags"][:, 0],
            #     dam1_vol,
            # )
            if river_basin.num_scenarios == 1:
                dam1_vol = dam1_vol.item()
            dam2_vol = np.clip(
                old_state["dam2"]["vol"]
                + (
                    state["dam1"]["turbined_flow"]
                    + old_state["dam2"]["next_unregulated_flow"]
                    - state["dam2"]["lags"][:, 0]
                )
                * river_basin.instance.get_time_step(),
                river_basin.instance.get_min_vol_of_dam("dam2"),
                river_basin.instance.get_max_vol_of_dam("dam2"),
            )
            # print(
            #     "dam2 calc in TEST",
            #     old_state["dam2"]["vol"],
            #     state["dam1"]["turbined_flow"],
            #     old_state["dam2"]["next_unregulated_flow"],
            #     state["dam2"]["lags"][:, 0],
            #     dam2_vol,
            # )
            if river_basin.num_scenarios == 1:
                dam2_vol = dam2_vol.item()

            if river_basin.num_scenarios == 1:
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
                ) in zip(
                    state["dam1"]["vol"], dam1_vol, state["dam2"]["vol"], dam2_vol
                ):
                    assert round(state_dam1_vol_scenario, 4) == round(
                        dam1_vol_scenario, 4
                    ), f"Volume of dam1 should be {dam1_vol_scenario} but it is {state_dam1_vol_scenario}"
                    assert round(state_dam2_vol_scenario, 4) == round(
                        dam2_vol_scanario, 4
                    ), f"Volume of dam2 should be {dam2_vol_scanario} but it is {state_dam2_vol_scenario}"

            old_state = state


if __name__ == "__main__":

    instance = Instance.from_json("../data/input_example1.json")
    paths_power_models = {
        "dam1": "../ml_models/model_E1.sav",
        "dam2": "../ml_models/model_E2.sav",
    }

    print("---- SCENARIO A ----")
    river_basin1 = RiverBasin(
        instance=instance,
        paths_power_models=paths_power_models,
        num_scenarios=1,
    )
    decisionsA = [[6.79, 6.58], [7.49, 6.73], [7.49, 6.73], [7.49, 6.73], [7.49, 6.73]]
    test_river_basin(
        river_basin1,
        decisions=decisionsA
    )

    print("---- SCENARIO A, WITH DEEP UPDATE ----")
    river_basin1.reset()
    income = river_basin1.deep_update(decisionsA)
    print(">>>> deep update")
    print(f"state after deep update: {river_basin1.get_state()}")
    print(f"accumulated income: {income}")

    print("---- SCENARIO A, WITH DEEP UPDATE THROUGH ALL PERIODS ----")
    river_basin1.reset()
    decisions_all_periods = [[0, 0] for _ in range(instance.get_num_time_steps())]
    print(f"number of time steps: {instance.get_num_time_steps()}")
    income = river_basin1.deep_update(decisions_all_periods)
    print(">>>> deep update")
    print(f"state after deep update: {river_basin1.get_state()}")
    print(f"accumulated income: {income}")

    print("---- SCENARIO B ----")
    river_basin1.reset()
    decisionsB = [[8, 7], [9, 8.5]]
    test_river_basin(
        river_basin1,
        decisions=decisionsB,
    )

    print("---- SCENARIOS A and B ----")
    river_basin2 = RiverBasin(
        instance=instance,
        paths_power_models=paths_power_models,
        num_scenarios=2,
    )
    decisionsAB = np.array([[[6.79, 8], [6.58, 7]], [[7.49, 9], [6.73, 8.5]]])
    test_river_basin(
        river_basin2,
        decisions=decisionsAB,
    )

    print("---- SCENARIOS A and B, WITH DEEP UPDATE ----")
    river_basin2.reset()
    income = river_basin2.deep_update(decisionsAB)
    print(">>>> deep update")
    print(f"state after deep update: {river_basin2.get_state()}")
    print(f"accumulated income: {income}")

