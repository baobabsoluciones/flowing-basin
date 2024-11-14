"""
rl_obs_histogram.py
This script plots the histogram of the RL agent's observations
"""

from flowing_basin.solvers.rl import ReinforcementLearning

if __name__ == "__main__":

    rl = ReinforcementLearning("rl-A1G1O221R1T2", verbose=3)
    rl.plot_histograms_agent_obs(
        show_lookback=False, filename=f"rl_obs_histogram/histograms_{rl.config_names['O']}.png"
    )
