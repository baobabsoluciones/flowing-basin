import papermill as pm
from wasabi import msg
import argparse
import datetime
import os
import re
import logging
logging.getLogger('papermill')
logging.basicConfig(level='INFO', format="%(message)s")

def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """
    parser = argparse.ArgumentParser(usage='Runs a RL agent training for a flowing basin problem.')

    parser.add_argument('--template', type=str, default = "train_model")

    parser.add_argument('--PATH-CONSTANTS', type=str, default = "../data/constants/constants_2dams.json")
    parser.add_argument('--PATH-TRAIN-DATA', type=str, default = "../data/history/historical_data_clean_train.pickle")
    parser.add_argument('--PATH-TEST-DATA', type=str, default = "../data/history/historical_data_clean_test.pickle")
    parser.add_argument('--PATH-OBSERVATIONS', type=str, default = "../analyses/rl_pca/observations_data/observationsO2.npy")
    parser.add_argument('--PATH-OBSERVATIONS-CONFIG', type=str, default = "../analyses/rl_pca/observations_data/observationsO2_config.json")

    parser.add_argument('--startups-penalty', type=int, default = 50)
    parser.add_argument('--limit-zones-penalty', type=int, default = 50)
    parser.add_argument('--flow-smoothing', type=int, default = 2)
    parser.add_argument('--flow-smoothing-penalty', type=int, default = 25)

    parser.add_argument('--flow-smoothing-clip', type=str, default = "false")
    parser.add_argument('--action-type', type=str, default = "exiting_flows")
    parser.add_argument('--features', help='List of features, separated by |', type=str, default = "|".join([
        "past_vols", "past_flows", "past_variations", "future_prices",
        "future_inflows", "past_turbined", "past_groups", "past_powers", "past_clipped",
    ]))
    parser.add_argument('--unique-features', help='List of features, separated by |', type=str, default = "|".join([
        "future_prices",
    ]))
    parser.add_argument('--obs-box-shape', type = str, default = "false")
    parser.add_argument('--feature-extractor', type = str, default = "mlp")

    parser.add_argument('--PLOT-TRAINING-CURVE', type = str, default = "true")
    parser.add_argument('--SAVE-OBSERVATIONS', type = str, default = "true")


    parser.add_argument('--length-episodes', type=int, default = 24 * 4 + 3)
    parser.add_argument('--log-ep-freq', type=int, default = 5)
    parser.add_argument('--eval-ep-freq', type=int, default = 5)
    parser.add_argument('--num-episodes', type=int, default = 100)
    parser.add_argument('--eval-num-episodes', type=int, default = 10)

    parser.add_argument('--do-history-updates', type=str, default = "false")
    parser.add_argument('--update-observation-record', type=str, default = "true")

    args = vars(parser.parse_args())
    return args

if __name__ == "__main__":

    args = get_arguments()
    args['experiment_name'] = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}_{args['feature_extractor']}_{re.sub('.ipynb','',args['template'])}"
    args['features'] = args['features'].split('|')
    args['unique_features'] = args['unique_features'].split('|')
    for k in ['flow_smoothing_clip', 'PLOT_TRAINING_CURVE', 'SAVE_OBSERVATIONS', 'do_history_updates',
              'update_observation_record', 'eval_num_episodes', 'PLOT_TRAINING_CURVE', 'SAVE_OBSERVATIONS', 'obs_box_shape']:
        args[k] = args[k] == "true"


    msg.info("Launching template", args['template'])
    os.makedirs(f"../studies/{args['experiment_name']}", exist_ok=True)

    pm.execute_notebook(
        f"./templates/{args['template']}{'' if args.get('template').endswith('ipynb') else'.ipynb'}",
        f"../studies/{args['experiment_name']}/experiment_report.ipynb",
        args,
        log_output=True, # This prints in the console the results as they would in the notebook cells
        report_mode=True
    )

    msg.good(f"Study finished. Results can be found in studies/{args['experiment_name']}")