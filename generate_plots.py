from cartpole_modified import CartPoleEnv
from bipedal_walker_modified import BipedalWalker
import utils
import numpy as np
from time import sleep
from nn import NeuralNetwork
import csv
import torch
import json
from utils import generate_morphologies
import sys
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from scipy import stats

train_cauchy1_path = "/home/edo/THESIS/evolving-generalist-controllers/Results_Acrobot/cauchy1/20231124114220"
train_cauchy2_path = "/home/edo/THESIS/evolving-generalist-controllers/Results_Acrobot/cauchy2/20231124114220"
train_gaussian1_path = "/home/edo/THESIS/evolving-generalist-controllers/Results_Acrobot/gaussian1/20231124114220"
train_gaussian2_path = "/home/edo/THESIS/evolving-generalist-controllers/Results_Acrobot/gaussian2/20231124114220"
train_incremental_path = "/home/edo/THESIS/evolving-generalist-controllers/Results_Acrobot/incremental/20231124114220"
train_uniform_path = "/home/edo/THESIS/evolving-generalist-controllers/Results_Acrobot/uniform/20231124114220"
all_train_folders = [train_gaussian1_path, train_gaussian2_path, train_cauchy1_path, train_cauchy2_path, train_incremental_path, train_uniform_path]

SEED=0

def plot_heatmap(json_filename, all_variations, scores, title, save_path=None):
    param1_values = all_variations[:, 0]
    param2_values = all_variations[:, 1]
    rounded_param1_values = [round(num, 2) for num in param1_values]
    rounded_param2_values = [round(num, 2) for num in param2_values]
    avg_scores = np.mean(scores, axis=0)
    data = {'Parameter 1': rounded_param1_values, 'Parameter 2': rounded_param2_values, 'Reward': avg_scores}
    df = pd.DataFrame(data)
    pivot_df = df.pivot(index='Parameter 2', columns='Parameter 1', values='Reward')

    plt.figure(figsize=(10, 7))
    if game=="CartPoleEnv":
        sns.heatmap(pivot_df, vmin=-1000, vmax=0)
    elif game=="AcrobotEnv":
        sns.heatmap(pivot_df, vmin=0, vmax=100)

    plt.title(title)
    plt.xlabel('Pole Mass')
    plt.ylabel('Pole Length')
    #plt.axvline(x=0.05, color='red', linestyle='--', linewidth=2)  # Adjust color, linestyle, and linewidth as needed
    plt.tight_layout()

    if save_path:
        # Split the path by "/"
        save_path_parts = save_path.split("/")

        # Remove empty strings from the split
        save_path_parts = [part for part in save_path_parts if part]

        # Initialize the training schedule
        training_schedule = None

        # Loop through the path parts
        for part in save_path_parts:
            if part.lower() in ["incremental", "gaussian1", "gaussian2", "cauchy1", "cauchy2", "uniform"]:
                training_schedule = part
                break

        save_path = os.path.join(save_path, "HeatMap_" + training_schedule + "_" + title + ".png")
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__": 
    with open(str(sys.argv[1])) as json_file:
        config = json.load(json_file)
    game=config['game']
    all_ts_avgs_IN = []
    #datafile_path = str(sys.argv[1])
    for datafile_path in all_train_folders:
        datafile_path = datafile_path + "/all_history_rewards_data.npz"
        data = np.load(datafile_path)
        plots_path = os.path.join(os.path.dirname(datafile_path), "plots") 
        os.makedirs(plots_path, exist_ok=True)


        all_history_rewards_IN = data['all_history_rewards_IN']
        all_history_rewards_OUT = data['avg_rewards_OUT']
        all_history_rewards_INOUT = data['avg_rewards_INOUT']

        all_avgs_IN = [np.mean(run_rewards) for run_rewards in all_history_rewards_IN]
        all_ts_avgs_IN.append(all_avgs_IN)
        print("Mean reward IN: ", np.mean(all_avgs_IN))

        all_avgs_OUT = [np.mean(run_rewards) for run_rewards in all_history_rewards_OUT]
        print("Mean reward OUT: ", np.mean(all_avgs_OUT))

        all_avgs_INOUT = [np.mean(run_rewards) for run_rewards in all_history_rewards_INOUT]
        print("Mean reward INOUT: ", np.mean(all_avgs_INOUT))

        # Specify the file path for the text file
        original_stdout = sys.stdout
        output_file_path = os.path.dirname(datafile_path) + "/results.txt"
        with open(output_file_path, 'w') as f:
            sys.stdout = f  
            print("Mean reward IN: ", np.mean(all_avgs_IN))
            print("Mean reward OUT: ", np.mean(all_avgs_OUT))
            print("Mean reward INOUT: ", np.mean(all_avgs_INOUT))
        sys.stdout = original_stdout

        IN_variations = utils.get_set(config, test_set="IN")
        OUT_variations = utils.get_set(config, test_set="OUT")
        INOUT_variations = utils.get_set(config, test_set="INOUT")

        plot_heatmap(game, IN_variations, all_history_rewards_IN, title='IN', save_path=plots_path)
        plot_heatmap(game, OUT_variations, all_history_rewards_OUT, title='OUT', save_path=plots_path)
        plot_heatmap(game, INOUT_variations, all_history_rewards_INOUT, title='INOUT', save_path=plots_path)

    print(stats.kruskal(all_ts_avgs_IN[0], all_ts_avgs_IN[1], all_ts_avgs_IN[2], all_ts_avgs_IN[3], all_ts_avgs_IN[4], all_ts_avgs_IN[5]))
    
    