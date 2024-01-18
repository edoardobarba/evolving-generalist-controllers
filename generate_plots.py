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
import matplotlib.patches as patches


# #training_schedules = ["incremental", "gaussian1", "gaussian2", "cauchy1", "cauchy2", "uniform", "RL", "beta1", "beta2", "betawalk1", "betawalk2"]

# #training_schedules = ["cauchy1", "cauchy2", "gaussian1", "gaussian2", "incremental",]

# train_cauchy1_path = "/home/edoardo.barba/Results_Biped/cauchy1/20231208165916"
# train_cauchy2_path = "/home/edoardo.barba/Results_Biped/cauchy2/20231208165916"
# train_gaussian1_path = "/home/edoardo.barba/Results_Biped/gaussian1/20231208165916"
# train_gaussian2_path = "/home/edoardo.barba/Results_Biped/gaussian2/20231208165916"
# train_incremental_path = "/home/edoardo.barba/Results_Biped/incremental/20231208165916"
# train_uniform_path = "/home/edoardo.barba/Results_Biped/uniform/20231208165916"
# train_RL_path = "/home/edoardo.barba/Results_Biped/RL/20231208165916"
# train_beta1 = "/home/edoardo.barba/Results_Biped/beta1/20231208165916"
# train_beta2 = "/home/edoardo.barba/Results_Biped/beta2/20231208165916"
# train_betawalk1 = "/home/edoardo.barba/Results_Biped/betawalk1/20231208165916"
# train_betawalk2 = "/home/edoardo.barba/Results_Biped/betawalk2/20231208165916"

# #all_train_folders = [train_incremental_path, train_gaussian1_path, train_gaussian2_path, train_cauchy1_path, train_cauchy2_path,   train_uniform_path, train_RL_path, train_beta1, train_beta2, train_betawalk1, train_betawalk2]


# train_cauchy1_path = "/home/edoardo.barba/Results_Biped/cauchy1/20231219091844"
# train_cauchy2_path = "/home/edoardo.barba/Results_Biped/cauchy2/20231219091844"
# train_gaussian1_path = "/home/edoardo.barba/Results_Cart/gaussian1/20240117115209"
# train_gaussian2_path = "/home/edoardo.barba/Results_Biped/gaussian2/20231223120018"
# train_incremental_path = "/home/edoardo.barba/Results_Biped/incremental/20231219025742"
# train_uniform_path = "/home/edoardo.barba/Results_Biped/uniform/20231219091844"
# train_RL_path = "/home/edoardo.barba/Results_Biped_old/RL/20231215114621" 
# train_beta01 = "/home/edoardo.barba/Results_Cart/beta01/20240117115209"
# train_beta02 = "/home/edoardo.barba/Results_Biped/beta02/20231219092337"
# train_betawalk01 = "/home/edoardo.barba/Results_Biped/betawalk01/20231221112601" 
# train_betawalk02 = "/home/edoardo.barba/Results_Biped/betawalk02/20231221112601" 
# train_gauss_dec = "/home/edoardo.barba/Results_Biped/gauss_dec/20231223120018"
# train_default_path = "/home/edoardo.barba/Results_Biped/default/20240103141803"
train_borderincr_path = "/home/edoardo.barba/Results_Walker/border_incr/20240117050517"
# train_random_path = "/home/edoardo.barba/Results_Biped/random/20240108110555"

# #training_schedules = ["incremental", "gaussian1", "gaussian2", "cauchy1", "cauchy2","uniform", "beta01", "beta02", "betawalk01", "betawalk02", "gauss_dec"]
# #all_train_folders = [train_incremental_path, train_gaussian1_path, train_gaussian2_path, train_cauchy1_path, train_cauchy2_path, train_uniform_path, train_beta01, train_beta02, train_betawalk01, train_betawalk02, train_gauss_dec]
# all_train_folders = [train_random_path]
# #all_train_folders = [train_default_path, train_borderincr_path]

#train_random_path = "/home/edoardo.barba/Results_Biped/random/20240108110555"
# train_incremental_path = "/home/edoardo.barba/Results_Walker/incremental/20240115180512"
# train_random_path  = "/home/edoardo.barba/Results_Walker/random/20240115180512"
# train_RL_path = "/home/edoardo.barba/Results_Biped/RL/20240115165851"


train_incremental_path = "/home/edoardo.barba/Results_Walker/incremental/20240118020055"
train_random_path  = "/home/edoardo.barba/Results_Walker/random/20240118020055"
# train_RL_path = "/home/edoardo.barba/Results_Biped/RL/20240115165851"


all_train_folders = [train_random_path, train_incremental_path, train_borderincr_path]

# train_random_path = "/home/edoardo.barba/Results_Walker/random/20240112123557"
# # train_incremental_path = "/home/edoardo.barba/Results_Walker/incremental/20240112123557"

# all_train_folders = [train_random_path]

#save_path = "/home/edoardo.barba/Results_Biped"

save_path = "/home/edoardo.barba/Results_Walker"
SEED=0

def plot_heatmap(json_filename, all_variations, scores, title, save_path=None):
    param1_values = all_variations[:, 0]
    param2_values = all_variations[:, 1]
    rounded_param1_values = [round(num, 2) for num in param1_values]
    rounded_param2_values = [round(num, 2) for num in param2_values]

    avg_scores = np.mean(scores, axis=0)
    data = {'Parameter 1': rounded_param1_values, 'Parameter 2': rounded_param2_values, 'Reward': avg_scores}
    df = pd.DataFrame(data)
    #print(df)
    pivot_df = df.pivot(index='Parameter 2', columns='Parameter 1', values='Reward')
    pivot_df = pivot_df[::-1]

    plt.figure(figsize=(10, 7))
    if game=="CartPoleEnv":
        sns.heatmap(pivot_df, vmin=-1000, vmax=0, annot=True, fmt=".0f")
        plt.xlabel('Pole Mass')
        plt.ylabel('Pole Length')
        rect = patches.Rectangle((1, 2), 4, 5, linewidth=3, edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect)
    elif game=="AcrobotEnv":
        sns.heatmap(pivot_df, vmin=0, vmax=100, annot=True)        
        plt.xlabel('MASS1')
        plt.ylabel('MASS2')
        # Add the red square
        rect = patches.Rectangle((2, 2), 6, 6, linewidth=2, edgecolor='black', facecolor='none')
        plt.gca().add_patch(rect)

    elif game=="BipedalWalker":
        sns.heatmap(pivot_df, vmin=-300, vmax=300, annot=True, fmt=".0f")        
        plt.xlabel('Leg Width')
        plt.ylabel('Leg Height')
        rect = patches.Rectangle((4, 4), 11, 11, linewidth=3, edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect)
        # Add the red square
        #rect = patches.Rectangle((2, 2), 6, 6, linewidth=2, edgecolor='black', facecolor='none')
        #plt.gca().add_patch(rect)
    elif game=="Walker2dEnv":
        sns.heatmap(pivot_df, vmin=-1700, vmax=0, annot=True, fmt=".0f")    
        plt.xlabel('Upper leg length')
        plt.ylabel('Lower leg length')
        rect = patches.Rectangle((3, 3), 6, 6, linewidth=3, edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect)
        
    plt.title(title)

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
            #print(part.lower())
            if part.lower() in ["incremental", "gaussian1", "gaussian2", "cauchy1", "cauchy2", "uniform", "rl", "beta01", "beta02", "betawalk01", "betawalk02", "gauss_dec", "border_incr", "default", "random"]:
                training_schedule = part
                break

        save_path = os.path.join(save_path, "HeatMap_" + training_schedule + "_" + title + ".png")
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__": 
    with open(str(sys.argv[1])) as json_file:
        config = json.load(json_file)
    game=config['game']
    all_ts_avgs_IN = []
    #datafile_path = str(sys.argv[1])
    for datafile_path in all_train_folders:
        print(datafile_path)
        datafile_path = datafile_path + "/all_history_rewards_data.npz"
        data = np.load(datafile_path)
        plots_path = os.path.join(os.path.dirname(datafile_path), "plots") 
        os.makedirs(plots_path, exist_ok=True)
        print()

        # all_history_rewards_IN = data['all_history_rewards_IN']
        # all_history_rewards_OUT = data['avg_rewards_OUT']
        all_history_rewards_INOUT = data['avg_rewards_INOUT']

        # all_avgs_IN = [np.mean(run_rewards) for run_rewards in all_history_rewards_IN]
        # all_ts_avgs_IN.append(all_avgs_IN)
        
        # print("Mean reward IN: ", np.mean(all_avgs_IN))

        # all_avgs_OUT = [np.mean(run_rewards) for run_rewards in all_history_rewards_OUT]
        
        # print("Mean reward OUT: ", np.mean(all_avgs_OUT))

        all_avgs_INOUT = [np.mean(run_rewards) for run_rewards in all_history_rewards_INOUT]

        
        print("Mean reward TRAIN+TEST: ", np.mean(all_avgs_INOUT))
        print("Median reward TRAIN+TEST: ", np.median(all_avgs_INOUT))
        print("standard deviation TRAIN+TEST: ", np.std(all_avgs_INOUT))

        # Specify the file path for the text file
        original_stdout = sys.stdout
        output_file_path = os.path.dirname(datafile_path) + "/results.txt"
        # with open(output_file_path, 'w') as f:
        #     # sys.stdout = f  
        #     # print("Mean reward IN: ", np.mean(all_avgs_IN))
        #     # print("std IN: ", np.std(all_avgs_IN))

        #     # print("Mean reward OUT: ", np.mean(all_avgs_OUT))
        #     # print("std OUT: ", np.std(all_avgs_OUT))

        #     print("Mean reward TRAIN+TEST: ", np.mean(all_avgs_INOUT))
        #     print("standard deviation TRAIN+TEST: ", np.std(all_avgs_INOUT))
            
        sys.stdout = original_stdout

        # IN_variations = utils.get_set(config, test_set="IN")
        # OUT_variations = utils.get_set(config, test_set="OUT")
        INOUT_variations = utils.get_set(config, test_set="INOUT")
        print(np.shape(all_history_rewards_INOUT))

        #plot_heatmap(game, IN_variations, all_history_rewards_IN, title='IN', save_path=plots_path)
        #plot_heatmap(game, OUT_variations, all_history_rewards_OUT, title='OUT', save_path=plots_path)
        plot_heatmap(game, INOUT_variations, all_history_rewards_INOUT, title='TRAIN+TEST', save_path=plots_path)

    #print(stats.kruskal(all_ts_avgs_IN[0], all_ts_avgs_IN[1], all_ts_avgs_IN[2], all_ts_avgs_IN[3], all_ts_avgs_IN[4], all_ts_avgs_IN[5]))
    
    