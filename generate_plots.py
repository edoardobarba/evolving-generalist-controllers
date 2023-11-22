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


npz_incremental_path = "/home/edo/THESIS/evolving-generalist-controllers/Results_Cart/incremental/20231122111009/all_history_rewards_data.npz"
npz_gaussian1_path = "/home/edo/THESIS/evolving-generalist-controllers/Results_Cart/gaussian1/20231122111104/all_history_rewards_data.npz"
npz_gaussian2_path = "/home/edo/THESIS/evolving-generalist-controllers/Results_Cart/gaussian2/20231122111134/all_history_rewards_data.npz"
npz_cauchy1_path = "/home/edo/THESIS/evolving-generalist-controllers/Results_Cart/cauchy1/20231122111151/all_history_rewards_data.npz"
npz_cauchy2_path = "/home/edo/THESIS/evolving-generalist-controllers/Results_Cart/cauchy2/20231122111322/all_history_rewards_data.npz"
npz_uniform_path = "/home/edo/THESIS/evolving-generalist-controllers/Results_Cart/uniform/20231122111359/all_history_rewards_data.npz"

all_datafile_paths = [npz_incremental_path, npz_gaussian1_path, npz_gaussian2_path, npz_cauchy1_path, npz_cauchy2_path, npz_uniform_path]


CARTPOLE_IN_LOWER_MASSPOLE = 0.05
CARTPOLE_IN_UPPER_MASSPOLE = 0.5
CARTPOLE_OUT_LOWER_MASSPOLE = 0.01
CARTPOLE_OUT_UPPER_MASSPOLE = 1.0

CARTPOLE_IN_LOWER_LENGTH = 0.25
CARTPOLE_IN_UPPER_LENGTH = 0.75
CARTPOLE_OUT_LOWER_LENGTH = 0.05
CARTPOLE_OUT_UPPER_LENGTH = 1.0

CARTPOLE_DEFAULT__MASS = 0.1
CARTPOLE_DEFAULT_LENGTH = 0.5

SEED=0

def plot_heatmap(all_variations, scores, title, save_path=None):
    print("QUI")
    param1_values = all_variations[:, 0]
    param2_values = all_variations[:, 1]
    rounded_param1_values = [round(num, 2) for num in param1_values]
    rounded_param2_values = [round(num, 2) for num in param2_values]
    avg_scores = np.mean(scores, axis=0)
    data = {'Parameter 1': rounded_param1_values, 'Parameter 2': rounded_param2_values, 'Reward': avg_scores}
    df = pd.DataFrame(data)
    pivot_df = df.pivot(index='Parameter 2', columns='Parameter 1', values='Reward')

    plt.figure(figsize=(10, 7))
    sns.heatmap(pivot_df, vmin=-1000, vmax=0)
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


def plot_boxplots(history_reward_in_distr, history_reward_out_distr, history_reward_in_out_distr, title, save_path = None):

    avg_scores_IN = np.mean(history_reward_in_distr, axis=1)

    avg_scores_OUT = np.mean(history_reward_out_distr, axis=1)
    avg_scores_INOUT = np.mean(history_reward_in_out_distr, axis=1)
    data = [avg_scores_IN, avg_scores_OUT, avg_scores_INOUT]
    labels = ['IN', 'OUT', 'IN+OUT']

    print(np.median(avg_scores_IN))
    print(np.median(avg_scores_OUT))
    print(np.median(avg_scores_INOUT))

    fig, ax = plt.subplots(figsize=(10, 7))

    # Creating boxplot with labels
    bp = ax.boxplot(data, labels=labels)

    # Adding scatter points
    for i, points in enumerate(data, 1):
        x = [i] * len(points)
        ax.scatter(x, points, alpha=0.7)#, label=f'Points {labels[i-1]}')

    # Set plot labels
    ax.set_title(title)
    #ax.set_xlabel('Groups')
    ax.set_ylabel('Reward')
    # Automatically adjust subplot parameters for a tight layout
    plt.tight_layout()


    # Show legend
    #ax.legend()

    # Show plot
    #plt.show()
    if save_path:
        save_path=os.path.join(save_path, "Boxplot_" + title + ".png")
        plt.savefig(save_path)
    else:
        plt.show()


# def get_variations(test_set="IN"):
#     if test_set=="IN":
#         parameter1_range = [CARTPOLE_IN_LOWER_MASSPOLE, CARTPOLE_IN_UPPER_MASSPOLE]
#         parameter2_range = [CARTPOLE_IN_LOWER_LENGTH, CARTPOLE_IN_UPPER_LENGTH]
    
#         return generate_morphologies(parameter1_range, parameter2_range, [0.1, 0.1]) 
    
#     if test_set=="OUT":
#         # parameter1_OUT_range = [CARTPOLE_OUT_LOWER_MASSPOLE, CARTPOLE_OUT_UPPER_MASSPOLE]
#         # parameter2_OUT_range = [CARTPOLE_OUT_LOWER_LENGTH, CARTPOLE_OUT_UPPER_LENGTH]
#         # parameter1_left_range = [CARTPOLE_OUT_LOWER_MASSPOLE, CARTPOLE_IN_LOWER_MASSPOLE]
#         # parameter1_rigth_range = [CARTPOLE_IN_UPPER_MASSPOLE, CARTPOLE_OUT_UPPER_MASSPOLE]
#         # parameter2_left_range = [CARTPOLE_OUT_LOWER_LENGTH, CARTPOLE_IN_LOWER_LENGTH]
#         # parameter2_rigth_range = [CARTPOLE_IN_UPPER_LENGTH, CARTPOLE_OUT_UPPER_LENGTH]
#         # all_variations_left = generate_morphologies(parameter1_left_range, parameter2_OUT_range, [0.1, 0.1])
#         # all_variations_up = generate_morphologies(parameter1_OUT_range, parameter2_rigth_range, [0.1, 0.1])
#         # all_variations_rigth = generate_morphologies(parameter1_rigth_range, parameter2_OUT_range, [0.1, 0.1])
#         # all_variations_bottom = generate_morphologies(parameter1_OUT_range, parameter2_left_range, [0.1, 0.1])
#         # all_variations = np.concatenate((all_variations_left, all_variations_up,all_variations_bottom, all_variations_rigth, ), axis=0)
#         all_variations
#         print(all_variations)
#         return all_variations
            
#     if test_set=="INOUT":
#         parameter1_range = [CARTPOLE_OUT_LOWER_MASSPOLE, CARTPOLE_OUT_UPPER_MASSPOLE]
#         parameter2_range = [CARTPOLE_OUT_LOWER_LENGTH, CARTPOLE_OUT_UPPER_LENGTH]

#         return generate_morphologies(parameter1_range, parameter2_range, [0.1, 0.1])


if __name__ == "__main__": 
    
    #datafile_path = str(sys.argv[1])
    for datafile_path in all_datafile_paths:
        data = np.load(datafile_path)
        plots_path = os.path.join(os.path.dirname(datafile_path), "plots") 
        os.makedirs(plots_path, exist_ok=True)


        all_history_rewards_IN = data['all_history_rewards_IN']
        all_history_rewards_OUT = data['avg_rewards_OUT']
        all_history_rewards_INOUT = data['avg_rewards_INOUT']

        all_avgs_IN = [np.mean(run_rewards) for run_rewards in all_history_rewards_IN]
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

        IN_variations = utils.get_set(test_set="IN", step_sizes=[0.1, 0.1])
        OUT_variations = utils.get_set(test_set="OUT", step_sizes=[0.1, 0.1])
        INOUT_variations = utils.get_set(test_set="INOUT", step_sizes=[0.1, 0.1])
        # print("IN_variations:" ,IN_variations)
        # print("OUT_variations:" ,OUT_variations)
        # print("INOUT_variations:" ,INOUT_variations)

        plot_heatmap(IN_variations, all_history_rewards_IN, title='IN', save_path=plots_path)
        plot_heatmap(OUT_variations, all_history_rewards_OUT, title='OUT', save_path=plots_path)
        plot_heatmap(INOUT_variations, all_history_rewards_INOUT, title='INOUT', save_path=plots_path)

        #plot_boxplots(all_history_rewards_IN, all_history_rewards_OUT, all_history_rewards_INOUT, title="incremental", save_path=plots_path)


    
    