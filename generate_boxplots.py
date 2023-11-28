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
from scipy.stats import ttest_ind

training_schedules = ["incremental", "gaussian1", "gaussian2", "cauchy1", "cauchy2", "uniform", "RL"]

train_cauchy1_path = "/home/edo/THESIS/evolving-generalist-controllers/Results_Cart/cauchy1/20231127163539"
train_cauchy2_path = "/home/edo/THESIS/evolving-generalist-controllers/Results_Cart/cauchy2/20231127163539"
train_gaussian1_path = "/home/edo/THESIS/evolving-generalist-controllers/Results_Cart/gaussian1/20231127163539"
train_gaussian2_path = "/home/edo/THESIS/evolving-generalist-controllers/Results_Cart/gaussian2/20231127163539"
train_incremental_path = "/home/edo/THESIS/evolving-generalist-controllers/Results_Cart/incremental/20231127163539"
train_uniform_path = "/home/edo/THESIS/evolving-generalist-controllers/Results_Cart/uniform/20231127163539"
train_RL_path = "/home/edo/THESIS/evolving-generalist-controllers/Results_Cart/RL/20231127172931"
all_train_folders = [train_gaussian1_path, train_gaussian2_path, train_cauchy1_path, train_cauchy2_path, train_incremental_path, train_uniform_path, train_RL_path]


save_path = "/home/edo/THESIS/evolving-generalist-controllers/Results_Acrobot/"

def plot(game, incremental_history_rewards, gaussian1_history_rewards, gaussian2_history_rewards, 
         cauchy1_history_rewards, cauchy2_history_rewards, uniform_history_rewards, RL_history_rewards,
         test, save_path):
    
    incremental_avg_scores = np.mean(incremental_history_rewards, axis=1)
    gaussian1_avg_scores = np.mean(gaussian1_history_rewards, axis=1)
    gaussian2_avg_scores = np.mean(gaussian2_history_rewards, axis=1)
    cauchy1_avg_scores = np.mean(cauchy1_history_rewards, axis=1)
    cauchy2_avg_scores = np.mean(cauchy2_history_rewards, axis=1)
    uniform_avg_scores = np.mean(uniform_history_rewards, axis=1)
    RL_avg_scores = np.mean(RL_history_rewards, axis=1)

    data = [incremental_avg_scores, gaussian1_avg_scores, gaussian2_avg_scores, 
            cauchy1_avg_scores, cauchy2_avg_scores, uniform_avg_scores, RL_avg_scores]
    
    labels = ['Incremental', 'Gaussian1', 'Gaussian2', 'Cauchy1', 'Cauchy2', 'Uniform', 'RL']

    fig, ax = plt.subplots(figsize=(10, 7))
    bp = ax.boxplot(data, labels=labels)
    
    for i, points in enumerate(data, 1):
        x = [i] * len(points)
        ax.scatter(x, points, alpha=0.7)

    ax.set_title(test + " TESTING")
    ax.set_ylabel('Reward')
    if game == "CartPoleEnv":
        ax.set_ylim(-1001, 0)
    elif game == "AcrobotEnv":
        ax.set_ylim(50, 70)

    plt.tight_layout()
    
    if save_path:
        save_path = os.path.join(save_path, "Boxplot_" + test + ".png")
        plt.savefig(save_path)
    else:
        plt.show()



def print_statistics(data, label):
    avg_scores = np.mean(data, axis=1)
    print(f"\nStatistics for {label}:")
    print(f"  Mean Reward: {np.mean(avg_scores)}")
    print(f"  Standard Deviation: {np.std(avg_scores)}")

def perform_ttest(data1, data2, label1, label2):
    avg_scores1 = np.mean(data1, axis=1)
    avg_scores2 = np.mean(data2, axis=1)

    print(f"\nT-test between {label1} and {label2}:")
    
    # Perform t-test
    t_stat, p_value = ttest_ind(avg_scores1, avg_scores2)
    print(f"  T-statistic: {t_stat}")
    print(f"  P-value: {p_value}")


def rank_array(arr):
    """
    Rank each element in the input array. Lower values are assigned lower ranks.

    Parameters:
    - arr: List or numpy array.

    Returns:
    - List of ranks.
    """
    # Create a list of (value, index) pairs
    indexed_arr = list(enumerate(arr, start=1))

    # Sort the list by values in ascending order
    sorted_arr = sorted(indexed_arr, key=lambda x: x[1])

    # Assign ranks based on the sorted order
    ranks = [item[0] for item in sorted_arr]

    return ranks


def print_rankings(incremental_history_rewards, gaussian1_history_rewards, gaussian2_history_rewards, cauchy1_history_rewards, cauchy2_history_rewards, uniform_history_rewards, RL_history_rewards, test):
    incremental_avg_scores = np.mean(incremental_history_rewards, axis=1)
    gaussian1_avg_scores = np.mean(gaussian1_history_rewards, axis=1)
    gaussian2_avg_scores = np.mean(gaussian2_history_rewards, axis=1)
    cauchy1_avg_scores = np.mean(cauchy1_history_rewards, axis=1)
    cauchy2_avg_scores = np.mean(cauchy2_history_rewards, axis=1)
    uniform_avg_scores = np.mean(uniform_history_rewards, axis=1)
    RL_avg_scores = np.mean(RL_history_rewards, axis=1)

    avgs = []
    avgs.append(np.mean(incremental_avg_scores))
    avgs.append(np.mean(gaussian1_avg_scores))
    avgs.append(np.mean(gaussian2_avg_scores))
    avgs.append(np.mean(cauchy1_avg_scores))
    avgs.append(np.mean(cauchy2_avg_scores))
    avgs.append(np.mean(uniform_avg_scores))
    avgs.append(np.mean(RL_avg_scores))

    ranks = rank_array(avgs)

    print(test, "test ranking: ")
    for rank in ranks: 
        print("  ", training_schedules[rank-1])

    print("\n")



with open(str(sys.argv[1])) as json_file:
    config = json.load(json_file)
game = config['game']

npz_cauchy1_path = train_cauchy1_path + "/all_history_rewards_data.npz"
npz_cauchy2_path = train_cauchy2_path + "/all_history_rewards_data.npz"
npz_gaussian1_path = train_gaussian1_path + "/all_history_rewards_data.npz"
npz_gaussian2_path = train_gaussian2_path + "/all_history_rewards_data.npz"
npz_incremental_path = train_incremental_path + "/all_history_rewards_data.npz"
npz_uniform_path = train_uniform_path + "/all_history_rewards_data.npz"
npz_RL_path = train_RL_path + "/all_history_rewards_data.npz"

incremental_data = np.load(npz_incremental_path)
gaussian1_data = np.load(npz_gaussian1_path)
gaussian2_data = np.load(npz_gaussian2_path)
cauchy1_data = np.load(npz_cauchy1_path)
cauchy2_data = np.load(npz_cauchy2_path)
uniform_data = np.load(npz_uniform_path)
RL_data = np.load(npz_RL_path)

# IN BOXPLOT 
incremental_history_rewards_IN = incremental_data['all_history_rewards_IN']
gaussian1_history_rewards_IN = gaussian1_data['all_history_rewards_IN']
gaussian2_history_rewards_IN = gaussian2_data['all_history_rewards_IN']
cauchy1_history_rewards_IN = cauchy1_data['all_history_rewards_IN']
cauchy2_history_rewards_IN = cauchy2_data['all_history_rewards_IN']
uniform_history_rewards_IN = uniform_data['all_history_rewards_IN']
RL_history_rewards_IN = RL_data['all_history_rewards_IN']

# OUT BOXPLOT 
incremental_history_rewards_OUT = incremental_data['avg_rewards_OUT']
gaussian1_history_rewards_OUT = gaussian1_data['avg_rewards_OUT']
gaussian2_history_rewards_OUT = gaussian2_data['avg_rewards_OUT']
cauchy1_history_rewards_OUT = cauchy1_data['avg_rewards_OUT']
cauchy2_history_rewards_OUT = cauchy2_data['avg_rewards_OUT']
uniform_history_rewards_OUT = uniform_data['avg_rewards_OUT']
RL_history_rewards_OUT = RL_data['avg_rewards_OUT']

# INOUT BOXPLOT 
incremental_history_rewards_INOUT = incremental_data['avg_rewards_INOUT']
gaussian1_history_rewards_INOUT = gaussian1_data['avg_rewards_INOUT']
gaussian2_history_rewards_INOUT = gaussian2_data['avg_rewards_INOUT']
cauchy1_history_rewards_INOUT = cauchy1_data['avg_rewards_INOUT']
cauchy2_history_rewards_INOUT = cauchy2_data['avg_rewards_INOUT']
uniform_history_rewards_INOUT = uniform_data['avg_rewards_INOUT']
RL_history_rewards_INOUT = RL_data['avg_rewards_INOUT']

# Plotting
plot(game, incremental_history_rewards_IN, gaussian1_history_rewards_IN, gaussian2_history_rewards_IN, cauchy1_history_rewards_IN, cauchy2_history_rewards_IN, uniform_history_rewards_IN, RL_history_rewards_IN, test="IN", save_path=save_path)

plot(game, incremental_history_rewards_OUT, gaussian1_history_rewards_OUT, gaussian2_history_rewards_OUT, cauchy1_history_rewards_OUT, cauchy2_history_rewards_OUT, uniform_history_rewards_OUT, RL_history_rewards_OUT, test="OUT", save_path=save_path)

plot(game, incremental_history_rewards_INOUT, gaussian1_history_rewards_INOUT, gaussian2_history_rewards_INOUT, cauchy1_history_rewards_INOUT, cauchy2_history_rewards_INOUT, uniform_history_rewards_INOUT, RL_history_rewards_INOUT, test="INOUT", save_path=save_path)

print_rankings(incremental_history_rewards_IN, gaussian1_history_rewards_IN, gaussian2_history_rewards_IN, cauchy1_history_rewards_IN, cauchy2_history_rewards_IN, uniform_history_rewards_IN, RL_history_rewards_IN, "IN")
print_rankings(incremental_history_rewards_OUT, gaussian1_history_rewards_OUT, gaussian2_history_rewards_OUT, cauchy1_history_rewards_OUT, cauchy2_history_rewards_OUT, uniform_history_rewards_OUT, RL_history_rewards_OUT, "OUT")
print_rankings(incremental_history_rewards_INOUT, gaussian1_history_rewards_INOUT, gaussian2_history_rewards_INOUT, cauchy1_history_rewards_INOUT, cauchy2_history_rewards_INOUT, uniform_history_rewards_INOUT, RL_history_rewards_INOUT, "INOUT")


# Print statistics for IN category
print_statistics(incremental_history_rewards_IN, "Incremental")
print_statistics(gaussian1_history_rewards_IN, "Gaussian1")
print_statistics(gaussian2_history_rewards_IN, "Gaussian2")
print_statistics(cauchy1_history_rewards_IN, "Cauchy1")
print_statistics(cauchy2_history_rewards_IN, "Cauchy2")
print_statistics(uniform_history_rewards_IN, "Uniform")
print_statistics(RL_history_rewards_IN, "RL")


# Perform t-test between Incremental and RL
perform_ttest(incremental_history_rewards_IN, RL_history_rewards_IN, "Incremental", "RL")

