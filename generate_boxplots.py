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

npz_incremental_path = "/home/edo/THESIS/evolving-generalist-controllers/Results_Cart/incremental/20231120164430/all_history_rewards_data.npz"
npz_gaussian1_path = "/home/edo/THESIS/evolving-generalist-controllers/Results_Cart/gaussian1/20231121115754/all_history_rewards_data.npz"
npz_gaussian2_path = "/home/edo/THESIS/evolving-generalist-controllers/Results_Cart/gaussian2/20231121125950/all_history_rewards_data.npz"

save_path = "/home/edo/THESIS/evolving-generalist-controllers/Results_Cart/"


def plot(incremental_history_rewards, gaussian1_history_rewards, gaussian2_history_rewards, test, save_path):

    incremental_avg_scores = np.mean(incremental_history_rewards, axis=1)
    gaussian1_avg_scores = np.mean(gaussian1_history_rewards, axis=1)
    gaussian2_avg_scores = np.mean(gaussian2_history_rewards, axis=1)

    data = [incremental_avg_scores, gaussian1_avg_scores, gaussian2_avg_scores]
    labels = ['Incremental', 'Gaussian1', 'Gaussian2']

    fig, ax = plt.subplots(figsize=(10, 7))
    bp = ax.boxplot(data, labels=labels)
    for i, points in enumerate(data, 1):
        x = [i] * len(points)
        ax.scatter(x, points, alpha=0.7)

    ax.set_title(test+" TESTING")
    ax.set_ylabel('Reward')
    ax.set_ylim(-1001, 0)
    plt.tight_layout()
    if save_path:
        save_path=os.path.join(save_path, "Boxplot_" + test + ".png")
        plt.savefig(save_path)
    else:
        plt.show()




incremental_data = np.load(npz_incremental_path)
gaussian1_data = np.load(npz_gaussian1_path)
gaussian2_data = np.load(npz_gaussian2_path)

# plots_path = os.path.join(os.path.dirname(datafile_path), "plots") 
# os.makedirs(plots_path, exist_ok=True)

#IN BOXPLOT 
incremental_history_rewards_IN = incremental_data['all_history_rewards_IN']
gaussian1_history_rewards_IN = gaussian1_data['all_history_rewards_IN']
gaussian2_history_rewards_IN = gaussian2_data['all_history_rewards_IN']

#OUT BOXPLOT 
incremental_history_rewards_OUT = incremental_data['avg_rewards_OUT']
gaussian1_history_rewards_OUT = gaussian1_data['avg_rewards_OUT']
gaussian2_history_rewards_OUT = gaussian2_data['avg_rewards_OUT']

#INOUT BOXPLOT 
incremental_history_rewards_INOUT = incremental_data['avg_rewards_INOUT']
gaussian1_history_rewards_INOUT = gaussian1_data['avg_rewards_INOUT']
gaussian2_history_rewards_INOUT = gaussian2_data['avg_rewards_INOUT']

plot(incremental_history_rewards_IN, gaussian1_history_rewards_IN, gaussian2_history_rewards_IN, test="IN", save_path=save_path)
plot(incremental_history_rewards_OUT, gaussian1_history_rewards_OUT, gaussian2_history_rewards_OUT, test="OUT", save_path=save_path)
plot(incremental_history_rewards_INOUT, gaussian1_history_rewards_INOUT, gaussian2_history_rewards_INOUT, test="IN+OUT", save_path=save_path)





# incremental_history_rewards_OUT = incremental_data['avg_rewards_OUT']
# incremental_history_rewards_INOUT = incremental_data['avg_rewards_INOUT']



# incremental_avg_scores_OUT = np.mean(incremental_history_rewards_OUT, axis=1)
# incremental_avg_scores_INOUT= np.mean(incremental_history_rewards_INOUT, axis=1)


# IN_data = [incremental_avg_scores_IN, incremental_avg_scores_OUT, incremental_avg_scores_INOUT]
# labels = ['IN', 'OUT', 'IN+OUT']


