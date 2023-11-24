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

train_cauchy1_path = "/home/edo/THESIS/evolving-generalist-controllers/Results_Acrobot/cauchy1/20231124114220"
train_cauchy2_path = "/home/edo/THESIS/evolving-generalist-controllers/Results_Acrobot/cauchy2/20231124114220"
train_gaussian1_path = "/home/edo/THESIS/evolving-generalist-controllers/Results_Acrobot/gaussian1/20231124114220"
train_gaussian2_path = "/home/edo/THESIS/evolving-generalist-controllers/Results_Acrobot/gaussian2/20231124114220"
train_incremental_path = "/home/edo/THESIS/evolving-generalist-controllers/Results_Acrobot/incremental/20231124114220"
train_uniform_path = "/home/edo/THESIS/evolving-generalist-controllers/Results_Acrobot/uniform/20231124114220"
all_train_folders = [train_gaussian1_path, train_gaussian2_path, train_cauchy1_path, train_cauchy2_path, train_incremental_path, train_uniform_path]

save_path = "/home/edo/THESIS/evolving-generalist-controllers/Results_Acrobot/"

def plot(game, incremental_history_rewards, gaussian1_history_rewards, gaussian2_history_rewards, 
         cauchy1_history_rewards, cauchy2_history_rewards, uniform_history_rewards, 
         test, save_path):
    
    incremental_avg_scores = np.mean(incremental_history_rewards, axis=1)
    gaussian1_avg_scores = np.mean(gaussian1_history_rewards, axis=1)
    gaussian2_avg_scores = np.mean(gaussian2_history_rewards, axis=1)
    cauchy1_avg_scores = np.mean(cauchy1_history_rewards, axis=1)
    cauchy2_avg_scores = np.mean(cauchy2_history_rewards, axis=1)
    uniform_avg_scores = np.mean(uniform_history_rewards, axis=1)

    data = [incremental_avg_scores, gaussian1_avg_scores, gaussian2_avg_scores, 
            cauchy1_avg_scores, cauchy2_avg_scores, uniform_avg_scores]
    
    labels = ['Incremental', 'Gaussian1', 'Gaussian2', 'Cauchy1', 'Cauchy2', 'Uniform']

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
        ax.set_ylim(0, 100)
    plt.tight_layout()
    
    if save_path:
        save_path = os.path.join(save_path, "Boxplot_" + test + ".png")
        plt.savefig(save_path)
    else:
        plt.show()


with open(str(sys.argv[1])) as json_file:
    config = json.load(json_file)
game=config['game']

npz_cauchy1_path = train_cauchy1_path + "/all_history_rewards_data.npz"
npz_cauchy2_path = train_cauchy2_path + "/all_history_rewards_data.npz"
npz_gaussian1_path = train_gaussian1_path + "/all_history_rewards_data.npz"
npz_gaussian2_path = train_gaussian2_path + "/all_history_rewards_data.npz"
npz_incremental_path = train_incremental_path + "/all_history_rewards_data.npz"
npz_uniform_path = train_uniform_path + "/all_history_rewards_data.npz"

incremental_data = np.load(npz_incremental_path)
gaussian1_data = np.load(npz_gaussian1_path)
gaussian2_data = np.load(npz_gaussian2_path)
cauchy1_data = np.load(npz_cauchy1_path)
cauchy2_data = np.load(npz_cauchy2_path)
uniform_data = np.load(npz_uniform_path)

#IN BOXPLOT 
incremental_history_rewards_IN = incremental_data['all_history_rewards_IN']
gaussian1_history_rewards_IN = gaussian1_data['all_history_rewards_IN']
gaussian2_history_rewards_IN = gaussian2_data['all_history_rewards_IN']
cauchy1_history_rewards_IN = cauchy1_data['all_history_rewards_IN']
cauchy2_history_rewards_IN = cauchy2_data['all_history_rewards_IN']
uniform_history_rewards_IN = uniform_data['all_history_rewards_IN']

#OUT BOXPLOT 
incremental_history_rewards_OUT = incremental_data['avg_rewards_OUT']
gaussian1_history_rewards_OUT = gaussian1_data['avg_rewards_OUT']
gaussian2_history_rewards_OUT = gaussian2_data['avg_rewards_OUT']
cauchy1_history_rewards_OUT = cauchy1_data['avg_rewards_OUT']
cauchy2_history_rewards_OUT = cauchy2_data['avg_rewards_OUT']
uniform_history_rewards_OUT = uniform_data['avg_rewards_OUT']

#INOUT BOXPLOT 
incremental_history_rewards_INOUT = incremental_data['avg_rewards_INOUT']
gaussian1_history_rewards_INOUT = gaussian1_data['avg_rewards_INOUT']
gaussian2_history_rewards_INOUT = gaussian2_data['avg_rewards_INOUT']
cauchy1_history_rewards_INOUT = cauchy1_data['avg_rewards_INOUT']
cauchy2_history_rewards_INOUT = cauchy2_data['avg_rewards_INOUT']
uniform_history_rewards_INOUT = uniform_data['avg_rewards_INOUT']


plot(game, incremental_history_rewards_IN, gaussian1_history_rewards_IN, gaussian2_history_rewards_IN, cauchy1_history_rewards_IN, cauchy2_history_rewards_IN, uniform_history_rewards_IN, test="IN", save_path=save_path)

plot(game, incremental_history_rewards_OUT, gaussian1_history_rewards_OUT, gaussian2_history_rewards_OUT, cauchy1_history_rewards_OUT, cauchy2_history_rewards_OUT, uniform_history_rewards_OUT, test="OUT", save_path=save_path)

plot(game, incremental_history_rewards_INOUT, gaussian1_history_rewards_INOUT, gaussian2_history_rewards_INOUT, cauchy1_history_rewards_INOUT, cauchy2_history_rewards_INOUT, uniform_history_rewards_INOUT, test="INOUT", save_path=save_path)

