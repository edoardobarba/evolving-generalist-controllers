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
import statsmodels.stats.api as sms
from scipy import stats



# train_cauchy1_path = "/home/edoardo.barba/Results_Biped/cauchy1/20231213105259"
# train_cauchy2_path = "/home/edoardo.barba/Results_Biped/cauchy2/20231213105259"
# train_gaussian1_path = "/home/edoardo.barba/Results_Biped/gaussian1/20231212113730"
# train_gaussian2_path = "/home/edoardo.barba/Results_Biped/gaussian2/20231212113730"
# train_incremental_path = "/home/edoardo.barba/Results_Biped/incremental/20231212113730"
# train_uniform_path = "/home/edoardo.barba/Results_Biped/uniform/20231213105259"
# train_RL_path = "/home/edoardo.barba/Results_Biped/RL/20231214142613"  #NEW RL WITH AVERAGE
# train_beta1 = "/home/edoardo.barba/Results_Biped/beta1/20231213105322"
# train_beta2 = "/home/edoardo.barba/Results_Biped/beta2/20231213105322"
# train_betawalk1 = "/home/edoardo.barba/Results_Biped/betawalk1/20231214142613" #NEW BETAWALK
# train_betawalk2 = "/home/edoardo.barba/Results_Biped/betawalk2/20231214142613" #NEW BETAWALK
# train_gauss_dec = "/home/edoardo.barba/Results_Biped/gauss_dec/20231214143701"

#NEW TRAIN SET: 

train_cauchy1_path = "/home/edoardo.barba/Results_Biped/cauchy1/20231219091844"
train_cauchy2_path = "/home/edoardo.barba/Results_Biped/cauchy2/20231219091844"
train_gaussian1_path = "/home/edoardo.barba/Results_Biped/gaussian1/20231215114440"
train_gaussian2_path = "/home/edoardo.barba/Results_Biped/gaussian2/20231215114440"
train_incremental_path = "/home/edoardo.barba/Results_Biped/incremental/20231219025742"
train_uniform_path = "/home/edoardo.barba/Results_Biped/uniform/20231219091844"
train_RL_path = "/home/edoardo.barba/Results_Biped/RL/20231215114621" 
train_beta1 = "/home/edoardo.barba/Results_Biped/beta01/20231219092337"
train_beta2 = "/home/edoardo.barba/Results_Biped/beta02/20231219092337"
train_betawalk01 = "/home/edoardo.barba/Results_Biped/betawalk01/20231221112601" 
train_betawalk02 = "/home/edoardo.barba/Results_Biped/betawalk02/20231221112601" 
train_gauss_dec = "/home/edoardo.barba/Results_Biped/gauss_dec/20231215114907"


#training_schedules = ["incremental", "gaussian1", "gaussian2","cauchy1","cauchy2","uniform", "RL", "beta1", "beta2", "betawalk1", "betawalk2", "gauss_dec"]
#all_train_folders = [train_incremental_path, train_gaussian1_path, train_gaussian2_path, train_cauchy1_path, train_cauchy2_path, train_uniform_path, train_RL_path, train_beta1, train_beta2, train_betawalk1, train_betawalk2, train_gauss_dec]

training_schedules = ["incremental", "cauchy1","cauchy2","uniform", "beta01", "beta02", "betawalk01", "betawalk02"]
all_train_folders = [train_incremental_path, train_cauchy1_path, train_cauchy2_path, train_uniform_path, train_beta1, train_beta2, train_betawalk01, train_betawalk02]

save_path = "/home/edoardo.barba/Results_Biped"



#print(np.shape(all_data))
arr_generations = [1000, 2000, 3000, 4000]
for generations in arr_generations:
    all_data = []
    for train_folder in all_train_folders:
        runs_folder = train_folder + "/runs/"
        all_runs = os.listdir(runs_folder)

        all_train_trends = []

        for run in all_runs:

            directory_path = runs_folder + run + "/" + str(generations) + "/generalist_evals"  

            if os.path.exists(directory_path):
                all_files = os.listdir(directory_path)

                desired_file = [file for file in all_files if "generalist_evals.csv" in file]
                #print(desired_file)

                file_path = directory_path + "/" + desired_file[0]
                df = pd.read_csv(file_path)
                first_column_values = df.iloc[:, 4].values

                all_train_trends.append(first_column_values)

        all_data.append(np.array(all_train_trends, dtype=object))
    all_means = []
    #print("n_schedules:")
    #print(len(all_data))
    n_runs_per_schedule = []
    actual_training_schedules = []
    # Generate x values from 0 to 500
    x_values = np.arange(0, generations)
    for i, schedule_trend in enumerate(all_data): 

        schedule_trend = [arr[:generations] for arr in schedule_trend if len(arr)>=generations]
        #print("n_runs")
        #print(len(schedule_trend))
        if len(schedule_trend)>0:

            for run in schedule_trend:
                plt.plot(x_values, run, linewidth=1, color = "b")
                    # Add labels and legend
            plt.xlabel('Generations')
            plt.ylabel('Fitness')
            plt.title('Generalization capability over generations')
            plt.tight_layout()
            plt.grid()

            fig_save_path = all_train_folders[i] + "/fitness_trends" + str(generations) + ".png"
            # Show the plot
            plt.savefig(fig_save_path)
            plt.close()

            overall_mean = np.mean(schedule_trend, axis=0)
            n_runs_per_schedule.append(len(schedule_trend))
            actual_training_schedules.append(training_schedules[i])

            # print(overall_mean)

            all_means.append(overall_mean)


    #print(np.array(all_means))
    all_means = np.array(all_means, dtype=object)



    line_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#FFA500', '#800080', '#00FFFF', '#FF00FF', 'purple']

    # Plot each row of the array
    for i in range(all_means.shape[0]):
        #print("all_means[i]")
        #print(all_means[i])
        #if not np.isnan(all_means[i]).any():
        plt.plot(x_values, all_means[i], label=f'{actual_training_schedules[i]}{" ("}{n_runs_per_schedule[i]}{")"}', linewidth=1, color=line_colors[i])

    # Add labels and legend
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.legend()
    plt.title('Generalization capability over generations')
    plt.tight_layout()
    plt.grid()

    fig_save_path = save_path + "/fitness_trends" + str(generations) + ".png"
    # Show the plot
    plt.savefig(fig_save_path, dpi=300)
    plt.close()
    print("plot saved in ", fig_save_path)
