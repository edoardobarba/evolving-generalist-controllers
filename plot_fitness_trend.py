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



train_cauchy1_path = r"C:\Users\edoar\Documents\GitHub\Results_Biped\cauchy1\20231219091844"
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

train_random_path = r"C:\Users\edoar\Documents\GitHub\Results_Biped\random\20240108110555"
train_incremental_path = r"C:\Users\edoar\Documents\GitHub\Results_Biped\incremental\20231219025742"
train_incremental_30_path = r"C:\Users\edoar\Documents\GitHub\Results_Biped\incremental\20240205224118"
train_incremental_50_path = r"C:\Users\edoar\Documents\GitHub\Results_Biped\incremental\20240204175622"
train_MAB_path = r"C:\Users\edoar\Documents\GitHub\Results_Biped\MAB\20240130055434"

# train_MAB_path = "/home/edoardo.barba/Results_Biped/MAB/20240130055434"
# training_schedules = ["border_incr", "random", "incremental", "gaussian1", "gaussian2", "cauchy1", "cauchy2","uniform", "beta01", "beta02", "betawalk01", "betawalk02"]
# all_train_folders = [train_borderincr_path, train_random_path, train_incremental_path, train_gaussian1_path, train_gaussian2_path, train_cauchy1_path, train_cauchy2_path, train_uniform_path, train_beta01, train_beta02, train_betawalk01, train_betawalk02]
training_schedules = ["20 neurons", "30 neurons", "50 neurons"]
all_train_folders = [train_incremental_path, train_incremental_30_path, train_incremental_50_path]



save_path = r"C:\Users\edoar\Documents\GitHub\Results_Biped"

#print(np.shape(all_data))
arr_generations = [1000, 2000, 3000, 4000]
for generations in arr_generations:
    all_data = []
    for train_folder in all_train_folders:
        runs_folder = os.path.join(train_folder, "runs")
        all_runs = os.listdir(runs_folder)

        all_train_trends = []

        for run in all_runs:

            directory_path = os.path.join(runs_folder, str(run),  str(generations), "generalist_evals")  

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

            overall_mean = np.median(schedule_trend, axis=0)
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
    
    plt.ylim(-300, 10)

    fig_save_path = save_path + "/fitness_trends" + str(generations) + ".png"
    # Show the plot
    plt.savefig(fig_save_path, dpi=300)
    plt.close()
    print("plot saved in ", fig_save_path)