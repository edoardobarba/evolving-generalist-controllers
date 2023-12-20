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


train_cauchy1_path = "/home/edoardo.barba/Results_Biped/cauchy1/20231213105259"
train_cauchy2_path = "/home/edoardo.barba/Results_Biped/cauchy2/20231213105259"
train_gaussian1_path = "/home/edoardo.barba/Results_Biped/gaussian1/20231212113730"
train_gaussian2_path = "/home/edoardo.barba/Results_Biped/gaussian2/20231212113730"
train_incremental_path = "/home/edoardo.barba/Results_Biped/incremental/20231212113730"
train_uniform_path = "/home/edoardo.barba/Results_Biped/uniform/20231213105259"
train_RL_path = "/home/edoardo.barba/Results_Biped/RL/20231212113849"
train_beta1 = "/home/edoardo.barba/Results_Biped/beta1/20231213105322"
train_beta2 = "/home/edoardo.barba/Results_Biped/beta2/20231213105322"
train_betawalk1 = "/home/edoardo.barba/Results_Biped/betawalk1/20231212114325"
train_betawalk2 = "/home/edoardo.barba/Results_Biped/betawalk2/20231212114325"


# training_schedules = ["cauchy1", "cauchy2", "uniform"] #, "RL", "beta1", "beta2", "betawalk1", "betawalk2"]
# all_train_folders = [train_cauchy1_path, train_cauchy2_path, train_uniform_path]# , train_RL_path, train_beta1, train_beta2, train_betawalk1, train_betawalk2]
training_schedules = ["incremental", "gaussian1", "gaussian2", "cauchy1", "cauchy2", "uniform", "RL", "beta1", "beta2", "betawalk1", "betawalk2"]
all_train_folders = [train_incremental_path, train_gaussian1_path, train_gaussian2_path, train_cauchy1_path, train_cauchy2_path, train_uniform_path, train_RL_path, train_beta1, train_beta2, train_betawalk1, train_betawalk2]


save_path = "/home/edoardo.barba/"


if __name__ == "__main__": 
    with open(str(sys.argv[1])) as json_file:
        config = json.load(json_file)
    game=config['game']

    if game == "CartPoleEnv":
        save_path = save_path + "Results_Cart/"

    elif game == "Acrobot":
        save_path = save_path + "Results_Acrobot/"

    elif game == "BipedalWalker":
        save_path = save_path + "Results_Biped/"


    for datafile_path in all_train_folders:
        print(datafile_path)  
        npz_file_name = datafile_path + "/all_history_rewards_data.npz"       
        data = np.load(npz_file_name)

        all_history_rewards_INOUT = data['avg_rewards_INOUT']

        # datafile_path = os.path.join(os.path.dirname(save_path), "results") 
        
        os.makedirs(datafile_path, exist_ok=True)
        csv_file_path = datafile_path + "/results.csv"



        INOUT_variations = utils.get_set(config, test_set="INOUT")
        column_names = [str(tuple(column)) for column in INOUT_variations]
        header=','.join(column_names)
        # Save the array to a CSV file
        np.savetxt(csv_file_path, all_history_rewards_INOUT, delimiter=',', header=','.join(column_names), comments='')

        print("Results saved in ", csv_file_path)

        IN_variations = utils.get_set(config, test_set="IN")
        OUT_variations = utils.get_set(config, test_set="OUT")
        

        



    
    