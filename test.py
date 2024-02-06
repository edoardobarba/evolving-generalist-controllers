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
from ant_v4_modified import AntEnv
from walker2d_v4_modified import Walker2dEnv
from bipedal_walker_modified import BipedalWalker
from cartpole_modified import CartPoleEnv
import gymnasium as gym
#from utils import gym_render, save_dataframes
import joblib



# # ANT 
# train_incremental_path = "/home/edoardo.barba/Results_Ant/incremental/20240119201533"
# train_random_path = "/home/edoardo.barba/Results_Ant/random/20240119201533"

# training_schedules = ["random", "incremental"]
# all_train_folders = [train_random_path, train_incremental_path]





# BIPEDAL WALKER 

# train_cauchy1_path = "/home/edoardo.barba/Results_Biped/cauchy1/20231219091844"
# train_cauchy2_path = "/home/edoardo.barba/Results_Biped/cauchy2/20231219091844"
# train_gaussian1_path = "/home/edoardo.barba/Results_Biped/gaussian1/20231223120018"
# train_gaussian2_path = "/home/edoardo.barba/Results_Biped/gaussian2/20231223120018"
# train_incremental_path = "/home/edoardo.barba/Results_Biped/incremental/20231219025742"
# train_uniform_path = "/home/edoardo.barba/Results_Biped/uniform/20231219091844"
# train_RL_path = "/home/edoardo.barba/Results_Biped_old/RL/20231215114621" 
# train_beta01 = "/home/edoardo.barba/Results_Biped_old/beta1"
# # train_beta02 = "/home/edoardo.barba/Results_Biped/beta02/20231219092337"
# # train_betawalk01 = "/home/edoardo.barba/Results_Biped/betawalk01/20231221112601" 
# # train_betawalk02 = "/home/edoardo.barba/Results_Biped/betawalk02/20231221112601" 
# # # train_gauss_dec = "/home/edoardo.barba/Results_Biped/gauss_dec/20231223120018"
# # train_default_path = "/home/edoardo.barba/Results_Biped/default/20240103141803"
# # train_borderincr_path = "/home/edoardo.barba/Results_Biped/border_incr/20240103134750"
# train_random_path = "/home/edoardo.barba/Results_Biped/random/20240125170701"
train_incremental_path = r"C:\Users\edoar\Documents\GitHub\Results_Biped\incremental\20240205224118"
train_beta01 = r"C:\Users\edoar\Documents\GitHub\Results_Biped\beta01\20231219092337"
# train_beta02 = "/home/edoardo.barba/Results_Biped/beta02/20231219092337"
# train_betawalk01 = "/home/edoardo.barba/Results_Biped/betawalk01/20231221112601" 
# train_betawalk02 = "/home/edoardo.barba/Results_Biped/betawalk02/20231221112601" 
# # train_gauss_dec = "/home/edoardo.barba/Results_Biped/gauss_dec/20231223120018"
# train_default_path = "/home/edoardo.barba/Results_Biped/default/20240103141803"
# train_borderincr_path = "/home/edoardo.barba/Results_Biped/border_incr/20240103134750"
# train_random_path = "/home/edoardo.barba/Results_Biped/random/20240125170701"
train_MAB_path = r"C:\Users\edoar\Documents\GitHub\Results_Biped\MAB\20240130055434"
# training_schedules = ["border_incr", "random", "incremental", "gaussian1", "gaussian2", "cauchy1", "cauchy2","uniform", "beta01", "beta02", "betawalk01", "betawalk02"]
# all_train_folders = [train_borderincr_path, train_random_path, train_incremental_path, train_gaussian1_path, train_gaussian2_path, train_cauchy1_path, train_cauchy2_path, train_uniform_path, train_beta01, train_beta02, train_betawalk01, train_betawalk02]
training_schedules = ["incremental"]
all_train_folders = [train_incremental_path]



# WALKER 2D

# # train_borderincr_path = "/home/edoardo.barba/Results_Walker/border_incr/20240117050517"

# train_incremental_path = "/home/edoardo.barba/Results_Walker/incremental/20240120035256"
# train_random_path  = "/home/edoardo.barba/Results_Walker/random/20240120035256"



# all_train_folders = [train_random_path, train_incremental_path]
# training_schedules = ["random", "incremental"]
# # all_train_folders = [train_borderincr_path, train_random_path, train_incremental_path]


# ACROBOT

# train_cauchy1_path = "/home/edoardo.barba/Results_Acrobot/cauchy1/20231130173608"
# train_cauchy2_path = "/home/edoardo.barba/Results_Acrobot/cauchy2/20231130173608"
# train_gaussian1_path = "/home/edoardo.barba/Results_Acrobot/gaussian1/20231130171943"
# train_gaussian2_path = "/home/edoardo.barba/Results_Acrobot/gaussian2/20231130173608"
# train_incremental_path = "/home/edoardo.barba/Results_Acrobot/incremental/20231130171943"


# training_schedules = ["incremental", "gaussian1", "gaussian2", "cauchy1", "cauchy2"]
# all_train_folders = [train_incremental_path, train_gaussian1_path, train_gaussian2_path, train_cauchy1_path, train_cauchy2_path]


#save_path = "/home/edoardo.barba/Results_Walker"

#save_path = "/home/edoardo.barba/Results_Walker"
ACTORS = -1


# CARTPOLE_IN_LOWER_MASSPOLE = 0.05
# CARTPOLE_IN_UPPER_MASSPOLE = 0.5
# CARTPOLE_OUT_LOWER_MASSPOLE = 0.01
# CARTPOLE_OUT_UPPER_MASSPOLE = 1.0

# CARTPOLE_IN_LOWER_LENGTH = 0.25
# CARTPOLE_IN_UPPER_LENGTH = 0.75
# CARTPOLE_OUT_LOWER_LENGTH = 0.05
# CARTPOLE_OUT_UPPER_LENGTH = 1.0

# CARTPOLE_DEFAULT_MASS = 1
# CARTPOLE_DEFAULT_LENGTH = 1



# ACROBOT_IN_LOWER_MASS1 = 0.3
# ACROBOT_IN_UPPER_MASS1 = 0.7
# ACROBOT_OUT_LOWER_MASS1 = 0.1
# ACROBOT_OUT_UPPER_MASS1 = 1.0

# ACROBOT_IN_LOWER_MASS2 = 0.3
# ACROBOT_IN_UPPER_MASS2 = 0.7
# ACROBOT_OUT_LOWER_MASS2 = 0.1
# ACROBOT_OUT_UPPER_MASS2 = 1.0

# ACROBOT_DEFAULT__MASS1 = 0.1
# ACROBOT_DEFAULT_MASS2 = 0.5


SEED=0


def gym_render_testing(game, agent, xml_path, parameters, topology, steps):
    s = 0
    total_reward = 0

    if game == AntEnv:
        xml_file = '{}/Ant_{:.2f}_hip_{:.2f}_ankle.xml'.format(xml_path, parameters[0], parameters[1])
        env = game(xml_file, render_mode=None, healthy_reward=0)
    elif game == Walker2dEnv:
        xml_file = '{}/Walker_{:.3f}_thigh_{:.3f}_leg.xml'.format(xml_path, parameters[0], parameters[1])
        env = game(xml_file, render_mode=None, healthy_reward=0)
    elif game == "AcrobotEnv":
        env = gym.make('Acrobot-v1', render_mode = None).unwrapped
        env.LINK_MASS_1 = parameters[0]  #: [kg] mass of link 1
        env.LINK_MASS_2 = parameters[1]  #: [kg] mass of link 2
    else:
        env = game(parameters)

    obs, info = env.reset(seed=SEED)
    done = False
 
    nn = agent
    weights = nn.reshape_layers(topology) 

    while not done:
        action = nn.feedforward(weights, topology, obs)

        if game == "AcrobotEnv":
            action = np.argmax(action)
        
        if game == Walker2dEnv:
            action = action.tolist()

        if game == AntEnv:
            action = action.tolist()
            
        obs, reward, terminated, truncated, info = env.step(action)

        s += 1
        total_reward += reward

        if s > steps:
            break

        done = terminated or truncated

    # print(-total_reward)

    env.close()
    return -total_reward

def comparison(game, agent, i, test_set, topology, max_steps, xml_path = None):

    fitness = gym_render_testing(game, agent, xml_path, test_set[i], topology, max_steps)
    return fitness


def do_default_episode(game, env, nn, weights, topology, max_steps, max_fitness):

    reward = 0
    obs, info = env.reset(seed=SEED)
    done = False
    step = 0

    while not done:
        action = nn.feedforward(weights, topology, obs)
        if game == "AcrobotEnv":
            action = np.argmax(action)
        obs, r, terminated, truncated, info = env.step(action)

        step += 1
        reward += -r
        if step > max_steps or reward<max_fitness:
            break
        
        done = terminated or truncated

    return reward

def get_NN(file):
    x = list(csv.reader(file))
    #weights_ANN_file.close()             
    x = np.asarray(x, dtype=float)
    x = list(x.flatten())
    x.pop(0)
    x = np.array(x)
    tensor = torch.tensor(x)
    nn = NeuralNetwork(tensor)
    weights = nn.reshape_layers(topology)
    return nn, weights


def set_test(config, nn, weights, max_steps, max_fitness, testing_set): 
    game = config["game"]
    topology = config["NN-struc"]
    #print("testing ", testing_set, "...")
    all_variations = utils.get_set(config, testing_set)

    xml_path = config["xml"]

    if game!="AcrobotEnv":
        game = eval(game)

    history_reward = []    

    
    history_reward = joblib.Parallel(n_jobs=ACTORS)(joblib.delayed(comparison)(game = game, agent=nn, i=i, test_set=all_variations, topology=topology, max_steps = max_steps, xml_path=xml_path)
                                                                for i in range(len(all_variations)))
    
    return history_reward, all_variations

def list_folders(path):
    # Get a list of all items (files and folders) in the specified path
    items = os.listdir(path)

    # Filter out only the directories
    folders = [item for item in items if os.path.isdir(os.path.join(path, item))]

    return folders


if __name__ == '__main__':
    
    json_file_name = str(sys.argv[1])

    for train_folder_path in all_train_folders:
        print(train_folder_path)
        save_path = train_folder_path
        runs_folder_path = save_path + "/runs"

        with open(json_file_name) as json_file:
            print('Running testing for', sys.argv[1])
            config = json.load(json_file)

        game = config['game']
        step_sizes = config['testing_step_sizes']
        topology = config["NN-struc"]
        max_steps = config["nStep"]
        max_fitness = config["maxFitness"]
        xml_path = config['xml']

        runs_folders = list_folders(runs_folder_path)

        all_history_rewards_IN = []
        all_history_rewards_OUT = []
        all_history_rewards_INOUT = []

        #generations = 3000
        #print("Testing ", generations, "generations")
        generations = 3000
        print("Generations: ", generations)
        for i, run_number in enumerate(runs_folders):
            run_path = os.path.join(runs_folder_path, run_number)
            training_txt_path = os.path.join(run_path,"training.txt")
            with open(training_txt_path, 'r') as file:
                content = file.read()
                generations_number = int(content[13:-1])
                #print("gen_number: ", generations_number)

            #if generations_number>=generations:
            #    continue

            print("Testing run number", i+1, "..." )
            
            generations_path = os.path.join(run_path, str(generations))
            generalist_folder_path = os.path.join(generations_path, "evals")
            path_generalist_ANN_weights = os.path.join(generalist_folder_path, os.listdir(generalist_folder_path)[0])
            file_generalist_ANN_weights = open(path_generalist_ANN_weights)
            nn, weights = get_NN(file_generalist_ANN_weights)
            file_generalist_ANN_weights.close()

            # history_reward_IN, all_var_IN = set_test(config, nn = nn, weights=weights, max_steps=max_steps, max_fitness=max_fitness, testing_set="IN")      
            # history_reward_OUT, all_var_OUT = set_test(config, nn = nn, weights=weights, max_steps=max_steps, max_fitness=max_fitness, testing_set="OUT")  
            history_reward_INOUT, all_var_INOUT = set_test(config, nn = nn, weights=weights, max_steps=max_steps, max_fitness=max_fitness, testing_set="INOUT")  

            # print("all_var_IN")
            # print(all_var_IN)
            # print("all_var_OUT")
            # print(all_var_OUT)

            # all_history_rewards_IN.append(np.array(history_reward_IN))
            # all_history_rewards_OUT.append(np.array(history_reward_OUT))
            all_history_rewards_INOUT.append(np.array(history_reward_INOUT))


        print("number of runs: ")
        print(len(all_history_rewards_IN))
        print("Results for ", train_folder_path)
        # all_avgs_IN = [np.mean(run_rewards) for run_rewards in all_history_rewards_IN]
        # print("Mean reward IN: ", np.mean(all_avgs_IN))

        # all_avgs_OUT = [np.mean(run_rewards) for run_rewards in all_history_rewards_OUT]
        # print("Mean reward OUT: ", np.mean(all_avgs_OUT))

        all_avgs_INOUT = [np.mean(run_rewards) for run_rewards in all_history_rewards_INOUT]
        print("Mean reward INOUT: ", np.mean(all_avgs_INOUT))

        save_path = os.path.join(save_path, "all_history_rewards_data.npz")
        #np.savez(save_path, all_history_rewards_IN=np.array(all_history_rewards_IN), avg_rewards_OUT=np.array(all_history_rewards_OUT), avg_rewards_INOUT=np.array(all_history_rewards_INOUT))
        np.savez(save_path, avg_rewards_INOUT=np.array(all_history_rewards_INOUT))