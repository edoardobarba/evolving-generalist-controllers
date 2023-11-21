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

"""
Default parameters
Cartpole:   masspole = 0.1
            length = 0.5

"""
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

def plot_heatmap(all_variations, scores, title):
    param1_values = all_variations[:, 0]
    param2_values = all_variations[:, 1]
    rounded_param1_values = [round(num, 2) for num in param1_values]
    rounded_param2_values = [round(num, 2) for num in param2_values]
    data = {'Parameter 1': rounded_param1_values, 'Parameter 2': rounded_param2_values, 'Reward': scores}
    df = pd.DataFrame(data)
    pivot_df = df.pivot(index='Parameter 2', columns='Parameter 1', values='Reward')
    #print(param1_values)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(pivot_df, vmin=-1000, vmax=0)
    plt.title(title)
    plt.xlabel('Parameter 1')
    plt.ylabel('Parameter 2')
    plt.show()


def plot_boxplots(history_reward_default, history_reward_in_distr, history_reward_out_distr, history_reward_in_out_distr):
    data = [history_reward_default, history_reward_in_distr, history_reward_out_distr, history_reward_in_out_distr]
    labels = ['Default', 'IN', 'OUT', 'IN+OUT']

    fig, ax = plt.subplots(figsize=(10, 7))

    # Creating boxplot with labels
    bp = ax.boxplot(data, labels=labels)

    # Adding scatter points
    for i, points in enumerate(data, 1):
        x = [i] * len(points)
        ax.scatter(x, points, alpha=0.7)#, label=f'Points {labels[i-1]}')

    # Set plot labels
    ax.set_title('Incremental')
    #ax.set_xlabel('Groups')
    ax.set_ylabel('Reward')

    # Show legend
    ax.legend()

    # Show plot
    plt.show()

# cartpole_default = [0.1, 0.5] #mass and pole-lenght
# cartpole_train_ranges = [[0.05, 0.5], [0.25, 0.75]]


def do_episode(env, nn, weights, topology, max_steps, max_fitness):

    reward = 0
    obs, info = env.reset(seed=SEED)
    done = False
    step = 0

    while not done:
        action = nn.feedforward(weights, topology, obs)

        obs, r, terminated, truncated, info = env.step(action)

        step += 1
        reward += r

        if step > max_steps or reward<max_fitness:
            break

        done = terminated or truncated

    return -reward


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

def get_default_param(game):
    if game == "CartPoleEnv":
        return [CARTPOLE_DEFAULT__MASS, CARTPOLE_DEFAULT_LENGTH]
    if game == "BipedalWalker":
        return [8, 34]
    # if game == "Walker2dEnv": 
    #     return 
    
def default_morph_test(game, nn, weights, topology, max_steps, max_fitness):
    print("Default morphology testing...")
    print("Parameter 1 range: ", CARTPOLE_DEFAULT__MASS)
    print("Parameter 2 range: ", CARTPOLE_DEFAULT_LENGTH)
    default_morph_param = get_default_param(game)

    if game == "AntEnv":
        xml_file = '{}/Ant_{:.2f}_hip_{:.2f}_ankle.xml'.format(xml_path, default_morph_param[0],
                                                                default_morph_param[1])
        env = game(xml_file, render_mode=None, healthy_reward=0)
    elif game == "Walker2dEnv":
        xml_file = '{}/Walker_{:.3f}_thigh_{:.3f}_leg.xml'.format(xml_path, default_morph_param[0],
                                                                    default_morph_param[1])
        env = game(xml_file, render_mode=None, healthy_reward=0)
    else:
        game = eval(game)
        env = game(default_morph_param)

    history_reward = []
    for run in range(1): 
        reward = do_episode(env, nn, weights, topology, max_steps, max_fitness)
        history_reward.append(reward)
    
    env.close()

    return history_reward


def in_distr_test(game, nn, weights, topology, max_steps, max_fitness): 
    print("IN distribution testing...")
    if game == "CartPoleEnv":
        parameter1_range = [CARTPOLE_IN_LOWER_MASSPOLE, CARTPOLE_IN_UPPER_MASSPOLE]
        parameter2_range = [CARTPOLE_IN_LOWER_LENGTH, CARTPOLE_IN_UPPER_LENGTH]
    game = eval(game)
        # train_M = generate_morphologies(parameter1_range, parameter2_range, step_sizes=)
        
    # elif game == "BipedalWalker":
    #     env = game(default_morph_param)
    print("Parameter 1 range: ", parameter1_range)
    print("Parameter 2 range: ", parameter2_range)

    all_variations = generate_morphologies(parameter1_range, parameter2_range, [0.05, 0.05])

    history_reward = []    
    param1_values_in = []
    param2_values_in = []
    for variation in all_variations:
        # param1 = np.random.uniform(parameter1_range[0], parameter1_range[1])
        # param2 = np.random.uniform(parameter2_range[0], parameter2_range[1])
        param1 = variation[0]
        param2 = variation[1]
        param1_values_in.append(param1)
        param2_values_in.append(param2)
        env = game([param1, param2])
        reward = do_episode(env, nn, weights, topology, max_steps, max_fitness)
        history_reward.append(reward)
        env.close()

    # Plot heatmaps

    #plot_heatmap(param1_values_in, param2_values_in, history_reward, 'In-Distribution Morphology')


    return history_reward, all_variations


def out_distr_test(game, nn, weights, topology, max_steps, max_fitness): 
    print("OUT distribution testing...")
    if game == "CartPoleEnv":
        parameter1_left_range = [CARTPOLE_OUT_LOWER_MASSPOLE, CARTPOLE_IN_LOWER_MASSPOLE]
        parameter1_rigth_range = [CARTPOLE_IN_UPPER_MASSPOLE, CARTPOLE_OUT_UPPER_MASSPOLE]
        parameter2_left_range = [CARTPOLE_OUT_LOWER_LENGTH, CARTPOLE_IN_LOWER_LENGTH]
        parameter2_rigth_range = [CARTPOLE_IN_UPPER_LENGTH, CARTPOLE_OUT_UPPER_LENGTH]

    print("Parameter 1 range: ", parameter1_left_range ," & ", parameter1_rigth_range)
    print("Parameter 2 range: ", parameter2_left_range ," & ", parameter2_rigth_range)
    game = eval(game)
        # train_M = generate_morphologies(parameter1_range, parameter2_range, step_sizes=)
        
    # elif game == "BipedalWalker":
    #     env = game(default_morph_param)


    all_variations_left = generate_morphologies(parameter1_left_range, parameter2_left_range, [0.05, 0.05])

    all_variations_rigth = generate_morphologies(parameter1_rigth_range, parameter2_rigth_range, [0.05, 0.05])
    all_variations = np.concatenate((all_variations_left, all_variations_rigth), axis=0)

    history_reward = []    
    param1_values_in = []
    param2_values_in = []
    for variation in all_variations:
        # param1 = np.random.uniform(parameter1_range[0], parameter1_range[1])
        # param2 = np.random.uniform(parameter2_range[0], parameter2_range[1])
        param1 = variation[0]
        param2 = variation[1]
        param1_values_in.append(param1)
        param2_values_in.append(param2)
        # param1_left = np.random.uniform(parameter1_left_range[0], parameter1_left_range[1])
        # param1_rigth = np.random.uniform(parameter1_rigth_range[0], parameter1_rigth_range[1])
        # param2_left = np.random.uniform(parameter2_left_range[0], parameter2_left_range[1])
        # param2_rigth = np.random.uniform(parameter2_rigth_range[0], parameter2_rigth_range[1])
        # param1 = random.choice([param1_left, param1_rigth])
        # param2 = random.choice([param2_left, param2_rigth])
        env = game([param1, param2])
        reward = do_episode(env, nn, weights, topology, max_steps, max_fitness)
        history_reward.append(reward)
        env.close()

    #plot_heatmap(param1_values_in, param2_values_in, history_reward, 'In-Distribution Morphology')
    return history_reward, all_variations


def in_out_distr_test(game, nn, weights, topology, max_steps, max_fitness): 
    print("IN OUT distribution testing...")
    if game == "CartPoleEnv":
        parameter1_range = [CARTPOLE_OUT_LOWER_MASSPOLE, CARTPOLE_OUT_UPPER_MASSPOLE]
        parameter2_range = [CARTPOLE_OUT_LOWER_LENGTH, CARTPOLE_OUT_UPPER_LENGTH]
    game = eval(game)
    print("Parameter 1 range: ", parameter1_range)
    print("Parameter 2 range: ", parameter2_range)
        # train_M = generate_morphologies(parameter1_range, parameter2_range, step_sizes=)
        
    # elif game == "BipedalWalker":
    #     env = game(default_morph_param)

    all_variations = generate_morphologies(parameter1_range, parameter2_range, [0.05, 0.05])

    history_reward = []    
    param1_values_in = []
    param2_values_in = []
    for variation in all_variations:
        param1 = variation[0]
        param2 = variation[1]
        # param1 = np.random.uniform(parameter1_range[0], parameter1_range[1])
        # param2 = np.random.uniform(parameter2_range[0], parameter2_range[1])
        param1_values_in.append(param1)
        param2_values_in.append(param2)
        env = game([param1, param2], render_mode=None)
        reward = do_episode(env, nn, weights, topology, max_steps, max_fitness)
        history_reward.append(reward)
        env.close()

    #plot_heatmap(param1_values_in, param2_values_in, history_reward, 'IN+OUT-Distribution')
    return history_reward, all_variations

def list_folders(path):
    # Get a list of all items (files and folders) in the specified path
    items = os.listdir(path)

    # Filter out only the directories
    folders = [item for item in items if os.path.isdir(os.path.join(path, item))]

    return folders


if __name__ == '__main__':
    runs_folder_path = str(sys.argv[2])

    with open(str(sys.argv[1])) as json_file:
        print('Running testing for', sys.argv[1])
        config = json.load(json_file)

    game = config['game']
    #parameter1_range, parameter2_range = config['parameter1'], config['parameter2']
    step_sizes = config['step_sizes']
    topology = config["NN-struc"]
    max_steps = config["nStep"]
    max_fitness = config["maxFitness"]
    xml_path = config['xml']

    # Get a list of all folders in the specified path
    runs_folders = list_folders(runs_folder_path)

    all_history_rewards_IN = []
    all_history_rewards_OUT = []
    all_history_rewards_INOUT = []

    for run_number in runs_folders:
        run_path = os.path.join(runs_folder_path, run_number)
        generalist_folder_path = os.path.join(run_path, "generalist")
        path_generalist_ANN_weights = os.path.join(generalist_folder_path, os.listdir(generalist_folder_path)[0])
        file_generalist_ANN_weights = open(path_generalist_ANN_weights)
        nn, weights = get_NN(file_generalist_ANN_weights)
        file_generalist_ANN_weights.close()

        history_reward_default = default_morph_test(game=game, nn = nn, weights=weights, topology=topology, max_steps=max_steps, max_fitness=max_fitness)
        history_reward_IN, all_var_IN = in_distr_test(game=game, nn = nn, weights=weights, topology=topology, max_steps=max_steps, max_fitness=max_fitness)      
        history_reward_OUT, all_var_OUT = out_distr_test(game=game, nn = nn, weights=weights, topology=topology, max_steps=max_steps, max_fitness=max_fitness)
        history_reward_INOUT, all_var_INOUT = in_out_distr_test(game=game, nn = nn, weights=weights, topology=topology, max_steps=max_steps, max_fitness=max_fitness)

        all_history_rewards_IN.append(np.array(history_reward_IN))
        all_history_rewards_OUT.append(np.array(history_reward_OUT))
        all_history_rewards_INOUT.append(np.array(history_reward_INOUT))

    save_path = os.path.join(runs_folder_path, "all_history_rewards_data.npz")
    np.savez(save_path, all_history_rewards_IN=np.array(all_history_rewards_IN), avg_rewards_OUT=np.array(all_history_rewards_OUT), avg_rewards_INOUT=np.array(all_history_rewards_INOUT))


        # print("Mean reward D: ", np.mean(np.array(history_reward_default)))
        # print("Mean reward IN: ", np.mean(np.array(history_reward_in_distr)))
        # print("Mean reward OUT: ", np.mean(np.array(history_reward_out_distr)))
        # print("Mean reward IN_OUT: ", np.mean(np.array(history_reward_in_out_distr)))

        # print("Median reward D: ", np.median(np.array(history_reward_default)))
        # print("Median reward IN: ", np.median(np.array(history_reward_in_distr)))
        # print("Median reward OUT: ", np.median(np.array(history_reward_out_distr)))
        # print("Median reward IN_OUT: ", np.median(np.array(history_reward_in_out_distr)))

        # plot_heatmap(all_var_IN, history_reward_in_distr, 'IN-Distribution')
        # plot_heatmap(all_var_OUT, history_reward_out_distr, 'OUT-Distribution')
        # plot_heatmap(all_var_INOUT, history_reward_in_out_distr, 'IN&OUT-Distribution')
        # plot_boxplots(history_reward_default, history_reward_in_distr, history_reward_out_distr, history_reward_in_out_distr)


    
    