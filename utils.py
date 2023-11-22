import pandas as pd
import numpy as np
from nn import NeuralNetwork
from ant_v4_modified import AntEnv
from walker2d_v4_modified import Walker2dEnv
from bipedal_walker_modified import BipedalWalker
from cartpole_modified import CartPoleEnv
import os
from scipy.stats import cauchy


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

def get_validation_set():
    parameter1_range = [CARTPOLE_IN_LOWER_MASSPOLE, CARTPOLE_IN_UPPER_MASSPOLE]
    parameter2_range = [CARTPOLE_IN_LOWER_LENGTH, CARTPOLE_IN_UPPER_LENGTH]
    
    return generate_morphologies(parameter1_range, parameter2_range, [0.1, 0.1]) 

def get_mean(parameter1_range, parameter2_range):
    mean_par1 = (parameter1_range[1] + parameter1_range[0]) / 2
    mean_par2 = (parameter2_range[1] + parameter2_range[0]) / 2

    return mean_par1, mean_par2

def get_std(parameter1_range, parameter2_range, distr):
    if distr == "gaussian1" or distr == "cauchy1":
        
        mid_p1 = (parameter1_range[1] + parameter1_range[0]) / 2
        right_mid_p1 = (parameter1_range[1] + mid_p1) / 2
        std_dev_par1 = (right_mid_p1 - mid_p1)
        mid_p2 = (parameter2_range[1] + parameter2_range[0]) / 2
        right_mid_p2 = (parameter2_range[1] + mid_p2) / 2
        std_dev_par2 = (right_mid_p2 - mid_p2)
        print("std 1:", std_dev_par1)
        print("std 2:", std_dev_par2)

        # percentage_of_range = 0.4
        # std_dev_par1 = (parameter1_range[1]-parameter1_range[0]) * percentage_of_range / 2
        # std_dev_par2 = (parameter2_range[1]-parameter2_range[0]) * percentage_of_range / 2

    elif distr == "gaussian2" or distr == "cauchy2":
        
        mid_p1 = (parameter1_range[1] + parameter1_range[0]) / 2
        right_mid_p1 = (parameter1_range[1] + mid_p1) / 2
        right_right_mid_p1 = (parameter1_range[1] + right_mid_p1) / 2
        std_dev_par1 = (right_right_mid_p1 - mid_p1)

        mid_p2 = (parameter2_range[1] + parameter2_range[0]) / 2
        right_mid_p2 = (parameter2_range[1] + mid_p2) / 2
        right_right_mid_p2 = (parameter2_range[1] + right_mid_p2) / 2
        std_dev_par2 = (right_right_mid_p2 - mid_p2)
        print("std 1:", std_dev_par1)
        print("std 2:", std_dev_par2)

    print(std_dev_par1)
    print(std_dev_par2)
    return std_dev_par1, std_dev_par2


# def clip_values(morphologies, parameter1_range, parameter2_range):
#     morphologies[:, 0] = np.clip(morphologies[:, 0], parameter1_range[0], parameter1_range[1])
#     morphologies[:, 1] = np.clip(morphologies[:, 1], parameter2_range[0], parameter2_range[1])
#     return morphologies

def generate_samples(parameter1_range, parameter2_range, num_samples, distr = "Gaussian1"):
    morphologies = []
    if distr=="gaussian1" or distr == "gaussian2":
        mean_p1, mean_p2 = get_mean(parameter1_range, parameter2_range)
        std_dev_p1, std_dev_p2 = get_std(parameter1_range, parameter2_range, distr)
        for _ in range(num_samples):
            p1 = np.random.normal(loc=mean_p1, scale=std_dev_p1)
            while(p1 < parameter1_range[0] or p1 > parameter1_range[1]):
                p1 = np.random.normal(loc=mean_p1, scale=std_dev_p1)

            p2 = np.random.normal(loc=mean_p2, scale=std_dev_p2)
            while(p2 < parameter2_range[0] or p2 > parameter2_range[1]):
                p2 = np.random.normal(loc=mean_p2, scale=std_dev_p2)

            morphologies.append([p1, p2])
        return np.array(morphologies)

    elif distr=="cauchy1" or distr=="cauchy2": 
        mean_p1, mean_p2 = get_mean(parameter1_range, parameter2_range)
        std_dev_p1, std_dev_p2 = get_std(parameter1_range, parameter2_range, distr)
        cauchy_dist_p1 = cauchy(loc=mean_p1, scale=std_dev_p1)
        cauchy_dist_p2 = cauchy(loc=mean_p2, scale=std_dev_p2)
        for _ in range(num_samples):
            p1 = cauchy_dist_p1.rvs(1)
            while(p1 < parameter1_range[0] or p1 > parameter1_range[1]):
                p1 = cauchy_dist_p1.rvs(1)

            p2 = cauchy_dist_p2.rvs(1)
            while(p2 < parameter2_range[0] or p2 > parameter2_range[1]):
                p2 = cauchy_dist_p2.rvs(1)


            morphologies.append([p1[0], p2[0]])
            
        return np.array(morphologies)
    
    elif distr == "uniform": 
        for _ in range(num_samples): 
            p1 = np.random.uniform(parameter1_range[0], parameter1_range[1])
            p2 = np.random.uniform(parameter2_range[0], parameter2_range[1])
            morphologies.append([p1, p2])
                
        return np.array(morphologies)
    

def get_set(test_set, step_sizes):
    if test_set == "IN": 
        parameter1_range = [CARTPOLE_IN_LOWER_MASSPOLE, CARTPOLE_IN_UPPER_MASSPOLE]
        parameter2_range = [CARTPOLE_IN_LOWER_LENGTH, CARTPOLE_IN_UPPER_LENGTH]
        return generate_morphologies(parameter1_range, parameter2_range, step_sizes) 
    
    if test_set == "INOUT": 
        parameter1_range = [CARTPOLE_OUT_LOWER_MASSPOLE, CARTPOLE_OUT_UPPER_MASSPOLE]
        parameter2_range = [CARTPOLE_OUT_LOWER_LENGTH, CARTPOLE_OUT_UPPER_LENGTH]

        return generate_morphologies(parameter1_range, parameter2_range, step_sizes)
    
    if test_set == "OUT": 
        OUT_set = []
        # IN_set = get_set("IN", step_sizes)
        INOUT_set  = get_set("INOUT", step_sizes)
        for element in INOUT_set:
            if (element[0] < CARTPOLE_IN_LOWER_MASSPOLE or element[0] > CARTPOLE_IN_UPPER_MASSPOLE) and (element[1] < CARTPOLE_IN_LOWER_LENGTH or element[1] > CARTPOLE_IN_UPPER_LENGTH):
                OUT_set.append(element)

        return np.array(OUT_set)


def generate_morphologies(parameter1_range, parameter2_range, step_sizes):
    parameter1_values = np.arange(parameter1_range[0], parameter1_range[1] + step_sizes[0], step_sizes[0])
    parameter2_values = np.arange(parameter2_range[0], parameter2_range[1] + step_sizes[1], step_sizes[1])

    morphologies = np.array(np.meshgrid(parameter1_values, parameter2_values)).T.reshape(-1, 2)

    return morphologies


def gym_render(game, agent, xml_path, parameters, topology, steps):
    s = 0
    total_reward = 0

    if game == AntEnv:
        xml_file = '{}/Ant_{:.2f}_hip_{:.2f}_ankle.xml'.format(xml_path, parameters[0], parameters[1])
        env = game(xml_file, render_mode=None, healthy_reward=0)
    elif game == Walker2dEnv:
        xml_file = '{}/Walker_{:.3f}_thigh_{:.3f}_leg.xml'.format(xml_path, parameters[0], parameters[1])
        env = game(xml_file, render_mode=None, healthy_reward=0)
    else:
        env = game(parameters)

    obs, info = env.reset(seed=s)
    done = False

    x = agent.cpu()
    nn = NeuralNetwork(x.numpy())
    weights = nn.reshape_layers(topology)

    while not done:
        action = nn.feedforward(weights, topology, obs)

        obs, reward, terminated, truncated, info = env.step(action)

        s += 1
        total_reward += reward

        if s > steps:
            break

        done = terminated or truncated

    env.close()

    return -total_reward


def save_dataframe(dataframe, directory, filename):
    dataframe.to_csv(os.path.join(directory, filename), index=False)


def create_directories(path, subdirectories):
    for subdir in subdirectories:
        os.makedirs(os.path.join(path, subdir), exist_ok=True)


def save_dataframes(evals, best, generalist, generalist_evals, info, path):
    subdirectories = ['xbest', 'generalist', 'evals', 'generalist_evals']

    create_directories(path, subdirectories)

    file_names = [
        '{}_evals.csv'.format(info),
        '{}_xbest.csv'.format(info),
        '{}_generalist.csv'.format(info),
        '{}_generalist_evals.csv'.format(info)
    ]

    dataframes = [evals, pd.DataFrame(best), pd.DataFrame(generalist), generalist_evals]

    for dataframe, subdir, filename in zip(dataframes, subdirectories, file_names):
        save_dataframe(dataframe, os.path.join(path, subdir), filename)
