import pandas as pd
import numpy as np
from nn import NeuralNetwork
#from ant_v4_modified import AntEnv
#from walker2d_v4_modified import Walker2dEnv
from bipedal_walker_modified import BipedalWalker
from cartpole_modified import CartPoleEnv
import os
from scipy.stats import cauchy
import gym
import matplotlib.pyplot as plt


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

# # ACROBOT_IN_LOWER_MASS1 = 0.3
# # ACROBOT_IN_UPPER_MASS1 = 0.7
# ACROBOT_IN_LOWER_MASS1 = 0.75
# ACROBOT_IN_UPPER_MASS1 = 1.25
# ACROBOT_OUT_LOWER_MASS1 = 0.5
# ACROBOT_OUT_UPPER_MASS1 = 1.5

# # ACROBOT_IN_LOWER_MASS2 = 0.3
# # ACROBOT_IN_UPPER_MASS2 = 0.7
# ACROBOT_IN_LOWER_MASS2 = 0.75
# ACROBOT_IN_UPPER_MASS2 = 1.25
# ACROBOT_OUT_LOWER_MASS2 = 0.5
# ACROBOT_OUT_UPPER_MASS2 = 1.5

# ACROBOT_DEFAULT__MASS1 = 1
# ACROBOT_DEFAULT_MASS2 = 1

# # def get_validation_set(game):
# #     if game == "CartPoleEnv":
# #         parameter1_range = [CARTPOLE_IN_LOWER_MASSPOLE, CARTPOLE_IN_UPPER_MASSPOLE]
# #         parameter2_range = [CARTPOLE_IN_LOWER_LENGTH, CARTPOLE_IN_UPPER_LENGTH]
    
# #         return generate_morphologies(parameter1_range, parameter2_range, [0.1, 0.1])

# #     if game == "AcrobotEnv":
# # #         parameter1_range = [ACROBOT_IN_LOWER_MASS1, ACROBOT_IN_UPPER_MASS1]
# # #         parameter2_range = [ACROBOT_IN_LOWER_MASS2, ACROBOT_IN_UPPER_MASS2]
    
# #         return generate_morphologies(parameter1_range, parameter2_range, [0.1, 0.1])  


def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtracting np.max(x) for numerical stability
    return e_x / e_x.sum()

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

    return std_dev_par1, std_dev_par2


# def clip_values(morphologies, parameter1_range, parameter2_range):
#     morphologies[:, 0] = np.clip(morphologies[:, 0], parameter1_range[0], parameter1_range[1])
#     morphologies[:, 1] = np.clip(morphologies[:, 1], parameter2_range[0], parameter2_range[1])
#     return morphologies

def generate_samples(parameter1_range, parameter2_range, num_samples, distr, samples_per_cycle):
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
    
    if distr=="gauss_dec":
        mean_p1 = parameter1_range[1]
        mean_p2 = parameter2_range[1]

        std_dev_p1, std_dev_p2 = get_std(parameter1_range, parameter2_range, "gaussian1")
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

    elif distr=="beta01" or distr =="beta02": 
        if distr == "beta01":
            a = b = 0.1
        if distr == "beta02":
            a = b = 0.2

        lower_bound_p1 = parameter1_range[0]
        upper_bound_p1 = parameter1_range[1]

        lower_bound_p2 = parameter2_range[0]
        upper_bound_p2 = parameter2_range[1]

        
        for _ in range(num_samples):
            sample1 = np.random.beta(a=a, b=b)
            p1 = lower_bound_p1 + (upper_bound_p1 - lower_bound_p1) * sample1

            while(p1 < parameter1_range[0] or p1 > parameter1_range[1]):
                sample1 = np.random.beta(a=a, b=b)
                p1 = lower_bound_p1 + (upper_bound_p1 - lower_bound_p1) * sample1

            sample2 = np.random.beta(a=a, b=b)
            p2 = lower_bound_p2 + (upper_bound_p2 - lower_bound_p2) * sample2

            while(p2 < parameter2_range[0] or p2 > parameter2_range[1]):
                sample2 = np.random.beta(a=a, b=b)
                p2 = lower_bound_p2 + (upper_bound_p2 - lower_bound_p2) * sample2

            morphologies.append([p1, p2])
        return np.array(morphologies)     

    
    elif distr == "uniform": 
        for _ in range(num_samples): 
            p1 = np.random.uniform(parameter1_range[0], parameter1_range[1])
            p2 = np.random.uniform(parameter2_range[0], parameter2_range[1])
            morphologies.append([p1, p2])
                
        return np.array(morphologies)

    elif distr == "betawalk01" or distr == "betawalk02":
        
        all_samples = []
        first_cycle = True
        while len(all_samples) < num_samples:

            if distr == "betawalk01":
                cycle_samples = generate_samples(parameter1_range, parameter2_range, samples_per_cycle, "beta01", None)
            elif distr == "betawalk02":
                cycle_samples = generate_samples(parameter1_range, parameter2_range, samples_per_cycle, "beta02", None)
            
            if first_cycle:
                random_element_idx = np.random.randint(0, len(cycle_samples))
                current = cycle_samples[random_element_idx]
                all_samples.append(current)
                cycle_samples = np.delete(cycle_samples, random_element_idx, axis=0)
                first_cycle = False

            while cycle_samples.size > 0:
                distances = np.linalg.norm(cycle_samples - current, axis=1)
                min_distance_index = np.argmin(distances)
                closest_element = cycle_samples[min_distance_index]
                all_samples.append(closest_element)
                cycle_samples = np.delete(cycle_samples, min_distance_index, axis=0)

                current = closest_element

                if len(all_samples) == num_samples: 
                    break
        
        return np.array(all_samples)
    
    elif distr == "default":
        return np.array([[8,34]])
    

def get_set(config, test_set):
    game = config["game"]
    step_sizes = config["testing_step_sizes"]

    if test_set == "VALIDATION":
        step_sizes = config["validation_step_sizes"]
        parameter1_range = config["IN_parameter1"]
        parameter2_range = config["IN_parameter2"]
        return generate_morphologies(parameter1_range, parameter2_range, step_sizes) 

    if test_set == "IN": 
        parameter1_range = config["IN_parameter1"]
        parameter2_range = config["IN_parameter2"]
        return generate_morphologies(parameter1_range, parameter2_range, step_sizes) 
    
    if test_set == "INOUT": 
        parameter1_range = config["OUT_parameter1"]
        parameter2_range = config["OUT_parameter2"]
        return generate_morphologies(parameter1_range, parameter2_range, step_sizes)
    
    if test_set == "OUT": 
        OUT_set = []
        INOUT_set  = get_set(config, "INOUT")
        IN_parameter1_range = config["IN_parameter1"]
        IN_parameter2_range = config["IN_parameter2"]


        for element in INOUT_set:
            if (element[0] < IN_parameter1_range[0] or element[0] > IN_parameter1_range[1]) or (element[1] < IN_parameter2_range[0] or element[1] > IN_parameter2_range[1]):
                OUT_set.append(element)

        return np.array(OUT_set)
    
    if test_set == "Validation": 
        parameter1_range = config["IN_parameter1"]
        parameter2_range = config["IN_parameter2"]
        step_sizes = config["validation_step_sizes"]
        return generate_morphologies(parameter1_range, parameter2_range, step_sizes) 


def generate_morphologies(parameter1_range, parameter2_range, step_sizes):
    parameter1_values = np.arange(parameter1_range[0], parameter1_range[1] + 0.001, step_sizes[0])
    parameter2_values = np.arange(parameter2_range[0], parameter2_range[1] + 0.001, step_sizes[1])

    morphologies = np.array(np.meshgrid(parameter1_values, parameter2_values)).T.reshape(-1, 2)

    return morphologies

def generate_border_morphologies(parameter1_range, parameter2_range, step_sizes):
    parameter1_values = np.arange(parameter1_range[0], parameter1_range[1] + 0.001, step_sizes[0])
    parameter2_values = np.arange(parameter2_range[0], parameter2_range[1] + 0.001, step_sizes[1])

    border_morphologies = []

    # Add the top and bottom borders
    for p1 in parameter1_values:
        border_morphologies.append([p1, parameter2_range[0]])
        border_morphologies.append([p1, parameter2_range[1]])

    # Add the left and right borders
    for p2 in parameter2_values:
        border_morphologies.append([parameter1_range[0], p2])
        border_morphologies.append([parameter1_range[1], p2])

    return np.array(border_morphologies)


def gym_render(game, agent, xml_path, parameters, topology, steps):
    s = 0
    total_reward = 0

    # if game == AntEnv:
    #     xml_file = '{}/Ant_{:.2f}_hip_{:.2f}_ankle.xml'.format(xml_path, parameters[0], parameters[1])
    #     env = game(xml_file, render_mode=None, healthy_reward=0)
    # elif game == Walker2dEnv:
    #     xml_file = '{}/Walker_{:.3f}_thigh_{:.3f}_leg.xml'.format(xml_path, parameters[0], parameters[1])
    #     env = game(xml_file, render_mode=None, healthy_reward=0)
    if game == "AcrobotEnv":
        env = gym.make('Acrobot-v1', render_mode = None).unwrapped
        env.LINK_MASS_1 = parameters[0]  #: [kg] mass of link 1
        env.LINK_MASS_2 = parameters[1]  #: [kg] mass of link 2
    else:
        env = game(parameters)

    obs, info = env.reset(seed=s)
    done = False

    x = agent.cpu()
    nn = NeuralNetwork(x.numpy())
    weights = nn.reshape_layers(topology)

    while not done:
        action = nn.feedforward(weights, topology, obs)
        #action = env.action_space.sample()

        if game == "AcrobotEnv":
            action = np.argmax(action)

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
