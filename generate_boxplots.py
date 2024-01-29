from cartpole_modified import CartPoleEnv
from bipedal_walker_modified import BipedalWalker
from walker2d_v4_modified import Walker2dEnv
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





# ANT 
# train_incremental_path = "/home/edoardo.barba/Results_Ant/incremental/20240119201533"
# train_random_path = "/home/edoardo.barba/Results_Ant/random/20240119201533"

# training_schedules = ["random", "incremental"]
# all_train_folders = [train_random_path, train_incremental_path]





# BIPED 

train_cauchy1_path = "/home/edoardo.barba/Results_Biped/cauchy1/20231219091844"
train_cauchy2_path = "/home/edoardo.barba/Results_Biped/cauchy2/20231219091844"
train_gaussian1_path = "/home/edoardo.barba/Results_Biped/gaussian1/20231223120018"
train_gaussian2_path = "/home/edoardo.barba/Results_Biped/gaussian2/20231223120018"
train_incremental_path = "/home/edoardo.barba/Results_Biped/incremental/20231219025742"
train_uniform_path = "/home/edoardo.barba/Results_Biped/uniform/20231219091844"
train_RL_path = "/home/edoardo.barba/Results_Biped_old/RL/20231215114621" 
train_beta01 = "/home/edoardo.barba/Results_Biped/beta01/20231219092337"
train_beta02 = "/home/edoardo.barba/Results_Biped/beta02/20231219092337"
train_betawalk01 = "/home/edoardo.barba/Results_Biped/betawalk01/20231221112601" 
train_betawalk02 = "/home/edoardo.barba/Results_Biped/betawalk02/20231221112601" 
# train_gauss_dec = "/home/edoardo.barba/Results_Biped/gauss_dec/20231223120018"
train_default_path = "/home/edoardo.barba/Results_Biped/default/20240103141803"
train_borderincr_path = "/home/edoardo.barba/Results_Biped/border_incr/20240103134750"
train_random_path = "/home/edoardo.barba/Results_Biped/random/20240123041607"
train_incremental_path = "/home/edoardo.barba/Results_Biped/incremental/20240123041607"

# training_schedules = ["border_incr", "random", "incremental", "gaussian1", "gaussian2", "cauchy1", "cauchy2","uniform", "beta01", "beta02", "betawalk01", "betawalk02"]
# all_train_folders = [train_borderincr_path, train_random_path, train_incremental_path, train_gaussian1_path, train_gaussian2_path, train_cauchy1_path, train_cauchy2_path, train_uniform_path, train_beta01, train_beta02, train_betawalk01, train_betawalk02]
training_schedules = ["random", "incremental"]
all_train_folders = [train_random_path, train_incremental_path]



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



save_path = "/home/edoardo.barba/"

def get_significant_combinations(data, save_path, title, significance_levels=[0.05]):
    # Initialise a list of combinations of groups that are significantly different
    significant_combinations = []
    # Check from the outside pairs of boxes inwards
    ls = list(range(0, len(data)))
    combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
    for combination in combinations:
        data1 = data[combination[0]]
        data2 = data[combination[1]]
        # Significance
        U, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        significant_combinations.append([combination, p])

    
    print(significant_combinations)
    return significant_combinations
    # path_combinations = os.path.join(save_path, "significant_combinations" + title + ".png")
    # print(path_combinations)
    # plt.savefig(path_combinations, dpi=300)

                
# def plot(game, incremental_history_rewards, gaussian1_history_rewards, gaussian2_history_rewards, 
#          cauchy1_history_rewards, cauchy2_history_rewards, uniform_history_rewards, RL_history_rewards, 
#          beta1_history_rewards, beta2_history_rewards, betawalk1_history_rewards, betawalk2_history_rewards,
#          test, save_path):
    
def plot(game, all_history_rewards, test, save_path):
    print("qui")
    print(np.shape(np.array(all_history_rewards)))
    avg_scores = [np.median(history_rewards, axis=1) for history_rewards in all_history_rewards]

    print(np.shape(np.array(avg_scores)))

    print(np.mean(avg_scores[0]))
    print(np.mean(avg_scores[1]))

    data=avg_scores



    print(np.shape(data))


    labels = training_schedules

    fig, ax = plt.subplots(figsize=(10, 7))
    bp = ax.boxplot(data, labels=labels)
    
    for i, points in enumerate(data, 1):
        x = [i] * len(points)
        ax.scatter(x, points, alpha=0.7)



    # Get the y-axis limits
    bottom, top = ax.get_ylim()
    y_range = top - bottom

    # print("significant_combinations: ")
    # for i, significant_combination in enumerate(significant_combinations):
    #     print("[", training_schedules[significant_combination[0][0]-1], training_schedules[significant_combination[0][1]-1] , "]")


    if test == "IN":
        title = "TRAIN"
    elif test == "OUT":
        title = "TEST"
    elif test =="INOUT":
        title = "TRAIN+TEST"


    ax.set_title(title + " set")
    ax.set_ylabel('Reward')
    if game == "CartPoleEnv":
        ax.set_ylim(-1001, 0)
        ax.axhline(y=-500, color='r', linestyle='--', label='Threshold')
    # elif game == "AcrobotEnv":
    #     if test == "IN":
    #         ax.set_ylim(50, 100)
    #     else:
    #         ax.set_ylim(50, 90)
    #     ax.axhline(y=100, color='r', linestyle='--', label='Threshold')

    elif game == "BipedalWalker":
        if test == "IN":
            ax.set_ylim(-280,-125)
        # elif test == "OUT":
        #     ax.set_ylim(-100,20)
        # else:
        #     ax.set_ylim(-150, 0)
        # ax.axhline(y=100, color='r', linestyle='--', label='Threshold')
            
    # elif game == Walker2dEnv:
    #     if test == "IN":
    #         ax.set_ylim(-250,-100)
    #     elif test == "OUT":
    #         ax.set_ylim(-100,20)
    #     else:
    #         ax.set_ylim(-150, 0)
    #     #ax.axhline(y=100, color='r', linestyle='--', label='Threshold')


    plt.tight_layout()
    plt.grid(axis='y')
    

    save_box_path = os.path.join(save_path, "Boxplot_" + title + ".png")
    print(save_box_path)
    plt.savefig(save_box_path, dpi=300)

    plt.close()

    significant_combinations = get_significant_combinations(data, save_path, test)
    # Create an empty 2D array for p-values
    p_values = np.ones((len(training_schedules), len(training_schedules)))

    # Fill the p-values array with values from the data
    for entry in significant_combinations:
        i, j = entry[0]
        p_value = entry[1]
        p_values[i][j] = p_value

    # Create a mask for p-values < 0.05
    mask = p_values < 0.05

    # Plot the heatmap
    plt.imshow(mask, cmap='Blues', origin='lower', interpolation='none', vmin=0, vmax=1)

    # Set axis labels and title
    plt.xticks(np.arange(len(training_schedules)) + 0.5, training_schedules, rotation=45, ha='right')
    plt.yticks(np.arange(len(training_schedules)) + 0.5, training_schedules, va='center')

    # Add grid with a 0.5 offset
    plt.grid(True, which='major', linestyle='-', linewidth=0.5, color='black', axis='both', alpha=0.5)
    # Add p-values to each cell
    for i in range(len(training_schedules)):
        for j in range(len(training_schedules)):
            if p_values[i, j] != 1:  # Only add p-values not equal to 1
                plt.text(j, i, round(p_values[i, j], 2), ha='center', va='center', color='black')
                
    plt.title('Heatmap of p-values (< 0.05)')

    plt.savefig(os.path.join(save_path, "significant_heatmap_" + title + ".png"), dpi=300)
    plt.close()




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


def get_rankings(all_history_rewards, test):
    
    # incremental_avg_scores = np.mean(incremental_history_rewards, axis=1)
    # gaussian1_avg_scores = np.mean(gaussian1_history_rewards, axis=1)
    # gaussian2_avg_scores = np.mean(gaussian2_history_rewards, axis=1)
    # cauchy1_avg_scores = np.mean(cauchy1_history_rewards, axis=1)
    # cauchy2_avg_scores = np.mean(cauchy2_history_rewards, axis=1)
    # uniform_avg_scores = np.mean(uniform_history_rewards, axis=1)
    # RL_avg_scores = np.mean(RL_history_rewards, axis=1)
    # beta1_avg_scores = np.mean(beta1_history_rewards, axis=1)
    # beta2_avg_scores = np.mean(beta2_history_rewards, axis=1)
    # betawalk1_avg_scores = np.mean(betawalk1_history_rewards, axis=1)
    # betawalk2_avg_scores = np.mean(betawalk2_history_rewards, axis=1)

    all_avg_scores = [np.mean(history_rewards, axis=1) for history_rewards in all_history_rewards]

    medians = [np.median(avg_scores) for avg_scores in all_avg_scores]
    # medians.append(np.median(incremental_avg_scores))
    # medians.append(np.median(gaussian1_avg_scores))
    # medians.append(np.median(gaussian2_avg_scores))
    # medians.append(np.median(cauchy1_avg_scores))
    # medians.append(np.median(cauchy2_avg_scores))
    # medians.append(np.median(uniform_avg_scores))
    # medians.append(np.median(RL_avg_scores))
    # medians.append(np.median(beta1_avg_scores))
    # medians.append(np.median(beta2_avg_scores))
    # medians.append(np.median(betawalk1_avg_scores))
    # medians.append(np.median(betawalk2_avg_scores))

    ranks = rank_array(medians)

    return ranks



with open(str(sys.argv[1])) as json_file:
    config = json.load(json_file)
game = config['game']



npz_paths = [train_path + "/all_history_rewards_data.npz" for train_path in all_train_folders]
# for train_path in all_train_folders:
#     data_path = train_path + "/all_history_rewards_data.npz"
#     npz_paths.append(data_path)



# npz_cauchy1_path = train_cauchy1_path + "/all_history_rewards_data.npz"
# npz_cauchy2_path = train_cauchy2_path + "/all_history_rewards_data.npz"
# npz_gaussian1_path = train_gaussian1_path + "/all_history_rewards_data.npz"
# npz_gaussian2_path = train_gaussian2_path + "/all_history_rewards_data.npz"
# npz_incremental_path = train_incremental_path + "/all_history_rewards_data.npz"
# npz_uniform_path = train_uniform_path + "/all_history_rewards_data.npz"
# npz_RL_path = train_RL_path + "/all_history_rewards_data.npz"
# npz_beta1_path = train_beta1_path + "/all_history_rewards_data.npz"
# npz_beta2_path = train_beta2_path + "/all_history_rewards_data.npz"
# npz_betawalk1_path = train_betawalk1_path + "/all_history_rewards_data.npz"
# npz_betawalk2_path = train_betawalk2_path + "/all_history_rewards_data.npz"


all_data_load = [np.load(npz_path) for npz_path in npz_paths]

# incremental_data = np.load(npz_incremental_path)
# gaussian1_data = np.load(npz_gaussian1_path)
# gaussian2_data = np.load(npz_gaussian2_path)
# cauchy1_data = np.load(npz_cauchy1_path)
# cauchy2_data = np.load(npz_cauchy2_path)
# uniform_data = np.load(npz_uniform_path)
# RL_data = np.load(npz_RL_path)
# beta1_data = np.load(npz_beta1_path)
# beta2_data = np.load(npz_beta2_path)
# betawalk1_data = np.load(npz_betawalk1_path)
# betawalk2_data = np.load(npz_betawalk2_path)

# all_history_rewards_IN = [data['all_history_rewards_IN'] for data in all_data_load]
# all_history_rewards_OUT = [data['avg_rewards_OUT'] for data in all_data_load]
all_history_rewards_INOUT = [data['avg_rewards_INOUT'] for data in all_data_load]

print(game)    
if game == "CartPoleEnv":
    save_path = save_path + "Results_Cart/"

elif game == "AcrobotEnv":
    save_path = save_path + "Results_Acrobot/"

elif game == "BipedalWalker":
    save_path = save_path + "Results_Biped/"

elif game == "Walker2dEnv":
    save_path = save_path + "Results_Walker/"

# print(np.mean(np.mean(all_history_rewards_IN[0], axis=1)))
# print(np.shape(np.mean(all_history_rewards_IN[0], axis=1)))
# print(np.shape(all_history_rewards_IN[0]))


IN_variations = utils.get_set(config, test_set="IN")
OUT_variations = utils.get_set(config, test_set="OUT")
INOUT_variations = utils.get_set(config, test_set="INOUT")

# print("IN")
# print(IN_variations)
# print("INOUT")
#print(len(INOUT_variations))

all_history_rewards_IN = all_history_rewards_INOUT
all_history_rewards_OUT = all_history_rewards_INOUT

#print(np.shape(all_history_rewards_INOUT))

indices_to_delete_IN = []
indices_to_delete_OUT = []
for i in range(len(INOUT_variations)): 
    IN = False
    for j in range(len(IN_variations)):
        if IN_variations[j][0] == INOUT_variations[i][0] and IN_variations[j][1] == INOUT_variations[i][1]:
            IN = True
    if not IN: 
        indices_to_delete_IN.append(i)
    else: 
        indices_to_delete_OUT.append(i)

#print(len(all_history_rewards_IN))
#print(indices_to_delete_IN)
all_history_rewards_IN = np.delete(all_history_rewards_IN, indices_to_delete_IN, axis=2)
all_history_rewards_OUT = np.delete(all_history_rewards_OUT, indices_to_delete_OUT, axis=2)
            

print(np.shape(all_history_rewards_IN))
print(np.shape(all_history_rewards_OUT))

# Plotting
plot(game, all_history_rewards_IN, test="IN", save_path=save_path)

plot(game, all_history_rewards_OUT, test="OUT", save_path=save_path)

plot(game, all_history_rewards_INOUT, test="INOUT", save_path=save_path)


ranks_IN = get_rankings(all_history_rewards_IN, "IN")
ranks_OUT = get_rankings(all_history_rewards_OUT, "OUT")
ranks_INOUT = get_rankings(all_history_rewards_INOUT, "INOUT")


original_stdout = sys.stdout
output_file_path = os.path.dirname(save_path) + "/Results.txt"
with open(output_file_path, 'w') as f:
    sys.stdout = f  
    print("TRAIN SET: ")
    print(np.array([np.std(history_rewards, axis=1) for history_rewards in all_history_rewards_IN]))

    all_avg_scores_IN = [np.mean(history_rewards, axis=1) for history_rewards in all_history_rewards_IN]
    medians_IN = [np.median(avg_scores) for avg_scores in all_avg_scores_IN]
    avgs_IN = [np.mean(avg_scores) for avg_scores in all_avg_scores_IN]
    std_IN = np.mean(np.array([np.std(history_rewards, axis=1) for history_rewards in all_history_rewards_IN]), axis=1)
    print("shape stss")
    print(np.shape(std_IN))
    print(std_IN)
    print("MEDIANS")
    for i, schedule in enumerate(training_schedules):
        print(schedule, medians_IN[i])

    print("AVGS")
    for i, schedule in enumerate(training_schedules):
        print(schedule, avgs_IN[i])

    print("STD")
    for i, schedule in enumerate(training_schedules):
        print(schedule, std_IN[i])

    print("")
    print("TEST SET: ")
    all_avg_scores_OUT = [np.mean(history_rewards, axis=1) for history_rewards in all_history_rewards_OUT]
    medians_OUT = [np.median(avg_scores) for avg_scores in all_avg_scores_OUT]
    avgs_OUT = [np.mean(avg_scores) for avg_scores in all_avg_scores_OUT]
    std_OUT = np.mean(np.array([np.std(history_rewards, axis=1) for history_rewards in all_history_rewards_OUT]), axis=1)
    print("MEDIANS")
    for i, schedule in enumerate(training_schedules):
        print(schedule, medians_OUT[i])

    print("AVGS")
    for i, schedule in enumerate(training_schedules):
        print(schedule, avgs_OUT[i])


    print("STD")
    for i, schedule in enumerate(training_schedules):
        print(schedule, std_OUT[i])

    print("")
    print("RANKING BASED ON MEDIAN")
    print("RANKING TRAIN:")
    for rank in ranks_IN: 
        print("  ", training_schedules[rank-1])

    print("\n")
    print("RANKING TEST:")
    for rank in ranks_OUT: 
        print("  ", training_schedules[rank-1])

    print("\n")
    print("RANKING TRAIN+TEST:")
    for rank in ranks_INOUT: 
        print("  ", training_schedules[rank-1])

    print("\n")

    
sys.stdout = original_stdout

# # Print statistics for IN category
# print_statistics(incremental_history_rewards_IN, "Incremental")
# print_statistics(gaussian1_history_rewards_IN, "Gaussian1")
# print_statistics(gaussian2_history_rewards_IN, "Gaussian2")
# print_statistics(cauchy1_history_rewards_IN, "Cauchy1")
# print_statistics(cauchy2_history_rewards_IN, "Cauchy2")
# print_statistics(uniform_history_rewards_IN, "Uniform")
# print_statistics(RL_history_rewards_IN, "RL")


# # Perform t-test between Incremental and RL
# perform_ttest(incremental_history_rewards_IN, RL_history_rewards_IN, "Incremental", "RL")

