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





train_cauchy1_path = "/home/edoardo.barba/Results_Biped_old/cauchy1/20231215114621"
train_cauchy2_path = "/home/edoardo.barba/Results_Biped_old/cauchy2/20231215114621"
train_gaussian1_path = "/home/edoardo.barba/Results_Biped_old/gaussian1/20231215114440"
train_gaussian2_path = "/home/edoardo.barba/Results_Biped_old/gaussian2/20231215114440"
train_incremental_path = "/home/edoardo.barba/Results_Biped_old/incremental/20231215114440"
train_uniform_path = "/home/edoardo.barba/Results_Biped_old/uniform/20231215114653"
train_RL_path = "/home/edoardo.barba/Results_Biped_old/RL/20231215114621" 
train_beta1 = "/home/edoardo.barba/Results_Biped_old/beta1/20231215114653"
train_beta2 = "/home/edoardo.barba/Results_Biped_old/beta2/20231215114653"
train_betawalk1 = "/home/edoardo.barba/Results_Biped_old/betawalk1/20231215114907" 
train_betawalk2 = "/home/edoardo.barba/Results_Biped_old/betawalk2/20231215114907" 
train_gauss_dec = "/home/edoardo.barba/Results_Biped_old/gauss_dec/20231215114907"


training_schedules = ["incremental", "gaussian1", "gaussian2","cauchy1","cauchy2","uniform", "RL", "beta1", "beta2", "betawalk1", "betawalk2", "gauss_dec"]
all_train_folders = [train_incremental_path, train_gaussian1_path, train_gaussian2_path, train_cauchy1_path, train_cauchy2_path, train_uniform_path, train_RL_path, train_beta1, train_beta2, train_betawalk1, train_betawalk2, train_gauss_dec]
#all_train_folders = [train_incremental_path]

save_path = "/home/edoardo.barba/"

def get_significant_combinations(data):

    # Initialise a list of combinations of groups that are significantly different
    significant_combinations = []
    # Check from the outside pairs of boxes inwards
    ls = list(range(1, len(data) + 1))
    combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
    for combination in combinations:
        data1 = data[combination[0] - 1]
        data2 = data[combination[1] - 1]
        # Significance
        U, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        if p < 0.05:
            significant_combinations.append([combination, p])

    return significant_combinations
                
# def plot(game, incremental_history_rewards, gaussian1_history_rewards, gaussian2_history_rewards, 
#          cauchy1_history_rewards, cauchy2_history_rewards, uniform_history_rewards, RL_history_rewards, 
#          beta1_history_rewards, beta2_history_rewards, betawalk1_history_rewards, betawalk2_history_rewards,
#          test, save_path):
    
def plot(game, all_history_rewards, test, save_path):
    # print(all_history_rewards)
    avg_scores = [np.mean(history_rewards, axis=1) for history_rewards in all_history_rewards]
    # print(avg_scores)
    # print(np.shape(avg_scores))

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

    # data = [incremental_avg_scores, gaussian1_avg_scores, gaussian2_avg_scores, 
    #         cauchy1_avg_scores, cauchy2_avg_scores, uniform_avg_scores, RL_avg_scores, beta1_avg_scores, beta2_avg_scores, betawalk1_avg_scores, betawalk2_avg_scores ]

    data=avg_scores

    #print(data)
    significant_combinations = get_significant_combinations(data)

    
    #labels = ['Incremental', 'Gaussian1', 'Gaussian2', 'Cauchy1', 'Cauchy2', 'Uniform', 'RL', 'Beta1', "Beta2", "BetaWalk1", "BetaWalk2"]
    labels = training_schedules


    fig, ax = plt.subplots(figsize=(10, 7))
    bp = ax.boxplot(data, labels=labels)
    
    for i, points in enumerate(data, 1):
        x = [i] * len(points)
        ax.scatter(x, points, alpha=0.7)



    # Get the y-axis limits
    bottom, top = ax.get_ylim()
    y_range = top - bottom


    # for i, significant_combination in enumerate(significant_combinations):
    #     # Columns corresponding to the datasets of interest
    #     x1 = significant_combination[0][0]
    #     x2 = significant_combination[0][1]
    #     # What level is this bar among the bars above the plot?
    #     level = len(significant_combinations) - i
    #     # Plot the bar
    #     bar_height = (y_range * 0.07 * level) + top
    #     bar_tips = bar_height - (y_range * 0.02)
    #     plt.plot(
    #         [x1, x1, x2, x2],
    #         [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k'
    #     )
    #     # Significance level
    #     p = significant_combination[1]
    #     if p < 0.001:
    #         sig_symbol = '***'
    #     elif p < 0.01:
    #         sig_symbol = '**'
    #     elif p < 0.05:
    #         sig_symbol = '*'
    #     text_height = bar_height + (y_range * 0.01)
    #     plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', va='bottom', c='k')

    print("significant_combinations: ")
    for i, significant_combination in enumerate(significant_combinations):
        print("[", training_schedules[significant_combination[0][0]-1], training_schedules[significant_combination[0][1]-1] , "]")


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
    elif game == "AcrobotEnv":
        if test == "IN":
            ax.set_ylim(50, 100)
        else:
            ax.set_ylim(50, 140)
        ax.axhline(y=100, color='r', linestyle='--', label='Threshold')

    elif game == "BipedalWalker":
        if test == "IN":
            ax.set_ylim(-250,-50)
        else:
            ax.set_ylim(-60, 60)
        #ax.axhline(y=100, color='r', linestyle='--', label='Threshold')


    plt.tight_layout()
    plt.grid(axis='y')
    
    if save_path:
        save_path = os.path.join(save_path, "Boxplot_" + title + ".png")
        print(save_path)
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()




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

#all_var_INOUT = all_data_load[0]['all_var_INOUT']
        


# IN BOXPLOT 
# incremental_history_rewards_IN = incremental_data['all_history_rewards_IN']
# gaussian1_history_rewards_IN = gaussian1_data['all_history_rewards_IN']
# gaussian2_history_rewards_IN = gaussian2_data['all_history_rewards_IN']
# cauchy1_history_rewards_IN = cauchy1_data['all_history_rewards_IN']
# cauchy2_history_rewards_IN = cauchy2_data['all_history_rewards_IN']
# uniform_history_rewards_IN = uniform_data['all_history_rewards_IN']
# RL_history_rewards_IN = RL_data['all_history_rewards_IN']
# beta1_history_rewards_IN = beta1_data['all_history_rewards_IN']
# beta2_history_rewards_IN = beta2_data['all_history_rewards_IN']
# betawalk1_history_rewards_IN = betawalk1_data['all_history_rewards_IN']
# betawalk2_history_rewards_IN = betawalk2_data['all_history_rewards_IN']

# OUT BOXPLOT 
# incremental_history_rewards_OUT = incremental_data['avg_rewards_OUT']
# gaussian1_history_rewards_OUT = gaussian1_data['avg_rewards_OUT']
# gaussian2_history_rewards_OUT = gaussian2_data['avg_rewards_OUT']
# cauchy1_history_rewards_OUT = cauchy1_data['avg_rewards_OUT']
# cauchy2_history_rewards_OUT = cauchy2_data['avg_rewards_OUT']
# uniform_history_rewards_OUT = uniform_data['avg_rewards_OUT']
# RL_history_rewards_OUT = RL_data['avg_rewards_OUT']
# beta1_history_rewards_OUT = beta1_data['avg_rewards_OUT']
# beta2_history_rewards_OUT = beta2_data['avg_rewards_OUT']
# betawalk1_history_rewards_OUT = betawalk1_data['avg_rewards_OUT']
# betawalk2_history_rewards_OUT = betawalk2_data['avg_rewards_OUT']

# # INOUT BOXPLOT 
# incremental_history_rewards_INOUT = incremental_data['avg_rewards_INOUT']
# gaussian1_history_rewards_INOUT = gaussian1_data['avg_rewards_INOUT']
# gaussian2_history_rewards_INOUT = gaussian2_data['avg_rewards_INOUT']
# cauchy1_history_rewards_INOUT = cauchy1_data['avg_rewards_INOUT']
# cauchy2_history_rewards_INOUT = cauchy2_data['avg_rewards_INOUT']
# uniform_history_rewards_INOUT = uniform_data['avg_rewards_INOUT']
# RL_history_rewards_INOUT = RL_data['avg_rewards_INOUT']
# beta1_history_rewards_INOUT = beta1_data['avg_rewards_INOUT']
# beta2_history_rewards_INOUT = beta2_data['avg_rewards_INOUT']
# betawalk1_history_rewards_INOUT = betawalk1_data['avg_rewards_INOUT']
# betawalk2_history_rewards_INOUT = betawalk2_data['avg_rewards_INOUT']


if game == "CartPoleEnv":
    save_path = save_path + "Results_Cart/"

elif game == "Acrobot":
    save_path = save_path + "Results_Acrobot/"

elif game == "BipedalWalker":
    save_path = save_path + "Results_Biped_old/"

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
output_file_path = os.path.dirname(save_path) + "/rankings.txt"
with open(output_file_path, 'w') as f:
    sys.stdout = f  
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

