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
import matplotlib
from utils import generate_border_morphologies

matplotlib.rcParams.update({'font.size': 16})  # Adjust the font size as needed


def count_changes(column):
    changes = np.sum(column.diff().fillna(0) != 0)
    return changes


def count_increases(column):
    increases = np.sum(column.diff().fillna(0) > 0)
    return increases

def count_decrease(column):
    increases = np.sum(column.diff().fillna(0) < 0)
    return increases

def list_folders(path):
    # Get a list of all items (files and folders) in the specified path
    items = os.listdir(path)

    # Filter out only the directories
    folders = [item for item in items if os.path.isdir(os.path.join(path, item))]

    return folders


train_MAB_path = r"C:\Users\edoar\Documents\GitHub\Results_Biped\MAB\36_var"

if __name__ == '__main__':
    with open(str(sys.argv[1])) as json_file:
        print('Running experiment for ', sys.argv[1])
        config = json.load(json_file)


    all_data_changes = []
    all_data_ratio = []
    all_values = []

    runs_folder = train_MAB_path + "/runs/"
    all_runs = os.listdir(runs_folder)

    all_count_array = []

    variations = generate_morphologies(config['IN_parameter1'], config['IN_parameter2'], config['incremental_step_sizes'])

    labels = [str(vec) for vec in variations]


    for run in all_runs:

        directory_path = runs_folder + run + "/3000"

        # List all files in the directory
        all_files = os.listdir(directory_path)

        desired_file = np.load(directory_path + "/history_beta.npz")
        count_array = desired_file['used_env']
        all_count_array.append(count_array)

    
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(np.array(all_count_array), cmap='Blues', annot=True, fmt="d")
    ax.set_xticks(range(len(labels)))  # Set the x-ticks to match the number of labels
    ax.set_xticklabels(labels)  # Set the x-tick labels
    plt.savefig(train_MAB_path + "/used_env.png")
    plt.close()





    mean_used = np.mean(np.array(all_count_array), axis=0)
    param1_values = variations[:, 0]
    param2_values = variations[:, 1]
    rounded_param1_values = [round(num, 2) for num in param1_values]
    rounded_param2_values = [round(num, 2) for num in param2_values]

    print(variations)
    data = {'Parameter 1': rounded_param1_values, 'Parameter 2': rounded_param2_values, 'Reward': mean_used}

    print(len(rounded_param1_values), len(rounded_param2_values), len(mean_used))
    df = pd.DataFrame(data)
    #print(df)
    pivot_df = df.pivot(index='Parameter 2', columns='Parameter 1', values='Reward')
    pivot_df = pivot_df[::-1]

    plt.figure(figsize=(10, 7))


    sns.heatmap(pivot_df, cmap='Blues', annot=True, fmt=".0f") 
    if config['game'] == "BipedalWalker":       
        plt.xlabel('Leg Width')
        plt.ylabel('Leg Height')
    elif config['game'] == "Walker2dEnv":
        plt.xlabel('Upper Leg Length')
        plt.ylabel('Lower Leg Length')

    elif config['game'] == "AntEnv":
        plt.xlabel('Upper Leg Length')
        plt.ylabel('Lower Leg Length')

    #plt.title("Morphologies Frequency Selection")

    #plt.axvline(x=0.05, color='red', linestyle='--', linewidth=2)  # Adjust color, linestyle, and linewidth as needed
    plt.tight_layout()
    plt.savefig(train_MAB_path + "/heatmap_used_env.pdf", format="pdf")
    plt.close()






    sns.scatterplot(x=variations[:, 0], y=variations[:, 1], size=np.mean(np.array(all_count_array), axis=0), sizes=(10, 200))

    plt.xlim(config['IN_parameter1'])
    plt.ylim(config['IN_parameter2'])
    # Remove the legend
    plt.legend().set_visible(False)

    save_plot_path = os.path.join(train_MAB_path, "Variations_used")
    plt.savefig(save_plot_path)
    plt.close()






    runs_folder_path = os.path.join(train_MAB_path, "runs")
    
    print("runs_folder_path", runs_folder_path)
    runs_folders = list_folders(runs_folder_path)

    all_mean_array = []
    for i, run_number in enumerate(runs_folders):
        run_path = os.path.join(runs_folder_path, run_number)
        generation_run_folder_path = os.path.join(run_path, "5000")
        #print("generation_run_folder_path", generation_run_folder_path)
        history_beta_file = np.load(generation_run_folder_path + "/history_beta.npz")
        history_beta = history_beta_file['history_beta']
        #mean_array = np.mean(history_beta, axis=1)
        all_mean_array.append(history_beta)

    print(np.shape(all_mean_array))
    mean_array = np.mean(np.array(all_mean_array), axis=0)
    print(np.shape(mean_array))

    sns.scatterplot(x=variations[:, 0], y=variations[:, 1], size=np.mean(np.array(all_count_array), axis=0), sizes=(10, 200))

    plt.xlim(config['IN_parameter1'])
    plt.ylim(config['IN_parameter2'])
    # Remove the legend
    plt.legend().set_visible(False)

    save_plot_path = os.path.join(train_MAB_path, "Variations_used")
    plt.savefig(save_plot_path)
    plt.close()


    sum_array = np.sum(mean_array, axis=2)
    ratio_array = mean_array[:, :, 0] / (mean_array[:, :, 0] + mean_array[:, :, 1])
            
    print(np.shape(ratio_array))






    plt.figure(figsize=(10, 7))
    plt.plot(ratio_array.T, label=labels) 


    plt.xlabel('Generations')
    plt.ylabel('Expected Value')
    plt.title("Expected Value for each task")
    plt.legend()
    plt.tight_layout()
    plt.savefig(train_MAB_path + "/exp_value.png")
    plt.close()

    # import plotly.graph_objects as go

    # # Assuming data_array is your array
    # # data_array = np.random.rand(9, 3001, 2)  # example array

    # # Create a figure
    # fig = go.Figure()
    # labels = [str(vec) for vec in variations]



    # # Add traces for each index along the second dimension
    # for i in range(mean_array.shape[1]):
    #     fig.add_trace(go.Scatter(visible=False, x=labels, y=mean_array[:, i, 0], name='alpha'))
    #     fig.add_trace(go.Scatter(visible=False, x=labels, y=mean_array[:, i, 1], name='beta'))

    # # Make the first traces visible
    # fig.data[0].visible = True
    # fig.data[1].visible = True

    # # Create slider steps
    # steps = []
    # for i in range(0, len(fig.data), 2):
    #     step = dict(
    #         method="update",
    #         args=[{"visible": [False] * len(fig.data)},
    #             {"title": "Plot for index " + str(i//2)}],
    #     )
    #     step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    #     step["args"][0]["visible"][i+1] = True  # Toggle i+1'th trace to "visible"
    #     steps.append(step)

    # # Create the slider
    # sliders = [dict(
    #     active=0,
    #     currentvalue={"prefix": "Index: "},
    #     pad={"t": 50},
    #     steps=steps
    # )]

    # fig.update_layout(sliders=sliders)

    # fig.show()




    

    # Plotting
    # plt.figure(figsize=(10, 6))
    # for i in range(len(variations)):
    #     x = np.linspace(0, 1, 100)
    #     y = x**(mean_array[i][0]) * (1 - x)**(mean_array[i][1])
    #     plt.plot(x, y, label=f'{labels[i]}')

    # plt.legend()
    # plt.title('Beta Distributions for Variations')
    # plt.xlabel('x')
    # plt.ylabel('Probability Density')

    # figure_name = "Betas.png"
    # plt.savefig(os.path.join(train_MAB_path, figure_name))
    # plt.close()
