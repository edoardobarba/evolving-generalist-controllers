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


def count_changes(column):
    changes = np.sum(column.diff().fillna(0) != 0)
    return changes


def count_increases(column):
    increases = np.sum(column.diff().fillna(0) > 0)
    return increases

def count_decrease(column):
    increases = np.sum(column.diff().fillna(0) < 0)
    return increases


train_RL_path = "/home/edoardo.barba/Results_Biped/RL/20231215114621"


all_data_changes = []
all_data_ratio = []
all_values = []

runs_folder = train_RL_path + "/runs/"
all_runs = os.listdir(runs_folder)

for run in all_runs:

    directory_path = runs_folder + run

    # List all files in the directory
    all_files = os.listdir(directory_path)

    desired_file = [file for file in all_files if "values_data.csv" in file]
    #print(desired_file)

    file_path = directory_path + "/" + desired_file[0]
    df = pd.read_csv(file_path)
    # Function to count changes in a column


    # Apply the function to each column in the DataFrame
    changes_count = df.apply(count_changes)
    increases_count = df.apply(count_increases)
    decreases_count = df.apply(count_decrease)
    ratio_incr_decr = increases_count/changes_count

    #print(changes_count)

    all_data_changes.append(changes_count)
    all_data_ratio.append(ratio_incr_decr)


    # Extracting the environments and values
    environments = df.columns[:]
    values = df.values[:, :]
    all_values.append(values)


# Combine the list of Series into a DataFrame
df_heatmap = pd.concat(all_data_changes, axis=1)

# Create a heatmap using seaborn
plt.figure(figsize=(12, 8))
sns.heatmap(df_heatmap, cmap='Blues', annot=True, fmt='d')

# Set labels and title
plt.xlabel('Run')
plt.ylabel('Environment')
plt.title('Heatmap of Values')


save_heatmap_path = train_RL_path + "/RL_NumberUsedEnv.png"
# Show the plot
print("plot saved in ", save_heatmap_path)
plt.savefig(save_heatmap_path)

plt.close()



df_ratio = pd.concat(all_data_ratio, axis=1)
# Create a heatmap using seaborn
plt.figure(figsize=(12, 8))
sns.heatmap(df_ratio, cmap='Blues', annot=True, fmt=".1f")

# Set labels and title
plt.xlabel('Run')
plt.ylabel('Environment')
plt.title('Heatmap of Values')


save_heatmap_path = train_RL_path + "/RL_ratio.png"
# Show the plot
print("plot saved in ", save_heatmap_path)
plt.savefig(save_heatmap_path)

plt.close()
















values_means = np.mean(all_values, axis=0)
values_stds = np.std(all_values, axis=0)

print(np.shape(values_means))
print(np.shape(values_stds))

# Get the number of rows (501) and columns (9)
num_rows, num_columns = values_means.shape

# Generate x values (row numbers)
x_values = np.arange(1, num_rows + 1)

# Set up a color map for different columns
colors = plt.cm.gist_rainbow(np.linspace(0, 1, num_columns))

# Plot each column with a different color
for i in range(num_columns):
    plt.plot(x_values, values_means[:, i], label=f'{environments[i]}', color=colors[i])
    plt.fill_between(x_values, values_means[:, i] - values_stds[:, i], values_means[:, i] + values_stds[:, i], alpha=0.1, color=colors[i])

# Add labels and legend
plt.xlabel('Generations')
plt.ylabel('Values')
plt.legend()


save_plot_path = train_RL_path + "/plot_values.png"
# Show the plot
print("plot saved in ", save_plot_path)
plt.savefig(save_plot_path)
plt.close()

