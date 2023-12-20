import json
from utils import generate_morphologies
from utils import generate_samples
from utils import get_set
from evo import Algo
import os
import sys
from datetime import datetime
import shutil
import time
from concurrent.futures import ProcessPoolExecutor

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def single_run(config, run_id, timestamp_str, training_schedule):
    print("Running single_run with run_id={}, training_schedule={}".format(run_id, training_schedule))
    mean = None
    cov = None

    #all_schedules = ['gaussian1', 'gaussian2', 'cauchy1', 'cauchy2', 'uniform']

    #training_schedule = config['training_schedule']

    samples_per_cycle = None
    if training_schedule == "betawalk1" or training_schedule == "betawalk2":
        samples_per_cycle = config["samples_per_cycle"]

    print("RUN id: ", run_id)
    
    if training_schedule == "incremental" or training_schedule == "RL":
        variations = generate_morphologies(config['IN_parameter1'], config['IN_parameter2'], config['incremental_step_sizes'])
    else:
        variations = generate_samples(config['IN_parameter1'], config['IN_parameter2'], num_samples=config['generations'], distr=training_schedule, samples_per_cycle = samples_per_cycle)

    folder_name = config['filename']
    path = folder_name
    path = os.path.join(path, training_schedule)
    #path = path + training_schedule + "/"
    
    path = os.path.join(path, timestamp_str)
    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)


    if run_id==0:
        if training_schedule == "betawalk1" or training_schedule == "betawalk2": 
            indices = np.arange(len(variations))
            hue_values = indices // samples_per_cycle
            sns.scatterplot(x=variations[:, 0], y=variations[:, 1], hue=hue_values, legend=False)

        sns.scatterplot(x=variations[:, 0], y=variations[:, 1])
        plt.title(training_schedule)
        plt.xlim(config['IN_parameter1'])
        plt.ylim(config['IN_parameter2'])
        if config["game"]=="CartPoleEnv":
            plt.xlabel('Pole Mass (parameter 1)')
            plt.ylabel('Pole Length (parameter 2)')
        elif config["game"]=="BipedalWalker":
            plt.xlabel('parameter 1')
            plt.ylabel('parameter 2')

        elif config["game"]=="AcrobotEnv":
            plt.xlabel('Mass1 (parameter 1)')
            plt.ylabel('Mass2 (parameter 2)')
        save_plot_path = os.path.join(path, "Variations_generated_" + training_schedule + ".png")
        plt.savefig(save_plot_path)
        plt.close()

    filename = "config.json"
    new_file_path = os.path.join(path, filename)
    #new_file_path = path + "/" + filename

    with open(new_file_path, 'w') as new_json_file:
        json.dump(config, new_json_file, indent=4)

    runs_folder_name = "runs"
    run_path = os.path.join(path, runs_folder_name)
    run_path = os.path.join(run_path, str(run_id))
    #run_path = path + "/runs/" + str(run_id)
    os.makedirs(run_path, exist_ok=True)
    cluster_count = 0
    generations = config['generations']

    # print("validation set:")
    # print(get_set(config, 'Validation'))

    run = Algo(game=config['game'], path=run_path, xml_path=config['xml'], variations=variations,
               config=config, generation=generations, run_id=run_id, cluster_id=cluster_count,
               validation_set=get_set(config, 'IN'), training_schedule=training_schedule, gauss_mean=mean, gauss_cov=cov)
    generation, _ = run.main()


def experiment_run_parallel(config):
    runs = config['runs']
    timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
    all_schedules = config['training_schedule']

    print(os.cpu_count())
    # Determine the maximum number of CPUs
    max_workers = os.cpu_count() or 1

    print("Using {} CPUs".format(max_workers))

    with ProcessPoolExecutor(max_workers=None) as executor:
        futures = [
            executor.submit(single_run, config, i, timestamp_str, training_schedule)
            for training_schedule in all_schedules
            for i in range(runs)
        ]

        # Wait for all futures to complete
        results = [future.result() for future in futures]

if __name__ == '__main__':
    with open(str(sys.argv[1])) as json_file:
        print('Running experiment for ', sys.argv[1])
        config = json.load(json_file)
        experiment_run_parallel(config)