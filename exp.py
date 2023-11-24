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
import joblib

import seaborn as sns
import matplotlib.pyplot as plt

PARALLEL_RUNS_ACTORS = -1

def single_run(config, run_id, timestamp_str, training_schedule):
    mean = None
    cov = None

    #all_schedules = ['gaussian1', 'gaussian2', 'cauchy1', 'cauchy2', 'uniform']

    #training_schedule = config['training_schedule']
    print("RUN id: ", run_id)
    
    if training_schedule == "incremental":
        variations = generate_morphologies(config['IN_parameter1'], config['IN_parameter1'], config['incremental_step_sizes'])
    else:
        variations = generate_samples(config['IN_parameter1'], config['IN_parameter1'], num_samples=config['generations'], distr=training_schedule)

    folder_name = config['filename']
    path = f"{folder_name}/"
    path = path + training_schedule + "/"
    
    path = path + timestamp_str
    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)

    if run_id==0:
        sns.scatterplot(x=variations[:, 0], y=variations[:, 1])
        plt.xlim(config['IN_parameter1'])
        plt.ylim(config['IN_parameter1'])
        plt.title(training_schedule)
        plt.xlabel('Pole Mass (parameter 1)')
        plt.ylabel('Pole Length (parameter 2)')
        save_plot_path = os.path.join(path, "Variations_generated_" + training_schedule + ".png")
        plt.savefig(save_plot_path)

    filename = "config.json"
    new_file_path = path + "/" + filename

    with open(new_file_path, 'w') as new_json_file:
        json.dump(config, new_json_file, indent=4)

    run_path = path + "/runs/" + str(run_id)
    os.makedirs(run_path, exist_ok=True)
    cluster_count = 0
    generations = config['generations']

    run = Algo(game=config['game'], path=run_path, xml_path=config['xml'], variations=variations,
               config=config, generation=generations, run_id=run_id, cluster_id=cluster_count,
               validation_set=get_set(config, 'IN'), gauss_mean=mean, gauss_cov=cov)
    generation, _ = run.main()

def experiment_run_parallel(config):
    runs = config['runs']
    timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
    # Parallelize the runs using joblib
    all_schedules = config['training_schedule']
    print("ALL SCHEDULES:", all_schedules)

    for training_schedule in all_schedules:
        print("TRAINING USING: ", training_schedule)
        joblib.Parallel(n_jobs=PARALLEL_RUNS_ACTORS)(joblib.delayed(single_run)(config, i, timestamp_str, training_schedule) for i in range(runs))

with open(str(sys.argv[1])) as json_file:
    print('Running experiment for ', sys.argv[1])
    config = json.load(json_file)
    experiment_run_parallel(config)