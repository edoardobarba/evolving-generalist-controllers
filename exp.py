import json
from utils import generate_morphologies
from utils import generate_samples
from evo import Algo
import os
import sys
from datetime import datetime
import shutil



import seaborn as sns
import matplotlib.pyplot as plt

def experiment_run(config):
    runs = config['runs']
    parameter1_range, parameter2_range = config['parameter1'], config['parameter2']
    step_sizes = config['step_sizes']
    training_schedule = config['training_schedule']

    mean = None 
    cov = None 
    if training_schedule == "incremental":
        variations = generate_morphologies(parameter1_range, parameter2_range, step_sizes)
    else: 
        #mean, cov = get_initial_mean_cov()
        variations = generate_samples(parameter1_range, parameter2_range, num_samples=66)
    
    
    # print("VARIATIONS: ")
    # print(variations)
    # # Create a scatter plot
    # sns.scatterplot(x=variations[:, 0], y=variations[:, 1])
    # # Set x-axis and y-axis limits
    # plt.xlim(parameter1_range)
    # plt.ylim(parameter2_range)

    # # Set plot labels
    # plt.title('Incremental discrete sampling')
    # plt.xlabel('Pole Mass (parameter 1)')
    # plt.ylabel('Pole Lenght (parameter 2)')

    # # Display the plot
    # plt.show()
    folder_name = config['filename']
    path = f"{folder_name}/" 
    path = path + training_schedule + "/"
    timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
    path = path + timestamp_str
    os.makedirs(path, exist_ok=True)

    filename = "config.json"
    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)
    # Combine the directory and filename to create the full path
    new_file_path = os.path.join(path, filename)

    # Save the config dictionary as a JSON file in the new directory
    with open(new_file_path, 'w') as new_json_file:
        json.dump(config, new_json_file, indent=4)

    for i in range(runs):
        variations = generate_morphologies(parameter1_range, parameter2_range, step_sizes)
        print("run: ", i)
        path = path + str(i)
        # Ensure the directory exists
        os.makedirs(path, exist_ok=True)
        cluster_count = 0
        generations = config['generations']
        
        while len(variations) != 0:
            cluster_count += 1




            run = Algo(game=config['game'], path=path, xml_path=config['xml'], variations=variations,
                       config=config, generation=generations, run_id=i, cluster_id=cluster_count, gauss_mean=mean, 
                       gauss_cov=cov)
            generation, variations = run.main()
            generations = generations - generation


with open(str(sys.argv[1])) as json_file:
    print('Running experiment for ', sys.argv[1])
    config = json.load(json_file)
    experiment_run(config)
