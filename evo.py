import joblib
import random
from evotorch import Problem, SolutionBatch
from evotorch.logging import StdOutLogger, PandasLogger
from evotorch.algorithms import XNES
import torch
import pandas as pd
import numpy as np
from nn import NeuralNetwork
from utils import gym_render, save_dataframes, softmax
from ant_v4_modified import AntEnv
from walker2d_v4_modified import Walker2dEnv
from bipedal_walker_modified import BipedalWalker
from cartpole_modified import CartPoleEnv
from utils import generate_samples
import seaborn as sns
import matplotlib.pyplot as plt
import gymnasium as gym
import time
import os
import sys

# class Eval(Problem):
#     def __init__(self, game, variations, topology, xml_path, steps, initial_bounds, counter):
#         super().__init__(
#             objective_sense="max",
#             solution_length=NeuralNetwork.calculate_total_connections(topology),
#             initial_bounds=(initial_bounds[0], initial_bounds[1]),
#         )

        
class Algo:
    def __init__(self, game, path, xml_path, variations, config, run_id, cluster_id, generation, validation_set, training_schedule, test_set, gauss_mean = None, gauss_cov = None):
        if game=="AcrobotEnv":
            self.game = game
        else:
            self.game = eval(game)
        self.variations = variations
        self.validation_set = validation_set
        self.test_set = test_set
        self.path = path
        self.xml_path = xml_path
        self.max_eval = generation
        self.cluster_id = cluster_id
        self.run_id = run_id
        self.max_fitness = config["maxFitness"]
        self.steps = config['nStep']
        self.topology = config['NN-struc']
        self.initial_stdev = config['stdev_init']
        self.initial_bounds = config["initial_bounds"]
        self.actors = config['actors']
        self.seed = random.randint(0, 1000000)
        self.parameters = self.variations[0]
        self.gauss_mean = gauss_mean
        self.gauss_cov = gauss_cov
        self.par1_range = config['IN_parameter1']
        self.par2_range = config['IN_parameter2']
        self.training_schedule = training_schedule
        self.V = np.empty((len(variations), generation+1))
        self.beta_param = np.empty((len(variations), generation+1, 2))
        self.gamma = 0.01
        self.alfa = 0.1  
        self.gscore_test = config["gscore_test"]
        self.evaluations = config['evals']
        if game == "AcrobotEnv":
            self.worst_score = 100
            self.best__score = 0
        elif game == "CartPoleEnv":
            self.worst_score = 0
            self.best__score = -1000

        elif game == "BipedalWalker":
            self.worst_score = 300
            self.best__score = -300


    def evaluate(self, agent: torch.Tensor) -> torch.Tensor:

        s = 0
        total_reward = 0
        if self.game == AntEnv:
            xml_file = '{}/Ant_{:.2f}_hip_{:.2f}_ankle.xml'.format(self.xml_path, self.parameters[0],
                                                                   self.parameters[1])
            env = self.game(xml_file, render_mode=None, healthy_reward=0)
        elif self.game == Walker2dEnv:
            xml_file = '{}/Walker_{:.3f}_thigh_{:.3f}_leg.xml'.format(self.xml_path, self.parameters[0],
                                                                      self.parameters[1])
            env = self.game(xml_file, render_mode=None, healthy_reward=0)
        elif self.game == "AcrobotEnv":
            env = gym.make('Acrobot-v1', render_mode = None).unwrapped
            env.LINK_MASS_1 = self.parameters[0]  #: [kg] mass of link 1
            env.LINK_MASS_2 = self.parameters[1]  #: [kg] mass of link 2
        else:
            env = self.game(self.parameters)

        obs, info = env.reset(seed=random.randint(0, 1000000))
        done = False

        x = agent.cpu()
        nn = NeuralNetwork(x.numpy())
        weights = nn.reshape_layers(self.topology)

        while not done:
            action = nn.feedforward(weights, self.topology, obs)
            #action = env.action_space.sample()
            if self.game == "AcrobotEnv":
                action = np.argmax(action)
            obs, reward, terminated, truncated, info = env.step(action)

            s += 1
            total_reward += reward

            if s > self.steps:
                break

            done = terminated or truncated

        #env.close()

        return -total_reward

    def multi_evaluate(self, agent: torch.Tensor) -> torch.Tensor:

        s = 0
        all_rewards = []
        for param in self.validation_set:

            total_reward = 0
            if self.game == AntEnv:
                xml_file = '{}/Ant_{:.2f}_hip_{:.2f}_ankle.xml'.format(self.xml_path, param[0],
                                                                    param[1])
                env = self.game(xml_file, render_mode=None, healthy_reward=0)
            elif self.game == Walker2dEnv:
                xml_file = '{}/Walker_{:.3f}_thigh_{:.3f}_leg.xml'.format(self.xml_path, param[0],
                                                                        param[1])
                env = self.game(xml_file, render_mode=None, healthy_reward=0)
            elif self.game == "AcrobotEnv":
                env = gym.make('Acrobot-v1', render_mode = None).unwrapped
                env.LINK_MASS_1 = param[0]  #: [kg] mass of link 1
                env.LINK_MASS_2 = param[1]  #: [kg] mass of link 2
            else:
                env = self.game(param
                )

            obs, info = env.reset(seed=random.randint(0, 1000000))
            done = False

            x = agent.cpu()
            nn = NeuralNetwork(x.numpy())
            weights = nn.reshape_layers(self.topology)

            while not done:
                action = nn.feedforward(weights, self.topology, obs)
                #action = env.action_space.sample()
                if self.game == "AcrobotEnv":
                    action = np.argmax(action)
                obs, reward, terminated, truncated, info = env.step(action)

                s += 1
                total_reward += reward

                if s > self.steps:
                    break

                done = terminated or truncated

            #env.close()
            all_rewards.append(-total_reward)

        print(all_rewards)
        return all_rewards


 

    def problem(self):
        #print("here1")
        if self.evaluations==1:
            objective_func=self.evaluate
        else: 
            objective_func=self.multi_evaluate
        problem = Problem(
            objective_sense ="min",  # minimize the fitness,
            objective_func=objective_func,  # evaluation function
            solution_length=NeuralNetwork.calculate_total_connections(self.topology),
            # NN topology determines the solution_length
            initial_bounds=(self.initial_bounds[0], self.initial_bounds[1]))
            # initial bounds limit the initial solutions
            # num_actors=self.actors)  
        #print("here2")
        searcher = XNES(problem, stdev_init=self.initial_stdev)#, obj_index = 0)
        #print("here3")
        return searcher

    def comparison(self, agent, i, test = False):
            if not test: 
                fitness = gym_render(self.game, agent, self.xml_path, self.validation_set[i], self.topology, self.steps)
            else: 
                fitness = gym_render(self.game, agent, self.xml_path, self.test_set[i], self.topology, self.steps)
            return fitness

    # main function to run the evolution
    def main(self):

        improved = 0  # Tracks whether there has been an improvement in the best fitness of the current population.
        generalist_old_dev = 0  # Represents the standard deviation of fitness scores across different environments for the generalist individual.
        prev_pop_best_fitness = 0  # Stores the best fitness of the previous population.
        xbest_weights = 0  # Stores the weights of the best individual in the current population.
        generalist_weights = 0  # Stores the weights of the generalist individual.
        generation = 0  # Tracks the current generation or iteration of the evolutionary algorithm.
        current_pop_best_fitness = self.max_eval  # Stores the best fitness in the current population.
        generalist_average_fitness = 1000000  # Represents the average fitness of the generalist individual across different environments.
        env_counter = 0  # Counts the current environment iteration in the loop.
        generalist_fitness_scores = np.zeros(len(self.validation_set))  # An array to store fitness scores of the generalist individual across different environments.
        good_fitness_scores = np.zeros(len(self.validation_set))  # Stores fitness scores for environments that are considered "good" based on certain criteria.
        number_environments = []  # Records the number of environments at each iteration.
        bad_environments = []  # Stores the environments that are considered "bad" and will be eliminated.
        generalist_average_fitness_history = []  # Records the history of average fitness of the generalist individual across iterations.
        fitness_std_deviation_history = []  # Records the history of standard deviation of fitness scores across different environments for the generalist individual.
        generalist_min_fitness_history = []  # Records the history of the minimum fitness of the generalist individual across iterations.
        generalist_max_fitness_history = []  # Records the history of the maximum fitness of the generalist individual across iterations.
        generalization_capability_history = []
        generalization_capability_test_history = []
        std_history =  []

        n_changed_gaussian_variance = 1
        fitness_step = self.max_fitness/3 
        #print("RUN searcher")

        searcher = self.problem()
        #print("here5")
        pandas_logger = PandasLogger(searcher)

        print('Number of Environments in Training Set: ', len(self.variations))
        print('Number of Environments in Validation Set: ', len(self.validation_set))
        logger = StdOutLogger(searcher, interval=50)
        if self.training_schedule == "RL":
            self.V[:, :] = 0.5
            
            G_diff = 0
            G_avg_score = None


        if self.training_schedule == "MAB":
            for k in range(len(self.variations)):
                self.beta_param[k, 0] = [1,1]


        iter = 1

        used_env = [0 for env in self.variations]
        start_time = time.time()
        while generation < self.max_eval:
            
            print("ITER: ", iter)
            if self.training_schedule == "RL":
                epsilon = 0.1  # Set your epsilon value here
                
                # With probability (1 - epsilon), choose the best action
                if np.random.rand() > epsilon:
                    max_indices = np.where(self.V[:, iter-1] == np.max(self.V[:, iter-1]))[0]
                    sampled_index = np.random.choice(max_indices)
                else:
                    # With probability epsilon, explore a random action
                    sampled_index = np.random.choice(len(self.variations))

                used_env[sampled_index] += 1
                self.parameters = self.variations[sampled_index]


            if self.training_schedule == "MAB":
                sampled_thetas = [np.random.beta(t[0], t[1]) for t in self.beta_param[:, iter-1]]
                
                max_indices = np.where(sampled_thetas == np.max(sampled_thetas))[0]
                sampled_index = np.random.choice(max_indices)

                used_env[sampled_index] += 1
                self.parameters = self.variations[sampled_index]


            searcher.step()
            index_best = searcher.population.argbest(obj_index=0)
            xbest_weights = searcher.population[index_best].values
            generation_best = float(searcher.population.evals[index_best][0])

            if current_pop_best_fitness > searcher.status.get('best_eval'):
                #print(searcher.status.get('best_eval'))
                current_pop_best_fitness = searcher.status.get('best_eval')
                xbest_weights_copy = xbest_weights.detach().clone()
                improved = searcher.status.get('iter')

            if len(self.validation_set) > 1:
                if self.game == "AcrobotEnv": 
                    gen_compute_Gscore = 50
                else:  
                    gen_compute_Gscore = self.gscore_test

                if self.evaluations == 1: 

                    if ((iter > self.max_eval-gen_compute_Gscore) or self.training_schedule == "RL" or self.training_schedule == "MAB"):
                        compare = joblib.Parallel(n_jobs=-1)(joblib.delayed(self.comparison)(xbest_weights, i)
                                                                    for i in range(len(generalist_fitness_scores)))

                        #compare = [self.comparison(xbest_weights, i) for i in range(len(generalist_fitness_scores))] 

                        generalist_fitness_scores = np.array(compare)

                        new_generalist_average_fitness = np.mean(generalist_fitness_scores)
                        new_std = np.std(generalist_fitness_scores)
                        
                        #new_generalist_average_fitness = new_generalist_average_fitness #+ (0.5*new_std)

                    else: 
                        new_generalist_average_fitness = 300
                        new_std = 300

                else:
                    new_generalist_average_fitness = generation_best
                    # print(new_generalist_average_fitness)
                    new_std = 0

                generalist_average_fitness_history.append(new_generalist_average_fitness)
                generalist_new_dev = np.std(generalist_fitness_scores)

                fitness_std_deviation_history.append(generalist_new_dev)
                generalist_min_fitness_history.append(np.min(generalist_fitness_scores))
                generalist_max_fitness_history.append(np.max(generalist_fitness_scores))

                #RL 
                if self.training_schedule == "RL":
                    G_score = new_generalist_average_fitness
                    #old_Gscore = generalization_capability_history[-1] if generalization_capability_history else self.worst_score

                    last_Gscores_mean = np.mean(np.array(generalist_average_fitness_history[-10:]))
                    #G_diff = G_score - old_Gscore
                    G_diff = G_score - last_Gscores_mean
                    r = 1 if G_diff<=0 else 0 #r=1 if improved

                    self.V[:, iter] = self.V[:, iter-1]
                    # if iter%50==0:
                    #     print(self.V[:, iter])
                    # print(self.V[:, 0])        
                    # print("G_score:", G_score)
                    # print("old_Gscore:", old_Gscore)
                    # print("G_diff:", G_diff)
                    # print("r:", r)
                    # print("V:", self.V[sampled_index, iter-1])
                    self.V[sampled_index, iter] = self.V[sampled_index, iter-1] + self.alfa*(r-self.V[sampled_index, iter-1])

                    if self.V[sampled_index, iter] < 0:
                        self.V[sampled_index, iter] = 0
                    if self.V[sampled_index, iter] > 1:
                        self.V[sampled_index, iter] = 1

                if self.training_schedule == "MAB": 
                    G_score = new_generalist_average_fitness

                    last_Gscores_mean = np.mean(np.array(generalist_average_fitness_history[-10:]))

                    G_diff = G_score - last_Gscores_mean

                    r = 1 if G_diff<=0 else 0 #r=1 if improved

                    for k in range(len(self.variations)): 
                        t = self.beta_param[k, iter-1]
                        t_alfa_beta_0 = self.beta_param[k, 0]
                        updated_alfa = (1-self.gamma)*t[0] + t_alfa_beta_0[0]*self.gamma
                        updated_beta = (1-self.gamma)*t[1] + t_alfa_beta_0[1]*self.gamma

                        self.beta_param[k, iter] = [updated_alfa, updated_beta]


                    # UPDATE alfa beta of sampled variation 
                    alfa = self.beta_param[sampled_index, iter][0]
                    beta = self.beta_param[sampled_index, iter][1]
                    updated_alfa = alfa + r 
                    updated_beta = beta + (1-r)

                    self.beta_param[sampled_index, iter] = [updated_alfa, updated_beta]

                # print("new generalist_average_fitness", new_generalist_average_fitness)
                # print("generalist_average_fitness", generalist_average_fitness)
                if new_generalist_average_fitness < generalist_average_fitness:
                    # test on test set 
                    # test_set_scores = joblib.Parallel(n_jobs=-1)(joblib.delayed(self.comparison)(xbest_weights, i, test=True)
                    #                                             for i in range(len(self.test_set)))
                    
                    # generalist_fitness_test_scores = np.array(test_set_scores)
                    # new_generalist_average_test_fitness = np.mean(generalist_fitness_test_scores)
                    # generalization_capability_test_history.append(new_generalist_average_test_fitness)

                    generalization_capability_history.append(new_generalist_average_fitness)
                    std_history.append(new_std)
                    generalist_average_fitness = new_generalist_average_fitness
                    generalist_old_dev = generalist_new_dev

                    

                    good_fitness_scores = generalist_fitness_scores.copy()
                    generalist_weights = xbest_weights.detach().clone()

                else:
                    generalization_capability_history.append(generalization_capability_history[-1])
                    # generalization_capability_test_history.append(generalization_capability_test_history[-1])
                    std_history.append(std_history[-1])


                if self.training_schedule == "incremental" or self.training_schedule == "default" or self.training_schedule == "border_incr":
                    env_counter += 1
                    if env_counter >= len(self.variations):
                        env_counter = 0
                    self.parameters = self.variations[env_counter] #INCREMENTAL TRAINING SCHEDULE 

                elif self.training_schedule == "random":
                    env_counter = random.randint(0, len(self.variations)-1)
                    self.parameters = self.variations[env_counter]

                elif self.training_schedule != "RL" and self.training_schedule != "MAB":
                    self.parameters = self.variations[iter-1] #iter-1 because iter starts from 1



            # elif len(self.variations) == 1 or self.evaluations>1:
            #     #print("current_pop_best_fitness: ", current_pop_best_fitness)
            #     generalist_average_fitness = current_pop_best_fitness
            #     xbest_weights = xbest_weights_copy
            #     generalist_weights = xbest_weights

            #     generalist_average_fitness_history.append(generalist_average_fitness)

            #     fitness_std_deviation_history.append(0)
            #     generalist_min_fitness_history.append(0)
            #     generalist_max_fitness_history.append(0)

            iter+=1

            number_environments.append(len(self.validation_set))
            generation = searcher.status.get('iter')

            #if generalist_average_fitness < self.max_fitness or generation > self.max_eval:
            if generation > self.max_eval:
                print("terzo break")
                break

            if generation%50==0:
                original_stdout = sys.stdout
                output_file_path = os.path.join(self.path, "training.txt")
                with open(output_file_path, 'w') as f:
                    sys.stdout = f  
                    print("Generation: ", iter)
                    sys.stdout = original_stdout
                    f.close()

            
            if self.game == CartPoleEnv or self.game == "AcrobotEnv":
                if generation%20==0:
                    print("Saving data...")
                    evals = pandas_logger.to_dataframe()
                    evals['no_envs'] = number_environments

                    save_path = os.path.join(self.path, str(generation)) 

                    generalist_evals = pd.DataFrame(
                        {'Mean': generalist_average_fitness_history, 'STD': fitness_std_deviation_history,
                        'Best': generalist_min_fitness_history, 'Worst': generalist_max_fitness_history, 'Gen_capability': generalization_capability_history, 'Gen_std': std_history})#, 'Gen_test_capability': generalization_capability_test_history})

                    info = '{}_{}_{}'.format(self.run_id, self.cluster_id, self.seed)

                    save_dataframes(evals, xbest_weights, generalist_weights, generalist_evals, info, save_path)

            else:
                if generation%1000==0:
                    print("Saving data...")
                    evals = pandas_logger.to_dataframe()
                    evals['no_envs'] = number_environments

                    save_path = os.path.join(self.path, str(generation)) 

                    generalist_evals = pd.DataFrame(
                        {'Mean': generalist_average_fitness_history, 'STD': fitness_std_deviation_history,
                        'Best': generalist_min_fitness_history, 'Worst': generalist_max_fitness_history, 'Gen_capability': generalization_capability_history, 'Gen_std': std_history})#, 'Gen_test_capability': generalization_capability_test_history})

                    info = '{}_{}_{}'.format(self.run_id, self.cluster_id, self.seed)

                    save_dataframes(evals, xbest_weights, generalist_weights, generalist_evals, info, save_path)


        end_time = time.time()
        print("Total execution time: ", end_time - start_time)
        if len(number_environments) != len(evals):
            number_environments.append(len(self.variations))

        if self.training_schedule == "RL":
            n_variations = len(self.V[:, 0])
            n_generations = len(self.V[0, :])
            x_axis = np.arange(0, n_generations)  # Assuming x-axis represents evaluations 1, 2, ..., n_evaluations

            plt.figure(figsize=(10, 6))

            # Create a heatmap using imshow
            heatmap = plt.imshow(self.V, aspect='auto', extent=[0, n_generations, 1, n_variations + 1], interpolation = "none", vmin=1, vmax=0)

            # Add colorbar
            cbar = plt.colorbar(heatmap, orientation='vertical')

            # Customize y-axis ticks
            custom_y_ticks = np.arange(1, len(self.variations) + 1)
            plt.yticks(custom_y_ticks, labels=[str(vec) for vec in self.variations])

            # Add horizontal lines
            for i in range(int(np.min(self.V)), int(np.max(self.V)) + 1):
                plt.axhline(y=i, linestyle='--', color='gray', linewidth=0.5)

            plt.xlabel('Evaluation')
            plt.ylabel('Values')
            plt.title('Values of Each Sample Across Evaluations')
            plt.tight_layout()
            # Save the heatmap plot
            filename = "Heatmap.png"
            plt.savefig(os.path.join(self.path, filename))
            plt.close()
            # Create a DataFrame to store the values
            values_df = pd.DataFrame(self.V.T, columns=[f'Env {vec}' for vec in self.variations])

            # Save the DataFrame to a CSV file
            csv_filename = 'values_data.csv'
            csv_path = os.path.join(self.path, csv_filename)
            values_df.to_csv(csv_path, index=False, float_format='%.2f')

            print(f'Values data saved to {csv_path}')

            for i in range(n_variations):
                self.V[i,:] = self.V[i, :] + i

            plt.figure(figsize=(10, 6))
            for i in range(n_variations):
                plt.plot(x_axis, self.V[i, :], label=f'{self.variations[i]}')

            custom_y_ticks = np.arange(1, len(self.variations)+1)
            plt.yticks(custom_y_ticks, labels=[str(vec) for vec in self.variations])

            for i in range(int(np.min(self.V)), int(np.max(self.V)) + 1):
                plt.axhline(y=i, linestyle='--', color='gray', linewidth=0.5)

            # Customize y-axis ticks
            # custom_y_ticks = [variation for variation in self.variations]
            # plt.yticks(custom_y_ticks)

            plt.xlabel('Evaluation')
            plt.ylabel('Values')
            plt.title('Values of Each Sample Across Evaluations')
            #plt.legend()
            figure_name = "Values.png"
            plt.savefig(os.path.join(self.path, figure_name))
            plt.close()
            # "heatmap"

        if self.training_schedule == "MAB":
            labels = [str(vec) for vec in self.variations]

            # Plotting
            plt.figure(figsize=(10, 6))
            for i in range(len(self.variations)):
                x = np.linspace(0, 1, 100)
                y = x**(self.beta_param[i, -1][0]) * (1 - x)**(self.beta_param[i, -1][1])
                plt.plot(x, y, label=f'{labels[i]}')

            plt.legend()
            plt.title('Beta Distributions for Variations')
            plt.xlabel('x')
            plt.ylabel('Probability Density')

            figure_name = "Betas.png"
            plt.savefig(os.path.join(self.path, figure_name))
            plt.close()


            beta_history_save_path = os.path.join(save_path, "history_beta.npz")
            #np.savez(save_path, all_history_rewards_IN=np.array(all_history_rewards_IN), avg_rewards_OUT=np.array(all_history_rewards_OUT), avg_rewards_INOUT=np.array(all_history_rewards_INOUT))
            np.savez(beta_history_save_path, history_beta=self.beta_param, used_env = np.array(used_env))
            
            sns.scatterplot(x=self.variations[:, 0], y=self.variations[:, 1], size=used_env, sizes=(10, 200))
            plt.title(self.training_schedule)
            plt.xlim(self.par1_range)
            plt.ylim(self.par2_range)
            # Remove the legend
            plt.legend().set_visible(False)

            save_plot_path = os.path.join(save_path, "Variations_generated_" + self.training_schedule + ".png")
            plt.savefig(save_plot_path)
            plt.close()


        return generation, self.variations#np.array(bad_environments)
