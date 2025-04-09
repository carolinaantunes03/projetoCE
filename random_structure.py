import numpy as np
import random
import copy
import gymnasium as gym
from evogym.envs import *

from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot, get_full_connectivity, is_connected
import utils
from fixed_controllers import *
import time

import matplotlib.pyplot as plt


#3.1 task

# ---- PARAMETERS ----
NUM_GENERATIONS = 10  # Number of generations to evolve
MIN_GRID_SIZE = (5, 5)  # Minimum size of the robot grid
MAX_GRID_SIZE = (5, 5)  # Maximum size of the robot grid
STEPS = 500  
POPULATION_SIZE = 20 # Number of robots per generation
MUTATION_RATE = 0.1 # Probability of mutation
NUM_RUNS = 5

# ---- TESTING SETTINGS ----
SCENARIOS = ['Walker-v0', 'BridgeWalker-v0'] #flat terrain locomotion AND soft ground that deforms under the robot

CONTROLLERS = {alternating_gait, sinusoidal_wave, hopping_motion} #we should choose only ONE but we can test all

SCENARIO = 'Walker-v0'
CONTROLLER = alternating_gait  #fixed controller 

# ---- VOXEL TYPES ----
VOXEL_TYPES = [0, 1, 2, 3, 4]  


def evaluate_fitness(robot_structure, view=False):   
    '''loads the robot into the environment, usin the alternating gait controller 
    and returns a fitness score based on how well the robot moves''' 
    '''O fitness score reflete o quão longe o robot se consegue deslocar'''
    if not is_connected(robot_structure):
        return 0.0  
    
    try:
        connectivity = get_full_connectivity(robot_structure)  #calcula a conectividades do robot (para verificar se os voxels estão conectados)
  
        env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
        env.reset()
        sim = env.sim
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')
        t_reward = 0
        action_size = sim.get_dim_action_space('robot')  # Get correct action size
        for t in range(STEPS):  
            # Update actuation before stepping
            actuation = CONTROLLER(action_size,t)
            if view:
                viewer.render('screen') 
            ob, reward, terminated, truncated, info = env.step(actuation)
            t_reward += reward

            if terminated or truncated:
                env.reset()
                break

        viewer.close()
        env.close()
        return t_reward
    except (ValueError, IndexError) as e:
        return 0.0


def create_random_robot():
    """Generate a valid random robot structure."""
    
    grid_size = (random.randint(MIN_GRID_SIZE[0], MAX_GRID_SIZE[0]), random.randint(MIN_GRID_SIZE[1], MAX_GRID_SIZE[1]))
    
    random_robot, _ = sample_robot(grid_size)
    return random_robot

def valid_robot():

    while True:
        robot = create_random_robot()  
        if is_connected(robot):
            return robot
        else:
            print("Estrutura desconectada, descartada.")


def flip_mutation (robot_structure, MUTATION_RATE):
    """Rabdomly modifies some voxels in a robot """
    mutated_robot = np.copy(robot_structure)

    for i in range (mutated_robot.shape[0]):
        for j in range (mutated_robot.shape[1]):
            if random.random() < MUTATION_RATE:
                original_voxel = mutated_robot[i, j]
                new_voxel = random.choice (VOXEL_TYPES)
            while new_voxel == original_voxel:
                    new_voxel = random.choice(VOXEL_TYPES)
                
            mutated_robot[i, j] = new_voxel

    return mutated_robot

def one_point_crossover (parent1, parent2): #passar para vetor 
    ''' combine features from two parents to create a new offspring (one point crossover)'''
    offspring = np.copy (parent1)

    crossover_point = random.randint(1, parent1.shape[0] - 1)

    offspring[crossover_point:,:] = parent2 [crossover_point:,:]

    return offspring

def tournament_selection (population, fitness_scores, k=5):
    selected_indices = random.sample (range(len(population)), k)
    best_idx = max (selected_indices, key=lambda idx: fitness_scores[idx])

    return population [best_idx]


def random_search():
    """Perform a random search to find the best robot structure."""
    """by evaluting the fitness of random robots, keeps track of the best-performing
    structure and simulates the best structure at the end """

    best_robot = None
    best_fitness = -float('inf')
    
    for it in range(NUM_GENERATIONS):
        robot = create_random_robot() 
        fitness_score = evaluate_fitness(robot)
        
        if fitness_score > best_fitness:
            best_fitness = fitness_score
            best_robot = robot
        
        print(f"Iteration {it + 1}: Fitness = {fitness_score}")
    
    return best_robot, best_fitness

def evolutionary_algorithm(elitism = False):
    start_time = time.time()

    #population = [valid_robot() for individual in range (POPULATION_SIZE)]
    population = [create_random_robot() for individual in range (POPULATION_SIZE)]
    fitness_scores = [evaluate_fitness(ind) for ind in population]

    best_fitness_overall = -float('inf')  
    best_robot_overall = None
    best_fitness_per_generation = []  
    avg_fitness_per_generation = []

    for generation in range (NUM_GENERATIONS):
        new_population = []
        if (elitism == True):
            sorted_indices = np.argsort(fitness_scores)[::-1]  #ordem os individuos pela fitness da maior para a menor
            elite_count = POPULATION_SIZE // 2  #guarda os 50% melhores indivíduos da população
            elites = [population[i] for i in sorted_indices[:elite_count]]
            new_population.extend(elites)

        while len(new_population) < POPULATION_SIZE:

            if elitism == True:
                parent1, parent2 = elites[:2]
            else:
                # Select parents using tournament selection
                parent1 = tournament_selection(population, fitness_scores)
                parent2 = tournament_selection(population, fitness_scores)

            # Apply crossover to produce offspring
            offspring = one_point_crossover(parent1, parent2)
            # Apply mutation
            offspring = flip_mutation(offspring, MUTATION_RATE)

            # If offspring is disconnected, discard it and generate a valid robot
            if not is_connected(offspring):
                #offspring = valid_robot()
                offspring = create_random_robot()

            new_population.append(offspring)

        population = new_population
        fitness_scores = [evaluate_fitness(ind) for ind in population]
        best_idx = np.argmax(fitness_scores)
        #best_fitnesses_per_generation = fitness_scores[best_idx]
        #avg_fitness_per_generation = np.mean (fitness_scores)
        best_fitness_per_generation.append(fitness_scores[best_idx])
        avg_fitness_per_generation.append(np.mean(fitness_scores))

        if fitness_scores[best_idx] > best_fitness_overall:
            best_fitness_overall = fitness_scores[best_idx]
            best_robot_overall = population[best_idx]
            
        print(f"Generation {generation+1}: Best Fitness = {fitness_scores[best_idx]}, Average Fitness = {avg_fitness_per_generation[-1]:.2f}")
    
   
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

    print(f"Best Fitness Overall: {best_fitness_overall:.2f}")

    #best_idx = np.argmax(fitness_scores)

    plt.figure(figsize=(10, 5))
    plt.plot(best_fitness_per_generation, label="Best Fitness")
    plt.plot(avg_fitness_per_generation, label="Average Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.title("Evolutionary Algorithm Progress")
    plt.show()

    return best_robot_overall, best_fitness_overall




# Choose which approach to run:
if __name__ == "__main__":
    
    approach = input("Enter 'random' for random search or 'ea' for evolutionary algorithm: ").strip().lower()
    if approach == 'random':
        best_robot, best_fitness = random_search()
    elif approach == 'ea':
        best_robot, best_fitness = evolutionary_algorithm(elitism=True) #alterar para True or False, if True não há torneio
    else:
        print("Invalid option.")
        exit()

    print("Best robot structure found:")
    print(best_robot)
    print("Best fitness score:")
    print(best_fitness)



    # Simulate and create a GIF for the best robot design
    for _ in range(10):
        utils.simulate_best_robot(best_robot, scenario=SCENARIO, steps=STEPS)
    utils.create_gif(best_robot, filename='best_robot.gif', scenario=SCENARIO, steps=STEPS, controller=CONTROLLER)

'''if __name__ == "__main__":
    for run in range(NUM_RUNS):
        #random.seed(run)
        np.random.seed(run)
        print(f"Run {run+1}/{NUM_RUNS}:")
        best_robot, best_fitness = evolutionary_algorithm(elitism=True)
        print("Best robot structure found:")
        print(best_robot)
        print("Best fitness score:")
        print(best_fitness)
        utils.simulate_best_robot(best_robot, scenario=SCENARIO, steps=STEPS)
        utils.create_gif(best_robot, filename=f'best_robot_run{run+1}.gif', scenario=SCENARIO, steps=STEPS, controller=CONTROLLER)'''


