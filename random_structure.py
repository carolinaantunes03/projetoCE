import numpy as np
import random
import copy
import gymnasium as gym
from evogym.envs import *

from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot, get_full_connectivity, is_connected
import utils
from fixed_controllers import *
import time


#3.1 task

# ---- PARAMETERS ----
NUM_GENERATIONS = 250  # Number of generations to evolve
MIN_GRID_SIZE = (5, 5)  # Minimum size of the robot grid
MAX_GRID_SIZE = (5, 5)  # Maximum size of the robot grid
STEPS = 500  
POPULATION_SIZE = 20 # Number of robots per generation
MUTATION_RATE = 0.1 # Probability of mutation

# ---- TESTING SETTINGS ----
SCENARIOS = ['Walker-v0', 'BridgeWalker-v0'] #flat terrain locomotion AND soft ground that deforms under the robot

CONTROLLERS = {alternating_gait, sinusoidal_wave, hopping_motion} #we should choose only ONE but we can test all

SCENARIO = 'Walker-v0'
CONTROLLER = alternating_gait  #fixed controller 

# ---- VOXEL TYPES ----
VOXEL_TYPES = [0, 1, 2, 3, 4]  # Empty, Rigid, Soft, Active (+/-)


def evaluate_fitness(robot_structure, view=False):   
    '''loads the robot into the environment, usin the alternating gait controller 
    and returns a fitness score based on how well the robot moves''' 
    try:
        connectivity = get_full_connectivity(robot_structure)
  
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
        robot = create_random_robot()  # Sua função que gera robôs aleatórios
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
                mutated_robot [i, j] = random.choice (VOXEL_TYPES)

    return mutated_robot

def one_point_crossover (parent1, parent2):
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

def evolutionary_algorithm():
     
    population = [valid_robot() for individual in range (POPULATION_SIZE)]
    fitness_scores = [evaluate_fitness(ind) for ind in population]

    for generation in range (NUM_GENERATIONS):
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            # Select parents using tournament selection
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)
            # Apply crossover to produce offspring
            offspring = one_point_crossover(parent1, parent2)
            # Apply mutation
            offspring = flip_mutation(offspring, MUTATION_RATE)
            # If offspring is disconnected, discard it and generate a valid robot
            if not is_connected(offspring):
                offspring = valid_robot()
            new_population.append(offspring)

        population = new_population
        fitness_scores = [evaluate_fitness(ind) for ind in population]
        best_idx = np.argmax(fitness_scores)
        best_fitnesses = fitness_scores[best_idx]
        avg_fitness = np.mean (fitness_scores)
        print(f"Generation {generation+1}: Best Fitness = {fitness_scores[best_idx]}, Average Fitness = {avg_fitness:.2f}")

    avg_best_fitness = np.mean(best_fitnesses)
    print(f"Média das Best Fitnesses ao longo das gerações: {avg_best_fitness:.2f}")
    best_idx = np.argmax(fitness_scores)
    return population[best_idx], fitness_scores[best_idx]


# Example usage

# Choose which approach to run:
if __name__ == "__main__":
    start_time = time.time()

    approach = input("Enter 'random' for random search or 'es' for evolutionary algorithm: ").strip().lower()
    if approach == 'random':
        best_robot, best_fitness = random_search()
    elif approach == 'es':
        best_robot, best_fitness = evolutionary_algorithm()
    else:
        print("Invalid option.")
        exit()

    end_time = time.time()  


    print("Best robot structure found:")
    print(best_robot)
    print("Best fitness score:")
    print(best_fitness)

    total_time = end_time - start_time
    print(f"Tempo total de execução: {total_time:.2f} segundos")

    # Simulate and create a GIF for the best robot design
    for _ in range(10):
        utils.simulate_best_robot(best_robot, scenario=SCENARIO, steps=STEPS)
    utils.create_gif(best_robot, filename='best_robot.gif', scenario=SCENARIO, steps=STEPS, controller=CONTROLLER)



