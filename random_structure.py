import json
import numpy as np
import random
import copy
import gymnasium as gym
from evogym.envs import *
import torch
import time

from evogym import (
    EvoWorld,
    EvoSim,
    EvoViewer,
    sample_robot,
    get_full_connectivity,
    is_connected,
)
import utils
from fixed_controllers import *
import time

import matplotlib.pyplot as plt


# ---- PARAMETERS ----
NUM_GENERATIONS = 5  # 250  # Number of generations to evolve
MIN_GRID_SIZE = (5, 5)  # Minimum size of the robot grid
MAX_GRID_SIZE = (5, 5)  # Maximum size of the robot grid
STEPS = 500
POPULATION_SIZE = 20  # Number of robots per generation
MUTATION_RATE = 0.1  # Probability of mutation


# ---- TESTING SETTINGS ----
SCENARIOS = [
    "Walker-v0",
    "BridgeWalker-v0",
]  # flat terrain locomotion AND soft ground that deforms under the robot

CONTROLLERS = {
    alternating_gait,
    sinusoidal_wave,
    hopping_motion,
}  # we should choose only ONE but we can test all

SCENARIO = "Walker-v0"
CONTROLLER = alternating_gait  # fixed controller

# ---- VOXEL TYPES ----
VOXEL_TYPES = [0, 1, 2, 3, 4]


# ----------Fitness Function--------------------------------------------------------
def evaluate_fitness(robot_structure, view=False):
    """loads the robot into the environment, usin the alternating gait controller
    and returns a fitness score based on how well the robot moves"""
    """O fitness score reflete o quão longe o robot se consegue deslocar"""
    if not is_connected(robot_structure):
        return 0.0

    try:
        connectivity = get_full_connectivity(
            robot_structure
        )  # calcula a conectividades do robot (para verificar se os voxels estão conectados)

        env = gym.make(
            SCENARIO,
            max_episode_steps=STEPS,
            body=robot_structure,
            connections=connectivity,
        )
        env.reset()
        sim = env.sim
        viewer = EvoViewer(sim)
        viewer.track_objects("robot")
        t_reward = 0
        action_size = sim.get_dim_action_space("robot")  # Get correct action size
        for t in range(STEPS):
            # Update actuation before stepping
            actuation = CONTROLLER(action_size, t)
            if view:
                viewer.render("screen")
            ob, reward, terminated, truncated, info = env.step(actuation)
            t_reward += reward

            if terminated or truncated:
                env.reset()
                break

        viewer.close()
        env.close()

        fitness_val = t_reward

        return fitness_val, t_reward  # return fitness and reward

    except (ValueError, IndexError) as e:
        return 0.0


def create_random_robot():
    """Generate a valid random robot structure."""

    grid_size = (
        random.randint(MIN_GRID_SIZE[0], MAX_GRID_SIZE[0]),
        random.randint(MIN_GRID_SIZE[1], MAX_GRID_SIZE[1]),
    )

    random_robot, _ = sample_robot(grid_size)
    return random_robot


def valid_robot():

    while True:
        robot = create_random_robot()
        if is_connected(robot):
            return robot
        else:
            print("Estrutura desconectada, descartada.")


def flip_mutation(robot_structure, MUTATION_RATE):
    """Rabdomly modifies some voxels in a robot"""
    mutated_robot = np.copy(robot_structure)

    for i in range(mutated_robot.shape[0]):
        for j in range(mutated_robot.shape[1]):
            if random.random() < MUTATION_RATE:
                original_voxel = mutated_robot[i, j]
                new_voxel = random.choice(VOXEL_TYPES)
            while new_voxel == original_voxel:
                new_voxel = random.choice(VOXEL_TYPES)

            mutated_robot[i, j] = new_voxel

    return mutated_robot


def one_point_crossover(parent1, parent2):
    """combine features from two parents to create a new offspring (one point crossover)"""
    # (5,5) -> vector
    parent1 = parent1.flatten()
    parent2 = parent2.flatten()

    crossover_point = random.randint(0, len(parent1) - 1)

    offspring = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))

    # vector -> (5,5)
    offspring = offspring.reshape((parent1.shape[0], parent1.shape[1]))

    return offspring


def tournament_selection(population, fitness_scores, k=5):
    selected_indices = random.sample(range(len(population)), k)
    best_idx = max(selected_indices, key=lambda idx: fitness_scores[idx])

    return population[best_idx]


# ----------Random Search--------------------------------------------------------


def random_search():
    """Perform a random search to find the best robot structure."""
    """by evaluting the fitness of random robots, keeps track of the best-performing
    structure and simulates the best structure at the end """

    best_fitness_history = []
    average_fitness_history = []
    best_reward_history = []
    average_reward_history = []

    best_robot = None
    best_fitness = -float("inf")
    best_reward = -float("inf")

    for it in range(NUM_GENERATIONS):
        robot = create_random_robot()
        fitness_score, reward = evaluate_fitness(robot)

        if fitness_score > best_fitness:
            best_fitness = fitness_score
            best_robot = robot

        # save best reward in task
        if reward > best_reward:
            best_reward = reward

        print(
            f"It. {it + 1}| Curr.Fit. = {fitness_score:.2f}| BestFit. = {best_fitness:.2f}| BestReward = {best_reward:.2f}"
        )

        best_fitness_history.append(best_fitness)
        average_fitness_history.append(fitness_score)
        best_reward_history.append(reward)
        average_reward_history.append(reward)

    return (
        best_robot,
        best_fitness,
        best_fitness_history,
        average_fitness_history,
        best_reward_history,
        average_reward_history,
    )


# ----------Evolutionary Algorithm--------------------------------------------------------


def evolutionary_algorithm(elitism=False):
    start_time = time.time()

    # population = [valid_robot() for individual in range (POPULATION_SIZE)]
    population = [create_random_robot() for individual in range(POPULATION_SIZE)]
    fitness_scores = [evaluate_fitness(ind) for ind in population]

    best_fitness_overall = -float("inf")
    best_robot_overall = None
    best_fitness_per_generation = []
    avg_fitness_per_generation = []

    for generation in range(NUM_GENERATIONS):
        new_population = []
        if elitism == True:
            sorted_indices = np.argsort(fitness_scores)[
                ::-1
            ]  # ordem os individuos pela fitness da maior para a menor
            elite_count = (
                POPULATION_SIZE // 2
            )  # guarda os 50% melhores indivíduos da população
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
                # offspring = valid_robot()
                offspring = create_random_robot()

            new_population.append(offspring)

        population = new_population
        fitness_scores = [evaluate_fitness(ind) for ind in population]
        best_idx = np.argmax(fitness_scores)
        # best_fitnesses_per_generation = fitness_scores[best_idx]
        # avg_fitness_per_generation = np.mean (fitness_scores)
        best_fitness_per_generation.append(fitness_scores[best_idx])
        avg_fitness_per_generation.append(np.mean(fitness_scores))

        if fitness_scores[best_idx] > best_fitness_overall:
            best_fitness_overall = fitness_scores[best_idx]
            best_robot_overall = population[best_idx]

        print(
            f"Generation {generation+1}: Best Fitness = {fitness_scores[best_idx]}, Average Fitness = {avg_fitness_per_generation[-1]:.2f}"
        )

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

    print(f"Best Fitness Overall: {best_fitness_overall:.2f}")

    # best_idx = np.argmax(fitness_scores)

    plt.figure(figsize=(10, 5))
    plt.plot(best_fitness_per_generation, label="Best Fitness")
    plt.plot(avg_fitness_per_generation, label="Average Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.title("Evolutionary Algorithm Progress")
    plt.show()

    return best_robot_overall, best_fitness_overall


def setup_run(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


# Choose which approach to run:
if __name__ == "__main__":
    RUN_SEEDS = [6363, 9374, 2003, 198, 2782]
    results_folder = "results\\task1"

    experiment_info = {
        # ***********************************************************************************
        # Change this to the name of the experiment. Will be used in the folder name.
        "name": "Random_Search_Test",
        # ***********************************************************************************
        "repetitions": len(RUN_SEEDS),
        "num_generations": NUM_GENERATIONS,
        "population_size": POPULATION_SIZE,
        "mutation_rate": MUTATION_RATE,
        "steps": STEPS,
        "controller": CONTROLLER.__name__,
        "scenario": SCENARIO,
        "time": time.strftime("D%d_M%m_%H_%M"),
    }

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Create a folder for the experiment
    experiment_folder = os.path.join(
        results_folder,
        experiment_info["scenario"],
        f"{experiment_info['name']}_{experiment_info['time']}",
    )

    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)

    # Save experiment info to experiment.json
    with open(os.path.join(experiment_folder, "experiment.json"), "w") as f:
        json.dump(experiment_info, f, indent=4)

    print("Starting experiment...")
    print("Experiment info:")
    print(experiment_info)

    for i, SEED in enumerate(RUN_SEEDS):
        run_info = {
            "seed": SEED,
            "best_robot_structure": None,
            "best_fitness": None,
            "best_fitness_history": None,
            "average_fitness_history": None,
            "best_reward_history": None,
            "average_reward_history": None,
            "execution_time": None,
        }

        print(f"Running experiment {i + 1} with seed {SEED}")

        # Set random seed for run
        setup_run(SEED)

        # Create folder for run
        run_folder = os.path.join(experiment_folder, f"run_{i + 1}")

        if not os.path.exists(run_folder):
            os.makedirs(run_folder)

        start_time = time.time()
        # **********************************************************************
        # Change the algorithm to be tested here.
        (
            best_robot,
            best_fitness,
            best_fitness_history,
            average_fitness_history,
            best_reward_history,
            average_reward_history,
        ) = random_search()
        # **********************************************************************
        end_time = time.time()

        print("Best robot structure found:")
        print(best_robot)
        print("Best fitness score:")
        print(best_fitness)

        # Save run info to run.json
        run_info["best_robot_structure"] = best_robot.tolist()
        run_info["best_fitness"] = best_fitness
        run_info["best_fitness_history"] = best_fitness_history
        run_info["average_fitness_history"] = average_fitness_history
        run_info["best_reward_history"] = best_reward_history
        run_info["average_reward_history"] = average_reward_history
        run_info["execution_time"] = end_time - start_time
        run_info["seed"] = SEED

        with open(os.path.join(run_folder, "run.json"), "w") as f:
            json.dump(run_info, f, indent=4)

        # Simulate and create a GIF for the best robot design
        # for _ in range(10):
        #    utils.simulate_best_robot(best_robot, scenario=SCENARIO, steps=STEPS)

        utils.create_gif(
            best_robot,
            filename=f"{run_folder}/best_robot.gif",
            scenario=SCENARIO,
            steps=STEPS,
            controller=CONTROLLER,
        )
