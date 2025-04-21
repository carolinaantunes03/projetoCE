import json
import numpy as np
import random
import copy
import gymnasium as gym
from evogym.envs import *
import torch
import time
import multiprocessing

from evogym import (
    EvoWorld,
    EvoSim,
    EvoViewer,
    sample_robot,
    get_full_connectivity,
    is_connected,
)
from crossover_operators import one_point_crossover, uniform_crossover
from genetic_programming import GPTree, ramped_half_and_half_init_pop
from mutation_operators import flip_mutation, swap_mutation
import utils
from fixed_controllers import *
import time

import matplotlib.pyplot as plt


# ---- PARAMETERS ----
NUM_GENERATIONS = 250  # Number of generations to evolve
MIN_GRID_SIZE = (5, 5)  # Minimum size of the robot grid
MAX_GRID_SIZE = (5, 5)  # Maximum size of the robot grid
STEPS = 500
POPULATION_SIZE = 20  # Number of robots per generation
MUTATION_RATE = 0.05  # Probability of mutation

TOURNAMENT_SIZE = 4  # Number of individuals in the tournament for selection
ELITISM = True  # Whether to use elitism or not
ELITE_SIZE = 2  # Number of elite individuals to carry over to the next generation

# -- Sim ---
MULTIPROCESSING = False  # Whether to use multiprocessing or not


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
CONTROLLER = hopping_motion  # fixed controller

# ---- VOXEL TYPES ----
VOXEL_TYPES = [0, 1, 2, 3, 4]


# ----------Fitness Function--------------------------------------------------------
def evaluate_fitness(robot_structure, view=False):
    """loads the robot into the environment, usin the alternating gait controller
    and returns a fitness score based on how well the robot moves"""
    """O fitness score reflete o quão longe o robot se consegue deslocar"""
    if not is_connected(robot_structure):
        return -15.0, 0

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
        t_velocity_x = 0.0
        t_velocity_y = 0.0
        max_step_reward = 0.0
        action_size = sim.get_dim_action_space(
            "robot")  # Get correct action size

        t = 0
        for t in range(STEPS):

            # Update actuation before stepping
            actuation = CONTROLLER(action_size, t)
            if view:
                viewer.render("screen")
            ob, reward, terminated, truncated, info = env.step(actuation)

            time = sim.get_time()

            vel = sim.object_vel_at_time(time, "robot")

            t_reward += reward

            # vel.shape = (2,n) velocities in x and y direction for each voxel

            # From paper:
            # let vo be a vector of length 2 that represents the velocity of the center of mass of an object
            # o in the simulation at time t. vox and voy denote the x and y components of this vector, respectively. vo
            # is computed by averaging the velocities of all the point-masses that make up object o at time t

            # Average velocity of the robot at time t
            vel_at_t = np.mean(vel, axis=1)

            t_velocity_x += vel_at_t[0]
            t_velocity_y += vel_at_t[1]

            max_step_reward = max(max_step_reward, reward)

            if terminated or truncated:
                env.reset()
                break

        viewer.close()
        env.close()

        total_time = sim.get_time()

        avg_velocity_x = t_velocity_x / t
        avg_velocity_y = t_velocity_y / t

        primary_obj_reward = reward

        secondary_obj_reward = avg_velocity_x

        time_penalty_factor = 0.05

        fitness_val = t_reward / (1 + time_penalty_factor * total_time)

        return fitness_val, t_reward  # return fitness and reward

    except (ValueError, IndexError) as e:
        return -15.0, 0  # Return a low fitness score if an error occurs


def evaluate_population_fitness(population) -> tuple:
    """Evaluate the fitness of a population of robots."""
    fitness_scores = []
    rewards = []

    if not MULTIPROCESSING:
        # Sequential execution if multiprocessing is not enabled
        for robot in population:
            fitness_score, reward = evaluate_fitness(robot)
            fitness_scores.append(fitness_score)
            rewards.append(reward)
        return fitness_scores, rewards

    # Use more than one process, e.g., the number of available CPU cores
    # Be mindful of resource limits and potential overhead
    # Use number of robots or CPU cores, whichever is smaller
    num_processes = min(len(population), os.cpu_count())
    if num_processes > 1:  # Only use multiprocessing if beneficial
        try:
            # Ensure the main script execution is guarded by if __name__ == "__main__":
            # This is crucial for multiprocessing on Windows.
            with multiprocessing.Pool(processes=8) as pool:
                results = pool.map(evaluate_fitness, population)

            for fitness_score, reward in results:
                fitness_scores.append(fitness_score)
                rewards.append(reward)
        except Exception as e:
            print(
                f"Multiprocessing failed: {e}. Falling back to sequential execution.")
            # Fallback to sequential execution if multiprocessing fails
            for robot in population:
                fitness_score, reward = evaluate_fitness(robot)
                fitness_scores.append(fitness_score)
                rewards.append(reward)
    else:
        # Execute sequentially if only one process is needed or available
        for robot in population:
            fitness_score, reward = evaluate_fitness(robot)
            fitness_scores.append(fitness_score)
            rewards.append(reward)

    return fitness_scores, rewards


def create_random_robot():
    """Generate a valid random robot structure."""

    grid_size = (
        random.randint(MIN_GRID_SIZE[0], MAX_GRID_SIZE[0]),
        random.randint(MIN_GRID_SIZE[1], MAX_GRID_SIZE[1]),
    )

    random_robot, _ = sample_robot(grid_size)
    return random_robot


def generate_valid_robot():
    while True:
        robot = create_random_robot()
        if is_connected(robot):
            return robot
        else:
            print("Estrutura desconectada, descartada.")


def tournament_selection(population, fitness_scores, k=5):
    selected_indices = random.sample(range(len(population)), k)
    best_idx = max(selected_indices, key=lambda idx: fitness_scores[idx])

    return population[best_idx]


# ----------Random Search--------------------------------------------------------


def random_search(population_size=POPULATION_SIZE):
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
        best_gen_fitness = -float("inf")
        best_gen_reward = -float("inf")

        population = [create_random_robot() for _ in range(population_size)]
        fitness_scores, rewards = evaluate_population_fitness(population)

        max_fitness_idx = np.argmax(fitness_scores)
        robot = population[max_fitness_idx]
        fitness_score = fitness_scores[max_fitness_idx]
        reward = rewards[max_fitness_idx]

        if fitness_score > best_fitness:
            best_fitness = fitness_score
            best_robot = robot

        if fitness_score > best_gen_fitness:
            best_gen_fitness = fitness_score

        # save best reward overall
        if reward > best_reward:
            best_reward = reward

        # save best reward for this generation
        if reward > best_gen_reward:
            best_gen_reward = reward

        print(
            f"Gen. {it + 1} | Curr.Fit. = {fitness_score:.2f} | BestFitGen. = {best_gen_fitness:.2f} | BestFitOvr. = {best_fitness:.2f} | Avg.Fit. = {np.mean(fitness_scores):.2f} | BestRewardGen = {best_gen_reward:.2f} | Avg.Reward = {np.mean(rewards):.2f}"
        )

        best_fitness_history.append(best_gen_fitness)
        average_fitness_history.append(np.mean(fitness_scores))
        best_reward_history.append(best_gen_reward)
        average_reward_history.append(np.mean(rewards))

    return (
        best_robot,
        best_fitness,
        best_fitness_history,
        average_fitness_history,
        best_reward_history,
        average_reward_history,
    )


# ----------Evolutionary Algorithm--------------------------------------------------------


def evolutionary_algorithm(elitism=ELITISM):

    population = [generate_valid_robot()
                  for individual in range(POPULATION_SIZE)]
    # population = [create_random_robot() for individual in range(POPULATION_SIZE)]
    fitness_scores, reward_scores = evaluate_population_fitness(population)

    # initialize overall best tracking
    best_initial_fitness_idx = np.argmax(fitness_scores)
    best_initial_reward_idx = np.argmax(reward_scores)

    best_reward = reward_scores[best_initial_reward_idx]
    best_fitness = fitness_scores[best_initial_fitness_idx]
    best_robot = population[best_initial_fitness_idx]

    best_fitness_history = []
    average_fitness_history = []
    best_reward_history = []
    average_reward_history = []

    for generation in range(NUM_GENERATIONS):
        best_gen_fitness = -float("inf")
        best_gen_reward = -float("inf")

        new_population = []

        if elitism == True:
            # Get indices sorted by fitness
            sorted_indices = np.argsort(fitness_scores)[::-1]

            elites = [population[i] for i in sorted_indices[:ELITE_SIZE]]

            # Add elites to the new population
            new_population.extend(elites)

        while len(new_population) < POPULATION_SIZE:

            # *************************************************************************************
            # Select parents using tournament selection
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)

            # Change here to create different types of evolutionary algorithms
            # Apply crossover to produce offspring
            offspring = one_point_crossover(parent1, parent2)
            # Apply mutation
            offspring = flip_mutation(offspring, MUTATION_RATE)

            # **************************************************************************************

            # If offspring is disconnected, discard it and generate a valid robot
            if not is_connected(offspring):
                offspring = generate_valid_robot()
                # offspring = create_random_robot()

            new_population.append(offspring)

        population = new_population
        fitness_scores, rewards = evaluate_population_fitness(population)

        best_fitness_idx = np.argmax(fitness_scores)
        best_reward_idx = np.argmax(rewards)

        if fitness_scores[best_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[best_fitness_idx]
            best_robot = population[best_fitness_idx]

        if fitness_scores[best_fitness_idx] > best_gen_fitness:
            best_gen_fitness = fitness_scores[best_fitness_idx]

        if rewards[best_reward_idx] > best_reward:
            best_reward = rewards[best_reward_idx]

        if rewards[best_reward_idx] > best_gen_reward:
            best_gen_reward = rewards[best_reward_idx]

        average_fitness = np.mean(fitness_scores)
        average_reward = np.mean(rewards)

        best_fitness_history.append(best_gen_fitness)
        average_fitness_history.append(average_fitness)
        best_reward_history.append(best_gen_reward)
        average_reward_history.append(average_reward)

        print(
            f"Gen. {generation + 1} | Curr.Fit. = {fitness_scores[best_fitness_idx]:.2f} | BestFitGen. = {best_fitness:.2f} | BestFitOvr. = {best_fitness:.2f} | Avg.Fit. = {average_fitness:.2f} | BestRewardGen = {best_reward:.2f} | Avg.Reward = {average_reward:.2f}"
        )

    return (
        best_robot,
        best_fitness,
        best_fitness_history,
        average_fitness_history,
        best_reward_history,
        average_reward_history,
    )

# ----------Genetic Programming - Generation Expression Trees--------------------------------------------------------


MIN_GP_TREE_DEPTH = 1
MAX_GP_TREE_DEPTH = 4


def standard_gp(elitism=ELITISM):
    population: List[GPTree] = ramped_half_and_half_init_pop(min_depth=MIN_GP_TREE_DEPTH, mutation_rate=MUTATION_RATE, crossover_rate=0.80,
                                                             max_depth=MAX_GP_TREE_DEPTH, pop_size=POPULATION_SIZE)

    robot_pop = [tree.decode_to_robot((5, 5)) for tree in population]

    fitness_scores, reward_scores = evaluate_population_fitness(robot_pop)

    # initialize overall best tracking

    best_initial_fitness_idx = np.argmax(fitness_scores)
    best_initial_reward_idx = np.argmax(reward_scores)

    best_reward = reward_scores[best_initial_reward_idx]
    best_fitness = fitness_scores[best_initial_fitness_idx]
    best_robot = robot_pop[best_initial_fitness_idx]
    best_robot_tree = population[best_initial_fitness_idx]

    best_fitness_history = []
    average_fitness_history = []
    best_reward_history = []
    average_reward_history = []

    for generation in range(NUM_GENERATIONS):
        nextgen_pop = []

        best_gen_fitness = -float("inf")
        best_gen_reward = -float("inf")

        # Elitism
        if elitism == True:
            # Get indices sorted by fitness
            sorted_indices = np.argsort(fitness_scores)[::-1]

            elites = [population[i] for i in sorted_indices[:ELITE_SIZE]]

            # Add elites to the new population
            nextgen_pop.extend(elites)

        while len(nextgen_pop) < POPULATION_SIZE:
            # Select parents using tournament selection
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)

            # Create copies to generate offspring
            offspring1 = copy.deepcopy(parent1)
            # Need a copy of parent2 for crossover
            offspring2 = copy.deepcopy(parent2)

            # Crossover modifies offspring1 in-place using offspring2
            offspring1.crossover(offspring2)
            offspring1.mutation()

            nextgen_pop.append(offspring1)

        # Generational Strategy?

        population = nextgen_pop

        robot_pop = [tree.decode_to_robot((5, 5)) for tree in population]

        # --- Add Logging Here ---
        connected_count = sum(1 for robot in robot_pop if is_connected(robot))
        disconnected_perc = (1 - (connected_count / POPULATION_SIZE)) * 100
        print(f"Gen. {generation + 1} Disconnected: {disconnected_perc:.1f}%")
        # --- End Logging ---

        fitness_scores, rewards = evaluate_population_fitness(robot_pop)
        best_fitness_idx = np.argmax(fitness_scores)
        best_reward_idx = np.argmax(rewards)

        # Update best fitness and robot if needed
        if fitness_scores[best_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[best_fitness_idx]
            best_robot = robot_pop[best_fitness_idx]
            best_robot_tree = population[best_fitness_idx]

        if fitness_scores[best_fitness_idx] > best_gen_fitness:
            best_gen_fitness = fitness_scores[best_fitness_idx]

        if rewards[best_reward_idx] > best_reward:
            best_reward = rewards[best_reward_idx]

        if rewards[best_reward_idx] > best_gen_reward:
            best_gen_reward = rewards[best_reward_idx]

        average_fitness = np.mean(fitness_scores)
        average_reward = np.mean(rewards)

        best_fitness_history.append(best_gen_fitness)
        average_fitness_history.append(average_fitness)
        best_reward_history.append(best_gen_reward)
        average_reward_history.append(average_reward)

        print(
            # Use best_gen_fitness and best_gen_reward for generation-specific reporting
            f"Gen. {generation + 1} | Curr.Fit. = {fitness_scores[best_fitness_idx]:.2f} | BestFitGen. = {best_gen_fitness:.2f} | BestFitOvr. = {best_fitness:.2f} | Avg.Fit. = {average_fitness:.2f} | BestRewardGen = {best_gen_reward:.2f} | Avg.Reward = {average_reward:.2f}"
        )

        print(f"Best robot tree:\n {best_robot_tree.print_tree()}")
        print(f"Best robot structure:\n {best_robot}")

    return (
        best_robot,
        best_fitness,
        best_fitness_history,
        average_fitness_history,
        best_reward_history,
        average_reward_history,
    )


def setup_run(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


# Choose which approach to run:
if __name__ == "__main__":

    multiprocessing.freeze_support()  # For Windows compatibility

    RUN_SEEDS = [6363, 9374, 2003, 198, 2782]
    results_folder = "results/task1"

    experiment_info = {
        # ***********************************************************************************
        # Change this to the name of the experiment. Will be used in the folder name.
        "name": "(0)RandomSearch",
        # ***********************************************************************************
        "repetitions": len(RUN_SEEDS),
        "num_generations": NUM_GENERATIONS,
        "population_size": POPULATION_SIZE,
        "mutation_rate": MUTATION_RATE,
        "steps": STEPS,
        "controller": CONTROLLER.__name__,
        "scenario": SCENARIO,
        "time": time.strftime("D%d_M%m_%H_%M"),
        "elitism": ELITISM,
        "elitism_size": ELITE_SIZE,
        "tournament_size": TOURNAMENT_SIZE,
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
        ) = random_search()        # **********************************************************************
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
