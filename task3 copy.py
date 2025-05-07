import numpy as np
import random
import gymnasium as gym
import torch
import time
import os
import json
import multiprocessing
from evogym.envs import *
from evogym import EvoViewer, get_full_connectivity, is_connected
from crossover_operators import arithmetic_crossover, binomial_crossover, uniform_crossover, one_point_crossover
from mutation_operators import flip_mutation, gaussian_dist_mutation
from neural_controller import *
from random_structure import tournament_selection
import utils
from copy import deepcopy


# ---- PARAMETERS ----
NUM_GENERATIONS = 5
STRUCTURE_POP_SIZE = 10
CONTROLLER_POP_SIZE = 10
STEPS = 500

# ---- STRUCTURE PARAMETERS ----
STRUCT_MUTATION_RATE = 0.15
VOXEL_TYPES = [0, 1, 2, 3, 4]
GRID_SIZE = (5, 5)


# ---- CONTROLLER PARAMETERS ----
CONTROLLER_MUTATION_RATE = 0.15
SIGMA = 0.1

# ---- SELECTION PARAMETERS ----
TOURNAMENT_SIZE = 4
ELITISM_CONTROLLERS = True
ELITE_SIZE_CONTROLLERS = 1
ELITISM_STRUCTURES = True
ELITE_SIZE_STRUCTURES = 1


MULTIPROCESSING = True

# ---- TESTING SETTINGS ----
SCENARIO = "GapJumper-v0"

SCENARIOS = [
    "GapJumper-v0",
    "CaveCrawler-v0",
]


# Input: Based on typical observation space.
# Output: Max possible actuators, e.g., every cell could be an actuator.
CONTROLLER_NN_INPUT_SIZE = 14 + 2 * np.prod(GRID_SIZE)
CONTROLLER_NN_OUTPUT_SIZE = np.prod(GRID_SIZE)  # Max possible actuators

template_structure = np.random.choice(VOXEL_TYPES, size=GRID_SIZE)
template_connectivity = get_full_connectivity(template_structure)
template_env = gym.make(
    SCENARIO,
    max_episode_steps=STEPS,
    body=template_structure,
    connections=template_connectivity,
)
CONTROLLER_NN_INPUT_SIZE = template_env.observation_space.shape[0]
CONTROLLER_NN_OUTPUT_SIZE = template_env.action_space.shape[0]
brain = NeuralController(CONTROLLER_NN_INPUT_SIZE, CONTROLLER_NN_OUTPUT_SIZE)
template_env.close()


def evaluate_fitness(controller, structure, connectivity, view=False):
    if not is_connected(structure):
        return -15.0, 0

    try:
        connectivity = get_full_connectivity(np.array(structure))
        env = gym.make(
            SCENARIO,
            max_episode_steps=STEPS,
            body=structure,
            connections=connectivity,
        )
        input_size = env.observation_space.shape[0]
        output_size = env.action_space.shape[0]
        brain = NeuralController(input_size, output_size)
        utils.set_weights(brain, utils.get_param_as_weights(
            np.array(controller), model=brain))
        sim = env.sim
        viewer = EvoViewer(sim)
        viewer.track_objects("robot")

        state = env.reset()[0]
        t_reward = 0
        t_velocity_x = 0.0
        t_velocity_y = 0.0

        for t in range(STEPS):
            state_tensor = torch.tensor(
                state, dtype=torch.float32).unsqueeze(0)
            action = brain(state_tensor).detach().numpy().flatten()

            if view:
                viewer.render('screen')

            state, reward, terminated, truncated, info = env.step(action)

            time = sim.get_time()
            vel = sim.object_vel_at_time(time, "robot")
            vel_at_t = np.mean(vel, axis=1)
            t_velocity_x += vel_at_t[0]
            t_velocity_y += vel_at_t[1]

            t_reward += reward

            if terminated or truncated:
                break

        viewer.close()
        env.close()

        total_time = sim.get_time()
        avg_velocity_x = t_velocity_x / (t + 1)
        avg_velocity_y = t_velocity_y / (t + 1)

        # Reward distance/time but penalize long runtimes slightly
        time_penalty_factor = 0.05
        fitness_val = t_reward

        return fitness_val, t_reward

    except (ValueError, IndexError) as e:
        return -15.0, 0  # Penalize invalid individuals


def generate_valid_robot():
    while True:
        structure = np.random.choice(VOXEL_TYPES, size=GRID_SIZE)
        if is_connected(structure):
            return structure

# evaluate (structure, controller) pairs


def evaluate_single_pair(pair: tuple) -> tuple:
    structure, controller = pair
    structure = np.array(structure).reshape(GRID_SIZE)
    if not is_connected(structure):
        return -15.0, 0

    if isinstance(structure, list) and len(structure) == 1:
        structure = structure[0]

    fitness, reward = evaluate_fitness(
        controller=controller,
        structure=structure,
        connectivity=get_full_connectivity(np.array(structure)),
    )

    return fitness, reward


def evaluate_pairs(pairs: tuple) -> tuple:
    fitness_scores = []
    rewards = []

    if not MULTIPROCESSING:
        # Sequential execution if multiprocessing is not enabled
        for pair in pairs:
            fitness_score, reward = evaluate_single_pair(pair)
            fitness_scores.append(fitness_score)
            rewards.append(reward)
        return fitness_scores, rewards

    num_processes = min(len(pairs), os.cpu_count())

    if num_processes > 1:  # Only use multiprocessing if beneficial
        try:
            # Ensure the main script execution is guarded by if __name__ == "__main__":
            # This is crucial for multiprocessing on Windows.
            with multiprocessing.Pool(processes=8) as pool:
                results = pool.map(evaluate_single_pair, pairs)

            for fitness_score, reward in results:
                fitness_scores.append(fitness_score)
                rewards.append(reward)
        except Exception as e:
            print(
                f"Multiprocessing failed: {e}. Falling back to sequential execution.")
            # Fallback to sequential execution if multiprocessing fails
            for pair in pairs:
                fitness_score, reward = evaluate_single_pair(pair)
                fitness_scores.append(fitness_score)
                rewards.append(reward)
    else:
        # Execute sequentially if only one process is needed or available
        for pair in pairs:
            fitness_score, reward = evaluate_single_pair(pair)
            fitness_scores.append(fitness_score)
            rewards.append(reward)

    return fitness_scores, rewards


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


# ---- PAIRING STRATEGIES ----
def get_interactions(pop1, pop2, type="random"):
    # possible types: random, all vs best, all vs all

    interactions = []  # (structure, controller) pairs

    match type:
        case "random":
            # Randomly pair all individuals from pop1 with individuals from pop2
            pop1_copy: List = deepcopy(pop1)
            pop2_copy: List = deepcopy(pop2)

            while len(pop1_copy) > 0:
                ind1 = random.choice(pop1_copy)
                ind2 = random.choice(pop2_copy)

                interactions.append((ind1, ind2))

                print(f"Pairing {ind1} with {ind2}")

                # Remove ind1 from pop1_copy
                idx1 = next(i for i, arr in enumerate(
                    pop1_copy) if np.array_equal(arr, ind1))
                pop1_copy.pop(idx1)
                # Remove ind2 from pop2_copy
                idx2 = next(i for i, arr in enumerate(
                    pop2_copy) if np.array_equal(arr, ind2))
                pop2_copy.pop(idx2)

        case "allVSall":
            pass
        case "allVSbest":
            pass

    return interactions


def run_ccea_evolution():
    # ---- POPULATION INITIALIZATION ----
    structure_population = [generate_valid_robot()
                            for _ in range(STRUCTURE_POP_SIZE)]

    controller_population = []
    for _ in range(CONTROLLER_POP_SIZE):
        param_vector = np.random.randn(
            sum(p.numel() for p in brain.parameters()))
        controller_population.append(param_vector)

    # pick single representatives (not lists)
    repr_structures = random.choice(structure_population)
    repr_controllers = random.choice(controller_population)

    # Tracking variables
    best_joint_fitness_so_far = -float('inf')
    best_robot_structure_ever = None
    best_robot_controller_params_ever = None

    best_joint_fitness_history = []
    best_structure_fitness_history = []
    best_controller_fitness_history = []
    avg_structure_fitness_history = []
    avg_controller_fitness_history = []
    best_reward_history = []
    average_reward_history = []
    best_reward_so_far = -float('inf')

    for generation in range(NUM_GENERATIONS):
        structure_best_fitness = -float('inf')
        controller_best_fitness = -float('inf')

        # Create new populations for each subproblem
        for component in range(2):
            if component == 0:  # structure evolution
                parents = structure_population
                other_repr = repr_controllers
                crossover_fn = one_point_crossover
                mutation_fn = flip_mutation
                mutation_rate = STRUCT_MUTATION_RATE
                population_size = STRUCTURE_POP_SIZE
                elite_size = ELITE_SIZE_STRUCTURES
                do_elitism = ELITISM_STRUCTURES
            else:  # controller evolution
                parents = controller_population
                other_repr = repr_structures
                crossover_fn = binomial_crossover
                mutation_fn = gaussian_dist_mutation
                mutation_rate = CONTROLLER_MUTATION_RATE
                population_size = CONTROLLER_POP_SIZE
                elite_size = ELITE_SIZE_CONTROLLERS
                do_elitism = ELITISM_CONTROLLERS

            # Make pairings for evaluation
            pairs = []
            if component == 0:
                pairs = [(parent, other_repr) for parent in parents]
            else:
                pairs = [(other_repr, parent) for parent in parents]

            fitness_vals, reward_vals = evaluate_pairs(pairs)

            # Track statistics
            avg_fitness = np.mean(fitness_vals)
            avg_reward = np.mean(reward_vals)
            best_idx = np.argmax(fitness_vals)
            best_fitness = fitness_vals[best_idx]
            best_reward = reward_vals[best_idx]

            if component == 0:  # structure stats
                avg_structure_fitness_history.append(avg_fitness)
                best_structure_fitness_history.append(best_fitness)
                structure_best_fitness = best_fitness
                structure_rewards = reward_vals
                best_structure = parents[best_idx]
            else:  # controller stats
                avg_controller_fitness_history.append(avg_fitness)
                best_controller_fitness_history.append(best_fitness)
                controller_best_fitness = best_fitness
                controller_rewards = reward_vals
                best_controller = parents[best_idx]

            # Create new population
            new_population = []

            # Apply elitism if enabled
            if do_elitism:
                elite_indices = np.argsort(fitness_vals)[-elite_size:]
                elite_individuals = [parents[i] for i in elite_indices]
                new_population.extend(elite_individuals)

            # Fill rest of population via selection, crossover, mutation
            while len(new_population) < population_size:
                # Select two parents via tournament
                parent1 = tournament_selection(
                    parents, fitness_vals, TOURNAMENT_SIZE)
                parent2 = tournament_selection(
                    parents, fitness_vals, TOURNAMENT_SIZE)

                # Apply crossover
                offspring = crossover_fn(parent1, parent2)

                # Apply mutation (fix parameter name)
                offspring = mutation_fn(offspring, mutation_rate)

                if component == 0:  # ensure structure is valid
                    # until connected
                    tries = 0
                    while not is_connected(offspring):
                        offspring = mutation_fn(offspring, mutation_rate)
                        tries += 1
                        if tries > 10:
                            # fallback to a new random valid robot
                            offspring = generate_valid_robot()
                            break

                new_population.append(offspring)

            # Replace old population
            if component == 0:
                structure_population = new_population
                # Update representative with best individual
                repr_structures = best_structure
            else:
                controller_population = new_population
                # Update representative with best individual
                repr_controllers = best_controller

        # After both components evolve, evaluate joint fitness
        joint_pairs = [(repr_structures, repr_controllers)]
        joint_fitness_vals, joint_reward_vals = evaluate_pairs(joint_pairs)
        joint_fitness = joint_fitness_vals[0]
        joint_reward = joint_reward_vals[0]

        # Track joint fitness
        best_joint_fitness_history.append(joint_fitness)

        average_reward_history.append(np.mean(joint_reward_vals))

        # Track best reward so far
        if joint_reward > best_reward_so_far:
            best_reward_so_far = joint_reward

        # Update overall best solution if better
        if joint_fitness > best_joint_fitness_so_far:
            best_joint_fitness_so_far = joint_fitness
            best_robot_structure_ever = repr_structures
            best_robot_controller_params_ever = repr_controllers

        # Print generation stats
        print(f"Gen {generation+1}/{NUM_GENERATIONS} | "
              f"Best Joint={joint_fitness:.4f} | "
              f"Best Struct={structure_best_fitness:.4f} | "
              f"Best Ctrl={controller_best_fitness:.4f} | "
              f"AvgStruct={avg_structure_fitness_history[-1]:.4f} | "
              f"AvgCtrl={avg_controller_fitness_history[-1]:.4f} | "
              f"Best Reward={joint_reward:.4f} | "
              f"Avg Reward={average_reward_history[-1]:.4f} | "
              f"Best Reward So Far={best_reward_so_far:.4f}")

    print(
        f"\nEvolution Finished. Best Overall Fitness: {best_joint_fitness_so_far:.2f}")

    return (best_robot_structure_ever,
            best_robot_controller_params_ever,
            best_joint_fitness_so_far,
            best_joint_fitness_history,
            avg_structure_fitness_history,
            avg_controller_fitness_history,
            best_reward_history,
            average_reward_history,
            best_reward_so_far)


# ---- SETUP FUNCTION ----
def setup_run(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


# ------ EXPERIMENTS ----------
if __name__ == "__main__":

    multiprocessing.freeze_support()  # For Windows compatibility

    RUN_SEEDS = [6363, 9374, 2003, 198, 2782]
    results_folder = "results/task3/"

    experiment_info = {
        # ***********************************************************************************
        # Change this to the name of the experiment. Will be used in the folder name.
        "name": "(0)TESTE",
        # ***********************************************************************************
        "repetitions": len(RUN_SEEDS),
        "num_generations": NUM_GENERATIONS,
        "sigma": SIGMA,
        "steps": STEPS,
        "population_size_structures": STRUCTURE_POP_SIZE,
        "population_size_controllers": CONTROLLER_POP_SIZE,
        "mutation_rate_structure": STRUCT_MUTATION_RATE,
        "mutation_rate_controller": CONTROLLER_MUTATION_RATE,
        "controller": NeuralController.__name__,
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
        run_info = {}

        print(f"Running experiment {i + 1} with seed {SEED}")
        setup_run(SEED)

        run_folder = os.path.join(experiment_folder, f"run_{i + 1}")
        if not os.path.exists(run_folder):
            os.makedirs(run_folder)

        start_time = time.time()

        (best_robot_structure_ever,
            best_robot_controller_params_ever,
            best_joint_fitness_so_far,
            best_joint_fitness_history,
            avg_structure_fitness_history,
            avg_controller_fitness_history,
            best_reward_history,
            average_reward_history,
            best_reward_so_far
         ) = run_ccea_evolution()

        # Recreate env from best structure to ensure consistent input/output sizes
        connectivity = get_full_connectivity(best_robot_structure_ever)

        env = gym.make(SCENARIO, max_episode_steps=STEPS,
                       body=best_robot_structure_ever, connections=connectivity)

        input_size = env.observation_space.shape[0]
        output_size = env.action_space.shape[0]

        brain = NeuralController(input_size, output_size)
        utils.set_weights(brain, best_robot_controller_params_ever)

        end_time = time.time()

        print(f"Best fitness found: {best_joint_fitness_so_far:.2f}")

        run_info["best_controller_params"] = best_robot_controller_params_ever.tolist()
        run_info["best_fitness"] = best_joint_fitness_so_far
        run_info["best_fitness_history"] = best_joint_fitness_history
        # run_info["average_fitness_history"] =
        run_info["best_reward_history"] = best_reward_history
        run_info["average_reward_history"] = average_reward_history
        run_info["best_structure"] = best_robot_structure_ever.tolist()
        run_info["best_structure_fitness_history"] = avg_structure_fitness_history

        run_info["best_controller_fitness_history"] = avg_controller_fitness_history
        run_info["best_reward_so_far"] = best_reward_so_far

        run_info["execution_time"] = end_time - start_time
        run_info["seed"] = SEED

        with open(os.path.join(run_folder, "run.json"), "w") as f:
            json.dump(run_info, f, indent=4)

        utils.create_gif_brain(
            robot_structure=best_robot_structure_ever,
            brain=brain,
            filename=os.path.join(run_folder, "best_robot.gif"),
            scenario=SCENARIO,
            steps=STEPS,
        )
