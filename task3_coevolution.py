from enum import Enum
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
NUM_GENERATIONS = 20
STRUCTURE_POP_SIZE = 20
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
TOURNAMENT_SIZE = 3
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


def generate_valid_robot():
    while True:
        structure = np.random.choice(VOXEL_TYPES, size=GRID_SIZE)
        if is_connected(structure):
            return structure


template_structure = generate_valid_robot()
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


# start from best structure found in task 3.1 and evolve a controller
best_structure_task_1 = [[2, 4, 2, 0, 0],
                         [0, 4, 1, 2, 2],
                         [4, 4, 0, 1, 3],
                         [2, 4, 3, 3, 1],
                         [3, 3, 0, 0, 3]]


def get_brain_for_structure(structure):
    connectivity = get_full_connectivity(structure)
    env = gym.make(SCENARIO, max_episode_steps=STEPS,
                   body=structure, connections=connectivity)

    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]

    brain = NeuralController(input_size, output_size)

    return brain


def robot_structure_to_key(structure: np.ndarray) -> str:
    return ''.join(map(str, structure.flatten().tolist()))


def _hill_climber_controller_learning_loop(structure, init_params, learn_steps=20, evaluate_fn=None):
    best_params = init_params.copy()
    best_fit = evaluate_fn(structure, best_params)
    for _ in range(1, learn_steps):
        candidate = best_params + np.random.randn(*best_params.shape) * 0.1
        fit = evaluate_fn(structure, candidate)
        if fit > best_fit:
            best_fit, best_params = fit, candidate.copy()
    return best_params, best_fit


def _genetic_algorithm_controller_learning_loop(structure, init_params, generations=20, evaluate_pop_fn=None):
    # init population from init_params and random, half-half
    population = [init_params.copy()]

    while len(population) < CONTROLLER_POP_SIZE:
        population.append(np.random.randn(len(init_params)))

    best_params = None
    best_fit = -np.inf
    for _ in range(generations):
        new_population = []

        # Evaluate population
        fitness_scores = evaluate_pop_fn(structure, population)

        # Elitism: keep the best individuals
        if ELITISM_CONTROLLERS:
            elite_indices = np.argsort(
                fitness_scores)[-ELITE_SIZE_CONTROLLERS:]
            elite_individuals = [population[i] for i in elite_indices]
            new_population.extend(elite_individuals)

        # Crossover and mutation to create new population
        while len(new_population) < CONTROLLER_POP_SIZE:
            parent1 = tournament_selection(
                population, fitness_scores, TOURNAMENT_SIZE)
            parent2 = tournament_selection(
                population, fitness_scores, TOURNAMENT_SIZE)

            offspring = binomial_crossover(parent1, parent2)

            offspring = gaussian_dist_mutation(
                offspring, CONTROLLER_MUTATION_RATE)

            new_population.append(offspring)

        # Replace old population with new one
        population = new_population

        # Track best params and fitness
        best_idx = np.argmax(fitness_scores)
        best_gen_fit = fitness_scores[best_idx]
        best_gen_params = population[best_idx]

        if best_gen_fit > best_fit:
            best_fit = best_gen_fit
            best_params = best_gen_params.copy()

        print(
            f"(Ctrl. Evo. Inner-Loop) Gen {_+1}/{generations} |  Best Fit: {best_fit:.4f} | Avg Fit: {np.mean(fitness_scores):.4f}")

    return best_params, best_fit


def step_coevolution(
        num_generations=NUM_GENERATIONS,
        controller_inner_loop_generations=20,
        structure_protection_window=3):

    # Archive: robot_structure_to_key -> (best_ctrl_params, best_fitness)
    #  avoids re-learning a controller from scratch every time a known structure reappears.
    controller_archive = {}

    structure_population = [generate_valid_robot()
                            for _ in range(STRUCTURE_POP_SIZE)]

    # Track age for innovation protection
    # track the "age" of each structure (how many generations it has been around and evaluated). This is used for "innovation protection."
    ages = {robot_structure_to_key(s): 0 for s in structure_population}

    eval_count = 0

    # Initialize all-time best tracking variables
    all_time_best_fit = -float('inf')
    all_time_best_struct = None
    all_time_best_params = None

    # Initialize history tracking lists
    best_fitness_history = []
    average_fitness_history = []
    best_structs_history = []

    for generation in range(num_generations):
        new_structs = []
        gen_fitness_values = []

        for struct in structure_population:
            # Mutate structure
            offspring = flip_mutation(struct.copy(), STRUCT_MUTATION_RATE)

            if not is_connected(offspring):
                offspring = generate_valid_robot()

            key = robot_structure_to_key(offspring)

            # init brain for specific structure
            struct_brain = get_brain_for_structure(offspring)

            # Initialize or lookup controller
            param_size = sum(p.numel() for p in struct_brain.parameters())

            # check if offspring is in archive
            if key in controller_archive:
                # if yes, best-found controller parameters for this structure to use as a starting point
                ctrl_init, _ = controller_archive[key]
            else:
                # else, random initialization
                ctrl_init = np.random.randn(param_size)

            # Inner-loop learning + evaluate
            def eval_fn(body, params):
                nonlocal eval_count
                fit, _ = evaluate_fitness(
                    params, body, get_full_connectivity(body))
                eval_count += 1
                return fit

            def eval_pop_fn(structure, population):
                nonlocal eval_count
                pairs = [(structure, p) for p in population]
                fitness_scores, _ = evaluate_pairs(pairs)
                eval_count += len(pairs)
                return fitness_scores

            # -------- Inner-loop Controller Optimization --------
            # trained_params, fit = _hill_climber_controller_learning_loop(offspring, ctrl_init,
            #                                                             learn_steps=controller_inner_loop_generations,
            #                                                             evaluate_fn=eval_fn)

            trained_params, fit = _genetic_algorithm_controller_learning_loop(
                offspring, ctrl_init,
                generations=controller_inner_loop_generations,
                evaluate_pop_fn=eval_pop_fn)

            # ----------------------------------------------------
            # Innovation protection - protect newer structures which have not evolved good controllers yet
            if ages.get(key, 0) < structure_protection_window:
                fit = max(fit, -1.0)
                ages[key] = ages.get(key, 0) + 1

            # Update archive
            prev_fit = controller_archive.get(key, (None, -np.inf))[1]
            if fit > prev_fit:
                controller_archive[key] = (trained_params.copy(), fit)

            # Track all-time best structure-controller combination
            if fit > all_time_best_fit:
                all_time_best_fit = fit
                all_time_best_struct = offspring.copy()
                all_time_best_params = trained_params.copy()
                print(
                    f"New all-time best: Fitness = {all_time_best_fit:.4f} (Gen {generation})")

            new_structs.append((offspring, fit))
            gen_fitness_values.append(fit)

        # Track history for this generation
        current_gen_best_idx = np.argmax([f for _, f in new_structs])
        best_fitness_history.append(new_structs[current_gen_best_idx][1])
        average_fitness_history.append(np.mean(gen_fitness_values))
        best_structs_history.append(
            new_structs[current_gen_best_idx][0].copy())

        # Select elites and fill structure pop
        new_structs.sort(key=lambda x: x[1], reverse=True)
        structure_population = [
            s for s, _ in new_structs[:ELITE_SIZE_STRUCTURES]]
        while len(structure_population) < STRUCTURE_POP_SIZE:
            structure_population.append(
                deepcopy(random.choice(new_structs)[0]))

        print(
            f"Gen. {generation} | #Evals: {eval_count} | Best fit = {new_structs[0][1]:.3f} | Avg fit = {np.mean([s[1] for s in new_structs]):.3f} | All-time best = {all_time_best_fit:.3f}"
        )

    # Return the all-time best
    return all_time_best_struct, all_time_best_params, all_time_best_fit, best_fitness_history, average_fitness_history, best_structs_history


def random_search_coevolution(num_generations=NUM_GENERATIONS, structure_pop_size=STRUCTURE_POP_SIZE, controller_pop_size=CONTROLLER_POP_SIZE):
    best_structure = None
    best_controller = None
    best_fitness_ever = -float('inf')
    best_fitness_history = []
    average_fitness_history = []

    for generation in range(num_generations):
        best_gen_fitness = -float('inf')
        # generate random structure population

        structure_population = [generate_valid_robot()
                                for _ in range(structure_pop_size)]

        # generate random controller for each structure

        pairs = []
        for structure in structure_population:
            struct_brain = get_brain_for_structure(structure)
            param_size = sum(p.numel() for p in struct_brain.parameters())
            random_controllers = [np.random.randn(
                param_size) for _ in range(controller_pop_size)]
            for controller in random_controllers:
                pairs.append((structure, controller))

        print(f"Evaluating {len(pairs)} pairs...")
        # evaluate fitness
        fitness_scores, rewards = evaluate_pairs(pairs)

        # track best fitness
        best_idx = np.argmax(fitness_scores)
        best_gen_fitness = fitness_scores[best_idx]
        best_gen_structure = pairs[best_idx][0]
        best_gen_controller = pairs[best_idx][1]

        # update best fitness ever
        if best_gen_fitness > best_fitness_ever:
            best_fitness_ever = best_gen_fitness
            best_structure = best_gen_structure
            best_controller = best_gen_controller

        # track history
        best_fitness_history.append(best_gen_fitness)
        average_fitness_history.append(np.mean(fitness_scores))

        print(f"Gen {generation+1}/{num_generations} | Best Fit: {best_gen_fitness:.4f} | Avg Fit: {np.mean(fitness_scores):.4f}")

    print(f"Best fitness ever: {best_fitness_ever:.4f}")
    return best_structure, best_controller, best_fitness_ever, best_fitness_history, average_fitness_history


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
        "name": "(0)CoEvGABiggerElitismImproveBestStructures",
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

        # **********************************************************************
        (all_time_best_struct,
         all_time_best_params,
         all_time_best_fit,
         best_fitness_history,
         average_fitness_history,
         ) = step_coevolution(
             num_generations=20,
             controller_inner_loop_generations=10,
             structure_protection_window=3,
        )
        # **********************************************************************

        # Recreate env from best structure to ensure consistent input/output sizes
        connectivity = get_full_connectivity(all_time_best_struct)

        env = gym.make(SCENARIO, max_episode_steps=STEPS,
                       body=all_time_best_struct, connections=connectivity)

        input_size = env.observation_space.shape[0]
        output_size = env.action_space.shape[0]

        brain = NeuralController(input_size, output_size)
        weights = utils.get_param_as_weights(
            np.array(all_time_best_params), model=brain)
        utils.set_weights(brain, weights)

        end_time = time.time()

        print(f"Best fitness found: {all_time_best_fit:.2f}")

        run_info["best_controller_params"] = all_time_best_params.tolist()
        run_info["best_fitness"] = all_time_best_fit
        run_info["best_structure"] = all_time_best_struct.tolist()
        run_info["best_fitness_history"] = best_fitness_history
        run_info["average_fitness_history"] = average_fitness_history
        run_info["execution_time"] = end_time - start_time
        run_info["seed"] = SEED

        with open(os.path.join(run_folder, "run.json"), "w") as f:
            json.dump(run_info, f, indent=4)

        utils.create_gif_brain(
            robot_structure=all_time_best_struct,
            brain=brain,
            filename=os.path.join(run_folder, "best_robot.gif"),
            scenario=SCENARIO,
            steps=STEPS,
        )
