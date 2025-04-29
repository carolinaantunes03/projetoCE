import numpy as np
import random
import gymnasium as gym
import torch
import time
import os
import json
import multiprocessing
from evogym.envs import *
from evogym import EvoViewer, get_full_connectivity
from crossover_operators import binomial_crossover
from neural_controller import *
import utils


# ---- PARAMETERS ----
NUM_GENERATIONS = 5  # Number of generations to evolve
POPULATION_SIZE = 10  # Number of robots per generation
STEPS = 500

# (1+1) Evolution Strategy Params
SIGMA = 0.1
ALPHA = 0.25

# (μ + λ) Evolution Strategy Params
MU = 5  # Number of parents
LAMBDA = 5  # Number of offspring

# Mutation Params
MUTATION_RATE = 0.15  # Probability of mutation

# Selection Params
TOURNAMENT_SIZE = 2  # Number of individuals in the tournament for selection
ELITISM = True  # Whether to use elitism or not
ELITE_SIZE = 1  # Number of elite individuals to carry over to the next generation

# -- For Robot Eval ---
MULTIPROCESSING = True  # Whether to use multiprocessing or not

# ---- Fixed Robot Structure ----
robot_structure = np.array([
    [1, 3, 1, 0, 0],
    [4, 1, 3, 2, 2],
    [3, 4, 4, 4, 4],
    [3, 0, 0, 3, 2],
    [0, 0, 0, 0, 2]
])

# ---- TESTING SETTINGS ----
SCENARIO = "DownStepper-v0"

SCENARIOS = [
    "DownStepper-v0",
    "ObstacleTraverser-v0",
]

connectivity = get_full_connectivity(robot_structure)
env = gym.make(SCENARIO, max_episode_steps=STEPS,
               body=robot_structure, connections=connectivity)
sim = env.sim
input_size = env.observation_space.shape[0]  # Observation size
output_size = env.action_space.shape[0]  # Action size

brain = NeuralController(input_size, output_size)

# ---- FITNESS FUNCTION ----


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


def evaluate_fitness(weights, view=False):
    utils.set_weights(brain, weights)  # Load weights into the network
    try:
        env = gym.make(SCENARIO, max_episode_steps=STEPS,
                       body=robot_structure, connections=connectivity)
        sim = env.sim
        viewer = EvoViewer(sim)
        viewer.track_objects("robot")

        state = env.reset()[0]  # Get initial state
        t_reward = 0
        t_velocity_x = 0.0
        t_velocity_y = 0.0
        max_step_reward = 0.0

        t = 0
        for t in range(STEPS):
            # Get action from NN controller
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
            max_step_reward = max(max_step_reward, reward)

            t += 1
            if terminated or truncated:
                env.reset()
                break

        viewer.close()
        env.close()

        total_time = sim.get_time()

        avg_velocity_x = t_velocity_x / t
        avg_velocity_y = t_velocity_y / t

        time_penalty_factor = 0.05
        fitness_val = t_reward / (1 + time_penalty_factor * total_time)

        return fitness_val, t_reward

    except (ValueError, IndexError) as e:
        return -15.0, 0

# ---- HELPER FUNCTIONS ----


def get_flat_params(brain):
    return np.concatenate([p.detach().numpy().flatten() for p in brain.parameters()])


# ---- RANDOM SEARCH ALGORITHM ----
def random_search_algorithm():
    best_fitness = -np.inf
    best_params = None
    best_reward = 0

    best_fitness_history = []
    average_fitness_history = []
    best_reward_history = []
    average_reward_history = []

    print("Starting random search...")

    for generation in range(NUM_GENERATIONS):
        # Generate population of random weights
        population = []
        for _ in range(POPULATION_SIZE):
            param_vector = np.random.randn(
                sum(p.numel() for p in brain.parameters()))
            population.append(param_vector)

        # Evaluate all individuals in the population
        fitness_scores, rewards = evaluate_population_fitness([
            utils.get_param_as_weights(params, model=brain) for params in population
        ])

        # Find the best individual in the current generation
        best_idx = np.argmax(fitness_scores)
        current_best_fitness = fitness_scores[best_idx]
        current_best_params = population[best_idx]
        current_best_reward = rewards[best_idx]

        # Update best overall solution if better was found
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_params = current_best_params.copy()
            best_reward = current_best_reward
            print(f"Gen {generation+1}: New best fitness = {best_fitness:.2f}")
        else:
            print(
                f"Gen {generation+1}: No improvement (best={best_fitness:.2f})")

        # Save history
        best_fitness_history.append(best_fitness)
        average_fitness_history.append(np.mean(fitness_scores))
        best_reward_history.append(best_reward)
        average_reward_history.append(np.mean(rewards))

    # Set the best weights found
    utils.set_weights(brain, utils.get_param_as_weights(
        best_params, model=brain))

    return (
        best_params,
        best_fitness,
        best_fitness_history,
        average_fitness_history,
        best_reward_history,
        average_reward_history,
    )


# ---- (1+1) EVOLUTION STRATEGY ----
def one_plus_one_es():
    param_vector = get_flat_params(brain)
    best_params = param_vector.copy()

    best_fitness, best_reward = evaluate_fitness(
        utils.get_param_as_weights(best_params, model=brain))
    best_fitness_history = [best_fitness]
    average_fitness_history = [best_fitness]
    best_reward_history = [best_reward]
    average_reward_history = [best_reward]

    print(f"Initial fitness: {best_fitness:.2f}")

    for generation in range(NUM_GENERATIONS):
        offspring_params = best_params.copy()
        # Mutation to create offspring
        # Apply mutation to each parameter with probability ALPHA
        # SIGMA is the mutation step size (the standard deviation of the Gaussian noise)
        for i in range(len(offspring_params)):
            if np.random.rand() < ALPHA:
                offspring_params[i] += SIGMA * np.random.randn()

        offspring_fitness, offspring_reward = evaluate_fitness(
            utils.get_param_as_weights(offspring_params, model=brain))

        if offspring_fitness > best_fitness:
            best_fitness = offspring_fitness
            best_params = offspring_params.copy()
            best_reward = offspring_reward
            print(f"Gen {generation+1}: New best fitness = {best_fitness:.2f}")
        else:
            print(
                f"Gen {generation+1}: No improvement (best={best_fitness:.2f})")

        best_fitness_history.append(best_fitness)
        average_fitness_history.append(best_fitness)
        best_reward_history.append(best_reward)
        average_reward_history.append(best_reward)

    utils.set_weights(brain, utils.get_param_as_weights(
        best_params, model=brain))

    return (
        best_params,
        best_fitness,
        best_fitness_history,
        average_fitness_history,
        best_reward_history,
        average_reward_history,
    )

# ---- (μ + λ) EVOLUTION STRATEGY ----


def mu_plus_lambda_es(mu=MU, lamb=LAMBDA):
    # Evaluate initial mu parents
    parents = []
    for _ in range(mu):
        param_vector = get_flat_params(brain)
        fitness, reward = evaluate_fitness(
            utils.get_param_as_weights(param_vector, model=brain))
        parents.append(
            {'params': param_vector, 'fitness': fitness, 'reward': reward})

    # Sort parents by fitness descending
    parents.sort(key=lambda x: x['fitness'], reverse=True)

    best_params = parents[0]['params'].copy()
    best_fitness = parents[0]['fitness']
    best_reward = parents[0]['reward']

    best_fitness_history = [best_fitness]
    average_fitness_history = [np.mean([p['fitness'] for p in parents])]
    best_reward_history = [best_reward]
    average_reward_history = [np.mean([p['reward'] for p in parents])]

    print(f"Initial best fitness: {best_fitness:.2f}")

    # Calculate max generations based on budget:
    max_generations = (NUM_GENERATIONS - mu) // lamb

    for generation in range(max_generations):
        offspring = []

        # For each offspring, choose a parent to mutate (can do uniform random or tournament)
        for _ in range(lamb):
            parent = random.choice(parents)
            child_params = parent['params'].copy()
            # Mutation like in 1+1 ES:
            for i in range(len(child_params)):
                if np.random.rand() < ALPHA:
                    child_params[i] += SIGMA * np.random.randn()

            child_fitness, child_reward = evaluate_fitness(
                utils.get_param_as_weights(child_params, model=brain))
            offspring.append(
                {'params': child_params, 'fitness': child_fitness, 'reward': child_reward})

        # Combine parents + offspring and select top mu individuals
        combined = parents + offspring
        combined.sort(key=lambda x: x['fitness'], reverse=True)
        parents = combined[:mu]

        # Track stats from the new parent population
        current_best = parents[0]
        if current_best['fitness'] > best_fitness:
            best_fitness = current_best['fitness']
            best_params = current_best['params'].copy()
            best_reward = current_best['reward']
            print(f"Gen {generation+1}: New best fitness = {best_fitness:.2f}")
        else:
            print(
                f"Gen {generation+1}: No improvement (best={best_fitness:.2f})")

        best_fitness_history.append(best_fitness)
        average_fitness_history.append(
            np.mean([p['fitness'] for p in parents]))
        best_reward_history.append(best_reward)
        average_reward_history.append(np.mean([p['reward'] for p in parents]))

    # Set best found weights in brain
    utils.set_weights(brain, utils.get_param_as_weights(
        best_params, model=brain))

    return (
        best_params,
        best_fitness,
        best_fitness_history,
        average_fitness_history,
        best_reward_history,
        average_reward_history,
    )


# ---- DIFFERENTIAL EVOLUTION ----
# Added cr parameter
def differential_evolution(pop_size=POPULATION_SIZE, scale=0.2, cr=0.5):
    # initialization
    # Generate population of random weights
    population = []
    for _ in range(pop_size):  # Use pop_size here
        param_vector = np.random.randn(
            sum(p.numel() for p in brain.parameters()))
        population.append(param_vector)

    # Evaluate initial population
    fitness_scores, rewards = evaluate_population_fitness([
        utils.get_param_as_weights(params, model=brain) for params in population
    ])

    best_fitness_idx = np.argmax(fitness_scores)
    best_fitness = fitness_scores[best_fitness_idx]
    best_params = population[best_fitness_idx].copy()
    # Correctly initialize best_reward based on initial population
    # Use the reward corresponding to the best fitness
    best_reward = rewards[best_fitness_idx]

    best_fitness_history = [best_fitness]  # Store initial best
    average_fitness_history = [
        np.mean(fitness_scores)]  # Store initial average
    best_reward_history = [best_reward]  # Store initial best reward
    average_reward_history = [np.mean(rewards)]  # Store initial average reward

    print(
        f"Initial best fitness: {best_fitness:.2f}, Initial avg fitness: {average_fitness_history[0]:.2f}")

    for generation in range(NUM_GENERATIONS):
        trial_vectors = []  # Collect trial vectors for batch evaluation

        # Generate trial vectors for the entire population
        for i in range(pop_size):
            # Mutation
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b, c = random.sample(idxs, 3)
            mutant_vector = population[a] + \
                scale * (population[b] - population[c])

            # Crossover
            trial_vector = binomial_crossover(
                population[i], mutant_vector, cr=cr)
            trial_vectors.append(trial_vector)

        # Evaluate all trial vectors using multiprocessing if enabled
        trial_fitness_scores, trial_rewards = evaluate_population_fitness([
            utils.get_param_as_weights(params, model=brain) for params in trial_vectors
        ])

        new_population = []
        new_fitness_scores = []
        new_rewards = []

        # Selection: Compare trial vectors with current population
        for i in range(pop_size):
            # Keep trial if better or equal
            if trial_fitness_scores[i] >= fitness_scores[i]:
                new_population.append(trial_vectors[i])
                new_fitness_scores.append(trial_fitness_scores[i])
                new_rewards.append(trial_rewards[i])

                # Check for new overall best
                if trial_fitness_scores[i] > best_fitness:
                    best_fitness = trial_fitness_scores[i]
                    best_params = trial_vectors[i].copy()
                    best_reward = trial_rewards[i]
                    # Print improvement immediately when found
                    # print(f"Gen {generation+1}: New best fitness = {best_fitness:.2f}")
            else:
                new_population.append(population[i])
                new_fitness_scores.append(fitness_scores[i])
                new_rewards.append(rewards[i])

        # Update population and scores for the next generation
        population = new_population
        fitness_scores = new_fitness_scores
        rewards = new_rewards

        # Update best fitness/reward for the generation (in case the best was replaced by a worse one but is still the best overall)
        current_gen_best_idx = np.argmax(fitness_scores)
        if fitness_scores[current_gen_best_idx] > best_fitness:
            best_fitness = fitness_scores[current_gen_best_idx]
            best_params = population[current_gen_best_idx].copy()
            best_reward = rewards[current_gen_best_idx]

        # Save history for the generation
        best_fitness_history.append(best_fitness)
        average_fitness_history.append(np.mean(fitness_scores))
        # Track the best reward found so far
        best_reward_history.append(best_reward)
        average_reward_history.append(np.mean(rewards))

        print(
            f"Gen {generation+1}: Best Fitness={best_fitness:.2f}, Avg Fitness={average_fitness_history[-1]:.2f}, Best Reward={best_reward:.2f}, Avg Reward={average_reward_history[-1]:.2f}")

    # Set the best weights found
    utils.set_weights(brain, utils.get_param_as_weights(
        best_params, model=brain))

    return (
        best_params,
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


# ------ EXPERIMENTS ----------
# Choose which approach to run:
if __name__ == "__main__":

    multiprocessing.freeze_support()  # For Windows compatibility

    RUN_SEEDS = [6363, 9374, 2003, 198, 2782]
    results_folder = "results/task2/"

    experiment_info = {
        # ***********************************************************************************
        # Change this to the name of the experiment. Will be used in the folder name.
        "name": "(2.1)DeRand1Bin",
        # ***********************************************************************************
        "repetitions": len(RUN_SEEDS),
        "num_generations": NUM_GENERATIONS,
        "sigma": SIGMA,
        "alpha": ALPHA,
        "steps": STEPS,
        "population_size": POPULATION_SIZE,
        "mutation_rate": MUTATION_RATE,
        "mu": MU,
        "lamb": LAMBDA,
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

        # ***********************************************************************************
        (
            best_controller_params,
            best_fitness,
            best_fitness_history,
            average_fitness_history,
            best_reward_history,
            average_reward_history,
        ) = differential_evolution(pop_size=POPULATION_SIZE, scale=0.2, cr=0.5)
        # ***********************************************************************************

        end_time = time.time()

        print(f"Best fitness found: {best_fitness:.2f}")

        run_info["best_controller_params"] = best_controller_params.tolist()
        run_info["best_fitness"] = best_fitness
        run_info["best_fitness_history"] = best_fitness_history
        run_info["average_fitness_history"] = average_fitness_history
        run_info["best_reward_history"] = best_reward_history
        run_info["average_reward_history"] = average_reward_history
        run_info["execution_time"] = end_time - start_time
        run_info["seed"] = SEED

        with open(os.path.join(run_folder, "run.json"), "w") as f:
            json.dump(run_info, f, indent=4)

        # Create a GIF of the best performing controller
        utils.create_gif_nn(
            weights=best_controller_params,
            robot_structure=robot_structure,
            filename=os.path.join(run_folder, "best_robot.gif"),
            scenario=SCENARIO,
            steps=STEPS,
            brain=brain,
        )
