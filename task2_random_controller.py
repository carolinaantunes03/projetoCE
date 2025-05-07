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
from crossover_operators import arithmetic_crossover, binomial_crossover, uniform_crossover
from mutation_operators import flip_mutation, gaussian_dist_mutation
from neural_controller import *
from random_structure import tournament_selection
import utils


# ---- PARAMETERS ----
NUM_GENERATIONS = 250  # Number of generations to evolve
POPULATION_SIZE = 20  # Number of robots per generation
STEPS = 500

# (1+1) Evolution Strategy Params
SIGMA = 0.1
ALPHA = 0.25

# (μ + λ) Evolution Strategy Params
MU = 5  # Number of parents
LAMBDA = 20  # Number of offspring

# Mutation Params
MUTATION_RATE = 0.15  # Probability of mutation

# Selection Params
TOURNAMENT_SIZE = 4  # Number of individuals in the tournament for selection
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


# ---- (μ + λ) EVOLUTION STRATEGY ----


def mu_plus_lambda_es(mu=MU, lamb=LAMBDA, alpha=ALPHA, sigma=SIGMA):

    # mu-> number of parent solutions in the population
    # lamb-> number of offspring solutions to be generated from the parents at each generation

    # 1. init pop with mu random solutions
    population = []
    for _ in range(mu):
        param_vector = np.random.randn(
            sum(p.numel() for p in brain.parameters()))
        population.append(param_vector)

    # 2. evaluate fitness of each solution in the population
    fitness_scores, rewards = evaluate_population_fitness([
        utils.get_param_as_weights(params, model=brain) for params in population
    ])

    best_fitness_idx = np.argmax(fitness_scores)
    best_reward_idx = np.argmax(rewards)

    best_fitness = fitness_scores[best_fitness_idx]
    best_reward = rewards[best_reward_idx]
    best_params = population[best_fitness_idx].copy()

    # init hist lists
    best_fitness_history = []
    average_fitness_history = []
    best_reward_history = []
    average_reward_history = []

    for generation in range(NUM_GENERATIONS):

        offspring_population = []
        for _ in range(lamb):
            # 3.1 select parent solution randomly from the population
            parent = random.choice(population)

            # 3.2 create offspring by applying mutation to the parent solution
            offspring = parent.copy()
            # Apply mutation to each parameter with probability ALPHA
            # SIGMA is the mutation step size (the standard deviation of the Gaussian noise)
            for i in range(len(offspring)):
                if np.random.rand() < alpha:
                    offspring[i] += sigma * np.random.randn()

            offspring_population.append(offspring)

        # 4. evaluate fitness of each offspring solution
        offspring_fitness_scores, offspring_rewards = evaluate_population_fitness([
            utils.get_param_as_weights(params, model=brain) for params in offspring_population
        ])

        # 5. select the best mu solutions from the combined population of parents and offspring
        combined_population = population + offspring_population
        combined_fitness_scores = fitness_scores + offspring_fitness_scores
        combined_rewards = rewards + offspring_rewards

        sorted_indices = np.argsort(combined_fitness_scores)[::-1]
        selected_indices = sorted_indices[:mu]
        population = [combined_population[i] for i in selected_indices]
        fitness_scores = [combined_fitness_scores[i] for i in selected_indices]
        rewards = [combined_rewards[i] for i in selected_indices]

        # 7. update the best solution found so far
        best_fitness_idx = np.argmax(fitness_scores)
        best_reward_idx = np.argmax(rewards)

        if fitness_scores[best_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[best_fitness_idx]
            best_params = population[best_fitness_idx].copy()

        if rewards[best_reward_idx] > best_reward:
            best_reward = rewards[best_reward_idx]

        # Save history
        best_fitness_history.append(best_fitness)
        average_fitness_history.append(np.mean(fitness_scores))
        best_reward_history.append(best_reward)
        average_reward_history.append(np.mean(rewards))

        print(f"Gen {generation+1}: Best Fitness={best_fitness:.2f}, Avg Fitness={np.mean(fitness_scores):.2f}, Best Reward={best_reward:.2f}, Avg Reward={np.mean(rewards):.2f}")

    return (
        best_params,
        best_fitness,
        best_fitness_history,
        average_fitness_history,
        best_reward_history,
        average_reward_history,
    )


# ---- CMA-ES ----
# from: https://algorithmafternoon.com/strategies/covariance_matrix_adaptation_evolution_strategy/
def cma_es(populationSize=POPULATION_SIZE, muRatio=0.5):

    best_fitness = -np.inf
    best_params = None
    best_reward = 0

    best_fitness_history = []
    average_fitness_history = []
    best_reward_history = []
    average_reward_history = []
    # Initialization

    # Initialize mean to random point in the search space
    mean = np.random.randn(sum(p.numel() for p in brain.parameters()))

    # Initialize covariance to identity matrix

    covariance = np.eye(len(mean))

    # Initialize step size to a suitable value in the search space bounds

    stepSize = 0.5 * np.std(mean)

    # Compute cumulationFactor and dampFactor based on muEffective and the dimensionality

    muEffective = int(muRatio * populationSize)

    dimensionality = len(mean)

    # Defaults from the source

    cumulationFactor = (muEffective + 2) / (dimensionality + muEffective + 5)

    dampFactor = 1 + 2 * \
        max(0, np.sqrt((muEffective - 1)/(dimensionality + 1)) - 1) + cumulationFactor

    # Small constant to add to the diagonal of the covariance matrix for numerical stability
    epsilon = 1e-8

    for generation in range(NUM_GENERATIONS):
        # Main Loop:
        # 1. Sample a population of populationSize candidate solutions from a multivariate normal distribution with mean and covariance * (stepSize^2)
        try:
            # Add regularization to the covariance matrix to prevent SVD errors
            regularized_covariance = covariance * \
                (stepSize**2) + epsilon * np.eye(dimensionality)
            population = np.random.multivariate_normal(
                mean, regularized_covariance, populationSize)
        except np.linalg.LinAlgError as e:
            print(
                f"Warning: LinAlgError during sampling: {e}. Resetting covariance and step size.")
            # Reset covariance and step size if sampling fails
            covariance = np.eye(dimensionality)
            # Avoid zero std dev
            stepSize = 0.5 * np.std(mean) if np.std(mean) > 1e-6 else 0.5
            regularized_covariance = covariance * \
                (stepSize**2) + epsilon * np.eye(dimensionality)
            population = np.random.multivariate_normal(
                mean, regularized_covariance, populationSize)

        # 2. Evaluate the fitness of each candidate solution using the objective function
        fitness_scores_list, rewards_list = evaluate_population_fitness([
            utils.get_param_as_weights(params, model=brain) for params in population
        ])
        # Convert lists to numpy arrays for advanced indexing
        fitness_scores = np.array(fitness_scores_list)
        rewards = np.array(rewards_list)

        # 3. Select the best muEffective solutions based on their rank in fitness scores

        sorted_indices = np.argsort(fitness_scores)[::-1]
        selected_indices = sorted_indices[:muEffective]
        selected_solutions = population[selected_indices]
        selected_fitness_scores = fitness_scores[selected_indices]
        selected_rewards = rewards[selected_indices]

        # 4. Update mean to the weighted average of the selected solutions

        mean = np.mean(
            selected_solutions, axis=0)

        # 5. Update covariance based on the selected solutions and the cumulative path of successful mutations

        # This is a simplified version of the covariance update step

        covariance = np.cov(selected_solutions, rowvar=False) 

        # 6. Update stepSize based on the cumulative path of successful mutations and the dampFactor

        stepSize *= np.exp((np.linalg.norm(mean) - 1) / dampFactor)

        # 7. Update the best solution found so far
        best_fitness_idx = np.argmax(fitness_scores)
        best_reward_idx = np.argmax(rewards)

        if fitness_scores[best_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[best_fitness_idx]
            best_params = population[best_fitness_idx].copy()

        if rewards[best_reward_idx] > best_reward:
            best_reward = rewards[best_reward_idx]

        # Save history
        best_fitness_history.append(best_fitness)
        average_fitness_history.append(np.mean(fitness_scores))
        best_reward_history.append(best_reward)
        average_reward_history.append(np.mean(rewards))

        # Print the best fitness and average fitness for the current generation
        print(f"Gen {generation+1}: Best Fitness={best_fitness:.2f}, Avg Fitness={np.mean(fitness_scores):.2f}, Best Reward={best_reward:.2f}, Avg Reward={np.mean(rewards):.2f}")

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


# --- GENETIC ALGORITHM FROM TASK 1 ---
def evolutionary_algorithm(elitism=ELITISM):

    population = []

    for _ in range(POPULATION_SIZE):
        param_vector = np.random.randn(
            sum(p.numel() for p in brain.parameters()))
        population.append(param_vector)

    fitness_scores, reward_scores = evaluate_population_fitness([
        utils.get_param_as_weights(params, model=brain) for params in population
    ])

    # initialize overall best tracking
    best_initial_fitness_idx = np.argmax(fitness_scores)
    best_initial_reward_idx = np.argmax(reward_scores)

    best_reward = reward_scores[best_initial_reward_idx]
    best_fitness = fitness_scores[best_initial_fitness_idx]
    best_params = population[best_initial_fitness_idx]

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
            parent1 = tournament_selection(
                population, fitness_scores, k=TOURNAMENT_SIZE)
            parent2 = tournament_selection(
                population, fitness_scores, k=TOURNAMENT_SIZE)

            # Change here to create different types of evolutionary algorithms
            # Apply crossover to produce offspring
            offspring = arithmetic_crossover(
                parent1, parent2, alpha=0.6)
            # Apply mutation
            offspring = gaussian_dist_mutation(weight_vector=offspring,
                                               MUTATION_RATE=MUTATION_RATE, sigma=0.3)

            # **************************************************************************************

            new_population.append(offspring)

        population = new_population
        fitness_scores, rewards = evaluate_population_fitness([
            utils.get_param_as_weights(params, model=brain) for params in population
        ])

        best_fitness_idx = np.argmax(fitness_scores)
        best_reward_idx = np.argmax(rewards)

        if fitness_scores[best_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[best_fitness_idx]
            best_params = population[best_fitness_idx].copy()

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
        best_params,
        best_fitness,
        best_fitness_history,
        average_fitness_history,
        best_reward_history,
        average_reward_history,
    )


# ---- DIFFERENTIAL EVOLUTION ----
def differential_evolution(pop_size=POPULATION_SIZE, scale=0.5, cr=0.5, mutant_selection="rand"):
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
            if mutant_selection == "rand":  # DE/rand/1
                idxs = list(range(pop_size))
                idxs.remove(i)
                a, b, c = random.sample(idxs, 3)
                mutant_vector = population[a] + \
                    scale * (population[b] - population[c])

            elif mutant_selection == "best":  # DE/best/1
                idxs = list(range(pop_size))
                idxs.remove(i)
                a, b = random.sample(idxs, 2)
                mutant_vector = population[best_fitness_idx] + \
                    scale * (population[a] - population[b])

            else:
                raise ValueError("Invalid mutant selection method.")

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
        "name": "(OPT)(2.1)DeRand1Bin32Neurons",
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
        ) = differential_evolution()
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
        utils.create_gif_brain(
            robot_structure=robot_structure,
            brain=brain,
            filename=os.path.join(run_folder, "best_robot.gif"),
            scenario=SCENARIO,
            steps=STEPS,
        )
