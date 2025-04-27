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
from neural_controller import *
import utils 

# ---- PARAMETERS ----
NUM_GENERATIONS = 25  # Number of generations to evolve
STEPS = 500

# (1+1) Evolution Strategy Params
SIGMA = 0.1
ALPHA = 0.25

#POPULATION_SIZE = 20  # Number of robots per generation
#MUTATION_RATE = 0.15  # Probability of mutation

#TOURNAMENT_SIZE = 2  # Number of individuals in the tournament for selection
#ELITISM = True  # Whether to use elitism or not
#ELITE_SIZE = 1  # Number of elite individuals to carry over to the next generation

# -- Sim ---
MULTIPROCESSING = False  # Whether to use multiprocessing or not

# ---- Fixed Robot Structure ----
robot_structure = np.array([ 
[1,3,1,0,0],
[4,1,3,2,2],
[3,4,4,4,4],
[3,0,0,3,2],
[0,0,0,0,2]
])

# ---- TESTING SETTINGS ----
SCENARIO = "DownStepper-v0"

SCENARIOS = [
    "DownStepper-v0",
    "ObstacleTraverser-v0",
]  




connectivity = get_full_connectivity(robot_structure)
env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
sim = env.sim
input_size = env.observation_space.shape[0]  # Observation size
output_size = env.action_space.shape[0]  # Action size

brain = NeuralController(input_size, output_size)

# ---- FITNESS FUNCTION ---- 

def evaluate_fitness(weights, view=False):
    set_weights(brain, weights)  # Load weights into the network

    try:
        env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
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
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
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
def get_param_as_weights(param_vector):
    shapes = [p.shape for p in brain.parameters()]
    new_weights = []
    idx = 0
    for shape in shapes:
        size = np.prod(shape)
        new_weights.append(param_vector[idx:idx+size].reshape(shape))
        idx += size
    return new_weights

def get_flat_params(brain):
    return np.concatenate([p.detach().numpy().flatten() for p in brain.parameters()])

def setup_run(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

# ---- RANDOM SEARCH ALGORITHM ----
def random_search_algorithm():
    best_fitness = -np.inf
    best_weights = None

    best_fitness_history = []
    average_fitness_history = []
    best_reward_history = []
    average_reward_history = []

    for generation in range(NUM_GENERATIONS):
        random_weights = [np.random.randn(*param.shape) for param in brain.parameters()]

        fitness, reward = evaluate_fitness(random_weights)

        if fitness > best_fitness:
            best_fitness = fitness
            best_weights = random_weights
            best_reward = reward
            print(f"Gen {generation+1}: New best fitness = {best_fitness:.2f}")
        else:
            print(f"Gen {generation+1}: No improvement (best={best_fitness:.2f})")

        best_fitness_history.append(best_fitness)
        average_fitness_history.append(best_fitness)
        best_reward_history.append(best_reward if 'best_reward' in locals() else reward)
        average_reward_history.append(best_reward if 'best_reward' in locals() else reward)

    # Set the best weights found
    set_weights(brain, best_weights)

    return (
        best_weights,
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

    best_fitness, best_reward = evaluate_fitness(get_param_as_weights(best_params))
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

        offspring_fitness, offspring_reward = evaluate_fitness(get_param_as_weights(offspring_params))

        if offspring_fitness > best_fitness:
            best_fitness = offspring_fitness
            best_params = offspring_params.copy()
            best_reward = offspring_reward
            print(f"Gen {generation+1}: New best fitness = {best_fitness:.2f}")
        else:
            print(f"Gen {generation+1}: No improvement (best={best_fitness:.2f})")

        best_fitness_history.append(best_fitness)
        average_fitness_history.append(best_fitness)
        best_reward_history.append(best_reward)
        average_reward_history.append(best_reward)

    set_weights(brain, get_param_as_weights(best_params))

    return (
        best_params,
        best_fitness,
        best_fitness_history,
        average_fitness_history,
        best_reward_history,
        average_reward_history,
    )

# ---- (μ + λ) EVOLUTION STRATEGY ----

def mu_plus_lambda_es(mu=5, lamb=5):
    # Evaluate initial mu parents
    parents = []
    for _ in range(mu):
        param_vector = get_flat_params(brain)
        fitness, reward = evaluate_fitness(get_param_as_weights(param_vector))
        parents.append({'params': param_vector, 'fitness': fitness, 'reward': reward})
    
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
            
            child_fitness, child_reward = evaluate_fitness(get_param_as_weights(child_params))
            offspring.append({'params': child_params, 'fitness': child_fitness, 'reward': child_reward})
        
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
            print(f"Gen {generation+1}: No improvement (best={best_fitness:.2f})")
        
        best_fitness_history.append(best_fitness)
        average_fitness_history.append(np.mean([p['fitness'] for p in parents]))
        best_reward_history.append(best_reward)
        average_reward_history.append(np.mean([p['reward'] for p in parents]))
    
    # Set best found weights in brain
    set_weights(brain, get_param_as_weights(best_params))
    
    return (
        best_params,
        best_fitness,
        best_fitness_history,
        average_fitness_history,
        best_reward_history,
        average_reward_history,
    )



# ------ EXPERIMENTS ----------

# Choose which approach to run:
if __name__ == "__main__":

    multiprocessing.freeze_support()  # For Windows compatibility

    RUN_SEEDS = [6363, 9374, 2003, 198, 2782]
    results_folder = "results/task2/"

    experiment_info = {
        # ***********************************************************************************
        # Change this to the name of the experiment. Will be used in the folder name.
        "name": "(1.1)(1+1) Evolution Strategy",
        # ***********************************************************************************
        "repetitions": len(RUN_SEEDS),
        "num_generations": NUM_GENERATIONS,
        "sigma": SIGMA,
        "alpha": ALPHA,
        "steps": STEPS,
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

        (
            best_controller_params,
            best_fitness,
            best_fitness_history,
            average_fitness_history,
            best_reward_history,
            average_reward_history,
        ) = one_plus_one_es()

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

        brain = NeuralController(input_size, output_size)
        set_weights(brain, best_controller_params)      

        # Create a GIF of the best performing controller
        utils.create_gif_nn(
            get_param_as_weights(np.array(best_controller_params)),
            brain,
            robot_structure,
            scenario=SCENARIO,
            steps=STEPS,
            filename=f"{run_folder}/best_robot.gif",
            #controller=brain,
        )