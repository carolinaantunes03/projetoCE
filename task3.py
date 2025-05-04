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


# ---- PARAMETERS ----
NUM_GENERATIONS = 5
POPULATION_SIZE = 20  
STEPS = 500

# ---- STRUCTURE PARAMETERS ----
GRID_SIZE = (5, 5)
VOXEL_TYPES = [0, 1, 2, 3, 4]
MUTATION_RATE = 0.15

# ---- CONTROLLER PARAMETERS ----
SIGMA = 0.1

# ---- SELECTION PARAMETERS ----
TOURNAMENT_SIZE = 4
ELITISM = True
ELITE_SIZE = 1


MULTIPROCESSING = False

# ---- TESTING SETTINGS ----
SCENARIO = "GapJumper-v0"

SCENARIOS = [
    "GapJumper-v0",
    "CaveCrawler-v0",
]




# ---- GENOTYPE REPRESENTATION ----
class StructureIndividual:
    def __init__(self, structure=None):
        while True:
            if structure is None:
                candidate = self.random_structure()
            else:
                candidate = structure
            if is_connected(candidate):
                self.structure = candidate
                break
        self.fitness = 0.0
        self.size = np.prod(self.structure.shape)  # Store the size of the structure

    def random_structure(self):
        return np.random.choice(VOXEL_TYPES, size=GRID_SIZE)

    def mutate(self, mutation_rate, voxel_types):
        self.structure = flip_mutation(self.structure , mutation_rate, voxel_types)
    
    def crossover(self, other):
        child_grid = one_point_crossover(self.structure, other.structure)
        if is_connected(child_grid):
            return StructureIndividual(child_grid)
        

    
    
class ControllerIndividual:
    def __init__(self, input_size, output_size):
        self.model = NeuralController(input_size, output_size)
        initialize_weights(self.model)
        self.controller_params = self.flatten_weights()

    def flatten_weights(self):
        return np.concatenate([p.detach().cpu().numpy().flatten() for p in self.model.parameters()])

    def set_weights(self, weights):
        with torch.no_grad():
            i = 0
            for param in self.model.parameters():
                n_params = param.numel()
                param.copy_(torch.tensor(weights[i:i+n_params].reshape(param.shape), dtype=torch.float32))
                i += n_params

    def forward(self, x):
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32)
            return self.model(x).numpy()
        
    def mutate(self):
        self.controller_params = gaussian_dist_mutation(self.controller_params,MUTATION_RATE,sigma=SIGMA)
        self.set_weights(self.controller_params)


    def crossover(self, other):
        child_params = arithmetic_crossover(self.controller_params, other.controller_params)
        child = ControllerIndividual(input_size=len(child_params), output_size=8)
        child.set_weights(child_params)
        return child


## here we can add the variation mechanism for evolution

# combination of both classes to create a full individual
class Individual:
    def __init__(self, input_size, output_size, structure=None):
        self.structure = StructureIndividual(structure)
        self.controller = ControllerIndividual(input_size, output_size)
        self.fitness = 0.0

    def mutate(self):
        self.structure.mutate(MUTATION_RATE, VOXEL_TYPES)
        self.controller.mutate()

    def crossover(self, other):
        child_structure = self.structure.crossover(other.structure)
        child_controller = self.controller.crossover(other.controller)
        child = Individual(input_size=len(self.controller.controller_params), 
                           output_size=8,  # adapt if dynamic
                           structure=child_structure.structure)
        child.controller = child_controller
        return child

# ---- FITNESS FUNCTION ----


def evaluate_fitness(weights, structure, connectivity, brain, view=False):
    if not is_connected(structure):
        return -15.0, 0

    try:
        utils.set_weights(brain, weights)  # Load weights into the neural controller

        connectivity = get_full_connectivity(structure)

        env = gym.make(
            SCENARIO,
            max_episode_steps=STEPS,
            body=structure,
            connections=connectivity,
        )
        sim = env.sim
        viewer = EvoViewer(sim)
        viewer.track_objects("robot")

        state = env.reset()[0]
        t_reward = 0
        t_velocity_x = 0.0
        t_velocity_y = 0.0

        for t in range(STEPS):
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

            if terminated or truncated:
                break

        viewer.close()
        env.close()

        total_time = sim.get_time()
        avg_velocity_x = t_velocity_x / (t + 1)
        avg_velocity_y = t_velocity_y / (t + 1)

        # Reward distance/time but penalize long runtimes slightly
        time_penalty_factor = 0.05
        fitness_val = t_reward / (1 + time_penalty_factor * total_time)

        return fitness_val, t_reward

    except (ValueError, IndexError) as e:
        return -15.0, 0  # Penalize invalid individuals



def evaluate_population_fitness(population, input_size, output_size):
    for ind in population:
        structure = ind.structure.structure
        controller = ind.controller
        connectivity = get_full_connectivity(structure)

        # Create new brain and load weights
        brain = NeuralController(input_size, output_size)
        brain.load_flat_weights(controller.controller_params)

        try:
            fitness, reward = evaluate_fitness(controller.controller_params, structure, connectivity, brain)
        except Exception as e:
            print(f"Error evaluating individual: {e}")
            fitness, reward = -15.0, 0

        ind.fitness = fitness
        ind.reward = reward


# ---- PAIRING STRATEGIES ----
def pairing (population, strategy="random"):
    if strategy == "random":
        pairings = []
        shuffled = random.sample(population, len(population))
        for i in range(0, len(shuffled) - 1, 2):
            pairings.append((shuffled[i], shuffled[i+1]))
        return pairings
    
    elif strategy == "best_so_far":
        sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)
        best = sorted_pop[0]
        for i in range(1, len(sorted_pop)):
            yield best, sorted_pop[i]

    elif strategy == "round_robin":
        for i in range(0, len(population), 2):
            yield population[i], population[(i + 1) % len(population)]
    else:
        raise ValueError(f"Unknown pairing strategy: {strategy}")

# ---- POPULATION INITIALIZATION ----

def initialize_population(pop_size, input_size, output_size):
    population = []
    for _ in range(pop_size):
        while True:
            try:
                structure = generate_valid_robot()
                if not is_connected(structure):
                    continue
                controller = ControllerIndividual(input_size, output_size)
                individual = Individual(input_size, output_size, structure)
                population.append(individual)
                break
            except Exception:
                continue
    return population

def generate_valid_robot():
    while True:
        # Generate a random structure (a grid of random voxel types)
        structure = np.random.choice(VOXEL_TYPES, size=GRID_SIZE)
        
        # Check if the structure is connected
        if is_connected(structure):
            return structure



# ---- EVOLUTIONARY ALGORITHM ----
def run_evolution():

    # ver se queremos deixar assim ou fazer de forma diferente
    input_size = 14 + 2 * np.prod(GRID_SIZE)
    output_size = 8
    '''
    # Use a temporary robot to infer input/output sizes
    dummy_structure = generate_valid_robot()
    connectivity = get_full_connectivity(dummy_structure)
    env = gym.make(SCENARIO, max_episode_steps=STEPS, body=dummy_structure, connections=connectivity)
    
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    
    env.close()'''

    population = initialize_population(POPULATION_SIZE, input_size, output_size)

    best_fitness_history = []
    average_fitness_history = []
    best_reward_history = []
    average_reward_history = []

    best_fitness = -float('inf')  # Initialize best fitness
    best_structure = None
    best_controller_params = None

    for gen in range(NUM_GENERATIONS):
        print(f"\n--- Generation {gen} ---")

        for ind in population:
            structure = ind.structure.structure
            controller = ind.controller

            # Initialize the neural controller (brain) for each individual
            connectivity = get_full_connectivity(structure)
            env = gym.make(SCENARIO, max_episode_steps=STEPS, body=structure, connections=connectivity)
            sim = env.sim
            input_size = env.observation_space.shape[0]
            output_size = env.action_space.shape[0]

            # Initialize the brain (neural controller) here
            brain = NeuralController(input_size, output_size)

            if not is_connected(structure):
                print("Disconnected robot detected, assigning low fitness.")
                ind.fitness, ind.reward = -15.0, 0
                continue

            # Now you pass the brain object to evaluate_fitness
            try:
                ind.fitness, ind.reward = evaluate_fitness(controller.controller_params, structure, connectivity, brain)

            except Exception as e:
                print(f"Error evaluating fitness: {e}")
                ind.fitness, ind.reward = -15.0, 0

        fitnesses = [ind.fitness for ind in population]
        rewards = [ind.reward for ind in population]

        best_fitness_gen = max(fitnesses)
        average_fitness = np.mean(fitnesses)
        best_reward = max(rewards)
        average_reward = np.mean(rewards)

        if best_fitness_gen > best_fitness:
            best_fitness = best_fitness_gen
            best_structure = population[fitnesses.index(best_fitness)]
            best_controller_params = best_structure.controller.controller_params

        print(f"Gen {gen+1}: Best Fitness={best_fitness:.2f}, Avg Fitness={average_fitness:.2f}, Best Reward={best_reward:.2f}, Avg Reward={average_reward:.2f}")

        best_fitness_history.append(best_fitness)
        average_fitness_history.append(average_fitness)
        best_reward_history.append(best_reward)
        average_reward_history.append(average_reward)

        population.sort(key=lambda ind: ind.fitness, reverse=True)
        next_gen = population[:ELITE_SIZE] if ELITISM else []

        pairings = pairing(population, strategy="random")
        for parent1, parent2 in pairings:
            child = parent1.crossover(parent2)
            child.mutate()
            next_gen.append(child)
            if len(next_gen) >= POPULATION_SIZE:
                break

        while len(next_gen) < POPULATION_SIZE:
            next_gen.append(Individual(input_size, output_size))

        population = next_gen

    # After the evolution is complete, return all the required values
    print(f"Best fitness found: {best_fitness:.2f}")

    return (
        best_controller_params,      # Best controller parameters
        best_fitness,                # Best fitness found
        best_fitness_history,        # History of best fitness over generations
        average_fitness_history,     # Average fitness history
        best_reward_history,         # Best reward history
        average_reward_history,      # Average reward history
        best_structure,             
    )



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
        #"alpha": ALPHA,
        "steps": STEPS,
        "population_size": POPULATION_SIZE,
        "mutation_rate": MUTATION_RATE,
        #"mu": MU,
        #"lamb": LAMBDA,
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
            best_structure,
        ) = run_evolution()

        # Recreate env from best structure to ensure consistent input/output sizes
        structure = best_structure.structure.structure
        connectivity = get_full_connectivity(structure)

        env = gym.make(SCENARIO, max_episode_steps=STEPS, body=structure, connections=connectivity)

        input_size = env.observation_space.shape[0]
        output_size = env.action_space.shape[0]

        # Now use these sizes to rebuild the brain correctly
        #brain = NeuralController(input_size, output_size)
        
        #brain.load_flat_weights(best_controller_params)
        brain = best_structure.controller.model  # Already initialized with correct sizes


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


        utils.create_gif_brain(
            robot_structure=best_structure,
            brain=brain,
            filename=os.path.join(run_folder, "best_robot.gif"),
            scenario=SCENARIO,
            steps=STEPS,
        )
