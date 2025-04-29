import random

import numpy as np


def one_point_crossover(parent1, parent2):
    # (5,5) -> vector
    parent1_vec = parent1.flatten()
    parent2_vec = parent2.flatten()

    crossover_point = random.randint(0, len(parent1_vec) - 1)

    offspring = np.concatenate(
        (parent1_vec[:crossover_point], parent2_vec[crossover_point:])
    )

    # vector -> (5,5)
    offspring = offspring.reshape(parent1.shape)

    return offspring


def uniform_crossover(parent1, parent2):
    # (5,5) -> vector
    parent1_vec = parent1.flatten()
    parent2_vec = parent2.flatten()

    offspring = np.zeros_like(parent1_vec)

    for i in range(len(parent1_vec)):
        if random.random() < 0.5:
            offspring[i] = parent1_vec[i]
        else:
            offspring[i] = parent2_vec[i]

    # vector -> (5,5)
    offspring = offspring.reshape(parent1.shape)

    return offspring


def binomial_crossover(vector1, vector2, cr):
    # Create a mask of the same shape as the input vectors
    mask = np.random.rand(*vector1.shape) <= cr

    # Create the offspring by combining elements from both parents based on the mask
    offspring = np.where(mask, vector1, vector2)

    return offspring
