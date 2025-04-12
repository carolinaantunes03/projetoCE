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
