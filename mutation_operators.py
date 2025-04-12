import random

import numpy as np

# ---- VOXEL TYPES ----
VOXEL_TYPES = [0, 1, 2, 3, 4]


def flip_mutation(robot_structure, MUTATION_RATE):
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
