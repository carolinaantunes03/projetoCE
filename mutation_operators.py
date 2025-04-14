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


def swap_mutation(robot_structure, MUTATION_RATE):
    # no mutation
    if random.random() > MUTATION_RATE:
        return robot_structure

    vector = robot_structure.flatten()

    vox_1 = 0
    vox_2 = 0

    # ensure swap is between different voxels
    while vox_1 == vox_2:
        vox_1 = random.randint(0, len(vector)-1)
        vox_2 = random.randint(0, len(vector)-1)

    # swap the voxels
    vector[vox_1], vector[vox_2] = vector[vox_2], vector[vox_1]

    # reshape the vector back to the original shape
    mutated_robot = vector.reshape(robot_structure.shape)

    return mutated_robot
