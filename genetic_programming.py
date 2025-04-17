import math
import random
from random import randint, random, uniform
# Adaptation of the Tiny-GP code available at https://github.com/moshesipper/tiny_gp/blob/master/tiny_gp.py
from random import random, randint, seed, sample
from statistics import mean
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------
# Function Definitions
# ---------------------------


def add(x, y):
    return x + y


add.arity = 2


def sub(x, y):
    return x - y


sub.arity = 2


def mul(x, y):
    return x * y


mul.arity = 2


def log_base(x, y):
    if x <= 0:
        return -1
    elif y <= 1:
        return -1
    else:
        return math.log(x, y)


log_base.arity = 2


def div(x, y):
    if y < 0.00001:
        return 1.0
    else:
        return x / y


div.arity = 2


def abs_f(x):
    return abs(x)


abs_f.arity = 1


def sin(x):
    return math.sin(x)


sin.arity = 1


def cos(x):
    return math.cos(x)


cos.arity = 1


def max_f(x, y):
    return max(x, y)


max_f.arity = 2


def min_f(x, y):
    return min(x, y)


min_f.arity = 2


# ---------------------------
# Terminals and Function List
# ---------------------------
ERC_MIN = -1.0
ERC_MAX = 1.0

TERMINALS = ["x", "y", 0, 1, 2, 3, 4, "ERC"]
FUNCTIONS = [add, sub, mul, div, sin, cos, abs_f]


# Implementation of a GP tree from PL4.


class GPTree:
    def __init__(self, mutation_rate, min_depth, max_depth, crossover_rate, data=None, children=None):
        self.data = data
        # Use a list for children; if not provided, initialize as empty.
        self.children = children if children is not None else []
        self.mutation_rate = mutation_rate
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.crossover_rate = crossover_rate

    def init_new_with_same_params(self):
        return GPTree(mutation_rate=self.mutation_rate, min_depth=self.min_depth,
                      crossover_rate=self.crossover_rate, max_depth=self.max_depth)

    def node_label(self):
        if self.data in FUNCTIONS:
            return self.data.__name__
        # Add check for float (ERC)
        elif isinstance(self.data, float):
            return f"{self.data:.3f}"  # Format the float ERC value
        else:
            return str(self.data)  # Handles 'x', 'y', and integers

    def print_tree(self, prefix=""):
        print(f"{prefix}{self.node_label()}")
        for child in self.children:
            child.print_tree(prefix + "   ")

    def compute_tree(self, x, y):
        # If the node holds a function, compute all its children first.
        if self.data in FUNCTIONS:
            args = [child.compute_tree(x, y) for child in self.children]
            return self.data(*args)
        elif self.data == 'x':
            return x
        elif self.data == 'y':
            return y
        else:
            return self.data

    def random_tree(self, grow, max_depth, depth=0):
        # Decide whether this node will be a function or a terminal.
        if depth < self.min_depth or (depth < max_depth and not grow):
            # Choose a function
            self.data = FUNCTIONS[randint(0, len(FUNCTIONS) - 1)]
        elif depth >= max_depth:
            # Must be a terminal
            terminal_choice = TERMINALS[randint(0, len(TERMINALS) - 1)]
            # If ERC is chosen, replace data with a random float, otherwise use the terminal directly
            if terminal_choice == "ERC":
                # Generate ERC value, e.g., between -1 and 1
                self.data = uniform(ERC_MIN, ERC_MAX)
            else:
                self.data = terminal_choice
        else:  # Intermediate depth, can be function or terminal
            if random() > 0.5:  # Choose terminal
                terminal_choice = TERMINALS[randint(0, len(TERMINALS) - 1)]
                if terminal_choice == "ERC":
                    self.data = uniform(ERC_MIN, ERC_MAX)
                else:
                    self.data = terminal_choice
            else:  # Choose function
                self.data = FUNCTIONS[randint(0, len(FUNCTIONS) - 1)]

        # If the node is a function, create children
        if self.data in FUNCTIONS:
            self.children = []
            for _ in range(self.data.arity):
                child = self.init_new_with_same_params()
                child.random_tree(grow, max_depth, depth + 1)
                self.children.append(child)

    def mutation(self):
        if random() < self.mutation_rate:
            # Replace this subtree with a new random tree (of limited depth)
            self.random_tree(grow=True, max_depth=2)
        else:
            for child in self.children:
                child.mutation()

    def size(self):
        if self.data not in FUNCTIONS:
            return 1
        return 1 + sum(child.size() for child in self.children)

    def build_subtree(self):
        t = self.init_new_with_same_params()
        t.data = self.data
        t.children = [child.build_subtree() for child in self.children]
        return t

    def scan_tree(self, count, second):
        count[0] -= 1
        if count[0] <= 1:
            if not second:
                return self.build_subtree()
            else:
                self.data = second.data
                self.children = second.children
        else:
            ret = None
            for child in self.children:
                if count[0] > 1:
                    ret = child.scan_tree(count, second)
            return ret

    def crossover(self, other):
        if random() < self.crossover_rate:
            second = other.scan_tree([randint(1, other.size())], None)
            self.scan_tree([randint(1, self.size())], second)

    def decode_to_robot(self, robot_shape: tuple):
        """Decodes individual to robot using expression tree."""
        robot = np.zeros(robot_shape, dtype=int)

        for i in range(robot_shape[0]):
            for j in range(robot_shape[1]):
                value = self.compute_tree(x=i, y=j)
                # voxel values are in range [0, 4]
                voxel_value = int(value) % 5
                robot[i, j] = voxel_value

        return robot

# Generate an individual: method full or grow
# Field Guide Genetic Programming: algorithm 2.1, pg.14


def ramped_half_and_half_init_pop(max_depth, pop_size, mutation_rate, min_depth, crossover_rate):

    pop = []
    for md in range(3, max_depth + 1):
        for i in range(int(pop_size/6)):
            t = GPTree(mutation_rate=mutation_rate, min_depth=min_depth,
                       crossover_rate=crossover_rate, max_depth=max_depth)
            t.random_tree(grow=True, max_depth=md)  # grow
            pop.append(t)
        for i in range(int(pop_size/6)):
            t = GPTree(mutation_rate=mutation_rate, min_depth=min_depth,
                       crossover_rate=crossover_rate, max_depth=max_depth)
            t.random_tree(grow=False, max_depth=md)  # full
            pop.append(t)
    # If we are missing some inviduals, just fill the rest of population with random trees with the full method
    print(len(pop))
    if len(pop) < pop_size:
        for i in range(pop_size - len(pop)):
            t = GPTree(mutation_rate=mutation_rate, min_depth=min_depth,
                       crossover_rate=crossover_rate, max_depth=max_depth)
            t.random_tree(grow=False, max_depth=md)  # full
            pop.append(t)
    elif len(pop) > pop_size:
        pop = pop[:pop_size]

    print(len(pop))
    return pop
