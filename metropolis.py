from numpy.random import rand
import random
import numpy as np
from numba import njit


@njit
def mcmove(lattice_size, config, beta):
    """Monte Carlo move using Metropolis algorithm """
    for i in range(lattice_size):
        for j in range(lattice_size):
            # select random spin from NxN system
            a = random.randint(0, lattice_size - 1)
            b = random.randint(0, lattice_size - 1)
            s = config[a, b]
            nb = config[(a + 1) % lattice_size, b] + config[a, (b + 1) % lattice_size] + config[
                (a - 1) % lattice_size, b] + config[a, (b - 1) % lattice_size]
            cost = 2 * s * nb
            if cost < 0:
                s *= -1
            elif rand() < np.exp(-cost * beta):
                s *= -1
            config[a, b] = s
    return config