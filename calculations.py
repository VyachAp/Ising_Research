import numpy as np
from numba import njit


@njit
def calcEnergy(lattice_size, config):
    """Energy of a given configuration"""
    energy = 0
    N = lattice_size
    for i in range(len(config)):
        for j in range(len(config)):
            S = config[i, j]
            nb = config[(i + 1) % N, j] + config[i, (j + 1) % N] + config[(i - 1) % N, j] + config[i, (j - 1) % N]
            energy += -nb * S
    return energy / 4.


@njit
def calc_spontaneous_mag(self, temp):
    result = []
    for each in temp:
        magnetization = np.power(1 - 1. / np.power(np.sinh((2 * self.interaction_energy) / each), 4), 1 / 8)
        result.append(magnetization ** 2)
    return result


@njit
def calcMag(config):
    """Magnetization of a given configuration"""
    mag = np.sum(config)
    return mag


@njit
def calculate_capacity_error(lattice_size, energy, anneal, temperature):
    nominator = np.var(energy[anneal:] ** 2) + 4 * np.var(energy[anneal:]) * np.var(energy[anneal:])
    return (np.sqrt(nominator)) / ((lattice_size * temperature) ** 2)
