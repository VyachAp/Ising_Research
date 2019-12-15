import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
import math
import random
import logging
import time

logging.basicConfig(level=logging.INFO)

BOLTSMAN_CONST = 1


class Lattice2D:
    def __init__(self):
        self.spins_amount = 250
        self.iterations = 500 * self.spins_amount
        self.spins = np.zeros((self.spins_amount, self.spins_amount), dtype=int)
        self.interaction_energy = 1  # J
        self.mu = 0.5
        self.B = 0
        self.temperature = 0.1
        self.capacities = []

    @staticmethod
    def probability(r):
        x = random.uniform(0, 1)
        if x <= r:
            return 1
        return 0

    # def clear_configuration(self):
    #     del self.spins[:]

    def initiate_start_configuration(self):
        for i in range(self.spins_amount):
            for j in range(self.spins_amount):
                g = random.uniform(0, 1)
                if g >= 0.5:
                    self.spins[i][j] = 1
                else:
                    self.spins[i][j] = -1

    def calculate_internal_energy(self):
        energy = 0
        squared_energy = 0
        for i in range(self.spins_amount):
            for j in range(self.spins_amount):
                spin_energy = self.single_spin_energy(i, j)
                energy += spin_energy
                squared_energy += spin_energy**2

        internal_energy = (1./self.spins_amount**2)*energy
        squared_internal_energy = (1./self.spins_amount**2)*squared_energy

        return internal_energy, squared_internal_energy

    def single_spin_energy(self, i, j):
        return -2 * self.spins[i, j] * (self.spins[i - 1, j]
                                        + self.spins[i + 1, j]
                                        + self.spins[i, j - 1]
                                        + self.spins[i, j + 1])

    def sweep_spin(self):
        n = random.randint(0, self.spins_amount - 1)
        k = random.randint(0, self.spins_amount - 1)
        self.spins[n][k] *= -1
        return n, k  # return ordered number of reverted spin

