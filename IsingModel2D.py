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
        self.spins_amount = 10
        self.iterations = 500 * (self.spins_amount ** 2)
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
        E = 0
        for i in range(self.spins_amount):
            for j in range(self.spins_amount):
                neighbours = self.spins[(i + 1) % self.spins_amount, j] + self.spins[i, (j + 1) % self.spins_amount] + \
                             self.spins[(i - 1) % self.spins_amount, j] + self.spins[i, (j - 1) % self.spins_amount]

                E += -self.interaction_energy * neighbours * self.spins[i, j] / 4

        return E - self.B * np.sum(self.spins)

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

    def metropolis(self):
        start_time = time.time()
        energies = []
        counter = 0
        for i in range(self.iterations):
            current_energy = self.calculate_internal_energy()
            n, k = self.sweep_spin()
            probe_energy = self.calculate_internal_energy()
            if probe_energy <= current_energy:
                # logging.info(f"State accepted - {i} iteration")
                energies.append(probe_energy)
                counter += 1
            else:
                p = self.probability(
                    math.exp(-1. * (probe_energy - current_energy) / (BOLTSMAN_CONST * self.temperature)))
                if p:
                    # logging.info(f"State accepted - {i} iteration")
                    energies.append(probe_energy)
                    counter += 1
                else:
                    self.spins[n][k] *= -1
                    # logging.info(f"State reverted - {i} iteration")
                    energies.append(current_energy)

        energy = np.array(energies)

        finish_time = time.time()
        logging.info(f'Runtime is {finish_time - start_time} seconds.')
        return energy

    def calculate_capacity_error(self, energy, anneal):
        nominator = np.var(energy[anneal:] ** 2) + 4 * np.var(energy[anneal:]) * np.var(energy[anneal:])
        return (np.sqrt(nominator)) / ((self.spins_amount * self.temperature) ** 2)

    def run(self):
        temperatures = np.linspace(self.temperature, 5, 50)
        anneal = 9 * self.iterations // 10
        energies = []
        heat_capacities = []
        magnetizations = []
        energies_error = []
        capacities_error = []
        magnetizations_error = []
        for index, each in enumerate(temperatures):
            self.temperature = each
            self.initiate_start_configuration()
            energy = self.metropolis()
            mean_energy = np.mean(energy[anneal:])
            mean_heat_capacity = (np.mean(energy[anneal:] ** 2) - mean_energy ** 2) / (
                (self.spins_amount * self.temperature) ** 2)
            mean_magnetization = np.mean(self.spins)
            logging.info(f'Temperature: {self.temperature} \n'
                         f'Mean energy: {mean_energy}; \n'
                         f'Mean heat capacity: {mean_heat_capacity} \n'
                         f'Mean magnetization: {mean_magnetization} \n')
            energies.append(mean_energy / (self.spins_amount ** 2))
            heat_capacities.append(mean_heat_capacity)
            magnetizations.append(mean_magnetization/ (self.spins_amount ** 2))
            energies_error.append(np.std(energy[anneal:]) / (self.spins_amount ** 2))
            tmp_capacity_error = self.calculate_capacity_error(energy, anneal)
            capacities_error.append(tmp_capacity_error)
            magnetizations_error.append(np.std(self.spins)/(self.spins_amount ** 2))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(temperatures, energies, yerr=energies_error, fmt='o-', ecolor='green')
        ax.set_xlabel(r'Temperature')
        ax.set_ylabel(r'Energy/Spins')
        ax.grid()
        ax.set_title(r"Energies comparison")
        plt.show()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(temperatures, heat_capacities, yerr=capacities_error, fmt='o-', ecolor='green')
        ax.set_xlabel(r'Temperature')
        ax.set_ylabel(r'Heat capacity')
        ax.grid()
        ax.set_title(r"Heat capacity changing due temperature")
        plt.show()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(temperatures, magnetizations, yerr=magnetizations_error, fmt='o-', ecolor='green')
        ax.set_xlabel(r'Temperature')
        ax.set_ylabel(r'Magnetization')
        ax.grid()
        ax.set_title(r"Magnetization due temperature")
        plt.show()



if __name__ == '__main__':
    lattice = Lattice2D()
    lattice.run()
