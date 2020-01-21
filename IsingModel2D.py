import matplotlib.pyplot as plt
import numpy as np
import math
import random
import logging
import time

logging.basicConfig(level=logging.INFO)

BOLTSMAN_CONST = 1


class Lattice2D:
    def __init__(self):
        self.spins_amount = 20
        self.iterations = 50 * (self.spins_amount ** 2)
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
                E += self.single_spin_energy(i, j)

        return E - self.B * np.sum(self.spins)

    def single_spin_energy(self, i, j):
        return -1.0 * self.spins[i, j] * (self.spins[(i + 1) % self.spins_amount, j] +
                                          self.spins[(i - 1 + self.spins_amount) % self.spins_amount, j] +
                                          self.spins[i, (j + 1) % self.spins_amount] +
                                          self.spins[i, (j - 1 + self.spins_amount) % self.spins_amount])

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

    def wolff_algorithm(self):
        indexes = np.random.randint(low=0, high=self.spins_amount - 1, size=(2, 1))
        i = indexes[0, 0]
        j = indexes[1, 0]
        root = (i, j)
        self.build_cluster(root, self.spins[root])

    def build_cluster(self, root, spin_value):

        self.spins[root] *= -1
        boundary_cond = self.spins_amount
        neighbors = [(0, 0), (0, 0), (0, 0), (0, 0)]
        (i, j) = root

        if i == boundary_cond - 1:
            neighbors[0] = (0, j)
        else:
            neighbors[0] = (i + 1, j)
        if i == 0:
            neighbors[1] = (boundary_cond - 1, j)
        else:
            neighbors[1] = (i - 1, j)
        if j == boundary_cond - 1:
            neighbors[2] = (i, 0)
        else:
            neighbors[2] = (i, j + 1)
        if j == 0:
            neighbors[3] = (i, boundary_cond - 1)
        else:
            neighbors[3] = (i, j - 1)

        for next_site in neighbors:
            if self.spins[next_site] == spin_value:
                if np.random.random() < 1 - np.exp(-2.0 / self.temperature):
                    self.build_cluster(next_site, spin_value)

    def run_wolff(self):
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
            tmp_energy = []
            for each in range(self.iterations):
                self.wolff_algorithm()
                tmp_energy.append(self.calculate_internal_energy())
            energy = np.array(tmp_energy)
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
            magnetizations.append(mean_magnetization / (self.spins_amount ** 2))
            energies_error.append(np.std(energy[anneal:]) / (self.spins_amount ** 2))
            tmp_capacity_error = self.calculate_capacity_error(energy, anneal)
            capacities_error.append(tmp_capacity_error)
            magnetizations_error.append(np.std(self.spins) / (self.spins_amount ** 2))

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
        ax.errorbar(temperatures, np.power(magnetizations, 2), yerr=magnetizations_error, fmt='o-', ecolor='green')
        ax.set_xlabel(r'Temperature')
        ax.set_ylabel(r'Squared magnetization')
        ax.grid()
        ax.set_title(r"Squared magnetization due temperature")
        plt.show()

    def run_metropolis(self):
        temperatures = np.linspace(self.temperature, 5, 50)
        anneal = 9 * self.iterations // 10
        energies = np.zeros(len(temperatures))
        energies_error = np.zeros(len(temperatures))
        heat_capacities = np.zeros(len(temperatures))
        heat_capacities_error = np.zeros(len(temperatures))
        magnetizations = np.zeros(len(temperatures))
        magnetizations_error = np.zeros(len(temperatures))
        for index, each in enumerate(temperatures):
            self.temperature = each
            self.initiate_start_configuration()
            energy = self.metropolis()
            energies[index] = np.mean(energy[anneal:])
            energies_error[index] = np.std(energy[anneal:]) / (self.spins_amount ** 2)
            heat_capacities[index] = (np.mean(np.power(energy[anneal:], 2)) - energies[index] ** 2) / \
                                     (self.spins_amount * self.spins_amount * (each ** 2))
            heat_capacities_error[index] = \
                np.sqrt(4 * np.var(energy[anneal:]) * np.var(energy[anneal:]) + np.var(energy[anneal:]) ** 2) / \
                (self.spins_amount * self.spins_amount * (each ** 2))
            magnetizations[index] = np.mean(self.spins) / (self.spins_amount ** 2)
            magnetizations_error[index] = np.std(self.spins) / (self.spins_amount ** 2)
            logging.info(f'Temperature: {self.temperature} \n'
                         f'Mean energy: {energies[index]}; \n'
                         f'Mean heat capacity: {heat_capacities[index]} \n'
                         f'Mean magnetization: {magnetizations[index]} \n')

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(temperatures, energies, yerr=energies_error, fmt='o-', ecolor='green')
        ax.set_xlabel(r'Temperature')
        ax.set_ylabel(r'Energy/Spins')
        ax.grid()
        ax.set_title(r"Energies comparison")
        plt.show()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(temperatures, heat_capacities, yerr=heat_capacities_error, fmt='o-', ecolor='green')
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
    lattice.run_metropolis()
