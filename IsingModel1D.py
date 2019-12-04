import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
import math
import random
import logging
import time

logging.basicConfig(level=logging.INFO)

BOLTSMAN_CONST = 1


class Lattice1D:
    def __init__(self):
        self.spins_amount = 500
        self.iterations = 500 * self.spins_amount
        self.spins = np.zeros(self.spins_amount)
        self.interaction_energy = 1  # J
        self.mu = 0.5
        self.B = 0
        self.temperature = 0.1

    @staticmethod
    def probability(r):
        x = random.uniform(0, 1)
        if x <= r:
            return 1
        return 0

    # def clear_configuration(self):
    #     del self.spins[:]

    def initiate_start_configuration(self):
        for each in range(self.spins_amount):
            g = random.uniform(0, 1)
            if g >= 0.5:
                self.spins[each] = 1
            else:
                self.spins[each] = -1

    def calculate_energy(self):
        firstTerm = self.sum_1()
        secondTerm = self.sum_2()

        firstTerm *= -self.interaction_energy
        secondTerm *= -self.B * self.mu
        return firstTerm + secondTerm

    def sum_1(self):
        b = np.array(self.spins[1:])
        b = np.append(b, [[self.spins[0]]])
        c = np.multiply(self.spins, b)
        return np.sum(c)

    def sum_2(self):
        return np.sum(self.spins)

    def sweep_spin(self):
        n = random.randint(0, self.spins_amount - 1)
        self.spins[n] *= -1
        return n  # return ordered number of reverted spin

    def metropolis_algorithm(self):
        start_time = time.time()
        energies = []
        energy_variates = []
        counter = 0
        for i in range(self.iterations):
            current_energy = self.calculate_energy()
            k = self.sweep_spin()
            probe_energy = self.calculate_energy()
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
                    self.spins[k] *= -1
                    # logging.info(f"State reverted - {i} iteration")
                    energies.append(current_energy)
            if i % self.spins_amount == 0:
                en = self.calculate_average_energy(energies)
                energy_variates.append(en)
        energy = self.calculate_average_energy(energies)
        heat_capacity = self.calculate_heat_capacity(energies)
        magnetization = np.mean(self.spins) / self.iterations
        finish_time = time.time()
        logging.info(f'Runtime is {finish_time - start_time} seconds. \n'
                     f'Temperature is {self.temperature}. \n'
                     f'Result of Metropolis algorithm: \n'
                     f'State changed {counter} times; '
                     f'Energy = {energy}; '
                     f'Heat capacity = {heat_capacity}; '
                     f'Magnetization = {magnetization}')
        # self.plot_energy(energy_variates, self.temperature)
        return energy, heat_capacity, magnetization

    def calculate_average_energy(self, energy):
        return np.sum(energy) / self.iterations

    def calculate_heat_capacity(self, energy):
        squared_energy = np.power(energy, 2)
        average_squared = self.calculate_average_energy(squared_energy)
        average_in_square = math.pow(self.calculate_average_energy(energy), 2)
        capacity = (average_squared - average_in_square) / (
                self.iterations * BOLTSMAN_CONST * math.pow(self.temperature, 2))
        return capacity

    @staticmethod
    def plot_energy(energy, temperature):
        plt.plot(np.arange(0, len(energy), 1), energy, 'b--')
        plt.title(f'Energy depending on iterations, temperature = {temperature}')
        plt.xlabel('Iterations')
        plt.ylabel('Energy')
        plt.show()

    def real_energy(self, temperature):
        return -self.interaction_energy * np.tanh(self.interaction_energy / temperature)

    def run(self):
        # self.clear_configuration()
        temperatures = np.linspace(self.temperature, 5, 50)
        energies = np.zeros(len(temperatures))
        heat_capacities = np.zeros(len(temperatures))
        magnetizations = np.zeros(len(temperatures))
        for index, each in enumerate(temperatures):
            self.temperature = each
            self.initiate_start_configuration()
            en, heat_capacitiy, magnetization = self.metropolis_algorithm()
            energies[index] = en
            heat_capacities[index] = heat_capacitiy
            magnetizations[index] = magnetization

        ## Plot average and real temperatures
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(temperatures, self.real_energy(temperatures), 'b', label=r"Exact value")
        ax.scatter(temperatures, energies / (self.interaction_energy * self.spins_amount), c='r', label="metropolis")
        ax.set_xlabel(r'Temperature')
        ax.set_ylabel(r'Energy/Spins')
        ax.grid()
        ax.set_title(r"Energies comparison")
        ax.legend(loc=2)
        plt.show()


if __name__ == '__main__':
    lattice = Lattice1D()
    lattice.run()
