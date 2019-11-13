import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
import math
import random
import logging

logging.basicConfig(level=logging.INFO)

BOLTSMAN_CONST = 1


class Lattice1D:
    def __init__(self):
        self.spins_amount = 100
        self.iterations = 10 * self.spins_amount
        self.spins = []
        self.interaction_energy = 1  # J
        self.mu = 0.50
        self.B = -10.0
        self.temperature = 1
        self.time = 10000.
        self.timeplots = self.time / 100.0
        self.timeplotsteps = int(self.time / self.timeplots)
        self.energy = 0.0

    @staticmethod
    def probability(r):  # input a probability, gives you a 1 with that probability, or a 0
        x = random.uniform(0, 1)
        if x <= r:
            return 1
        return 0

    def clear_configuration(self):
        del self.spins[:]

    def initiate_start_configuration(self):
        for each in range(self.spins_amount):
            g = random.uniform(0, 1)
            if g >= 0.5:
                self.spins.append(-1)
            else:
                self.spins.append(1)
        print(len(self.spins))

    def calculate_energy(self):
        firstTerm = 0.0
        secondTerm = 0.0
        for i in range(len(self.spins) - 1):
            firstTerm += self.spins[i] * self.spins[i + 1]
        # logging.info(f'First term is {firstTerm}')
        firstTerm *= -self.interaction_energy
        for i in range(len(self.spins)):
            secondTerm += self.spins[i]
        # logging.info(f'Second term is {secondTerm}')
        secondTerm *= -self.B * self.mu
        return firstTerm + secondTerm

    def sweep_spin(self):
        n = random.randint(0, self.spins_amount - 1)
        self.spins[n] *= -1
        return n  # return ordered number of reverted spin

    def metropolis_algorithm(self):
        energies = []
        counter = 0
        for i in range(self.iterations):
            current_energy = self.calculate_energy()
            k = self.sweep_spin()
            probe_energy = self.calculate_energy()
            if probe_energy <= current_energy:
                logging.info("State accepted")
                energies.append(probe_energy)
                counter += 1
            else:
                p = self.probability(
                    math.exp(-1. * (probe_energy - current_energy) / (BOLTSMAN_CONST * self.temperature)))
                if p:
                    logging.info("State accepted")
                    energies.append(probe_energy)
                    counter += 1
                else:
                    self.spins[k] *= -1
                    logging.info("State reverted")
                    energies.append(current_energy)

        energy = self.calculate_average_energy(energies)
        heat_capacity = self.calculate_heat_capacity(energies)
        magnetization = np.sum(self.spins) / self.iterations

        logging.info(f'Result of Metropolis algorithm: \n'
                     f'State changed {counter} times '
                     f'Energy = {energy} '
                     f'Heat capacity = {heat_capacity} '
                     f'Magnetization = {magnetization}')
        self.plot_energy(energies)

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
    def plot_energy(energy):
        plt.plot(np.arange(0, len(energy), 1), energy, 'b--')
        plt.title('Energy depending on iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Energy')
        plt.show()

    def run(self):
        self.clear_configuration()
        self.initiate_start_configuration()
        self.metropolis_algorithm()


if __name__ == '__main__':
    lattice = Lattice1D()
    lattice.run()
