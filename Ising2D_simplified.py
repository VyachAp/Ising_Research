import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import random
import time
import logging

logging.basicConfig(level=logging.INFO)


class Model2D:

    def __init__(self):
        self.measurements_number = 2 ** 8  # number of temperature points
        self.lattice_size = 2 ** 4  # size of the lattice, N x N
        self.equilibration_steps = 2 ** 10  # number of MC sweeps for equilibration
        self.calculation_steps = 2 ** 10  # number of MC sweeps for calculation

        self.denominator_1 = 1.0 / (self.calculation_steps * self.lattice_size * self.lattice_size)
        self.denominator_2 = 1.0 / (
                    self.calculation_steps * self.calculation_steps * self.lattice_size * self.lattice_size)
        # Generate a random distribution of temperatures to make an exploration
        self.critical_temperature = 2.269

    # Generation of a random initial state for NxN spins
    def initialstate(self, N):
        """ generates a random spin configuration for initial condition"""
        state = 2 * np.random.randint(2, size=(N, N)) - 1
        return state

    # Here we define the interactions of the model (2D spin Ising model)
    # and the solution method (Metropolis Monte Carlo)
    def mcmove(self, config, beta):
        """Monte Carlo move using Metropolis algorithm """
        for i in range(self.lattice_size):
            for j in range(self.lattice_size):
                # select random spin from NxN system
                a = random.randint(0, self.lattice_size - 1)
                b = random.randint(0, self.lattice_size - 1)
                s = config[a, b]
                # calculate energy cost of this new configuration (the % is for calculation of periodic boundary condition)
                nb = config[(a + 1) % self.lattice_size, b] + config[a, (b + 1) % self.lattice_size] + config[
                    (a - 1) % self.lattice_size, b] + config[a, (b - 1) % self.lattice_size]
                cost = 2 * s * nb
                # flip spin or not depending on the cost and its Boltzmann factor
                ## (acceptance probability is given by Boltzmann factor with beta = 1/kBT
                if cost < 0:
                    s *= -1
                elif rand() < np.exp(-cost * beta):
                    s *= -1
                config[a, b] = s
        return config

    # This function calculates the energy of a given configuration for the plots of Energy as a function of T
    def calcEnergy(self, config):
        """Energy of a given configuration"""
        energy = 0
        N = self.lattice_size
        for i in range(len(config)):
            for j in range(len(config)):
                S = config[i, j]
                nb = config[(i + 1) % N, j] + config[i, (j + 1) % N] + config[(i - 1) % N, j] + config[i, (j - 1) % N]
                energy += -nb * S
        return energy / 4.

    # This function calculates the magnetization of a given configuration
    def calcMag(self, config):
        """Magnetization of a given configuration"""
        mag = np.sum(config)
        return mag

    def run(self):
        temperatures = np.random.normal(self.critical_temperature, .64, self.measurements_number)
        temperatures = temperatures[(temperatures > 0.7) & (temperatures < 3.8)]
        temperatures = np.sort(temperatures)
        measurements_number = np.size(temperatures)

        energy = np.zeros(self.measurements_number)
        magnetization = np.zeros(self.measurements_number)
        specific_heat = np.zeros(self.measurements_number)

        for m in range(len(temperatures)):
            start_time = time.time()
            total_energy = total_magnetization = 0
            total_energy_2 = total_magnetization_2 = 0
            config = self.initialstate(self.lattice_size)
            iT = 1.0 / temperatures[m]
            iT2 = iT * iT

            for i in range(self.equilibration_steps):  # equilibrate
                self.mcmove(config, iT)  # Monte Carlo moves

            for i in range(self.measurements_number):
                self.mcmove(config, iT)
                temp_energy = self.calcEnergy(config)  # calculate the energy
                temp_magnetization = self.calcMag(config)  # calculate the magnetisation

                total_energy += temp_energy
                total_magnetization += temp_magnetization
                total_magnetization_2 += temp_magnetization * temp_magnetization
                total_energy_2 += temp_energy * temp_energy

                energy[m] = self.denominator_1 * total_energy
                magnetization[m] = self.denominator_1 * total_magnetization
                specific_heat[m] = (self.denominator_1 * total_energy_2 - self.denominator_2 * total_energy * total_energy) * iT2
            finish_time = time.time()
            logging.info(f'Runtime is {finish_time - start_time} seconds. \n'
                         f'Temperature is {temperatures[m]}. \n'
                         f'Result of Metropolis algorithm: \n'
                         f'Energy = {energy[m]}; '
                         f'Heat capacity = {specific_heat[m]}; '
                         f'Magnetization = {magnetization[m]} \n')

            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(temperatures, energy, 'o', color="#A60628")
            ax.set_xlabel("Temperature (T)", fontsize=20)
            ax.set_ylabel("Energy ", fontsize=20)
            ax.grid()
            ax.set_title(r"Energy changing")
            ax.legend(loc=2)
            plt.show()

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(temperatures, abs(magnetization), 'o', color="#348ABD")
            ax.set_xlabel("Temperature (T)", fontsize=20)
            ax.set_ylabel("Magnetization ", fontsize=20)
            ax.grid()
            ax.set_title(r"Magnetization changing")
            ax.legend(loc=2)
            plt.show()

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(temperatures, specific_heat, 'o', color="#A60628")
            ax.set_xlabel("Temperature (T)")
            ax.set_ylabel("Heat capacity ")
            ax.grid()
            ax.set_title(r"Heat capacity changing")
            ax.legend(loc=2)
            plt.show()


if __name__ == '__main__':
    Lattice = Model2D()
    Lattice.run()
