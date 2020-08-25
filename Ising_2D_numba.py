import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import random
import time
import logging
from funcs import plot_graphics, plot_graphics_with_error_bar
from numba import int32, float32, int64
from Wolff_2D import wolff_move
from Swendsen_Wang_2D import sw_move
from calculations import calcEnergy, calcMag, calculate_capacity_error
from metropolis import mcmove

spec = [
    ('measurements_number', int32),
    ('lattice_size', int32),
    ('equilibration_steps', int32),
    ('calculation_steps', int32),
    ('denominator_1', float32),
    ('denominator_2', float32),
    ('critical_temperature', float32),
    ('interaction_energy', int32),
    ('state', int64[:, :]),
    ('wolffs_epochs', int32),
    ('sw_iterations', int32),
    ('bc_simulation', int32),
    ('values', int64[:])
]
logging.basicConfig(level=logging.INFO)
plt.rcParams.update({'font.size': 16})


class Model2D(object):

    def __init__(self):
        self.measurements_number = 2 ** 10  # number of temperature points
        self.lattice_size = 50  # size of the lattice, N x N
        self.equilibration_steps = 2 ** 10  # number of MC sweeps for equilibration
        self.calculation_steps = 2 ** 10  # number of MC sweeps for calculation

        self.denominator_1 = 1.0 / (self.calculation_steps * self.lattice_size * self.lattice_size)
        self.denominator_2 = 1.0 / (
                self.calculation_steps * self.calculation_steps * self.lattice_size * self.lattice_size)
        self.critical_temperature = 2.269
        self.interaction_energy = 1  # J
        self.values = np.array([-1, 1])
        self.state = np.random.choice(self.values, (self.lattice_size, self.lattice_size))
        self.wolffs_epochs = 500
        self.sw_iterations = 35
        self.bc_simulation = 3000

    def run_metropolis(self):
        temperatures = np.random.normal(self.critical_temperature, .64, self.measurements_number)
        temperatures = temperatures[(temperatures > 0.7) & (temperatures < 3.8)]
        temperatures = np.sort(temperatures)

        energy = np.zeros(self.measurements_number)
        magnetization = np.zeros(self.measurements_number)
        specific_heat = np.zeros(self.measurements_number)

        for m in range(len(temperatures)):
            start_time = time.time()
            total_energy = total_magnetization = 0
            total_energy_2 = total_magnetization_2 = 0
            self.state = np.random.choice([-1, 1], (self.lattice_size, self.lattice_size))
            iT = 1.0 / temperatures[m]
            iT2 = iT * iT

            for i in range(self.equilibration_steps):  # equilibrate
                mcmove(self.lattice_size, self.state, iT)  # Monte Carlo moves

            for i in range(self.measurements_number):
                mcmove(self.lattice_size, self.state, iT)
                temp_energy = calcEnergy(self.lattice_size, self.state)  # calculate the energy
                temp_magnetization = calcMag(self.state)  # calculate the magnetisation

                total_energy += temp_energy
                total_magnetization += temp_magnetization
                total_magnetization_2 += temp_magnetization * temp_magnetization
                total_energy_2 += temp_energy * temp_energy

                energy[m] = self.denominator_1 * total_energy
                magnetization[m] = self.denominator_1 * total_magnetization
                specific_heat[m] = (self.denominator_1 * total_energy_2 -
                                    self.denominator_2 * total_energy * total_energy) * iT2
            finish_time = time.time()
            logging.info(f'Runtime is {finish_time - start_time} seconds. \n'
                         f'Temperature is {temperatures[m]}. \n'
                         f'Result of Metropolis algorithm: \n'
                         f'Energy = {energy[m]}; '
                         f'Heat capacity = {specific_heat[m]}; '
                         f'Magnetization = {magnetization[m]} \n')

            plot_graphics(temperatures, energy, "Temperature (T)", "Energy", "Energy changing")
            plot_graphics(temperatures, abs(magnetization), "Temperature (T)", "Magnetization",
                          "Magnetization changing")
            plot_graphics(temperatures, specific_heat, "Temperature (T)", "Heat capacity ",
                          "Heat capacity changing")

    def run_SW_algorithm(self):
        temperatures = np.random.normal(self.critical_temperature, .64, self.measurements_number)
        temperatures = temperatures[(temperatures > 0.7) & (temperatures < 3.8)]
        temperatures = np.sort(temperatures)
        energies = []
        magnetizations = []
        heat_capacities = []
        energies_error = []
        capacities_error = []
        magnetizations_error = []
        anneal = 7 * self.sw_iterations // 10
        for temp in temperatures:
            tmp_energy = []
            self.state = np.random.choice([-1, 1], (self.lattice_size, self.lattice_size))
            for bc in range(self.sw_iterations):  # equilibrate
                self.state = sw_move(state=self.state, temp=temp)
                tmp_energy.append(calcEnergy(self.lattice_size, self.state))

            energy = np.array(tmp_energy)
            mean_energy = np.mean(energy[anneal:])
            mean_heat_capacity = (np.mean(energy[anneal:] ** 2) - mean_energy ** 2) / (
                    (self.lattice_size * temp) ** 2)
            mean_magnetization = np.mean(self.state)
            logging.info(f'Temperature: {temp} \n'
                         f'Mean energy: {mean_energy}; \n'
                         f'Mean heat capacity: {mean_heat_capacity} \n'
                         f'Squared mean magnetization: {mean_magnetization ** 2} \n')
            energies.append(mean_energy / (self.lattice_size ** 2))
            heat_capacities.append(mean_heat_capacity)
            magnetizations.append(mean_magnetization)
            energies_error.append(np.std(energy[anneal:]) / (self.lattice_size ** 2))
            tmp_capacity_error = calculate_capacity_error(self.lattice_size, energy, anneal, temp)
            capacities_error.append(tmp_capacity_error)
            magnetizations_error.append(np.std(self.state) / (self.lattice_size ** 2))

        plot_graphics(temperatures, energies, "Temperature", "Energy/Spins", "Energies comparison")
        plot_graphics(temperatures, heat_capacities, "Temperature", "Heat capacity",
                      "Heat capacity changing due temperature")
        plot_graphics(temperatures, np.power(magnetizations, 2), "Temperature", "Squared magnetization",
                      "Squared magnetization due temperature")

    def wolff_algorithm(self):
        temperatures = np.random.normal(self.critical_temperature, .64, self.measurements_number)
        temperatures = temperatures[(temperatures > 0.7) & (temperatures < 3.8)]
        temperatures = np.sort(temperatures)
        # temperatures = [self.critical_temperature, 3.209610]
        energies = []
        magnetizations = []
        heat_capacities = []
        energies_error = []
        capacities_error = []
        magnetizations_error = []
        anneal = 7 * self.sw_iterations // 10

        for temp in temperatures:
            tmp_energy = []
            self.state = np.random.choice([-1, 1], (self.lattice_size, self.lattice_size))
            for bc in range(self.wolffs_epochs):  # equilibrate
                self.state = wolff_move(state=self.state, temp=temp)
                tmp_energy.append(calcEnergy(self.lattice_size, self.state))
            energy = np.array(tmp_energy)
            mean_energy = np.mean(energy[anneal:])
            mean_heat_capacity = (np.mean(energy[anneal:] ** 2) - mean_energy ** 2) / (
                    (self.lattice_size * temp) ** 2)
            mean_magnetization = np.mean(self.state)
            logging.info(f'Temperature: {temp} \n'
                         f'Mean energy: {mean_energy}; \n'
                         f'Mean heat capacity: {mean_heat_capacity} \n'
                         f'Squared mean magnetization: {mean_magnetization ** 2} \n')
            energies.append(mean_energy / (self.lattice_size ** 2))
            heat_capacities.append(mean_heat_capacity)
            magnetizations.append(mean_magnetization)
            energies_error.append(np.std(energy[anneal:]) / (self.lattice_size ** 2))
            tmp_capacity_error = calculate_capacity_error(self.lattice_size, energy, anneal, temp)
            capacities_error.append(tmp_capacity_error)
            magnetizations_error.append(np.std(self.state) / (self.lattice_size ** 2))

        plot_graphics(temperatures, energies, "Temperatures", "Energy/Spins", "Energies comparison (Wolff)")
        plot_graphics(temperatures, heat_capacities, "Temperatures", "Heat capacity", "Heat capacity changing due "
                                                                                      "temperature (Wolff)")
        plot_graphics(temperatures, np.power(magnetizations, 2), "Temperatures", "Squared magnetization",
                      "Squared magnetization due temperature (Wolff)")

    def binders_cummulants(self):
        temperatures = [i for i in np.arange(2.05, 2.2, 0.015)]
        sizes = [8, 16, 32]
        # palette = sns.color_palette()  # To get colors
        # labels = {
        #     8: '8',
        #     16: '16',
        #     32: '32',
        #     64: '64'
        # }
        for each in sizes:
            cummulants = []
            magnetizations_4 = []
            magnetizations_2 = []
            cummulants_error = []
            self.lattice_size = each
            for temp in temperatures:
                magnetizations = []
                for i in range(self.bc_simulation):
                    print(i)
                    self.state = np.random.choice(np.array([-1, 1]), (self.lattice_size, self.lattice_size))
                    for bc in range(self.sw_iterations):  # equilibrate
                        sw_move(self.state, temp)
                        magnetizations.append(np.mean(self.state))
                magnetizations_4.append(np.mean(np.power(magnetizations, 4)))
                magnetizations_2.append(np.mean(np.power(magnetizations, 2)))
                cummulant_value = 1 - np.mean(np.power(magnetizations, 4)) / (
                        3 * np.power(np.mean(np.power(magnetizations, 2)), 2))
                m_4_variance = np.var(np.power(magnetizations, 4)) / np.sqrt(len(magnetizations))
                m_2_variance = np.var(np.power(magnetizations, 2)) / np.sqrt(len(magnetizations))
                cummulant_variance = np.sqrt(m_4_variance ** 2 + m_2_variance ** 2)
                cummulants_error.append(cummulant_variance)
                cummulants.append(cummulant_value)
                logging.info('Temperature: {} \n'
                             'Lattice size: {} \n'.format(temp, self.lattice_size))

            # plt.title("Binder's cummulants")
            # plt.errorbar(temperatures, cummulants, yerr=cummulants_error, ls="--", marker="o", label=labels[each],
            #              color=palette.pop(0))
            # plt.legend(loc="upper right")
            # plt.savefig(f'Binders cummulant')
        # plt.show()


if __name__ == '__main__':
    Lattice = Model2D()
    Lattice.binders_cummulants()
    # Lattice.run_SW_algorithm()
    # Lattice.wolff_algorithm()
