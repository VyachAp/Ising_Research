import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
from math import exp
from random import randrange, random, choice
import logging

logging.basicConfig(level=logging.INFO)


class Lattice2D:
    def __init__(self, lattice_size, temperature, magnetic_field_force=0):

        self.lattice_size = lattice_size  # size of lattice
        self.magnetic_field_force = magnetic_field_force  # strength of magnetic field
        self.temperature = temperature  # temperature
        self.magnetization = 0.  # total magnetization
        self.energy = 0.  # total energy
        self.lattice_values = self.init_lattice()
        self.magnetization_total()  # total magnetization
        self.energy_total()  # total energy

        self.magnetization_values = []  # list that holds magnetization values
        self.energy_values = []  # list that holds energy values

    def init_lattice(self):
        lattice = np.zeros((self.lattice_size, self.lattice_size), dtype=int)
        for y in range(self.lattice_size):
            for x in range(self.lattice_size):
                lattice[x, y] = choice([1, -1])
        return lattice

    # calculates E for one element
    def single_element_energy(self, x, y):
        if self.magnetic_field_force == 0:  # no magnetic field
            return (-1.0 * self.lattice_values[x, y] * (self.lattice_values[(x + 1) % self.lattice_size, y] +
                                                        self.lattice_values[
                                                            (x - 1 + self.lattice_size) % self.lattice_size, y] +
                                                        self.lattice_values[x, (y + 1) % self.lattice_size] +
                                                        self.lattice_values[
                                                            x, (y - 1 + self.lattice_size) % self.lattice_size]))
        else:  # there is a magnetic field
            return (-1.0 * self.lattice_values[x, y] * (self.lattice_values[(x + 1) % self.lattice_size, y] +
                                                        self.lattice_values[
                                                            (x - 1 + self.lattice_size) % self.lattice_size, y] +
                                                        self.lattice_values[x, (y + 1) % self.lattice_size] +
                                                        self.lattice_values[
                                                            x, (y - 1 + self.lattice_size) % self.lattice_size]) +
                    self.lattice_values[x, y] * self.magnetic_field_force)
            # sums up potential of lattice

    def energy_total(self):
        energy = 0
        for y in range(self.lattice_size):
            for x in range(self.lattice_size):
                energy += self.single_element_energy(x, y)
        self.energy = energy

    # sums up magnetization of lattice
    def magnetization_total(self):
        magnetization = 0
        for y in range(self.lattice_size):
            for x in range(self.lattice_size):
                magnetization += self.lattice_values[x, y]
        self.magnetization = magnetization

    def metropolis(self, steps):
        for i in range(steps):
            logging.info(f'Metropolis {i} iteration')
            # choose random atom
            x = randrange(self.lattice_size)
            y = randrange(self.lattice_size)

            energy_difference = -2 * self.single_element_energy(x, y)

            if energy_difference <= 0:
                self.lattice_values[x, y] *= -1
                self.magnetization += 2 * self.lattice_values[x, y]
                self.energy += energy_difference
                self.magnetization_values.append(self.magnetization)
                self.energy_values.append(self.energy)

            elif random() < exp(-1.0 * energy_difference / self.temperature):
                self.lattice_values[x, y] *= -1
                self.magnetization += 2 * self.lattice_values[x, y]
                self.energy += energy_difference
                self.magnetization_values.append(self.magnetization)
                self.energy_values.append(self.energy)


class Plots:
    def __init__(self, lattice_size=10, magnetic_field_force=10, x0=1, x1=5, inc=0.1, steps=50000, temperature=1):
        self.label_y = ''
        self.label_x = ''
        self.data = []
        self.lattice_size = lattice_size  # size of lattice
        self.magnetic_field_force = magnetic_field_force  # strength of magnetic field
        self.inc = inc  # size of increments in plots
        self.x0 = x0  # starting point of plots
        self.x1 = x1  # final point of plots
        self.steps = steps
        self.temperature = temperature
        self.title = f'atoms: {self.lattice_size},' \
                     f' steps: {self.steps},' \
                     f' Magnetic field: {self.magnetic_field_force},' \
                     f' Temperature increments by {self.inc}'

    def normalization(self, array):  # normalizes an array
        normalized_array = [1.0 * i / self.lattice_size ** 2 for i in array]
        return normalized_array

    def calculate_specific_heat_capacity(self, energy_list, temperature):
        average_energy = np.average(energy_list)
        squared_average_energy = average_energy ** 2  # (average of E)^2
        average_of_squared_energy = sum([i ** 2 for i in energy_list]) / len(energy_list)  # average of (E^2)
        capacity = (self.lattice_size ** (-2) * (1.0 / temperature) ** 2) * (
                    average_of_squared_energy - squared_average_energy)  # specific heat capacity
        return capacity

    def calculate_magnetic_susceptibility(self, s_list, T):
        avS = np.average(s_list)
        avS2 = avS ** 2  # (average of S)^2
        av_S2 = sum([i ** 2 for i in s_list]) / len(s_list)  # average of (S^2)
        X = (self.lattice_size ** (-2) * (1.0 / T)) * (av_S2 - avS2)  # magnetic susceptibilty
        return X

    def plot_array(self):
        plt.imshow(self.data)
        plt.title(self.title)

    def plot(self):
        plt.xlabel(self.label_x)
        plt.ylabel(self.label_y)
        plt.title(self.title)
        plt.scatter(np.arange(self.x0, self.x1, self.inc), self.data)  # ,label='B field is {}'.format(self.B))

    @staticmethod
    def average(values):  # values is a list of lists
        num = len(values) - 1
        new_list = []
        for i in range(len(values[0])):
            sum_ = 0
            for j in range(num):
                sum_ += values[j][i]
            new_list.append(sum_ / num)
        return new_list

    def f_plot(self):
        plt.xlabel(self.label_x)
        plt.ylabel(self.label_y)
        # plt.title(self.title)
        x = np.arange(self.x0, self.x1, self.inc)
        # x2=np.arange(self.x0,self.x1-1,self.inc/10.0)
        y = self.data
        fit = interpolate.interp1d(x, y)
        plt.plot(x, y, 'o', x, fit(x), '-')

    def show(self):
        plt.legend()
        plt.show()

    def magnetization(self):
        t_list = []
        for T in np.arange(self.x0, self.x1, self.inc):
            m = Lattice2D(self.lattice_size, T, magnetic_field_force=self.magnetic_field_force)
            m.metropolis(self.steps)
            t_list.append(np.absolute(np.average(m.magnetization_values)))
        self.data = Plots.normalization(self, t_list)
        self.label_x = 'Temperature in units of kB/J'
        self.label_y = 'Energy per atom'
        Plots.plot(self)

    def energy(self):
        t_list = []
        for T in np.arange(self.x0, self.x1, self.inc):
            m = Lattice2D(self.lattice_size, T, magnetic_field_force=self.magnetic_field_force)
            m.metropolis(self.steps)
            t_list.append(np.average(m.energy_values))
        self.data = Plots.normalization(self, t_list)
        self.label_x = 'Temperature in units of kB/J'
        self.label_y = 'Energy per atom'
        Plots.plot(self)

    def specific_heat(self):
        t_list = []
        for T in np.arange(self.x0, self.x1, self.inc):
            m = Lattice2D(self.lattice_size, T, magnetic_field_force=self.magnetic_field_force)
            m.metropolis(self.steps)
            t_list.append(Plots.calculate_specific_heat_capacity(self, m.energy_values, T))
        self.data = Plots.normalization(self, t_list)
        self.label_x = 'Temperature in units of kB/J'
        self.label_y = 'Specific heat capacity per atom'
        Plots.plot(self)

    def magnetic_susceptibility(self):
        t_list = []
        for T in np.arange(self.x0, self.x1, self.inc):
            m = Lattice2D(self.lattice_size, T, magnetic_field_force=self.magnetic_field_force)
            m.metropolis(self.steps)
            t_list.append(Plots.calculate_magnetic_susceptibility(self, m.magnetization_values, T))
        self.data = Plots.normalization(self, t_list)
        self.label_x = 'Temperature in units of kB/J'
        self.label_y = 'Magnetic susceptibility per atom'
        Plots.plot(self)

    def lattice(self):
        self.title = f'A 2D lattice of atomic spins equilibrated a fixed temperature of {self.temperature},' \
                     f' atoms: {self.lattice_size},' \
                     f'Magnetic field: {self.magnetic_field_force}'
        m = Lattice2D(self.lattice_size, self.temperature,
                      magnetic_field_force=self.magnetic_field_force)
        m.metropolis(self.steps)
        self.data = m.lattice_values
        Plots.plot_array(self)

    def fixed_temperature_magnetization(self, T):  # magnetization at fixed temperature
        self.title = f'magnetization at a fixed temperature of {T} with a varied magnetic field,' \
                     f'atoms: {self.lattice_size}'
        self.label_x = 'magnetic field'
        self.label_y = 'magnetization'
        dump_list1 = []
        dump_list2 = []
        t_list = []
        for B in np.arange(self.x0, self.x1, self.inc):
            m = Lattice2D(self.lattice_size, T, magnetic_field_force=B)
            m.metropolis(self.steps)
            t_list.append(np.average(m.magnetization_values))
            # dump_list1.append(plots.Norm(self,T_list)[::-1])
        self.data = Plots.normalization(self, t_list)[::-1]
        Plots.f_plot(self)
        t_list = []
        for B in np.arange(self.x1, 2 * self.x1, self.inc):
            self.magnetic_field_force = B
            m = Lattice2D(self.lattice_size, T, magnetic_field_force=B)
            m.metropolis(self.steps)
            t_list.append(np.average(m.magnetization_values))
            # ~ #dump_list2.append(plots.Norm(self,T_list)[::-1])
        self.data = Plots.normalization(self, t_list)
        Plots.f_plot(self)

    def fixed_temperature_energy(self, T):  # Energy at fixed temperature
        self.title = f'Interaction energy at a fixed temperature of {T}' \
                     f' atoms: {self.lattice_size}'
        self.label_x = 'number of steps'
        self.label_y = 'Energy per atom'
        for B in np.arange(self.x0, self.x1, self.inc):
            self.magnetic_field_force = B
            m = Lattice2D(self.lattice_size, T, magnetic_field_force=self.magnetic_field_force)
            m.metropolis(self.steps)
            t_list = m.energy_values[::-1]
            self.data = Plots.normalization(self, t_list)
            Plots.f_plot(self)

    def magnetization_varied_field(self, T):  # magnetization at fixed temperature
        self.title = f'magnetization at a fixed temperature of {T} with a varied magnetic field,' \
                     f'atoms: {self.lattice_size}'
        self.label_x = 'magnetic field'
        self.label_y = 'magnetization'
        t_list = []
        for B in np.arange(self.x0, self.x1 / 2, self.inc):
            m = Lattice2D(self.lattice_size, T, magnetic_field_force=B)
            m.metropolis(self.steps)
            t_list.append(np.average(m.magnetization_values))
        for i in np.arange(self.x1 / 2, self.x1, self.inc):
            m = Lattice2D(self.lattice_size, T, magnetic_field_force=0)
            m.metropolis(self.steps)
            t_list.append(np.average(m.magnetization_values))
        self.data = Plots.normalization(self, t_list)[::-1]
        plt.plot(self.data)


