import pyximport

pyximport.install()

import Cython_Ising
import numpy as np
from scipy.special import ellipe, ellipk


class IsingModel:
    """A Monte Carlo simulation of the Ising model."""

    def __init__(self, lattice_size, bond_energy, temperature, sweeps):
        """Initialize variables and the lattice."""
        self.rng_seed = int(lattice_size * temperature * 1000)
        self.lattice_size = lattice_size
        self.number_of_sites = lattice_size ** 2
        self.bond_energy = bond_energy
        self.temperature = temperature
        self.beta = 1 / self.temperature
        self.sweeps = sweeps
        self.lattice = self.init_lattice()
        self.energy_log = np.empty(self.sweeps)
        self.magnetization_log = np.empty(self.sweeps)

    def init_lattice(self):
        ground_state = np.random.choice([-1, 1])
        lattice = np.full((self.lattice_size, self.lattice_size), ground_state, dtype="int64")

        return lattice

    def calculate_lattice_energy(self):
        energy = 0
        for y in range(self.lattice_size):
            offset_y = (y + 1) % self.lattice_size
            current_row = self.lattice[y]
            next_row = self.lattice[offset_y]
            for x in range(self.lattice_size):
                center = current_row[x]
                offset_x = (x + 1) % self.lattice_size
                if current_row[offset_x] == center:
                    energy -= self.bond_energy
                else:
                    energy += self.bond_energy
                if next_row[x] == center:
                    energy -= self.bond_energy
                else:
                    energy += self.bond_energy
        return energy

    def metropolis(self):
        self.energy_log, self.magnetization_log = Cython_Ising.cy_metropolis(self.lattice,
                                                                             self.lattice_size,
                                                                             self.bond_energy, self.beta,
                                                                             self.sweeps)

    def python_metropolis(self):
        """Implentation of the Metropolis alogrithm."""
        exponents = {2 * self.bond_energy * x: np.exp(-self.beta * 2 * self.bond_energy * x) for x in range(-4, 5, 2)}
        energy = self.calculate_lattice_energy()
        magnetization = np.sum(self.lattice)
        for t in range(self.sweeps):
            np.put(self.energy_log, t, energy)
            np.put(self.magnetization_log, t, magnetization)
            for k in range(self.lattice_size ** 2):
                rand_y = np.random.randint(0, self.lattice_size)
                rand_x = np.random.randint(0, self.lattice_size)

                spin = self.lattice[rand_y, rand_x]

                neighbours = [
                    (rand_y, (rand_x - 1) % self.lattice_size),
                    (rand_y, (rand_x + 1) % self.lattice_size),
                    ((rand_y - 1) % self.lattice_size, rand_x),
                    ((rand_y + 1) % self.lattice_size, rand_x)]
                spin_sum = 0
                for n in neighbours:
                    spin_sum += self.lattice[n]
                energy_delta = 2 * self.bond_energy * spin * spin_sum

                if energy_delta <= 0:
                    acceptance_probability = 1
                else:
                    acceptance_probability = exponents[energy_delta]
                if np.random.random() <= acceptance_probability:
                    # Flip the spin and change the energy.
                    self.lattice[rand_y, rand_x] = -1 * spin
                    energy += energy_delta
                    magnetization += -2 * spin

    def wolff(self):
        self.energy_log, self.magnetization_log, cluster_sizes = Cython_Ising.cy_wolff(self.lattice,
                                                                                       self.lattice_size,
                                                                                       self.bond_energy,
                                                                                       self.beta,
                                                                                       self.sweeps)
        return cluster_sizes

    def python_wolff(self):
        """Simulate the lattice using the Wolff algorithm."""
        padd = 1 - np.exp(-2 * self.beta * self.bond_energy)
        cluster_sizes = []
        energy = self.calculate_lattice_energy()
        for t in range(self.sweeps):
            np.put(self.energy_log, t, energy)
            np.put(self.magnetization_log, t, np.sum(self.lattice))

            stack = []

            seed_y = np.random.randint(0, self.lattice_size)
            seed_x = np.random.randint(0, self.lattice_size)

            seed_spin = self.lattice[seed_y, seed_x]  # Get spin at the seed location.
            stack.append((seed_y, seed_x))
            self.lattice[seed_y, seed_x] *= -1  # Flip the spin.
            cluster_size = 1

            while stack:
                current_y, current_x = stack.pop()
                neighbours = [
                    (current_y, (current_x - 1) % self.lattice_size),
                    (current_y, (current_x + 1) % self.lattice_size),
                    ((current_y - 1) % self.lattice_size, current_x),
                    ((current_y + 1) % self.lattice_size, current_x)]

                for n in neighbours:
                    if self.lattice[n] == seed_spin:
                        if np.random.random() < padd:
                            stack.append(n)
                            self.lattice[n] *= -1
                            cluster_size += 1

            energy = Cython_Ising.calculate_lattice_energy(self.lattice, self.lattice_size, self.bond_energy)
            cluster_sizes.append(cluster_size)

        return cluster_sizes

    @staticmethod
    def exact_magnetization(bond_energy, lower_temperature, higher_temperature, step=0.001):
        exact_mag = []

        for t in np.arange(lower_temperature, higher_temperature, step):
            m = (1 - np.sinh((2 / t) * bond_energy) ** (-4)) ** (1 / 8)
            if np.isnan(m):
                m = 0
            exact_mag.append((t, m))

        return exact_mag

    @staticmethod
    def exact_internal_energy(bond_energy, lower_temperature, higher_temperature, step=0.001):
        j = bond_energy
        exact_energy = []
        for t in np.arange(lower_temperature, higher_temperature, step):
            b = 1 / t
            k = 2 * np.sinh(2 * b * j) / np.cosh(2 * b * j) ** 2
            u = -j * (1 / np.tanh(2 * b * j)) * (1 + (2 / np.pi) * (2 * np.tanh(2 * b * j) ** 2 - 1) * ellipk(k ** 2))
            exact_energy.append((t, u))

        return exact_energy

    @staticmethod
    def exact_heat_capacity(bond_energy, lower_temperature, higher_temperature, step=0.001):
        j = bond_energy
        exact_hc = []
        for t in np.arange(lower_temperature, higher_temperature, step):
            b = 1 / t
            k = 2 * np.sinh(2 * b * j) / np.cosh(2 * b * j) ** 2
            kprime = np.sqrt(1 - k ** 2)
            c = (b * j / np.tanh(2 * b * j)) ** 2 * (2 / np.pi) * (
                    2 * ellipk(k ** 2) - 2 * ellipe(k ** 2) - (1 - kprime) * (np.pi / 2 + kprime * ellipk(k ** 2)))
            exact_hc.append((t, c))

        return exact_hc
