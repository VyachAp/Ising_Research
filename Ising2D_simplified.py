import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import random
import time
import logging
from copy import deepcopy
import seaborn as sns
from funcs import plot_graphics
from numba import jit

logging.basicConfig(level=logging.INFO)
plt.rcParams.update({'font.size': 16})


class Model2D:

    def __init__(self, walk_length):
        self.measurements_number = 2 ** 10  # number of temperature points
        self.lattice_size = 10  # size of the lattice, N x N
        self.equilibration_steps = 2 ** 10  # number of MC sweeps for equilibration
        self.calculation_steps = 2 ** 10  # number of MC sweeps for calculation

        self.denominator_1 = 1.0 / (self.calculation_steps * self.lattice_size * self.lattice_size)
        self.denominator_2 = 1.0 / (
                self.calculation_steps * self.calculation_steps * self.lattice_size * self.lattice_size)
        self.critical_temperature = 2.269
        self.interaction_energy = 1  # J
        self.wolffs_epochs = 500
        self.sw_iterations = 35
        self.bc_simulation = 300
        self.lattice_dim = (self.lattice_size, self.lattice_size)
        self.generate_walk(walk_length)
        self.neighbours = self.tabulate_neighbors(self.lattice_dim)

    @staticmethod
    def get_site(coord, L):
        return coord[1] * L[0] + coord[0]

    @staticmethod
    def get_coord(site, L):
        """Get the 2-vector of coordinates from the site index."""
        if site // L[0] == 0:
            y = site // L[0] - 1
            x = site - (L[0] * y)
            return x, y
        else:
            y = site // (L[0])
            x = site - (L[0] * y)
        return x, y

    def get_neighbors(self, site, L):
        neighb = set()
        x, y = self.get_coord(site, L)
        neighb_coords = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        for each in neighb_coords:
            x1 = (x + each[0]) % L[0]
            y1 = (y + each[1]) % L[1]
            neighb.add(self.get_site([x1, y1], L))

        return list(neighb)

    def tabulate_neighbors(self, L):
        neighb = np.empty((self.lattice_size * self.lattice_size, 4), dtype=int)
        for site in range(self.lattice_size * self.lattice_size):
            neighb[site, :] = self.get_neighbors(site, L)
        return neighb

    @staticmethod
    def get_nearest_neighbour(site_indices, site_ranges, nearest_neighbour_number):
        """
            site_indices: [i,j], site to get NN of
            site_ranges: [Nx,Ny], boundaries of the grid
            num_NN: number of nearest neighbors, usually 1
            function which gets NN on any d dimensional cubic grid
            with a periodic boundary condition
        """

        nearest_neighbours = list()
        for i in range(len(site_indices)):
            for j in range(-nearest_neighbour_number, nearest_neighbour_number + 1):  # of nearest neighbors to include
                if j == 0:
                    continue
                NN = list(deepcopy(site_indices))
                NN[i] = (NN[i] + j) % (site_ranges[i])
                nearest_neighbours.append(tuple(NN))
        return nearest_neighbours

    def mcmove(self, config, beta):
        """Monte Carlo move using Metropolis algorithm """
        for i in range(self.lattice_size):
            for j in range(self.lattice_size):
                # select random spin from NxN system
                a = random.randint(0, self.lattice_size - 1)
                b = random.randint(0, self.lattice_size - 1)
                s = config[a, b]
                nb = config[(a + 1) % self.lattice_size, b] + config[a, (b + 1) % self.lattice_size] + config[
                    (a - 1) % self.lattice_size, b] + config[a, (b - 1) % self.lattice_size]
                cost = 2 * s * nb
                if cost < 0:
                    s *= -1
                elif rand() < np.exp(-cost * beta):
                    s *= -1
                config[a, b] = s
        return config

    def sw_move(self, temp):
        bonded = np.zeros(self.lattice_size ** 2)
        beta = 1.0 / temp
        clusters = dict()  # keep track of bonds

        for i in range(self.lattice_size):
            for j in range(self.lattice_size):
                coord = self.get_site([i, j], self.lattice_dim)
                bonded, clusters, visited = self.SW_BFS(bonded, clusters, coord, beta,
                                                        nearest_neighbors=1)

        for cluster_index in clusters.keys():
            r = np.random.rand()
            if r < 0.5:
                for coords in clusters[cluster_index]:
                    self.state[coords] = -1 * self.state[coords]

    def calcEnergy(self, config):
        """Energy of a given configuration"""
        energy = 0
        for i in range(self.lattice_size - 1):
            for j in range(self.lattice_size - 1):
                coord = self.get_site([i, j], self.lattice_dim)
                S = config[coord]
                nb = 0
                for each in self.neighbours[coord]:
                    nb += each
                # nb = config[(i + 1) % N, j] + config[i, (j + 1) % N] + config[(i - 1) % N, j] + config[i, (j - 1) % N]
                energy += -nb * S
        return energy / 4.

    def calculate_total_energy(self):
        pass

    def calc_spontaneous_mag(self, temp):
        result = []
        for each in temp:
            magnetization = np.power(1 - 1. / np.power(np.sinh((2 * self.interaction_energy) / each), 4), 1 / 8)
            result.append(magnetization ** 2)
        return result

    @staticmethod
    def calcMag(config):
        """Magnetization of a given configuration"""
        mag = np.sum(config)
        return mag

    def calculate_capacity_error(self, energy, anneal, temperature):
        nominator = np.var(energy[anneal:] ** 2) + 4 * np.var(energy[anneal:]) * np.var(energy[anneal:])
        return (np.sqrt(nominator)) / ((self.lattice_size * temperature) ** 2)

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
                self.mcmove(self.state, iT)  # Monte Carlo moves

            for i in range(self.measurements_number):
                self.mcmove(self.state, iT)
                temp_energy = self.calcEnergy(self.state)  # calculate the energy
                temp_magnetization = self.calcMag(self.state)  # calculate the magnetisation

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

    def SW_BFS(self, bonded, clusters, start, beta, nearest_neighbors=1):
        """
        :param bonded: 1 or 0, indicates whether a site has been assigned to a cluster
               or not
        :param clusters: dictionary containing all existing clusters, keys are an integer
                denoting natural index of root of cluster
        :param start: root node of graph (x,y)
        :param beta: temperature
        :param nearest_neighbors: number or NN to probe
        :return:
        """
        visited = np.zeros(self.lattice_size ** 2)  # indexes whether we have visited nodes during
        # this particular BFS search
        if bonded[start] != 0:  # cannot construct a cluster from this site
            return bonded, clusters, visited

        p = 1 - np.exp(-2 * beta * self.interaction_energy)  # bond forming probability

        queue = [start]
        index = start
        clusters[index] = [index]
        cluster_spin = self.state[index]
        color = np.max(bonded) + 1

        # whatever the input coordinates are
        while len(queue) > 0:
            r = queue.pop(0)
            if visited[r] == 0:  # if not visited
                visited[r] = 1
                # to see clusters, always use different numbers
                bonded[r] = color
                NN = self.neighbours[r]
                for nn_coords in NN:
                    if self.state[nn_coords] == cluster_spin and bonded[nn_coords] == 0 and visited[nn_coords] == 0:
                        random_val = np.random.rand()
                        if random_val < p:  # accept bond proposal
                            queue.append(nn_coords)  # add coordinate to search
                            clusters[index].append(nn_coords)  # add point to the cluster
                            bonded[nn_coords] = color  # indicate site is no longer available

        return bonded, clusters, visited

    def run_SW_algorithm(self):
        temperatures = [i for i in np.arange(1.35, 2.8, 0.025)]
        energies = []
        magnetizations = []
        heat_capacities = []
        energies_error = []
        capacities_error = []
        magnetizations_error = []
        anneal = 7 * self.sw_iterations // 10
        palette = sns.color_palette()  # To get colors
        color = palette.pop(0)
        for temp in temperatures:
            tmp_energy = []
            for i in range(self.bc_simulation):
                print(f'Temp: {temp}, iter: {i}')
                self.initiate_state()
                tmp_energy = []
                for bc in range(self.sw_iterations):  # equilibrate
                    self.sw_move(temp)

                    tmp_energy.append(self.calcEnergy(self.state))

            energy = np.array(tmp_energy)
            mean_energy = np.mean(energy[anneal:])
            mean_heat_capacity = (np.mean(energy[anneal:] ** 2) - mean_energy ** 2) / (
                    (self.steps * temp) ** 2)
            mean_magnetization = np.mean(self.state)
            logging.info(f'Temperature: {temp} \n'
                         f'Mean energy: {mean_energy}; \n'
                         f'Mean heat capacity: {mean_heat_capacity} \n'
                         f'Squared mean magnetization: {mean_magnetization ** 2} \n')
            energies.append(mean_energy / self.steps)
            heat_capacities.append(mean_heat_capacity)
            magnetizations.append(mean_magnetization)
            energies_error.append(np.std(energy[anneal:]) / self.steps)
            tmp_capacity_error = self.calculate_capacity_error(energy, anneal, temp)
            capacities_error.append(tmp_capacity_error)
            magnetizations_error.append(np.std(self.state) / (self.steps ** 2))

            plt.title("Energy")
            plt.errorbar(temperatures[:len(energies)], energies, yerr=energies_error, ls="--", marker="o",
                         color=color)
            plt.savefig(f'Binders cummulant')
            plt.grid(True)
            plt.xlabel('Temperature')
            plt.ylabel('Energy/Spins')
            plt.show()
            # plot_graphics(temperatures[:len(energies)], energies, "Temperature", "Energy/Spins", "Energies comparison")
            # plot_graphics(temperatures[:len(heat_capacities)], heat_capacities, "Temperature", "Heat capacity",
            #               "Heat capacity changing due temperature")
            # plot_graphics(temperatures[:len(magnetizations)], np.power(magnetizations, 2), "Temperature",
            #               "Squared magnetization",
            #               "Squared magnetization due temperature")

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
                N = self.state.shape
                change_tracker = np.ones(N)
                visited = np.zeros(N)
                root = []  # generate random coordinate by sampling from uniform random...
                for i in range(len(N)):
                    root.append(np.random.randint(0, N[i], 1)[0])
                root = tuple(root)
                visited[root] = 1
                C = [root]  # denotes cluster coordinates
                F_old = [root]  # old frontier
                change_tracker[root] = -1
                while len(F_old) != 0:
                    F_new = []
                    for site in F_old:
                        site_spin = self.state[tuple(site)]
                        # get neighbors
                        NN_list = self.get_nearest_neighbour(site, N, nearest_neighbour_number=1)
                        for NN_site in NN_list:
                            nn = tuple(NN_site)
                            if self.state[nn] == site_spin and visited[nn] == 0:
                                if np.random.rand() < 1 - np.exp(-2 * self.interaction_energy / temp):
                                    F_new.append(nn)
                                    visited[nn] = 1
                                    C.append(nn)
                                    change_tracker[nn] = -1
                    F_old = F_new
                for each in C:
                    self.state[each] *= -1
                tmp_energy.append(self.calcEnergy(self.state))
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
            tmp_capacity_error = self.calculate_capacity_error(energy, anneal, temp)
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
        plt.figure(figsize=(20, 12))

        palette = sns.color_palette()  # To get colors
        labels = {
            8: '8',
            16: '16',
            32: '32',
            64: '64'
        }
        for each in sizes:
            cummulants = []
            magnetizations_4 = []
            magnetizations_2 = []
            cummulants_error = []
            self.lattice_size = each
            for temp in temperatures:
                magnetizations = []
                for i in range(self.bc_simulation):
                    self.state = np.random.choice([-1, 1], (self.lattice_size, self.lattice_size))
                    for bc in range(self.sw_iterations):  # equilibrate
                        self.sw_move(temp)

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
                logging.info(f'Temperature: {temp} \n'
                             f'Lattice size: {self.lattice_size} \n')

            plt.title("Binder's cummulants")
            plt.errorbar(temperatures, cummulants, yerr=cummulants_error, ls="--", marker="o", label=labels[each],
                         color=palette.pop(0))
            plt.legend(loc="upper right")
            plt.savefig(f'Binders cummulant')
        plt.show()

    def myopic_saw(self, n):
        """
        Tries to generate a SAW of length n using the myopic algorithm

        Args:
            n (int): the length of the walk
        Returns:
            (x, y, stuck, steps) (list, list, bool, int):
                (x,y) is a SAW of length <= n
                stuck is 1 if the walk could not terminate
                steps is the number of sites of the final walk
        """
        x, y = [0], [0]
        positions = {(0, 0)}  # positions is a set that stores all sites visited by the walk
        stuck = 0
        steps = 0
        for i in range(n):
            deltas = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            deltas_feasible = []  # deltas_feasible stores the available directions
            for dx, dy in deltas:
                if (x[-1] + dx, y[-1] + dy) not in positions:  # checks if direction leads to a site not visited before
                    deltas_feasible.append((dx, dy))
            if deltas_feasible:  # checks if there is a direction available
                dx, dy = deltas_feasible[
                    np.random.randint(0, len(deltas_feasible))]  # choose a direction at random among available ones
                positions.add((x[-1] + dx, y[-1] + dy))
                x.append(x[-1] + dx)
                y.append(y[-1] + dy)
            else:  # in that case the walk is stuck
                stuck = 1
                steps = i + 1
                break  # terminate the walk prematurely
            steps = n + 1
        return x, y, stuck, steps

    def generate_walk(self, n):
        x, y, stuck, self.steps = self.myopic_saw(n)
        min_x, max_x, min_y, max_y = min(x), max(x), min(y), max(y)
        self.lattice_size = max(abs(min_x) + abs(max_x), abs(min_y) + abs(max_y)) + 1
        self.lattice_dim = self.lattice_size, self.lattice_size
        if min_x < 0:
            for ind, val in enumerate(x):
                x[ind] = val + abs(min_x)
        if max_y > 0:
            for ind, val in enumerate(y):
                y[ind] = val - abs(max_y)
        self.plot_walk(x, y, stuck, self.steps, n)
        self.points = [(coord_x, abs(coord_y)) for (coord_x, coord_y) in zip(x, y)]
        self.initiate_state()

    def initiate_state(self):
        self.state = np.zeros(self.lattice_size ** 2)
        for each in self.points:
            self.state[self.get_site(each, (self.lattice_size, self.lattice_size))] = random.choice([-1, 1])

    def plot_walk(self, x, y, stuck, steps, n):
        """
        Plots a simple random walk of length n

        Args:
            n (int): the length of the walk
        Returns:
            Plot of a simple random walk of length n
        """
        # x, y, stuck, steps = self.myopic_saw(n)
        plt.figure(figsize=(8, 8))
        plt.plot(x, y, 'bo-', linewidth=1)
        plt.plot(x[0], y[0], 'go', ms=12, label='Start')
        plt.plot(x[-1], y[-1], 'ro', ms=12, label='End')
        plt.axis('equal')
        plt.legend()
        if stuck:
            plt.title('Figure 2: Walk stuck at step ' + str(steps), fontsize=14, fontweight='bold', y=1.05)
        else:
            plt.title('Figure 2: SAW of length ' + str(n), fontsize=14, fontweight='bold', y=1.05)
        plt.show()


if __name__ == '__main__':
    Lattice = Model2D(500)
    Lattice.run_SW_algorithm()
    # Lattice.binders_cummulants()
    # Lattice.run_SW_algorithm()
    # Lattice.wolff_algorithm()
