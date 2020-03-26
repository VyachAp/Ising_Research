import numpy as np
from numpy.random import rand
import numpy.random as rnd
import matplotlib.pyplot as plt
import random
import time
import logging
from copy import deepcopy

logging.basicConfig(level=logging.INFO)


class Model2D:

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
        self.state = np.random.choice([-1, 1], (self.lattice_size, self.lattice_size))
        self.wolffs_epochs = 50
        self.sw_iterations = 100

    @staticmethod
    def getNN(site_indices, site_ranges, num_NN):
        """
            site_indices: [i,j], site to get NN of
            site_ranges: [Nx,Ny], boundaries of the grid
            num_NN: number of nearest neighbors, usually 1
            function which gets NN on any d dimensional cubic grid
            with a periodic boundary condition
        """

        Nearest_Neighbors = list()
        for i in range(len(site_indices)):
            for j in range(-num_NN, num_NN + 1):  # of nearest neighbors to include
                if j == 0:
                    continue
                NN = list(deepcopy(site_indices))
                NN[i] = (NN[i] + j) % (site_ranges[i])
                Nearest_Neighbors.append(tuple(NN))
        return Nearest_Neighbors

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

    def calcMag(self, config):
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
        measurements_number = np.size(temperatures)

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

    def SW_BFS(self, bonded, clusters, start, beta, nearest_neighbors=1):
        """
        :param lattice: lattice
        :param bonded: 1 or 0, indicates whether a site has been assigned to a cluster
               or not
        :param clusters: dictionary containing all existing clusters, keys are an integer
                denoting natural index of root of cluster
        :param start: root node of graph (x,y)
        :param beta: temperature
        :param J: strength of lattice coupling
        :param nearest_neighbors: number or NN to probe
        :return:
        """
        N = self.state.shape
        visited = np.zeros(N)  # indexes whether we have visited nodes during
        # this particular BFS search
        if bonded[tuple(start)] != 0:  # cannot construct a cluster from this site
            return bonded, clusters, visited

        p = 1 - np.exp(-2 * beta * self.interaction_energy)  # bond forming probability

        queue = [start]

        index = tuple(start)
        clusters[index] = [index]
        cluster_spin = self.state[index]
        color = np.max(bonded) + 1

        # whatever the input coordinates are
        while len(queue) > 0:
            # print(queue)
            r = tuple(queue.pop(0))
            if visited[r] == 0:  # if not visited
                visited[r] = 1
                # to see clusters, always use different numbers
                bonded[r] = color
                NN = self.getNN(r, N, nearest_neighbors)
                for nn_coords in NN:
                    rn = tuple(nn_coords)
                    if self.state[rn] == cluster_spin and bonded[rn] == 0 and visited[rn] == 0:
                        random_val = np.random.rand()
                        if random_val < p:  # accept bond proposal
                            queue.append(rn)  # add coordinate to search
                            clusters[index].append(rn)  # add point to the cluster
                            bonded[rn] = color  # indicate site is no longer available

        # print(len(clusters))
        # print(bonded)
        # xr = list(range(N[0]))
        # yr = list(range(N[1]))
        # [Xr, Yr] = np.meshgrid(xr, yr)
        # plt.figure()
        # plt.scatter(Xr.flatten(), Yr.flatten(), c=bonded.ravel(), cmap='jet')
        #
        # plt.show()

        return bonded, clusters, visited

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

        Nx, Ny = self.state.shape

        # scan through every element of the lattice

        # propose a random lattice site to generate a cluster
        for temp in temperatures:
            tmp_energy = []
            for bc in range(150):  # equilibrate
                bonded = np.zeros((Nx, Ny))
                beta = 1.0 / temp
                clusters = dict()  # keep track of bonds

                for i in range(Nx):
                    for j in range(Ny):
                        # print(f"Iter: {bc} x: {i}, y:{j}")
                        bonded, clusters, visited = self.SW_BFS(bonded, clusters, [i, j], beta, nearest_neighbors=1)

                # print(f'Bonded: {bonded} \nClusters: {clusters} \nVisited:{visited}')
                # xr = list(range(Nx))
                # yr = list(range(Ny))
                # [Xr, Yr] = np.meshgrid(xr, yr)
                # plt.figure()
                # plt.title(f"Clusters, iteration {bc}")
                # plt.scatter(Xr.flatten(), Yr.flatten(), c=bonded.ravel(), cmap='jet'
                # plt.show()
                for cluster_index in clusters.keys():
                    r = np.random.rand()
                    if r < 0.5:
                        for coords in clusters[cluster_index]:
                            [x, y] = coords
                            self.state[x, y] = -1 * self.state[x, y]

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

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(temperatures, energies, fmt='.c')
        ax.set_xlabel(r'Temperature')
        ax.set_ylabel(r'Energy/Spins')
        ax.grid()
        ax.set_title(r"Energies comparison")
        plt.show()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(temperatures, heat_capacities, fmt='.k')
        ax.set_xlabel(r'Temperature')
        ax.set_ylabel(r'Heat capacity')
        ax.grid()
        ax.set_title(r"Heat capacity changing due temperature")
        plt.show()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(temperatures, np.power(magnetizations, 2), fmt='.r')
        ax.set_xlabel(r'Temperature')
        ax.set_ylabel(r'Squared magnetization')
        ax.grid()
        ax.set_title(r"Squared magnetization due temperature")
        plt.show()

    def Wolff_simulation(self, thermalization_epochs=5, num_views=10):
        """
        :param thermalization_epochs:
        :param num_views:
        :return:
        """
        plt.ion()
        N = self.state.shape
        # generate random particle
        beta = 1. / self.critical_temperature
        p = 1 - np.exp(-2 * beta * self.interaction_energy)
        data = list()
        for t in range(self.wolffs_epochs):
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
                    NN_list = self.getNN(site, N, num_NN=1)
                    for NN_site in NN_list:
                        nn = tuple(NN_site)
                        if self.state[nn] == site_spin and visited[nn] == 0:
                            if np.random.rand() < p:
                                F_new.append(nn)
                                visited[nn] = 1
                                C.append(nn)
                                change_tracker[nn] = -1
                    # print(f'Visited: {visited},\n F_new: {F_new},\n F_old: {F_old},\n C:{C}')
                F_old = F_new
            # update the cluster
            if t > thermalization_epochs:  # calculate only after some correlation iterations
                value = abs(np.sum(self.state) / np.prod(self.state.shape))
                print(value)
                data.append(value)
            for site in C:
                self.state[site] = -1 * self.state[site]
            if t % int(self.wolffs_epochs / num_views) == 0:
                print('epoch: ' + str(t))
                plt.imshow(self.state)
                plt.pause(0.05)
        plt.imshow(self.state)
        plt.show()


if __name__ == '__main__':
    Lattice = Model2D()
    Lattice.run_SW_algorithm()
    # Lattice.Wolff_simulation()
