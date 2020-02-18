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
        self.state = np.random.choice([-1, 1], (self.lattice_size, self.lattice_size))
        self.measurements_number = 2 ** 8  # number of temperature points
        self.lattice_size = 2 ** 4  # size of the lattice, N x N
        self.equilibration_steps = 2 ** 10  # number of MC sweeps for equilibration
        self.calculation_steps = 2 ** 10  # number of MC sweeps for calculation

        self.denominator_1 = 1.0 / (self.calculation_steps * self.lattice_size * self.lattice_size)
        self.denominator_2 = 1.0 / (
                self.calculation_steps * self.calculation_steps * self.lattice_size * self.lattice_size)
        self.critical_temperature = 2.269
        self.interaction_energy = 1  # J

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
                NN = list(deepcopy(site_indices))  # don't want to overwite;
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
                specific_heat[m] = (
                                           self.denominator_1 * total_energy_2 - self.denominator_2 * total_energy * total_energy) * iT2
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
        function currently cannot generalize to dimensions higher than 2...
        main idea is that we populate a lattice with clusters according to SW using a BFS from a root coord
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

        return bonded, clusters, visited

    def run_SW_algorithm(self):
        temperatures = np.random.normal(self.critical_temperature, .64, self.measurements_number)
        temperatures = temperatures[(temperatures > 0.7) & (temperatures < 3.8)]
        temperatures = np.sort(temperatures)
        energy = np.zeros([self.measurements_number, 1])
        magnetization = np.zeros([self.measurements_number, 2])
        heat_capacity = np.zeros([self.measurements_number, 2])

        Nx, Ny = self.state.shape

        # scan through every element of the lattice

        # propose a random lattice site to generate a cluster
        bonded = np.zeros((Nx, Ny))
        beta = 1.0 / 2.238
        clusters = dict()  # keep track of bonds
        ## iterate through the entire lattice to assign bonds
        ## and clusters
        for i in range(Nx):
            for j in range(Ny):
                print("Nx, Ny", i, j)
                ## at this point, we do a BFS search to create the cluster
                bonded, clusters, visited = self.SW_BFS(bonded, clusters, [i, j], beta, nearest_neighbors=1)

        print("finish with SW_BFS!")
        for cluster_index in clusters.keys():
            [x0, y0] = np.unravel_index(cluster_index, (Nx, Ny))
            r = np.random.rand()
            if (r < 0.5):
                for coords in clusters[cluster_index]:
                    [x, y] = coords
                    # print(Lattice[x,y], end=', '); #check clusters
                    self.state[x, y] = -1 * self.state[x, y]

        return self.state


if __name__ == '__main__':
    Lattice = Model2D()
    Lattice.run_metropolis()
