import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
import math
from random import randrange, random, choice
import logging

logging.basicConfig(level=logging.INFO)


class Lattice1D:
    def __init__(self):
        self.spins = 100
        self.aveSpinGroup = 10
        self.ising = []
        self.alignmentE = -0.20
        self.mu = 0.50
        self.B = -10.0
        self.kB = 1.0
        self.temperature = 2.5
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
        del self.ising[:]

    def fill_ising_configuration(self):
        for each in range(self.spins):  # spin up is 1, spin down is -1
            self.ising.append(-1)

    def calculate_energy(self):
        firstTerm = 0.0
        secondTerm = 0.0
        for i in range(len(self.ising) - 1):
            firstTerm += self.ising[i] * self.ising[i + 1]
        logging.info(f'First term is {firstTerm}')
        firstTerm *= self.alignmentE
        for i in range(len(self.ising)):
            secondTerm += self.ising[i]
        logging.info(f'Second term is {secondTerm}')
        firstTerm *= -self.B * self.mu
        return firstTerm + secondTerm

    def flip_spins(self, n, T, isingI):
        flipped = []
        isingO = isingI[:]
        energy = self.calculate_energy()  # original energy
        for i in range(n):
            flipped.append(int(random.uniform(0, 1) * self.spins))  # choose a array of terms
        for i in range(len(flipped)):  # flip those random terms in the output array
            isingO[flipped[i]] = -1 * isingO[flipped[i]]
        Etrial = self.calculate_energy()  # new energy
        dE = float(Etrial - energy)  # you keep it if the new energy is lower than the old
        # print "de is equal to %f" %dE
        # print "dE is equal to %f" %dE
        if dE > 0.0:  # if it is higher energy, then you don't automatically take it
            logging.info(math.exp(-1. * dE / (self.kB * T)))
            p = self.probability(math.exp(-1. * dE / (self.kB * T)))
            if p > 0:  # with probablity (exp(-dE/ kbT)), keep the new arrangement
                logging.info('Keeping arrangement')
            else:  # otherwise, flip everything back
                for i in range(len(flipped)):
                    isingO[flipped[i]] *= -1
                # print "keep the old arrangement00000"
        count = 0
        for i in range(len(isingI)):
            if isingO[i] * isingI[i] == -1.:
                count = count + 1
        return isingO

    def average_spin(self, array, i, spingroupsize):
        summed = 0.0
        for x in range(i, spingroupsize, 1):
            summed += array[x]
        logging.info(f'Average spins is {summed}')
        return summed

    def plot_ising_model(self):
        timeIsing = []
        self.clear_configuration()
        self.fill_ising_configuration()
        for i in range(int(self.time)):
            ising = self.flip_spins(1, self.temperature)
            timeIsing.append(ising)

        print
        "len ising %d" % len(timeIsing)
        print
        "len(timeIsing[1]) %d" % len(timeIsing[1])

        fig1 = plt.figure(1)
        scatter = []
        scatterplot = []

        # scatter1 = []
        # this is to make the plots

        for t in range(int(self.time)):
            #  del scatter1[:]
            for i in range(len(self.ising)):
                if timeIsing[t][i] == -1:
                    # scatter1.append((t, i))
                    scatter.append((t, i))
                    if t % self.timeplotsteps == 0:
                        scatterplot.append((t, i))
        #  scatter.append(scatter1)
        # for i in range(len(scatter)):
        #  # iterate through columns
        # for j in range(len(scatter[0])):
        #    scatterA.append(scatter[i][j])

        diff = []
        for i in range(len(timeIsing[0])):
            diff.append(timeIsing[0][i] - timeIsing[len(timeIsing) - 1][i])
        # sum(diff)
        # print diff

        plt.scatter([pt[0] for pt in scatterplot], [pt[1] for pt in scatterplot])
        # plt.scatter(*zip(*scatter))
        fig1.show()
