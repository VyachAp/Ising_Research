import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import random
import time
import logging


logging.basicConfig(level=logging.INFO)


# Generation of a random initial state for NxN spins
def initialstate(N):
    """ generates a random spin configuration for initial condition"""
    state = 2 * np.random.randint(2, size=(N, N)) - 1
    return state


# Here we define the interactions of the model (2D spin Ising model)
# and the solution method (Metropolis Monte Carlo)
def mcmove(config, beta):
    """Monte Carlo move using Metropolis algorithm """
    for i in range(N):
        for j in range(N):
            # select random spin from NxN system
            a = random.randint(0, N-1)
            b = random.randint(0, N-1)
            s = config[a, b]
            # calculate energy cost of this new configuration (the % is for calculation of periodic boundary condition)
            nb = config[(a + 1) % N, b] + config[a, (b + 1) % N] + config[(a - 1) % N, b] + config[a, (b - 1) % N]
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
def calcEnergy(config):
    """Energy of a given configuration"""
    energy = 0
    for i in range(len(config)):
        for j in range(len(config)):
            S = config[i, j]
            nb = config[(i + 1) % N, j] + config[i, (j + 1) % N] + config[(i - 1) % N, j] + config[i, (j - 1) % N]
            energy += -nb * S
    return energy / 4.


# This function calculates the magnetization of a given configuration
def calcMag(config):
    """Magnetization of a given configuration"""
    mag = np.sum(config)
    return mag


nt = 2 ** 8  # number of temperature points
N = 2 ** 4  # size of the lattice, N x N
eqSteps = 2 ** 10  # number of MC sweeps for equilibration
mcSteps = 2 ** 10  # number of MC sweeps for calculation

n1, n2 = 1.0 / (mcSteps * N * N), 1.0 / (mcSteps * mcSteps * N * N)
# Generate a random distribution of temperatures to make an exploration
tm = 2.269
T = np.random.normal(tm, .64, nt)
T = T[(T > 1.2) & (T < 3.8)]
nt = np.size(T)

Energy = np.zeros(nt)
Magnetization = np.zeros(nt)
SpecificHeat = np.zeros(nt)
Susceptibility = np.zeros(nt)


for m in range(len(T)):
    start_time = time.time()
    E1 = M1 = E2 = M2 = 0
    config = initialstate(N)
    iT = 1.0 / T[m]
    iT2 = iT * iT

    for i in range(eqSteps):  # equilibrate
        mcmove(config, iT)  # Monte Carlo moves

    for i in range(mcSteps):
        mcmove(config, iT)
        Ene = calcEnergy(config)  # calculate the energy
        Mag = calcMag(config)  # calculate the magnetisation

        E1 = E1 + Ene
        M1 = M1 + Mag
        M2 = M2 + Mag * Mag
        E2 = E2 + Ene * Ene

        Energy[m] = n1 * E1
        Magnetization[m] = n1 * M1
        SpecificHeat[m] = (n1 * E2 - n2 * E1 * E1) * iT2
    finish_time = time.time()
    logging.info(f'Runtime is {finish_time - start_time} seconds. \n'
                 f'Temperature is {T[m]}. \n'
                 f'Result of Metropolis algorithm: \n'
                 f'Energy = {Energy[m]}; '
                 f'Heat capacity = {SpecificHeat[m]}; '
                 f'Magnetization = {Magnetization[m]} \n')


f = plt.figure(figsize=(18, 10))  # plot the calculated values

sp = f.add_subplot(2, 2, 1)
plt.plot(T, Energy, 'o', color="#A60628")
plt.xlabel("Temperature (T)", fontsize=20)
plt.ylabel("Energy ", fontsize=20)
plt.show()

sp = f.add_subplot(2, 2, 2)
plt.plot(T, abs(Magnetization), 'o', color="#348ABD")
plt.xlabel("Temperature (T)", fontsize=20)
plt.ylabel("Magnetization ", fontsize=20)
plt.show()

sp = f.add_subplot(2, 2, 3)
plt.plot(T, SpecificHeat, 'o', color="#A60628")
plt.xlabel("Temperature (T)", fontsize=20)
plt.ylabel("Specific Heat ", fontsize=20)
plt.show()