from libc.math cimport exp as c_exp
from libc.stdlib cimport malloc, free

import numpy as np
import cython
cimport numpy as np
from mc_lib.rndm cimport RndmWrapper

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef calculate_lattice_energy(np.ndarray[np.int_t, ndim=2] lattice, int lattice_size, int bond_energy):
    cdef int energy = 0
    cdef int y, x, center, offset_y, offset_x, xnn, ynn
    for y in range(lattice_size):
        offset_y = y + 1
        if y + 1 >= lattice_size:
            offset_y = offset_y - lattice_size
        for x in range(lattice_size):
            offset_x = x + 1
            if x + 1 >= lattice_size:
                offset_x = offset_x - lattice_size
            center = lattice[y, x]
            xnn = lattice[y, offset_x]
            ynn = lattice[offset_y, x]
            if xnn == center:
                energy += -1 * bond_energy
            else:
                energy += bond_energy
            if ynn == center:
                energy += -1 * bond_energy
            else:
                energy += bond_energy
    return energy


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cy_metropolis(np.ndarray[np.int_t, ndim=2] lattice, int lattice_size, int bond_energy, double beta, int sweeps):
    """
    Implentation of the Metropolis alogrithm.
    """
    cdef int t, k, spin, rand_y, rand_x, spin_sum, prev_x, next_x, prev_y, next_y
    cdef double energy_delta, acceptance_probability
    cdef double energy = calculate_lattice_energy(lattice, lattice_size, bond_energy)
    cdef double magnetization = np.sum(lattice)
    cdef np.ndarray[np.float_t, ndim=1] energy_history = np.empty(sweeps, dtype=np.float64)
    cdef np.ndarray[np.float_t, ndim=1] magnetization_history = np.empty(sweeps, dtype=np.float64)
    cdef RndmWrapper rndm = RndmWrapper((1234, 0))
    for t in range(sweeps):
        energy_history[t] = energy
        magnetization_history[t] = magnetization
        for k in range(lattice_size**2):
            rand_y = int(<double> rndm.uniform() * lattice_size)
            rand_x = int(<double> rndm.uniform() * lattice_size)

            spin = lattice[rand_y, rand_x]

            spin_sum = 0

            prev_x = rand_x - 1
            if prev_x < 0:
                prev_x += lattice_size
            spin_sum += lattice[rand_y, prev_x]

            next_x = rand_x + 1
            if next_x >= lattice_size:
                next_x -= lattice_size
            spin_sum += lattice[rand_y, next_x]

            prev_y = rand_y - 1
            if prev_y < 0:
                prev_y += lattice_size
            spin_sum += lattice[prev_y, rand_x]

            next_y = rand_y + 1
            if next_y >= lattice_size:
                next_y -= lattice_size
            spin_sum += lattice[next_y, rand_x]

            energy_delta = 2 * bond_energy * spin * spin_sum

            if energy_delta <= 0:
                acceptance_probability = 1
            else:
                acceptance_probability = c_exp(-beta * energy_delta)
            if <double> rndm.uniform() <= acceptance_probability:
                lattice[rand_y, rand_x] = -1 * spin
                energy += energy_delta
                magnetization += -2 * spin
    return energy_history, magnetization_history


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cy_wolff(np.ndarray[np.int_t, ndim=2] lattice, int lattice_size, int bond_energy, double beta, int sweeps):
    """Simulate the lattice using the Wolff algorithm."""
    cdef int seed_x, seed_y, seed_spin, cluster_size, prev_x, next_x, prev_y, next_y, current_x, current_y, stack_counter
    cdef double padd = 1 - c_exp(-2 * beta * bond_energy)
    cdef np.ndarray[np.float_t, ndim=1] energy_history = np.empty(sweeps, dtype=np.float64)
    cdef np.ndarray[np.float_t, ndim=1] magnetization_history = np.empty(sweeps, dtype=np.float64)
    cdef np.ndarray[np.int_t, ndim=1] cluster_sizes = np.empty(sweeps, dtype=np.int)
    cdef double energy = calculate_lattice_energy(lattice, lattice_size, bond_energy)
    cdef int magnetization = np.sum(lattice)

    cdef int *stack_x = <int *>malloc(lattice_size**2 * sizeof(int))
    cdef int *stack_y = <int *>malloc(lattice_size**2 * sizeof(int))
    cdef RndmWrapper rndm = RndmWrapper((1234, 0))

    for t in range(sweeps):
        # Measurement every sweep.
        energy_history[t] = energy
        magnetization_history[t] = magnetization

        stack_counter = 1

        # Pick a random location on the lattice as the seed.
        seed_y = int(<double> rndm.uniform() * lattice_size)
        seed_x = int(<double> rndm.uniform() * lattice_size)

        seed_spin = lattice[seed_y, seed_x]  # Get spin at the seed location.
        stack_x[0] = seed_x  # Locations for which the neighbours still have to be checked.
        stack_y[0] = seed_y
        lattice[seed_y, seed_x] *= -1  # Flip the spin.
        cluster_size = 1

        while stack_counter:
            stack_counter -= 1
            current_x = stack_x[stack_counter]
            current_y = stack_y[stack_counter]
            prev_x = current_x - 1
            if prev_x < 0:
                prev_x += lattice_size
            if lattice[current_y, prev_x] == seed_spin and (<double> rndm.uniform() < padd):
                stack_x[stack_counter] = prev_x
                stack_y[stack_counter] = current_y
                lattice[current_y, prev_x] *= -1
                cluster_size += 1
                stack_counter += 1

            next_x = current_x + 1
            if next_x >= lattice_size:
                next_x -= lattice_size
            if lattice[current_y, next_x] == seed_spin and (<double> rndm.uniform() < padd):
                stack_x[stack_counter] = next_x
                stack_y[stack_counter] = current_y
                lattice[current_y, next_x] *= -1
                cluster_size += 1
                stack_counter += 1

            prev_y = current_y - 1
            if prev_y < 0:
                prev_y += lattice_size
            if lattice[prev_y, current_x] == seed_spin and (<double> rndm.uniform() < padd):
                stack_x[stack_counter] = current_x
                stack_y[stack_counter] = prev_y
                lattice[prev_y, current_x] *= -1
                cluster_size += 1
                stack_counter += 1

            next_y = current_y + 1
            if next_y >= lattice_size:
                next_y -= lattice_size
            if lattice[next_y, current_x] == seed_spin and (<double> rndm.uniform() < padd):
                stack_x[stack_counter] = current_x
                stack_y[stack_counter] = next_y
                lattice[next_y, current_x] *= -1
                cluster_size += 1
                stack_counter += 1


        energy = calculate_lattice_energy(lattice, lattice_size, bond_energy)
        magnetization -= 2 * seed_spin * cluster_size
        cluster_sizes[t] = cluster_size

    free(stack_x)
    free(stack_y)
    return energy_history, magnetization_history, cluster_sizes