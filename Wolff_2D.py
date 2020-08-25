import numpy as np
from BFS_2D import get_nearest_neighbour
from numba import njit


@njit
def wolff_move(state, temp, interaction_energy=1):
    N = state.shape
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
            site_spin = state[tuple(site)]
            # get neighbors
            NN_list = get_nearest_neighbour(site, N, nearest_neighbour_number=1)
            for NN_site in NN_list:
                nn = tuple(NN_site)
                if state[nn] == site_spin and visited[nn] == 0:
                    if np.random.rand() < 1 - np.exp(-2 * interaction_energy / temp):
                        F_new.append(nn)
                        visited[nn] = 1
                        C.append(nn)
                        change_tracker[nn] = -1
        F_old = F_new
    for each in C:
        state[each] *= -1

    return state
