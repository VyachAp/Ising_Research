import numpy as np
from copy import deepcopy
from numba import njit


@njit
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
            nearest_neighbours.append(np.array(NN))
    return nearest_neighbours


@njit
def SW_BFS(bonded, clusters, start, beta, state, interaction_energy=1, nearest_neighbors=1):
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
    N = state.shape
    visited = np.zeros(N)  # indexes whether we have visited nodes during
    # this particular BFS search
    if bonded[np.array(start)] != 0:  # cannot construct a cluster from this site
        return bonded, clusters, visited

    p = 1 - np.exp(-2 * beta * interaction_energy)  # bond forming probability

    queue = [start]

    index = np.array(start)
    clusters[index] = np.array(index)
    cluster_spin = state[index]
    color = np.max(bonded) + 1

    # whatever the input coordinates are
    while len(queue) > 0:
        # print(queue)
        r = np.array(queue.pop(0))
        if visited[r] == 0:  # if not visited
            visited[r] = 1
            # to see clusters, always use different numbers
            bonded[r] = color
            NN = get_nearest_neighbour(r, N, nearest_neighbors)
            for nn_coords in NN:
                rn = np.array(nn_coords)
                if state[rn] == cluster_spin and bonded[rn] == 0 and visited[rn] == 0:
                    random_val = np.random.rand()
                    if random_val < p:  # accept bond proposal
                        queue.append(rn)  # add coordinate to search
                        clusters[index].append(rn)  # add point to the cluster
                        bonded[rn] = color  # indicate site is no longer available
    return bonded, clusters, visited
