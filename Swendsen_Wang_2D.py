import numpy as np
from numba import int32, float32, int64
from numba import types
from numba.typed import Dict
from BFS_2D import SW_BFS
from numba import njit


@njit
def sw_move(state, temp):
    Nx = state.shape[0]
    Ny = state.shape[1]
    bonded = np.zeros((Nx, Ny))
    beta = 1.0 / temp
    clusters = Dict.empty(key_type=types.unicode_type,
                          value_type=int32[:],
                          )  # keep track of bonds

    for i in range(Nx):
        for j in range(Ny):
            print(1)
            bonded, clusters, visited = SW_BFS(bonded, clusters, [i, j], beta, state, nearest_neighbors=1)
    for cluster_index in clusters.keys():
        r = np.random.rand()
        if r < 0.5:
            for coords in clusters[cluster_index]:
                [x, y] = coords
                state[x, y] = -1 * state[x, y]

    return state
