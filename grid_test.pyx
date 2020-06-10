import numpy as np

def grid_generator_c(length):
    cdef int step
    cdef int start_x, start_y
    cdef double[:, :] grid = np.zeros((30, 30), dtype=float)
    start_x = np.random.randint(low=0, high=100)
    start_y = np.random.randint(low=0, high=100)
    grid[start_x, start_y] = 1
    cur_pos = start_x, start_y
    filled = []
    for i in range(length):
        step = np.random.randint(low=0, high=3)
        if step == 0:
            cur_pos[0] += 1
            if cur_pos not in filled:
                grid[cur_pos] = i
                filled.append(cur_pos)
        elif step == 1:
            cur_pos[0] -= 1
            if cur_pos not in filled:
                grid[cur_pos] = i
                filled.append(cur_pos)
        elif step == 2:
            cur_pos[1] += 1
            if cur_pos not in filled:
                grid[cur_pos] = i
                filled.append(cur_pos)
        elif step == 3:
            cur_pos[1] -= 1
            if cur_pos not in filled:
                grid[cur_pos] = i
                filled.append(cur_pos)
    print(filled)
    print(grid)

if __name__ == '__main__':
    grid_generator_c(5)