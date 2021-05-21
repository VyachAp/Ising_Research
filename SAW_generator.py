import numpy as np
import matplotlib.pyplot as plt
import random
import time


# Self-avoiding random walk generator
def randomwalk(n):
    x, y = [0], [0]
    positions = [(0, 0)]  # Stores all coordinates visited by the walk
    stuck = 0
    for i in range(n):
        deltas = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # Directions to go
        deltas_possible = []  # Possible directions given condition

        for dx, dy in deltas:
            if (x[-1] + dx, y[-1] + dy) not in positions:  # If we haven't already visited this coordinate
                deltas_possible.append((dx, dy))
        if deltas_possible:  # Checks if there is one or more directions to go
            dx, dy = deltas_possible[
                np.random.randint(0, len(deltas_possible))]  # Randomly choose one of the possible directions
            positions.append((x[-1] + dx, y[-1] + dy))  # Save the new coordinate
            x.append(x[-1] + dx)
            y.append(y[-1] + dy)
        else:  # In that case the walk is stuck
            stuck = 1
            steps = i + 1
            break
        steps = n + 1
    return x, y, stuck, steps


# Plot function
def plot(n):
    steps = 0
    while steps < int(n * 0.8):
        x, y, stuck, steps = randomwalk(n)
    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)
    x_width = abs(min_x) + abs(max_x)
    y_height = abs(min_y) + abs(max_y)
    lattice_size = max(x_width, y_height) + 1

    # Affine movement of lattice
    new_x = [each + abs(min_x) for each in x]
    new_y = [each + abs(min_y) for each in y]
    plt.figure()
    plt.plot(x, y, 'bo-', linewidth=1)
    plt.plot(0, 0, 'go', label='Start')
    plt.plot(x[-1], y[-1], 'ro', label='End')
    plt.axis('equal')
    plt.legend()
    if stuck:
        plt.title('Self avoiding walk stuck at step ' + str(steps))
    else:
        plt.title('Self avoiding random walk of length ' + str(n))
    plt.show()
    return new_x, new_y, lattice_size

