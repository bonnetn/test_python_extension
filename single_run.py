import numpy

import random
import time

import numpy
import pathfinder
from pathfinding.core.grid import Grid

# Side of the grid.
x = 100
y = 100

# Generate map.
obstacles = numpy.zeros((x, y), dtype=bool)
for i in range(x):
    for j in range(y):
        obstacles[i, j] = random.random() > 0.8

obstacles[0, 0] = False
obstacles[x - 1, y - 1] = False

path = pathfinder.dijkstra(obstacles, (0, 0), (x - 1, y - 1))
start_time = time.time()
grid = Grid(matrix=numpy.logical_not(obstacles))
t = time.time() - start_time


print(grid.grid_str(path=path, start=(0,0), end=(x-1,y-1)))
print(f"{t*1000} ms")

