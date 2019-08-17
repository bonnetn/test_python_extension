import numpy

import random
import time

import numpy
import pathfinder
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.dijkstra import DijkstraFinder

# Side of the grid.
x = 100
y = 100
p = 0.8

# Generate map.
obstacles = numpy.random.choice(a=[False, True], size=(x, y), p=[p, 1 - p])

obstacles[0, 0] = False
obstacles[x - 1, y - 1] = False

grid = Grid(matrix=numpy.logical_not(obstacles))
start = grid.node(0, 0)
end = grid.node(x - 1, y - 1)
finder = DijkstraFinder(diagonal_movement=DiagonalMovement.always)

start_time = time.time()
path = pathfinder.dijkstra(obstacles, (0, 0), (x - 1, y - 1))
t = time.time() - start_time

start_time = time.time()
finder.find_path(start, end, grid)
t2 = time.time() - start_time


print(grid.grid_str(path=path, start=(0,0), end=(x-1,y-1)))
print(f"Ext {t*1000:>10f} ms")
print(f"Lib {t2*1000:>10f} ms")

