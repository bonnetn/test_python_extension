import timeit
import numpy
import matplotlib.pyplot as plt
REPEATS = 500
NUMBER = 100

setup = """
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

# Generate map.
obstacles = numpy.zeros((x, y), dtype=bool)
for i in range(x):
    for j in range(y):
        obstacles[i, j] = random.random() > 0.6

obstacles[0, 0] = False
obstacles[x - 1, y - 1] = False

# Other lib
grid = Grid(matrix=numpy.logical_not(obstacles))
start = grid.node(0, 0)
end = grid.node(x - 1, y - 1)
finder = DijkstraFinder(diagonal_movement=DiagonalMovement.always)
"""

stmp_extension = "pathfinder.dijkstra(obstacles, (0, 0), (x - 1, y - 1))"
stmp_lib_python = "finder.find_path(start, end, grid)"



t_extension = timeit.repeat(stmp_extension, setup=setup, repeat=REPEATS, number=NUMBER)
print("Extension   {:>10.3f} milliseconds for {} iterations".format(min(t_extension) * 1000, NUMBER))

t_lib_python = timeit.repeat(stmp_lib_python, setup=setup, repeat=REPEATS, number=NUMBER)
print("Python lib  {:>10.3f} milliseconds for {} iterations".format(min(t_lib_python) * 1000, NUMBER))
print("----------------------------------")
print(f"Extension is {numpy.median(t_lib_python) / numpy.median(t_extension)} times faster than python.")

t_extension_ms = [t*1000 for t in t_extension]
t_lib_python_ms = [t*1000 for t in t_lib_python]

plt.hist(t_extension_ms, 40, facecolor='g', alpha=0.75)
plt.xlabel(f'Time [ms] for {NUMBER} executions')
plt.title(f"CPython extension m={numpy.median(t_extension_ms):.3f} avg={numpy.mean(t_extension_ms):.3f} σ={numpy.nanstd(t_extension_ms):.3f}")
plt.grid(True)

plt.figure()
plt.hist(t_lib_python_ms, 40, facecolor='r', alpha=0.75)
plt.xlabel(f'Time [ms] for {NUMBER} executions')
plt.title(f"Library Pathfinding python m={numpy.median(t_lib_python_ms):.3f} avg={numpy.mean(t_lib_python_ms):.3f} σ={numpy.nanstd(t_lib_python_ms):.3f}")
plt.grid(True)

plt.show()
