import timeit
import numpy
import matplotlib.pyplot as plt
import random
import time

import numpy
import pathfinder
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.dijkstra import DijkstraFinder

TEST_COUNT = int(1e2)
t_extension = []
t_lib_python = []

# Side of the grid.
x = 100
y = 100
p = 0.8

# Benchmark
print(f"Running {TEST_COUNT} tests")
print("----------------------------------")
for i in range(TEST_COUNT):
    ext_path = None
    while not ext_path:
        # Generate map.
        obstacles = numpy.random.choice(a=[False, True], size=(x, y), p=[p, 1 - p])

        # Make sure start and end are free.
        obstacles[0, 0] = False
        obstacles[x - 1, y - 1] = False

        # Initialize python library.
        grid = Grid(matrix=numpy.logical_not(obstacles))
        start = grid.node(0, 0)
        end = grid.node(x - 1, y - 1)
        finder = DijkstraFinder(diagonal_movement=DiagonalMovement.always)

        # Benchmark extension.
        start_time = time.time()
        ext_path = pathfinder.dijkstra(obstacles, (0, 0), (x - 1, y - 1))
        ext_time = time.time() - start_time

        # Benchmark python library.
        start_time = time.time()
        lib_path, runs = finder.find_path(start, end, grid)
        lib_time = time.time() - start_time

        if lib_path and ext_path:
            t_lib_python.append(lib_time)
            t_extension.append(ext_time)
            print(f"{i+1:>3}/{TEST_COUNT} - Done: {lib_time*1000:.3f}ms / {ext_time*1000:.3f}ms")

        elif not lib_path and not ext_path:
            print(f"{i+1:>3}/{TEST_COUNT} - No path, retrying...")

        else:
            raise RuntimeError("Bad state!")

t_extension_ms = [t * 1000 for t in t_extension]
t_lib_python_ms = [t * 1000 for t in t_lib_python]

print("Median:")
print("Extension   {:>10.3f} milliseconds".format(numpy.median(t_extension_ms)))
print("Python lib  {:>10.3f} milliseconds".format(numpy.median(t_lib_python_ms)))
print("----------------------------------")
print(f"Extension is {numpy.median(t_lib_python) / numpy.median(t_extension):.1f} times faster than python.")

t_extension_ms.sort()
t_lib_python_ms.sort()

plt.hist(t_extension_ms, 40, facecolor='g', alpha=0.75)
plt.xlabel(f'Time [ms]')
plt.title("CPython extension m={:.3f} avg={:.3f} σ = {:.3f}".format(
    numpy.median(t_extension_ms),
    numpy.mean(t_extension_ms),
    numpy.nanstd(t_extension_ms),
))
plt.grid(True)

plt.figure()
plt.hist(t_lib_python_ms, 40, facecolor='r', alpha=0.75)
plt.xlabel(f'Time [ms]')
plt.title("Library Pathfinding python m={:.3f} avg={:.3f} σ = {:.3f}".format(
    numpy.median(t_lib_python_ms),
    numpy.mean(t_lib_python_ms),
    numpy.nanstd(t_lib_python_ms),
))
plt.grid(True)

plt.show()
