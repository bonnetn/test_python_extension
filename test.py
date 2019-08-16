import random
import time

import numpy
import pathfinder

# Side of the grid.
x = 100
y = 100

# Put some obstacles.
obstacles = numpy.zeros((x, y), dtype=bool)
for i in range(x):
    for j in range(y):
        obstacles[i, j] = random.random() > 0.75

obstacles[0, 0] = False
obstacles[x - 1, y - 1] = False


# Find the path.
start_time = time.time()
path = pathfinder.dijkstra(obstacles, (0, 0), (x - 1, y - 1))
print(f"{time.time()-start_time} seconds")


# Print the path.
path_map = numpy.full((x, y), ' ', dtype=str)
for i in range(x):
    for j in range(y):
        if obstacles[i,j]:
            path_map[i,j] = '#'

if path:
    for p in path:
        path_map[p] = '0'

for i in range(y):
    print("".join(path_map[i]))

if path is None:
    print("No path.")
