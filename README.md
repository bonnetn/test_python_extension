# Pathfinder

> This is a small project to use CPython extensions.

I have re-implemented Dijkstra's pathfinding algorithm in C++ and made an extension out of it to use it with python.

The `test.py` file contains some benchmark which compares the performance of this extension against the [pathfinding](https://pypi.org/project/pathfinding/) library.


## How to build?

```bash
virtualenv venv
source venv/bin/activate

make clean build install
make benchmark # If you want to run the benchmark.
make single_run # If you want to run it once and see the results.
```

**OR**

```bash
virtualenv venv
source venv/bin/activate

python3 setup.py build install

python3 test.py # If you want to run the benchmark.
python3 single_run.py # If you want to run it once and see the results.
```

## How to use?

```python
import pathfinder
import numpy

# Create a ndarray of bools which represents your map.
map = numpy.zeros((100,100), dtype=bool)

# True represents an obstacle
map[1,3] = True

# Find the path between (0,0) and (99,99):
path = pathfinder.dijkstra(map, (0,0), (99,99))

# If there is no path, the function returns None, otherwise it returns a list with all the positions from the start to the end.
print(path)
```

## Results

To see the results, you can run `test.py`. Here are the results I got running it on my laptop.

The benchmarks were run on a grid of **100x100** cells, with obstacles placed randomly (uniform distribution) on the map with a probility of **p=0.8**.
The algorithm was tested on 1000 times, with a different map for each test.

Both algorithms (pathfinding lib and cpython extention) were tested against the same set of map.


```
Median:
Extension        2.109 milliseconds
Python lib     258.623 milliseconds
----------------------------------
Extension is 122.7 times faster than python.
```

Histogram using CPython extension:
![Extension](img/extension.png?raw=true "Extension")



Histogram using "pathfinding" python library:
![Python library](img/python_lib.png?raw=true "Python library")

Histogram using A* algorithm and CPython extension:
![A* using extension](img/extension_astar.png?raw=true "Extension A star")
