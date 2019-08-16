# Pathfinder

> This is a small project to use CPython extensions.

I have re-implemented Dijkstra's pathfinding algorithm in C++ and made an extension out of it to use it with python.

The `test.py` file contains some benchmark which compares the performance of this extension against the [pathfinding](https://pypi.org/project/pathfinding/) library.


## How to use?

```bash
virtualenv venv
source venv/bin/activate

python3 setup.py build install
python3 test.py
```

## Results

To see the results, you can run `test.py`. Here are the results I got running it on my laptop.

The benchmarks were run on a grid of **100x100** cells, with obstacles placed randomly (uniform) on the map with a probility **p=0.6**.
The algorithm are tested on 500 different random maps. The algorithm is run 100 times. 


```
Minimum times:
Extension        0.069 milliseconds for 100 iterations
Python lib       0.762 milliseconds for 100 iterations
----------------------------------
Median time:
Extension is 1805 times faster than python.
```

Histogram using CPython extension:
![Extension](img/extension.png?raw=true "Extension")



Histogram using "pathfinding" python library:
![Python library](img/python_lib.png?raw=true "Python library")
