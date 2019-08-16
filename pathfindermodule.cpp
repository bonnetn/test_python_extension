#define PY_SSIZE_T_CLEAN

#include <Python.h>

#include <iostream>
#include <sstream>
#include <queue>
#include <thread>
#include <vector>
#include <optional>
#include <numpy/arrayobject.h>
#include <unordered_set>

struct Vec2 {
    long x;
    long y;
};

bool operator==(Vec2 a, Vec2 b) {
    return a.x == b.x && a.y == b.y;
}

bool operator!=(Vec2 a, Vec2 b) {
    return !(a == b);
}

std::ostream &operator<<(std::ostream &os, Vec2 const &v) {
    return os << "Vec2[" << v.x << ", " << v.y << "]";
}


template<typename T>
class Grid2D {
public:
    Grid2D(T *grid, long strideX, long strideY, long lenX, long lenY) :
            _grid(grid), _strideX(strideX), _strideY(strideY), _lenX(lenX), _lenY(lenY) {};

    long lengthX() { return _lenX; }

    long lengthY() { return _lenY; }

    T get(long x, long y) {
        if (x < 0 || x >= _lenX) {
            throw std::runtime_error("out of bounds");
        }
        if (y < 0 || y >= _lenY) {
            throw std::runtime_error("out of bounds");
        }
        return *(_grid + _strideX * x + _strideY * y);
    }

private:
    T *_grid;
    long _strideX, _strideY;
    long _lenX, _lenY;
};

double norm(Vec2 a) {
    return a.x * a.x + a.y * a.y;
}

double distance(Vec2 a, Vec2 b) {
    Vec2 c{
            a.x - b.x,
            a.y - b.y,
    };

    return norm(c);
}

bool inGrid(Vec2 p, long lenX, long lenY) {
    return p.x >= 0 && p.x < lenX && p.y >= 0 && p.y < lenY;
}

std::vector<Vec2> get_neighbors(Vec2 p, long lenX, long lenY) {
    std::vector<Vec2> result;
    for (auto x = -1; x <= 1; x++) {
        for (auto y = -1; y <= 1; y++) {
            if (x == 0 && y == 0) {
                continue;
            }
            auto v = Vec2{
                    p.x + x,
                    p.y + y,
            };
            if (inGrid(v, lenX, lenY)) {
                result.emplace_back(v);
            }
        }
    }

    return result;
}

struct PriorityQueueElement {
    Vec2 v;
    double dist;
};

bool operator<(PriorityQueueElement a, PriorityQueueElement b) {
    return a.dist > b.dist;
}

std::optional<std::vector<Vec2>> do_dijkstra(Grid2D<bool> grid, Vec2 start, Vec2 end) {
    /*
     * From https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm :
1  function Dijkstra(Graph, source):
2      dist[source] ← 0                           // Initialization
3
4      create vertex priority queue Q
5
6      for each vertex v in Graph:
7          if v ≠ source
8              dist[v] ← INFINITY                 // Unknown distance from source to v
9          prev[v] ← UNDEFINED                    // Predecessor of v
10
11         Q.add_with_priority(v, dist[v])
12
13
14     while Q is not empty:                      // The main loop
15         u ← Q.extract_min()                    // Remove and return best vertex
16         for each neighbor v of u:              // only v that are still in Q
17             alt ← dist[u] + length(u, v)
18             if alt < dist[v]
19                 dist[v] ← alt
20                 prev[v] ← u
21                 Q.decrease_priority(v, alt)
22
23     return dist, prev     */
    double dist[grid.lengthX()][grid.lengthY()];
    for(long i{0}; i<grid.lengthX(); i++) {
        for(long j{0}; j<grid.lengthY(); j++) {
            dist[i][j] = std::numeric_limits<double>::infinity();
        }
    }
    Vec2 prev[grid.lengthX()][grid.lengthY()];
    std::priority_queue<PriorityQueueElement> Q;
    Q.push(PriorityQueueElement{.v=start,.dist=0});

    dist[start.x][start.y] = 0;

    while (!Q.empty()) {
        auto element = Q.top();
        auto u = element.v;
        Q.pop();
        if (element.dist != dist[u.x][u.y])
            continue;

        for (auto v: get_neighbors(u, grid.lengthX(), grid.lengthY())) {
            if (grid.get(v.x, v.y))
                continue;

            auto alt = dist[u.x][u.y] + distance(u, v);
            if (alt < dist[v.x][v.y]) {
                dist[v.x][v.y] = alt;
                prev[v.x][v.y] = u;
                Q.push(PriorityQueueElement{.v=v, .dist=alt});
            }
        }
    }
    if (dist[end.x][end.y] == std::numeric_limits<double>::infinity()) {
        return {};
    }

    std::vector<Vec2> result;
    Vec2 cur = end;
    while (cur != start) {
        result.emplace_back(cur);
        cur = prev[cur.x][cur.y];
    }
    result.emplace_back(start);
    return {result};
}

static PyObject *dijkstra(PyObject *self, PyObject *args) {
    PyObject *arrayObject;
    Vec2 start{};
    Vec2 end{};
    int ok = PyArg_ParseTuple(args, "O(ll)(ll)", &arrayObject, &start.x, &start.y, &end.x, &end.y);
    if (!ok) {
        return nullptr;
    }

    auto isNDArray = PyArray_Check(arrayObject);
    if (!isNDArray) {
        PyErr_SetString(PyExc_TypeError, "argument must be a numpy ndarray");
        return nullptr;
    }

    auto arr = PyArray_FROM_OTF(arrayObject, NPY_BOOL, NPY_IN_ARRAY);
    if (arr == nullptr) {
        return nullptr;
    }

    auto grid = Grid2D<bool>{
            static_cast<bool *>(PyArray_DATA(arr)),
            PyArray_STRIDE(arr, 0),
            PyArray_STRIDE(arr, 1),
            PyArray_DIM(arr, 0),
            PyArray_DIM(arr, 1),
    };

    auto path = do_dijkstra(grid, start, end);
    if (!path) {
        return Py_None;
    }

    auto list = PyList_New((*path).size());
    if (list == nullptr) {
        return PyErr_NoMemory();
    }

    auto n = (*path).size();
    for (auto i{0lu}; i < n; ++i) {
        auto cell = (*path)[i];
        auto xLong = PyLong_FromLong(cell.x);
        if (xLong == nullptr) {
            return nullptr;
        }

        auto yLong = PyLong_FromLong(cell.y);
        if (yLong == nullptr) {
            return nullptr;
        }

        auto tuple = PyTuple_Pack(2, xLong, yLong);
        if (tuple == nullptr) {
            return PyErr_NoMemory();
        }

        if (PyList_SetItem(list, i, tuple)) {
            return nullptr;
        }
    }

    if (PyList_Reverse(list)) {
        return nullptr;
    }
    return list;
}

static PyMethodDef pathfinderMethods[] = {
        {"dijkstra", dijkstra, METH_VARARGS, "Find the shortest path between two points"},
        {nullptr,    nullptr, 0,             nullptr}
};

static struct PyModuleDef pathfinderModule = {
        PyModuleDef_HEAD_INIT,
        "pathfinder",
        "Pathfinder Module",
        -1,
        pathfinderMethods
};

PyMODINIT_FUNC PyInit_pathfinder(void) {
    import_array();
    return PyModule_Create(&pathfinderModule);
}

