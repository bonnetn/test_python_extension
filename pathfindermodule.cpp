#define PY_SSIZE_T_CLEAN

#include <Python.h>

#include <iostream>
#include <sstream>
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

struct Hash {
    std::size_t operator()(Vec2 const &v) const noexcept {
        std::size_t h1 = std::hash<long>{}(v.x);
        std::size_t h2 = std::hash<long>{}(v.y);
        return h1 ^ (h2 << 1);
    }
};


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
    return sqrt(a.x * a.x + a.y * a.y);
}

double distance(Vec2 a, Vec2 b) {
    Vec2 c{
            a.x - b.x,
            a.y - b.y,
    };

    return norm(c);
}

Vec2 get_min_dist(std::unordered_set<Vec2, Hash> const &Q, Vec2 start) {
    if (Q.empty()) {
        throw std::runtime_error{"empty set"};
    }

    Vec2 minVec{};
    double minDist = std::numeric_limits<double>::infinity();
    for (auto v: Q) {
        auto d = distance(start, v);
        if (d < minDist) {
            minDist = d;
            minVec = v;
        }
    }

    return minVec;
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

std::optional<std::vector<Vec2>> do_dijkstra(Grid2D<bool> grid, Vec2 start, Vec2 end) {
    /*
 1  function Dijkstra(Graph, source):
 2
 3      create vertex set Q
 4
 5      for each vertex v in Graph:
 6          dist[v] ← INFINITY
 7          prev[v] ← UNDEFINED
 8          add v to Q
10      dist[source] ← 0
11
12      while Q is not empty:
13          u ← vertex in Q with min dist[u]
14
15          remove u from Q
16
17          for each neighbor v of u:           // only v that are still in Q
18              alt ← dist[u] + length(u, v)
19              if alt < dist[v]:
20                  dist[v] ← alt
21                  prev[v] ← u
22
23      return dist[], prev[]
     */
    std::unordered_set<Vec2, Hash> Q{};
    double dist[grid.lengthX()][grid.lengthY()];
    Vec2 prev[grid.lengthX()][grid.lengthY()];
    for (long y = 0; y < grid.lengthY(); ++y) {
        for (long x = 0; x < grid.lengthX(); ++x) {
            dist[x][y] = std::numeric_limits<double>::infinity();
            prev[x][y] = Vec2{0, 0};
            Q.emplace(Vec2{x, y});
        }
    }
    dist[start.x][start.y] = 0;

    while (!Q.empty()) {
        auto u = get_min_dist(Q, start);
        Q.erase(u);
        if (u == end) {
            break;
        }

        for (auto v: get_neighbors(u, grid.lengthX(), grid.lengthY())) {
            if (grid.get(v.x, v.y)) {
                continue;
            }
            if (Q.find(v) == Q.end()) {
                continue;
            }
            auto alt = dist[u.x][u.y] + distance(u, v);
            if (alt < dist[v.x][v.y]) {
                dist[v.x][v.y] = alt;
                prev[v.x][v.y] = u;
            }
        }
    }


    auto d = dist[end.x][end.y];
    if (d == std::numeric_limits<double>::infinity()) {
        return {};
    }

    std::vector<Vec2> path;
    Vec2 cur = end;
    while (cur != start) {
        path.emplace_back(cur);
        cur = prev[cur.x][cur.y];
    }
    path.emplace_back(start);
    return {path};
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
        {nullptr,     nullptr, 0,              nullptr}
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

