#define PY_SSIZE_T_CLEAN

#include <Python.h>

#include <vector>
#include <optional>
#include <numpy/arrayobject.h>
#include "pathfinding/grid.h"
#include "pathfinding/dijkstra.h"


static PyObject *dijkstra_handler(PyObject *self, PyObject *args) {
    PyObject *arrayObject;
    Vector2<long> start{};
    Vector2<long> end{};
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

    auto grid = MapGrid<long, bool>{
            static_cast<bool *>(PyArray_DATA(arr)),
            PyArray_STRIDE(arr, 0),
            PyArray_STRIDE(arr, 1),
            PyArray_DIM(arr, 0),
            PyArray_DIM(arr, 1),
    };

    std::optional<std::vector<Vec2>> path;

    // Release the GIL while finding the path.
    Py_BEGIN_ALLOW_THREADS

    path = find_path_with_dijkstra(grid, start, end);

    // Acquire back the GIL.
    Py_END_ALLOW_THREADS

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
        {"dijkstra", dijkstra_handler, METH_VARARGS, "Find the shortest path between two points"},
        {nullptr,    nullptr, 0,                     nullptr}
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

