#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <pthread.h>
#include <stdlib.h>

typedef struct thread_args {
    int i;
    long v;
} thread_args;

void *thread_func(void *vargp)
{
    thread_args *args = (thread_args*) vargp;
    printf("list[%d]=%ld\n", args->i, args->v);

    free(args);
    return NULL;
}

static PyObject* find_path(PyObject *self, PyObject *args)
{
    PyObject *obj;
    int ok = PyArg_ParseTuple(args, "O", &obj);
    if (!ok) {
        return NULL;
    }

    if (!PyList_Check(obj)) {
        PyErr_SetString(PyExc_TypeError, "argument must be a list");
        return NULL;
    }

    int len = PyList_Size(obj);
    pthread_t *threadList = (pthread_t*) malloc(sizeof(pthread_t)*len);

    for (int i=0; i<len; i++) {
        PyObject *v = PyList_GetItem(obj, i);
        if (v == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "could not get value in list");
            goto err;
        }
        if (!PyLong_Check(v)) {
            PyErr_SetString(PyExc_TypeError, "all elements of the list must be ints");
            goto err;
        }
        thread_args *args = malloc(sizeof(thread_args));
        if (args == NULL) {
            PyErr_NoMemory();
            goto err;
        }
        args->i = i;
        args->v = PyLong_AsLong(v);
        if (pthread_create(&threadList[i], NULL, thread_func, (void*) args) != 0) {
            PyErr_SetString(PyExc_RuntimeError, "could not create thread");
            goto err;
        }

    }

    for (int i=0; i<len; i++) {
        pthread_join(threadList[i], NULL);
    }

    free(threadList);
    return Py_None;

err:
    free(threadList);
    return NULL;
}

static PyMethodDef pathfinderMethods[] = {
        { "find_path", find_path, METH_VARARGS, "Find the shortest path between two points" },
        { NULL, NULL, 0, NULL }
};

static struct PyModuleDef pathfinderModule = {
        PyModuleDef_HEAD_INIT,
        "pathfinder",
        "Pathfinder Module",
        -1,
        pathfinderMethods
};

PyMODINIT_FUNC PyInit_pathfinder(void)
{
    return PyModule_Create(&pathfinderModule);
}

