#include <Python.h>
#include <numpy/arrayobject.h>
#include "euclidean.h"
 
/* Docstrings */
static char module_docstring[] =
    "This module provides an interface for calculating squared euclidean distance";
static char euclidean_docstring[] =
    "Calculate the squared euclidean distance of two 128-dimensional vectors";
 
/* Available functions */
static PyObject *euclidean_euclidean(PyObject *self, PyObject *args);
 
/* Module specification */
static PyMethodDef module_methods[] = {
    {"euclidean", euclidean_euclidean, METH_VARARGS, euclidean_docstring},
    {NULL, NULL, 0, NULL}
};
 
/* Initialize the module */
PyMODINIT_FUNC init_euclidean(void)
{
    PyObject *m = Py_InitModule3("_euclidean", module_methods, module_docstring);
    if (m == NULL)
        return;
 
    /* Load `numpy` functionality. */
    import_array();
}
 
static PyObject *euclidean_euclidean(PyObject *self, PyObject *args)
{
    PyObject *x_obj, *y_obj;
 
    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OO", &x_obj, &y_obj))
        return NULL;
 
    /* Interpret the input objects as numpy arrays. */
    PyObject *x_array = PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *y_array = PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_IN_ARRAY);
 
    /* If that didn't work, throw an exception. */
    if (x_array == NULL || y_array == NULL) {
        Py_XDECREF(x_array);
        Py_XDECREF(y_array);
        return NULL;
    }
 
    /* Get pointers to the data as C-types. */
    double *x    = (double*)PyArray_DATA(x_array);
    double *y    = (double*)PyArray_DATA(y_array);
 
    /* Call the external C function to compute the distance. */
    double value = euclidean(x, y);
 
    /* Clean up. */
    Py_DECREF(x_array);
    Py_DECREF(y_array);
 
    if (value < 0.0) {
        PyErr_SetString(PyExc_RuntimeError,
                    "Euclidean returned an impossible value.");
        return NULL;
    }
 
    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("d", value);
    return ret;
}
