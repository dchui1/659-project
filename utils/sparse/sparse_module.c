#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

static void init(void) {
    if (PyArray_API == NULL) {
        if (_import_array() < 0) {
            PyErr_Print();
            PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        }
    }
}

static void FillLongVectorFromList(long* out, PyObject *list, int size) {
    int i = 0;
    for (i = 0; i < size; ++i) {
        PyObject* o = PyList_GetItem(list, i);
        out[i] = PyLong_AsLong(o);
    }
}

static PyObject* VectorMatrixProduct(PyObject *self, PyObject *args) {
    // Parse arguments
    PyArrayObject* raw_np;
    PyObject *listObj;
    if (!PyArg_ParseTuple(args, "O!O!", &PyList_Type, &listObj, &PyArray_Type, &raw_np)) return NULL;
    const int num_ones = PyList_Size(listObj);
    const npy_intp* data_dims = PyArray_DIMS(raw_np);
    long *ones = malloc(num_ones * sizeof(double));
    FillLongVectorFromList(ones, listObj, num_ones);

    const int cols = data_dims[1];
    int np_dims[] = {1, cols};
    PyArrayObject* out_array = (PyArrayObject*) PyArray_FromDims(2, np_dims, NPY_DOUBLE);

    // Perform multiplication
    double* out_data = PyArray_DATA(out_array);
    int c = 0;
    int j = 0;
    for (c = 0; c < cols; ++c) {
        double sum = 0.0;
        for (j = 0; j < num_ones; ++j) {
            int one = ones[j];
            npy_intp ind[] = {one, c};
            double* dat = (double*)PyArray_GetPtr(raw_np, ind);
            sum += *dat;
        }
        out_data[c] = sum;
    }

    // Py_INCREF(out_array);
    free(ones);
    return (PyObject*) out_array;
}

static PyObject* VectorDotProduct(PyObject *self, PyObject *args) {
    PyArrayObject* raw_np;
    PyObject* listObj;
    if (!PyArg_ParseTuple(args, "O!O!", &PyList_Type, &listObj, &PyArray_Type, &raw_np)) return NULL;
    const int num_ones = PyList_Size(listObj);
    long *ones = malloc(num_ones * sizeof(double));
    FillLongVectorFromList(ones, listObj, num_ones);

    double sum = 0.0;
    int c = 0;
    for (c = 0; c < num_ones; ++c) {
        npy_intp ind[] = {0, ones[c]};
        double* dat = (double*)PyArray_GetPtr(raw_np, ind);
        sum += *dat;
    }

    free(ones);
    return PyFloat_FromDouble(sum);
}

static PyObject* MatrixDotProduct(PyObject *self, PyObject *args) {
    PyArrayObject* raw_np;
    PyObject* listObj;
    if (!PyArg_ParseTuple(args, "O!O!", &PyList_Type, &listObj, &PyArray_Type, &raw_np)) return NULL;
    const int num_ones = PyList_Size(listObj);
    long *ones = malloc(num_ones * sizeof(double));
    FillLongVectorFromList(ones, listObj, num_ones);

    double sum = 0.0;
    int c = 0;
    for (c = 0; c < num_ones; ++c) {
        npy_intp ind[] = {ones[c], 0};
        double* dat = (double*)PyArray_GetPtr(raw_np, ind);
        sum += *dat;
    }

    free(ones);
    return PyFloat_FromDouble(sum);
}

//--------------------------------
//--------------------------------



static PyMethodDef SparseMethods[] = {
    {"VectorMatrixProduct",  VectorMatrixProduct, METH_VARARGS, "Do a sparse vector by dense matrix product."},
    {"VectorDotProduct", VectorDotProduct, METH_VARARGS, "Do a sparse vector by dense vector dot product."},
    {"MatrixDotProduct", MatrixDotProduct, METH_VARARGS, "Do a sparse vector by dense (n, 1) matrix dot product."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef sparseMod = {
    PyModuleDef_HEAD_INIT,
    "sparse",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    SparseMethods
};

PyMODINIT_FUNC PyInit_sparse(void) {
    PyObject *m = PyModule_Create(&sparseMod);
    init();
    return m;
}
