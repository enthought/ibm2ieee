#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

/*
 * single_type_logit.c
 * This is the C code for creating your own
 * NumPy ufunc for a logit function.
 *
 * In this code we only define the ufunc for
 * a single dtype. The computations that must
 * be replaced to create a ufunc for
 * a different function are marked with BEGIN
 * and END.
 *
 * Details explaining the Python-C API can be found under
 * 'Extending and Embedding' and 'Python/C API' at
 * docs.python.org .
 */

static PyMethodDef LogitMethods[] = {
  {NULL, NULL, 0, NULL}
};

/* The loop definition must precede the PyMODINIT_FUNC. */

static void double_logit(char **args, npy_intp *dimensions,
                         npy_intp* steps, void* data)
{
  npy_intp i;
  npy_intp n = dimensions[0];
  char *in = args[0], *out = args[1];
  npy_intp in_step = steps[0], out_step = steps[1];

  double tmp;

  for (i = 0; i < n; i++) {
    /*BEGIN main ufunc computation*/
    tmp = *(double *)in;
    tmp /= 1-tmp;
    *((double *)out) = log(tmp);
    /*END main ufunc computation*/

    in += in_step;
    out += out_step;
  }
}

/*This a pointer to the above function*/
PyUFuncGenericFunction funcs[1] = {&double_logit};

/* These are the input and return dtypes of logit.*/
static char types[2] = {NPY_DOUBLE, NPY_DOUBLE};

static void *data[1] = {NULL};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "ibm2ieee",
  NULL,
  -1,
  LogitMethods,
  NULL,
  NULL,
  NULL,
    NULL
};

PyMODINIT_FUNC PyInit_ibm2ieee(void)
{
  PyObject *m, *logit, *d;
  m = PyModule_Create(&moduledef);
  if (!m) {
    return NULL;
  }

  import_array();
  import_umath();

  logit = PyUFunc_FromFuncAndData(funcs, data, types, 1, 1, 1,
                                  PyUFunc_None, "logit",
                                  "logit_docstring", 0);

  d = PyModule_GetDict(m);

  PyDict_SetItemString(d, "logit", logit);
  Py_DECREF(logit);

  return m;
}
#else
PyMODINIT_FUNC initibm2ieee(void)
{
  PyObject *m, *logit, *d;


  m = Py_InitModule("ibm2ieee", LogitMethods);
  if (m == NULL) {
    return;
  }

  import_array();
  import_umath();

  logit = PyUFunc_FromFuncAndData(funcs, data, types, 1, 1, 1,
                                  PyUFunc_None, "logit",
                                  "logit_docstring", 0);

  d = PyModule_GetDict(m);

  PyDict_SetItemString(d, "logit", logit);
  Py_DECREF(logit);
}
#endif
