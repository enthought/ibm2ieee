#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"


static PyMethodDef IBM2IEEEMethods[] = {
  {NULL, NULL, 0, NULL}
};


static void ibm2ieee32(char **args, npy_intp *dimensions,
                       npy_intp* steps, void* data)
{
  npy_intp i;
  npy_intp n = dimensions[0];
  char *in = args[0], *out = args[1];
  npy_intp in_step = steps[0], out_step = steps[1];

  npy_uint32 tmp, significand, exponent, sign, result;

  for (i = 0; i < n; i++) {
    tmp = *(npy_uint32 *)in;

    significand = tmp & (npy_uint32)0x00FFFFFFU;
    exponent = ((tmp & (npy_uint32)0x7F000000U) >> 22) - 130;
    sign = tmp & (npy_uint32)0x80000000U;

    if (significand) {
      /* normalise */
      while (significand < (npy_uint32)0x00800000U) {
        significand <<= 1;
        exponent -= 1U;
      }

      if ((npy_int32)exponent <= 0) {
        if ((npy_int32)exponent < -23) {
          /* underflow to zero */
          significand = 0;
          exponent = 0;
        }
        else {
          /* underflow; apply round-ties-to-even */
          npy_uint32 shift, mask, has_remainder, is_odd;
          shift = -exponent;
          mask = ~((~0U) << shift);
          has_remainder = !!(significand & mask);
          significand >>= shift;
          is_odd = !!(significand & 2);
          significand += (is_odd | has_remainder);
          significand >>= 1;
          /* not possible for significand to be more than 2**23 at this point,
             due to limitations of IBM single precision */
          exponent = 0U;
        }
      }
      else if (exponent >= 255U) {
        /* overflow to infinity */
        significand = 0U;
        exponent = 255U;
      }
      else {
        significand -= 0x00800000U;
      }
    }
    else {
      /* zero */
      exponent = 0;
    }
    result = sign | (exponent << 23) | significand;
    *((npy_uint32 *)out) = result;

    in += in_step;
    out += out_step;
  }
}

/*This a pointer to the above function*/
PyUFuncGenericFunction funcs[1] = {&ibm2ieee32};

static char types[2] = {NPY_UINT32, NPY_FLOAT32};

static void *data[1] = {NULL};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "ibm2ieee",
  NULL,
  -1,
  IBM2IEEEMethods,
  NULL,
  NULL,
  NULL,
    NULL
};

PyMODINIT_FUNC PyInit_ibm2ieee(void)
{
  PyObject *m, *ibm2float32, *d;
  m = PyModule_Create(&moduledef);
  if (!m) {
    return NULL;
  }

  import_array();
  import_umath();

  ibm2float32 = PyUFunc_FromFuncAndData(funcs, data, types, 1, 1, 1,
                                        PyUFunc_None, "ibm2float32",
                                        "docstring", 0);

  d = PyModule_GetDict(m);

  PyDict_SetItemString(d, "ibm2float32", ibm2float32);
  Py_DECREF(ibm2float32);

  return m;
}
#else
PyMODINIT_FUNC initibm2ieee(void)
{
  PyObject *m, *ibm2ieee32, *d;


  m = Py_InitModule("ibm2ieee", IBM2IEEEMethods);
  if (m == NULL) {
    return;
  }

  import_array();
  import_umath();

  ibm2ieee32 = PyUFunc_FromFuncAndData(funcs, data, types, 1, 1, 1,
                                  PyUFunc_None, "ibm2ieee32",
                                  "ibm2ieee32_docstring", 0);

  d = PyModule_GetDict(m);

  PyDict_SetItemString(d, "ibm2ieee32", ibm2ieee32);
  Py_DECREF(ibm2ieee32);
}
#endif
