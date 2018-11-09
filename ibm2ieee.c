#include "Python.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"


static int bitlength32(npy_uint32 n)
{
  int n_bits = 0;
  while (n) {
    n >>= 1;
    n_bits += 1;
  }
  return n_bits;
}

static npy_uint32 rshift_ties_to_even(npy_uint32 n, int shift) {
  npy_uint32 mask, has_remainder, is_odd;

  if (shift > 32) {
    return 0U;
  }
  mask = ~((~0U) << (shift - 1));
  has_remainder = !!(n & mask);
  n >>= shift - 1;
  is_odd = !!(n & 2);
  n += (is_odd | has_remainder);
  return n >> 1;
}



static PyMethodDef IBM2IEEEMethods[] = {
  {NULL, NULL, 0, NULL}
};


static void ibm32ieee32(char **args, npy_intp *dimensions,
                       npy_intp* steps, void* data)
{
  npy_intp i;
  npy_intp n = dimensions[0];
  char *in = args[0], *out = args[1];
  npy_intp in_step = steps[0], out_step = steps[1];

  for (i = 0; i < n; i++) {
    int target_exponent;
    npy_uint32 tmp = *(npy_uint32 *)in;
    npy_uint32 significand = tmp & (npy_uint32)0x00FFFFFFU;
    npy_uint32 sign = tmp & (npy_uint32)0x80000000U;

    if (significand) {
      int exponent = ((tmp & (npy_uint32)0x7F000000U) >> 22) - 155;
      target_exponent = exponent + bitlength32(significand);
      if (target_exponent >= 254) {
        target_exponent = 254;
        significand = 0x800000U;
      }
      else {
        int shift;
        target_exponent = target_exponent >= 0 ? target_exponent : 0;
        shift = exponent + 24 - target_exponent;
        significand = shift >= 0 ? significand << shift :
          rshift_ties_to_even(significand, -shift);
      }
    }
    else {
      target_exponent = 0;
    }
    *((npy_uint32 *)out) = sign + ((npy_uint32)target_exponent << 23) + significand;

    in += in_step;
    out += out_step;
  }
}


static void ibm64ieee32(char **args, npy_intp *dimensions,
                        npy_intp* steps, void* data)
{
  npy_intp i;
  npy_intp n = dimensions[0];
  char *in = args[0], *out = args[1];
  npy_intp in_step = steps[0], out_step = steps[1];

  for (i = 0; i < n; i++) {
    npy_uint64 tmp, significand;
    npy_uint32 sign, significand_out, result;
    int exponent;

    tmp = *(npy_uint64 *)in;

    significand = tmp & (npy_uint64)0x00FFFFFFFFFFFFFFU;
    exponent = ((tmp & (npy_uint64)0x7F00000000000000U) >> 54) - 130;
    sign = (tmp & (npy_uint64)0x8000000000000000U) >> 32;

    if (significand) {
      /* normalise */
      while (significand < (npy_uint64)0x0080000000000000U) {
        significand <<= 1;
        exponent -= 1;
      }

      if (exponent <= 0) {
        /* underflow */
        if (exponent < -23) {
          /* underflow to zero */
          significand_out = 0U;
          exponent = 0;
        }
        else {
          /* underflow; apply round-ties-to-even */
          npy_uint64 shift, mask, has_remainder, is_odd;
          shift = 32 - exponent;
          mask = ~((~(npy_uint64)0) << shift);
          has_remainder = !!(significand & mask);
          significand >>= shift;
          is_odd = !!(significand & 2);
          significand += (is_odd | has_remainder);
          significand >>= 1;
          significand_out = significand;
          exponent = 0;
        }
      }
      else if (exponent >= 255) {
        /* overflow to infinity */
        significand_out = 0U;
        exponent = 255;
      }
      else {
        npy_uint64 mask, has_remainder, is_odd;
        significand -= 0x0080000000000000U;
        /* Now have a 55-bit fraction; need a 23-bit fraction, so remove
           32 bits, and round. */
        mask = 0x7fffffffULL;
        has_remainder = !!(significand & mask);
        significand >>= 31;
        is_odd = !!(significand & 2);
        significand += (is_odd | has_remainder);
        significand >>= 1;
        significand_out = significand;
      }
    }
    else {
      /* zero */
      significand_out = 0;
      exponent = 0;
    }
    result = sign + ((npy_uint32)exponent << 23) + significand_out;

    *((npy_uint32 *)out) = result;

    in += in_step;
    out += out_step;
  }
}

PyUFuncGenericFunction funcs[2] = {
  &ibm32ieee32,
  &ibm64ieee32
};

static char types[4] = {
  NPY_UINT32, NPY_FLOAT32,
  NPY_UINT64, NPY_FLOAT32
};

static void *data[2] = {
  NULL,
  NULL
};

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

  ibm2float32 = PyUFunc_FromFuncAndData(funcs, data, types, 2, 1, 1,
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
