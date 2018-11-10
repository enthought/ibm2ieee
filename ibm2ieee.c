#include "Python.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

/* masks */
#define IBM32_SIGN ((npy_uint32)0x80000000U)
#define IBM32_EXPT ((npy_uint32)0x7f000000U)
#define IBM32_FRAC ((npy_uint32)0x00ffffffU)

#define IBM64_SIGN ((npy_uint64)0x8000000000000000U)
#define IBM64_EXPT ((npy_uint64)0x7f00000000000000U)
#define IBM64_FRAC ((npy_uint64)0x00ffffffffffffffU)


static int bitlength32(npy_uint32 n)
{
  int n_bits = 0;
  while (n) {
    n >>= 1;
    n_bits += 1;
  }
  return n_bits;
}

static int bitlength64(npy_uint64 n)
{
  int n_bits = 0;
  while (n) {
    n >>= 1;
    n_bits += 1;
  }
  return n_bits;
}

/* right shift with rounded result, using round-ties-to-even.

   Returns the closest integer to n / 2**shift, rounding ties
   to even. shift must be positive. */

static npy_uint32 rshift_ties_to_even32(npy_uint32 n, int shift) {
  npy_uint32 mask, has_remainder, is_odd;

  if (shift > 32) {
    return 0;
  }
  mask = ~((~(npy_uint32)0) << (shift - 1));
  has_remainder = !!(n & mask);
  n >>= shift - 1;
  is_odd = !!(n & 2);
  n += (is_odd | has_remainder);
  return n >> 1;
}

static npy_uint64 rshift_ties_to_even64(npy_uint64 n, int shift) {
  npy_uint64 mask, has_remainder, is_odd;

  if (shift > 64) {
    return 0;
  }
  mask = ~((~(npy_uint64)0) << (shift - 1));
  has_remainder = !!(n & mask);
  n >>= shift - 1;
  is_odd = !!(n & 2);
  n += (is_odd | has_remainder);
  return n >> 1;
}

static npy_uint32 ibm32ieee32(npy_uint32 ibm)
{
  int exponent, target_exponent, shift;
  npy_uint32 significand = ibm & IBM32_FRAC;
  npy_uint32 sign = ibm & IBM32_SIGN;

  /* Quick return for zeros. */
  if (!significand) {
    return sign;
  }

  exponent = ((ibm & IBM32_EXPT) >> 22) - 155;
  target_exponent = exponent + bitlength32(significand);
  if (target_exponent >= 254) {
    target_exponent = 254;
    significand = 0x800000U;
  }
  else {
    target_exponent = target_exponent >= 0 ? target_exponent : 0;
    shift = exponent + 24 - target_exponent;
    significand = shift >= 0 ? significand << shift :
      rshift_ties_to_even32(significand, -shift);
  }
  return sign + ((npy_uint32)target_exponent << 23) + significand;
}

static npy_uint32 ibm64ieee32(npy_uint64 ibm)
{
  int exponent, target_exponent, shift;
  npy_uint64 significand = ibm & IBM64_FRAC;
  npy_uint32 sign = (ibm & IBM64_SIGN) >> 32;

  /* Quick return for zeros. */
  if (!significand) {
    return sign;
  }

  exponent = ((ibm & IBM64_EXPT) >> 54) - 187;
  target_exponent = exponent + bitlength64(significand);
  if (target_exponent >= 254) {
    target_exponent = 254;
    significand = 0x800000U;
  }
  else {
    target_exponent = target_exponent >= 0 ? target_exponent : 0;
    shift = exponent + 24 - target_exponent;
    significand = shift >= 0 ? significand << shift :
      rshift_ties_to_even64(significand, -shift);
  }
  return sign + ((npy_uint32)target_exponent << 23) + (npy_uint32)significand;
}



static PyMethodDef IBM2IEEEMethods[] = {
  {NULL, NULL, 0, NULL}
};


static void ibm32ieee32_ufunc(char **args, npy_intp *dimensions,
                       npy_intp* steps, void* data)
{
  npy_intp i;
  npy_intp n = dimensions[0];
  char *in = args[0], *out = args[1];
  npy_intp in_step = steps[0], out_step = steps[1];

  for (i = 0; i < n; i++) {
    *((npy_uint32 *)out) = ibm32ieee32(*(npy_uint32 *)in);
    in += in_step;
    out += out_step;
  }
}


static void ibm64ieee32_ufunc(char **args, npy_intp *dimensions,
                        npy_intp* steps, void* data)
{
  npy_intp i;
  npy_intp n = dimensions[0];
  char *in = args[0], *out = args[1];
  npy_intp in_step = steps[0], out_step = steps[1];

  for (i = 0; i < n; i++) {
    *((npy_uint32 *)out) = ibm64ieee32(*(npy_uint64 *)in);
    in += in_step;
    out += out_step;
  }
}

PyUFuncGenericFunction funcs[2] = {
  &ibm32ieee32_ufunc,
  &ibm64ieee32_ufunc,
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
