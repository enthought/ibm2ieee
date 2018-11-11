#include "Python.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"

/* Various format-related masks and constants */

#define IBM32_SIGN ((npy_uint32)0x80000000U)
#define IBM32_EXPT ((npy_uint32)0x7f000000U)
#define IBM32_FRAC ((npy_uint32)0x00ffffffU)
#define IBM32_BIAS 64
#define IBM32_PREC 24

#define IBM64_SIGN ((npy_uint64)0x8000000000000000U)
#define IBM64_EXPT ((npy_uint64)0x7f00000000000000U)
#define IBM64_FRAC ((npy_uint64)0x00ffffffffffffffU)
#define IBM64_BIAS 64
#define IBM64_PREC 56

#define IEEE32_PREC 24
#define IEEE32_MAXEXP 254     /* Maximum biased exponent for finite values. */
#define IEEE32_EXP_MIN (-149) /* Exponent of smallest power of two. */
#define IEEE32_EXP1 ((npy_uint32)0x00800000U)

#define IEEE64_PREC 53
#define IEEE64_EXP_MIN (-1074) /* Exponent of smallest power of two. */

/* Minimum number of bits needed to represent n. Assumes n positive. */

static int
bitlength32(npy_uint32 n)
{
    int n_bits = 0;
    while (n) {
        n >>= 1;
        n_bits += 1;
    }
    return n_bits;
}

/* Minimum number of bits needed to represent n. Assumes n positive. */

static int
bitlength64(npy_uint64 n)
{
    int n_bits = 0;
    while (n) {
        n >>= 1;
        n_bits += 1;
    }
    return n_bits;
}

/* Right shift with result rounded using round-ties-to-even.

   Returns the closest integer to n / 2**shift, rounding ties to even. shift
   must be positive, but is permitted to exceed 31. */

static npy_uint32
rshift_ties_to_even32(npy_uint32 n, int shift)
{
    npy_uint32 trailing;

    if (shift > 32) {
        return 0U;
    }
    trailing = n & ~((~(npy_uint32)0) << (shift - 1));
    n >>= shift - 1;
    return (n + (trailing + (n & 2) > 0U)) >> 1;
}

/* Right shift with result rounded using round-ties-to-even.

   Returns the closest integer to n / 2**shift, rounding ties to even. shift
   must be positive, and is permitted to exceed 63. */

static npy_uint64
rshift_ties_to_even64(npy_uint64 n, int shift)
{
    npy_uint64 trailing;

    if (shift > 64) {
        return 0U;
    }
    trailing = n & ~((~(npy_uint64)0) << (shift - 1));
    n >>= shift - 1;
    return (n + (trailing + (n & 2) > 0U)) >> 1;
}

/* Convert IBM single-precision bit pattern to IEEE single-precision bit
   pattern. */

static npy_uint32
ibm32ieee32(npy_uint32 ibm)
{
    /* Overflow and underflow possible; rounding can only happen
       in subnormal cases. */
    int shift, shift_expt, shift_frac;
    npy_uint32 ibm_frac;
    npy_uint32 ieee_sign, ieee_expt, ieee_frac;

    ieee_sign = ibm & IBM32_SIGN;
    ibm_frac = ibm & IBM32_FRAC;

    /* Quick return for zeros. */
    if (!ibm_frac) {
        return ieee_sign;
    }

    shift_expt = ((ibm & IBM32_EXPT) >> (IBM32_PREC - 2)) -
                 (4 * IBM32_BIAS + IBM32_PREC + IEEE32_EXP_MIN);
    shift_frac = IEEE32_PREC - bitlength32(ibm_frac);
    shift = shift_frac <= shift_expt ? shift_frac : shift_expt;

    ieee_expt = shift_expt - shift;
    ieee_frac = shift >= 0 ? ibm_frac << shift
                           : rshift_ties_to_even32(ibm_frac, -shift);
    if (ieee_expt >= IEEE32_MAXEXP) {
        /* overflow */
        ieee_expt = IEEE32_MAXEXP;
        ieee_frac = IEEE32_EXP1;
    }
    return ieee_sign + (ieee_expt << (IEEE32_PREC - 1)) + ieee_frac;
}

/* Convert IBM double-precision bit pattern to IEEE single-precision bit
   pattern. */

static npy_uint32
ibm64ieee32(npy_uint64 ibm)
{
    /* Overflow and underflow possible; rounding can occur in both
       normal and subnormal cases. */
    int shift, shift_expt, shift_frac;
    npy_uint64 ibm_frac;
    npy_uint32 ieee_sign, ieee_expt, ieee_frac;

    ieee_sign = (ibm & IBM64_SIGN) >> 32;
    ibm_frac = ibm & IBM64_FRAC;

    /* Quick return for zeros. */
    if (!ibm_frac) {
        return ieee_sign;
    }

    shift_expt = ((ibm & IBM64_EXPT) >> (IBM64_PREC - 2)) -
                 (4 * IBM64_BIAS + IBM64_PREC + IEEE32_EXP_MIN);
    shift_frac = IEEE32_PREC - bitlength64(ibm_frac);
    shift = shift_frac <= shift_expt ? shift_frac : shift_expt;

    ieee_expt = shift_expt - shift;
    ieee_frac = shift >= 0 ? ibm_frac << shift
                           : rshift_ties_to_even64(ibm_frac, -shift);
    if (ieee_expt >= IEEE32_MAXEXP) {
        /* overflow */
        ieee_expt = IEEE32_MAXEXP;
        ieee_frac = IEEE32_EXP1;
    }
    return ieee_sign + (ieee_expt << (IEEE32_PREC - 1)) + ieee_frac;
}

/* Convert IBM single-precision bit pattern to IEEE double-precision bit
   pattern. */

static npy_uint64
ibm32ieee64(npy_uint32 ibm)
{
    /* This is the simplest of the four cases: there's no need to check for
       overflow or underflow, no possibility of subnormal output, and never
       any rounding. */
    int shift;
    npy_uint32 ibm_frac;
    npy_uint64 ieee_sign, ieee_expt, ieee_frac;

    ieee_sign = (npy_uint64)(ibm & IBM32_SIGN) << 32;
    ibm_frac = ibm & IBM32_FRAC;

    /* Quick return for zeros. */
    if (!ibm_frac) {
        return ieee_sign;
    }

    shift = IEEE64_PREC - bitlength32(ibm_frac);
    ieee_expt = ((ibm & IBM32_EXPT) >> (IBM32_PREC - 2)) - shift -
                (4 * IBM32_BIAS + IBM32_PREC + IEEE64_EXP_MIN);
    ieee_frac = (npy_uint64)ibm_frac << shift;
    return ieee_sign + (ieee_expt << (IEEE64_PREC - 1)) + ieee_frac;
}

/* Convert IBM double-precision bit pattern to IEEE double-precision bit
   pattern. */

static npy_uint64
ibm64ieee64(npy_uint64 ibm)
{
    /* No overflow or underflow possible, but the precision of the
       IBM double-precision format exceeds that of its IEEE counterpart,
       so we'll frequently need to round. */
    int shift;
    npy_uint64 ibm_frac;
    npy_uint64 ieee_sign, ieee_expt, ieee_frac;

    ieee_sign = ibm & IBM64_SIGN;
    ibm_frac = ibm & IBM64_FRAC;

    /* Quick return for zeros. */
    if (!ibm_frac) {
        return ieee_sign;
    }

    shift = IEEE64_PREC - bitlength64(ibm_frac);
    ieee_expt = ((ibm & IBM64_EXPT) >> (IBM64_PREC - 2)) -
                (4 * IBM64_BIAS + IBM64_PREC + IEEE64_EXP_MIN) - shift;
    ieee_frac = shift >= 0 ? ibm_frac << shift
                           : rshift_ties_to_even64(ibm_frac, -shift);
    return ieee_sign + (ieee_expt << (IEEE64_PREC - 1)) + ieee_frac;
}

/* NumPy ufunc wrapper for ibm32ieee32 */

static void
ibm32ieee32_ufunc(char **args, npy_intp *dimensions, npy_intp *steps,
                  void *data)
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

/* NumPy ufunc wrapper for ibm64ieee32 */

static void
ibm64ieee32_ufunc(char **args, npy_intp *dimensions, npy_intp *steps,
                  void *data)
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

/* NumPy ufunc wrapper for ibm32ieee64 */

static void
ibm32ieee64_ufunc(char **args, npy_intp *dimensions, npy_intp *steps,
                  void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];

    for (i = 0; i < n; i++) {
        *((npy_uint64 *)out) = ibm32ieee64(*(npy_uint32 *)in);
        in += in_step;
        out += out_step;
    }
}

/* NumPy ufunc wrapper for ibm64ieee64 */

static void
ibm64ieee64_ufunc(char **args, npy_intp *dimensions, npy_intp *steps,
                  void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];

    for (i = 0; i < n; i++) {
        *((npy_uint64 *)out) = ibm64ieee64(*(npy_uint64 *)in);
        in += in_step;
        out += out_step;
    }
}

PyDoc_STRVAR(ibm2float32_docstring,
             "\
Convert IBM-format single- or double-precision float data, represented\n\
using types np.uint32 or np.uint64 (respectively), to np.float32.\n\
\n\
Examples\n\
--------\n\
>>> ibm2float32(np.uint32(0xc1180000))\n\
-1.5\n\
>>> ibm2float32(np.uint64(0x413243f6a8885a31))\n\
3.1415927\n\
>>> ibm2float32(np.uint32(0x61100000))\n\
inf\n\
");

PyUFuncGenericFunction ibm2float32_funcs[2] = {
    &ibm32ieee32_ufunc,
    &ibm64ieee32_ufunc,
};

static char ibm2float32_types[4] = {NPY_UINT32, NPY_FLOAT32, NPY_UINT64,
                                    NPY_FLOAT32};

static void *ibm2float32_data[2] = {NULL, NULL};

PyDoc_STRVAR(ibm2float64_docstring,
             "\
Convert IBM-format single- or double-precision float data, represented\n\
using types np.uint32 or np.uint64 (respectively), to np.float64.\n\
\n\
Examples\n\
--------\n\
>>> ibm2float64(np.uint32(0xc1180000))\n\
-1.5\n\
>>> ibm2float64(np.uint64(0x413243f6a8885a31))\n\
3.141592653589793\n\
>>> ibm2float64(np.uint32(0x61100000))\n\
3.402823669209385e+38\n\
");

PyUFuncGenericFunction ibm2float64_funcs[2] = {
    &ibm32ieee64_ufunc,
    &ibm64ieee64_ufunc,
};

static char ibm2float64_types[4] = {NPY_UINT32, NPY_FLOAT64, NPY_UINT64,
                                    NPY_FLOAT64};

static void *ibm2float64_data[2] = {NULL, NULL};

static PyMethodDef IBM2IEEEMethods[] = {{NULL, NULL, 0, NULL}};

#if PY_VERSION_HEX >= 0x03000000

static struct PyModuleDef moduledef = {PyModuleDef_HEAD_INIT,
                                       "_ibm2ieee",
                                       NULL,
                                       -1,
                                       IBM2IEEEMethods,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL};

PyMODINIT_FUNC
PyInit__ibm2ieee(void)
{
    PyObject *m, *d;
    PyObject *ibm2float32, *ibm2float64;

    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    d = PyModule_GetDict(m);

    import_array();
    import_umath();

    ibm2float32 = PyUFunc_FromFuncAndData(
        ibm2float32_funcs, ibm2float32_data, ibm2float32_types, 2, 1, 1,
        PyUFunc_None, "ibm2float32", ibm2float32_docstring, 0);
    PyDict_SetItemString(d, "ibm2float32", ibm2float32);
    Py_DECREF(ibm2float32);

    ibm2float64 = PyUFunc_FromFuncAndData(
        ibm2float64_funcs, ibm2float64_data, ibm2float64_types, 2, 1, 1,
        PyUFunc_None, "ibm2float64", ibm2float64_docstring, 0);
    PyDict_SetItemString(d, "ibm2float64", ibm2float64);
    Py_DECREF(ibm2float64);

    return m;
}

#else /* PY_VERSION_HEX >= 0x03000000 */

PyMODINIT_FUNC
init_ibm2ieee(void)
{
    PyObject *m, *d;
    PyObject *ibm2float32, *ibm2float64;

    m = Py_InitModule("_ibm2ieee", IBM2IEEEMethods);
    if (m == NULL) {
        return;
    }
    d = PyModule_GetDict(m);

    import_array();
    import_umath();

    ibm2float32 = PyUFunc_FromFuncAndData(
        ibm2float32_funcs, ibm2float32_data, ibm2float32_types, 2, 1, 1,
        PyUFunc_None, "ibm2float32", ibm2float32_docstring, 0);
    PyDict_SetItemString(d, "ibm2float32", ibm2float32);
    Py_DECREF(ibm2float32);

    ibm2float64 = PyUFunc_FromFuncAndData(
        ibm2float64_funcs, ibm2float64_data, ibm2float64_types, 2, 1, 1,
        PyUFunc_None, "ibm2float64", ibm2float64_docstring, 0);
    PyDict_SetItemString(d, "ibm2float64", ibm2float64);
    Py_DECREF(ibm2float64);
}

#endif /* PY_VERSION_HEX >= 0x03000000 */
