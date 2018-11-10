#include "Python.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"

/* Various format-related masks and constants */

#define IBM_BIAS 64

#define IBM32_SIGN ((npy_uint32)0x80000000U)
#define IBM32_EXPT ((npy_uint32)0x7f000000U)
#define IBM32_FRAC ((npy_uint32)0x00ffffffU)
#define IBM32_PREC 24

#define IBM64_SIGN ((npy_uint64)0x8000000000000000U)
#define IBM64_EXPT ((npy_uint64)0x7f00000000000000U)
#define IBM64_FRAC ((npy_uint64)0x00ffffffffffffffU)
#define IBM64_PREC 56

#define IEEE32_PREC 24
#define IEEE32_MAXEXP 254
#define IEEE32_EXP_MIN (-149)
#define IEEE32_EXP1 ((npy_uint32)0x00800000U)

#define IEEE64_PREC 53
#define IEEE64_MAXEXP 2046
#define IEEE64_EXP_MIN (-1074)
#define IEEE64_EXP1 ((npy_uint64)0x0010000000000000U)

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
   must be positive. */

static npy_uint32
rshift_ties_to_even32(npy_uint32 n, int shift)
{
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

static npy_uint64
rshift_ties_to_even64(npy_uint64 n, int shift)
{
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

/* Convert IBM single-precision bit pattern to IEEE single-precision bit
 * pattern. */

static npy_uint32
ibm32ieee32(npy_uint32 ibm)
{
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
                 (4 * IBM_BIAS + IBM32_PREC + IEEE32_EXP_MIN);
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
 * pattern. */

static npy_uint32
ibm64ieee32(npy_uint64 ibm)
{
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
                 (4 * IBM_BIAS + IBM64_PREC + IEEE32_EXP_MIN);
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
 * pattern. */

static npy_uint64
ibm32ieee64(npy_uint32 ibm)
{
    int shift, shift_expt, shift_frac;
    npy_uint32 ibm_frac;
    npy_uint64 ieee_sign, ieee_expt, ieee_frac;

    ieee_sign = (npy_uint64)(ibm & IBM32_SIGN) << 32;
    ibm_frac = ibm & IBM32_FRAC;

    /* Quick return for zeros. */
    if (!ibm_frac) {
        return ieee_sign;
    }

    shift_expt = ((ibm & IBM32_EXPT) >> (IBM32_PREC - 2)) -
                 (4 * IBM_BIAS + IBM32_PREC + IEEE64_EXP_MIN);
    shift_frac = IEEE64_PREC - bitlength32(ibm_frac);
    shift = shift_frac <= shift_expt ? shift_frac : shift_expt;

    ieee_expt = shift_expt - shift;
    ieee_frac = shift >= 0 ? (npy_uint64)ibm_frac << shift
                           : rshift_ties_to_even32(ibm_frac, -shift);
    if (ieee_expt >= IEEE64_MAXEXP) {
        /* overflow */
        ieee_expt = IEEE64_MAXEXP;
        ieee_frac = IEEE64_EXP1;
    }
    return ieee_sign + (ieee_expt << (IEEE64_PREC - 1)) + ieee_frac;
}

/* Convert IBM double-precision bit pattern to IEEE double-precision bit
 * pattern. */

static npy_uint64
ibm64ieee64(npy_uint64 ibm)
{
    int shift, shift_expt, shift_frac;
    npy_uint64 ibm_frac;
    npy_uint64 ieee_sign, ieee_expt, ieee_frac;

    ieee_sign = ibm & IBM64_SIGN;
    ibm_frac = ibm & IBM64_FRAC;

    /* Quick return for zeros. */
    if (!ibm_frac) {
        return ieee_sign;
    }

    shift_expt = ((ibm & IBM64_EXPT) >> (IBM64_PREC - 2)) -
                 (4 * IBM_BIAS + IBM64_PREC + IEEE64_EXP_MIN);
    shift_frac = IEEE64_PREC - bitlength64(ibm_frac);
    shift = shift_frac <= shift_expt ? shift_frac : shift_expt;

    ieee_expt = shift_expt - shift;
    ieee_frac = shift >= 0 ? ibm_frac << shift
                           : rshift_ties_to_even64(ibm_frac, -shift);
    if (ieee_expt >= IEEE64_MAXEXP) {
        /* overflow */
        ieee_expt = IEEE64_MAXEXP;
        ieee_frac = IEEE64_EXP1;
    }
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

PyUFuncGenericFunction float32_funcs[2] = {
    &ibm32ieee32_ufunc,
    &ibm64ieee32_ufunc,
};

static char float32_types[4] = {NPY_UINT32, NPY_FLOAT32, NPY_UINT64,
                                NPY_FLOAT32};

static void *float32_data[2] = {NULL, NULL};

PyUFuncGenericFunction float64_funcs[2] = {
    &ibm32ieee64_ufunc,
    &ibm64ieee64_ufunc,
};

static char float64_types[4] = {NPY_UINT32, NPY_FLOAT64, NPY_UINT64,
                                NPY_FLOAT64};

static void *float64_data[2] = {NULL, NULL};

static PyMethodDef IBM2IEEEMethods[] = {{NULL, NULL, 0, NULL}};

#if PY_VERSION_HEX >= 0x03000000

static struct PyModuleDef moduledef = {PyModuleDef_HEAD_INIT,
                                       "ibm2ieee",
                                       NULL,
                                       -1,
                                       IBM2IEEEMethods,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL};

PyMODINIT_FUNC
PyInit_ibm2ieee(void)
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
        float32_funcs, float32_data, float32_types, 2, 1, 1, PyUFunc_None,
        "ibm2float32", "ibm2float32_docstring", 0);
    PyDict_SetItemString(d, "ibm2float32", ibm2float32);
    Py_DECREF(ibm2float32);

    ibm2float64 = PyUFunc_FromFuncAndData(
        float64_funcs, float64_data, float64_types, 2, 1, 1, PyUFunc_None,
        "ibm2float64", "ibm2float64_docstring", 0);
    PyDict_SetItemString(d, "ibm2float64", ibm2float64);
    Py_DECREF(ibm2float64);

    return m;
}

#else /* PY_VERSION_HEX >= 0x03000000 */

PyMODINIT_FUNC
initibm2ieee(void)
{
    PyObject *m, *d;
    PyObject *ibm2float32, *ibm2float64;

    m = Py_InitModule("ibm2ieee", IBM2IEEEMethods);
    if (m == NULL) {
        return;
    }
    d = PyModule_GetDict(m);

    import_array();
    import_umath();

    ibm2float32 = PyUFunc_FromFuncAndData(
        float32_funcs, float32_data, float32_types, 2, 1, 1, PyUFunc_None,
        "ibm2float32", "ibm2float32_docstring", 0);
    PyDict_SetItemString(d, "ibm2float32", ibm2float32);
    Py_DECREF(ibm2float32);

    ibm2float64 = PyUFunc_FromFuncAndData(
        float64_funcs, float64_data, float64_types, 2, 1, 1, PyUFunc_None,
        "ibm2float64", "ibm2float64_docstring", 0);
    PyDict_SetItemString(d, "ibm2float64", ibm2float64);
    Py_DECREF(ibm2float64);
}

#endif /* PY_VERSION_HEX >= 0x03000000 */
