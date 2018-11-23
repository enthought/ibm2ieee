/*
Copyright (c) 2018, Enthought, Inc.
All rights reserved.
*/

#include "Python.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"


/* Format-related masks */

#define IBM32_SIGN ((npy_uint32)0x80000000U)
#define IBM32_EXPT ((npy_uint32)0x7f000000U)
#define IBM32_FRAC ((npy_uint32)0x00ffffffU)
#define IBM32_TOP  ((npy_uint32)0x00f00000U)
#define TIES_TO_EVEN_MASK32 ((npy_uint32)0xfffffffd)

#define IBM64_SIGN ((npy_uint64)0x8000000000000000U)
#define IBM64_EXPT ((npy_uint64)0x7f00000000000000U)
#define IBM64_FRAC ((npy_uint64)0x00ffffffffffffffU)
#define IBM64_TOP  ((npy_uint64)0x00f0000000000000U)
#define TIES_TO_EVEN_MASK64 ((npy_uint64)0xfffffffffffffffd)

#define IEEE32_MAXEXP 254     /* Maximum biased exponent for finite values. */
#define IEEE32_INFINITY ((npy_uint32)0x7f800000U)

/* Constant used to count number of leading bits in a nonzero hex digit
   via `(BITCOUNT_MAGIC >> (hex_digit*2)) & 3U`. */
#define BITCOUNT_MAGIC ((npy_uint32)0x000055afU)


/* IBM single-precision bit pattern to IEEE single-precision bit pattern. */

static npy_uint32
ibm32ieee32(npy_uint32 ibm)
{
    /* Overflow and underflow possible; rounding can only happen
       in subnormal cases. */
    int ibm_expt, ieee_expt, leading_zeros;
    npy_uint32 ibm_frac, top_digit;
    npy_uint32 ieee_sign, ieee_frac;

    ieee_sign = ibm & IBM32_SIGN;
    ibm_frac = ibm & IBM32_FRAC;

    /* Quick return for zeros. */
    if (!ibm_frac) {
        return ieee_sign;
    }

    /* Reduce shift by 2 to get a binary exponent from the hex exponent. */
    ibm_expt = (ibm & IBM32_EXPT) >> 22;

    /* Normalise significand, then count leading zeros in top hex digit. */
    top_digit = ibm_frac & IBM32_TOP;
    while (top_digit == 0) {
        ibm_frac <<= 4;
        ibm_expt -= 4;
        top_digit = ibm_frac & IBM32_TOP;
    }
    leading_zeros = (BITCOUNT_MAGIC >> (top_digit >> 19)) & 3U;
    ibm_frac <<= leading_zeros;

    /* Adjust exponents for the differing biases of the formats: the IBM bias
       is 64 hex digits, or 256 bits. The IEEE bias is 127. The difference is
       -129; we get an extra -1 from the different significand representations
       (0.f for IBM versus 1.f for IEEE), and another -1 to compensate for an
       evil trick that saves an operation on the fast path: we don't remove the
       hidden 1-bit from the IEEE significand, so in the final addition that
       extra bit ends in incrementing the exponent by one. */
    ieee_expt = ibm_expt - 131 - leading_zeros;

    if (ieee_expt >= 0 && ieee_expt < IEEE32_MAXEXP) {
        /* normal case; no shift needed */
        ieee_frac = ibm_frac;
        return ieee_sign + ((npy_uint32)ieee_expt << 23) + ieee_frac;
    }
    else if (ieee_expt >= IEEE32_MAXEXP) {
        /* overflow */
        return ieee_sign + IEEE32_INFINITY;
    }
    else if (ieee_expt >= -32) {
        /* possible subnormal result; shift significand right by -ieee_expt
           bits, rounding the result with round-ties-to-even.

           The round-ties-to-even code deserves some explanation: out of the
           bits we're shifting out, let's call the most significant bit the
           "rounding bit", and the rest the "trailing bits". We'll call the
           least significant bit that *isn't* shifted out the "parity bit".
           So for an example 5-bit shift right, we'd label the bits as follows:

           Before the shift:

                   ...xxxprtttt
                              ^
              msb            lsb

           After the shift:

                        ...xxxp
                              ^
              msb            lsb

           with the result possibly incremented by one.

           For round-ties-to-even, we need to round up if both (a) the rounding
           bit is 1, and (b) either the parity bit is 1, or at least one of the
           trailing bits is 1. We construct a mask that has 1-bits in the
           parity bit position and trailing bit positions, and use that to
           check condition (b). So for example in the 5-bit shift right, the
           mask looks like this:

                   ...000101111 : mask
                   ...xxxprtttt : ibm_frac
                              ^
              msb            lsb

           We then shift right by (shift - 1), add 1 if (ibm & mask) is
           nonzero, and then do a final shift by one to get the rounded
           value. Note that this approach avoids the possibility of
           trying to shift a width-32 value by 32, which would give
           undefined behaviour (see C99 6.5.7p3).
         */
        npy_uint32 mask = ~(TIES_TO_EVEN_MASK32 << (-1 - ieee_expt));
        npy_uint32 round_up = (ibm_frac & mask) > 0U;
        ieee_frac = ((ibm_frac >> (-1 - ieee_expt)) + round_up) >> 1;
        return ieee_sign + ieee_frac;
    }
    else {
        /* underflow to zero */
        return ieee_sign;
    }
}


/* IBM double-precision bit pattern to IEEE single-precision bit pattern. */

static npy_uint32
ibm64ieee32(npy_uint64 ibm)
{
    /* Overflow and underflow possible; rounding can occur in both
       normal and subnormal cases. */
    int ibm_expt, ieee_expt, leading_zeros;
    npy_uint64 ibm_frac, top_digit;
    npy_uint32 ieee_sign, ieee_frac;

    ieee_sign = (ibm & IBM64_SIGN) >> 32;
    ibm_frac = ibm & IBM64_FRAC;

    /* Quick return for zeros. */
    if (!ibm_frac) {
        return ieee_sign;
    }

    /* Reduce shift by 2 to get a binary exponent from the hex exponent. */
    ibm_expt = (ibm & IBM64_EXPT) >> 54;

    /* Normalise significand, then count leading zeros in top hex digit. */
    top_digit = ibm_frac & IBM64_TOP;
    while (top_digit == 0) {
        ibm_frac <<= 4;
        ibm_expt -= 4;
        top_digit = ibm_frac & IBM64_TOP;
    }
    leading_zeros = (BITCOUNT_MAGIC >> (top_digit >> 51)) & 3U;

    ibm_frac <<= leading_zeros;
    ieee_expt = ibm_expt - 131 - leading_zeros;

    if (ieee_expt >= 0 && ieee_expt < IEEE32_MAXEXP) {
        /* normal case; shift right 32, with round-ties-to-even */
        npy_uint32 round_up = (ibm_frac & (npy_uint64)(0x17fffffff)) > 0U;
        ieee_frac = ((ibm_frac >> 31) + round_up) >> 1;
        return ieee_sign + ((npy_uint32)ieee_expt << 23) + ieee_frac;
    }
    else if (ieee_expt >= IEEE32_MAXEXP) {
        /* overflow */
        return ieee_sign + IEEE32_INFINITY;
    }
    else if (ieee_expt >= -32) {
        /* possible subnormal; shift right with round-ties-to-even */
        npy_uint64 mask = ~(TIES_TO_EVEN_MASK64 << (31 - ieee_expt));
        npy_uint32 round_up = (ibm_frac & mask) > 0U;
        ieee_frac = ((ibm_frac >> (31 - ieee_expt)) + round_up) >> 1;
        return ieee_sign + ieee_frac;
    }
    else {
        /* underflow to zero */
        return ieee_sign;
    }
}


/* IBM single-precision bit pattern to IEEE double-precision bit pattern. */

static npy_uint64
ibm32ieee64(npy_uint32 ibm)
{
    /* This is the simplest of the four cases: there's no need to check for
       overflow or underflow, no possibility of subnormal output, and never
       any rounding. */
    int ibm_expt, ieee_expt, leading_zeros;
    npy_uint32 ibm_frac, top_digit;
    npy_uint64 ieee_sign, ieee_frac;

    ieee_sign = (npy_uint64)(ibm & IBM32_SIGN) << 32;
    ibm_frac = ibm & IBM32_FRAC;

    /* Quick return for zeros. */
    if (!ibm_frac) {
        return ieee_sign;
    }

    /* Reduce shift by 2 to get a binary exponent from the hex exponent. */
    ibm_expt = (ibm & IBM32_EXPT) >> 22;

    /* Normalise significand, then count leading zeros in top hex digit. */
    top_digit = ibm_frac & IBM32_TOP;
    while (top_digit == 0) {
        ibm_frac <<= 4;
        ibm_expt -= 4;
        top_digit = ibm_frac & IBM32_TOP;
    }
    leading_zeros = (BITCOUNT_MAGIC >> (top_digit >> 19)) & 3U;

    /* Adjust exponents for the differing biases of the formats: the IBM bias
       is 64 hex digits, or 256 bits. The IEEE bias is 1023. The difference is
       767; we get an extra -1 from the different significand representations
       (0.f for IBM versus 1.f for IEEE), and another -1 to compensate for an
       evil trick that saves an operation: we don't remove the hidden 1-bit
       from the IEEE significand, so in the final addition that extra bit ends
       in incrementing the exponent by one. */
    ieee_expt = ibm_expt + 765 - leading_zeros;
    ieee_frac = (npy_uint64)ibm_frac << (29 + leading_zeros);
    return ieee_sign + ((npy_uint64)ieee_expt << 52) + ieee_frac;
}


/* IBM double-precision bit pattern to IEEE double-precision bit pattern. */

static npy_uint64
ibm64ieee64(npy_uint64 ibm)
{
    /* No overflow or underflow possible, but the precision of the
       IBM double-precision format exceeds that of its IEEE counterpart,
       so we'll frequently need to round. */
    int ibm_expt, ieee_expt, leading_zeros;
    npy_uint64 ibm_frac, top_digit;
    npy_uint64 ieee_sign, ieee_frac, round_up;

    ieee_sign = ibm & IBM64_SIGN;
    ibm_frac = ibm & IBM64_FRAC;

    /* Quick return for zeros. */
    if (!ibm_frac) {
        return ieee_sign;
    }

    /* Reduce shift by 2 to get a binary exponent from the hex exponent. */
    ibm_expt = (ibm & IBM64_EXPT) >> 54;

    /* Normalise significand, then count leading zeros in top hex digit. */
    top_digit = ibm_frac & IBM64_TOP;
    while (top_digit == 0) {
        ibm_frac <<= 4;
        ibm_expt -= 4;
        top_digit = ibm_frac & IBM64_TOP;
    }
    leading_zeros = (BITCOUNT_MAGIC >> (top_digit >> 51)) & 3U;

    ibm_frac <<= leading_zeros;
    ieee_expt = ibm_expt + 765 - leading_zeros;

    /* Right-shift by 3 bits (the difference between the IBM and IEEE
       significand lengths), rounding with round-ties-to-even. */
    round_up = (ibm_frac & (npy_uint64)0xb) > 0U;
    ieee_frac = ((ibm_frac >> 2) + round_up) >> 1;
    return ieee_sign + ((npy_uint64)ieee_expt << 52) + ieee_frac;
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
