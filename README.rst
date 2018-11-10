The **ibm2ieee** package provides NumPy universal functions ("ufuncs") for
converting IBM single-precision and double-precision hexadecimal floats
to the IEEE 754-format floats used by NumPy.

Usage
=====
There are two functions: ``ibm2float32`` converts IBM single- or
double-precision data to ``numpy.float32`` format, while ``ibm2float64``
converts IBM single- or double-precision data to ``numpy.float64`` format.

For both functions, IBM single-precision input data must be represented
using the ``numpy.uint32`` dtype, while IBM double-precision inputs must
be represented using ``numpy.uint64``.

All conversions are correctly rounded, using the standard round-ties-to-even
rounding mode. Where the converted value is out of range for the target type,
an appropriately-signed infinity is returned. Subnormal results are handled
correctly, with no double rounding.

Examples
--------
>>> import numpy
>>> from ibm2ieee import ibm2float32, ibm2float64
>>> ibm2float32(numpy.uint32(0xc1180000))
-1.5
>>> type(ibm2float32(numpy.uint32(0xc1180000)))
<class 'numpy.float32'>
>>> ibm2float32(numpy.uint64(0x413243f6a8885a31))
3.1415927
>>> ibm2float32(numpy.uint32(0x61100000))
inf
>>> ibm2float64(numpy.uint32(0xc1180000))
-1.5
>>> ibm2float64(numpy.uint64(0x413243f6a8885a31))
3.141592653589793
>>> ibm2float64(numpy.uint32(0x61100000))
3.402823669209385e+38
>>> input_array = numpy.arange(
        0x40fffffe, 0x41000002, dtype=numpy.uint32).reshape(2, 2)
>>> input_array
array([[1090519038, 1090519039],
       [1090519040, 1090519041]], dtype=uint32)
>>> ibm2float64(input_array)
array([[9.99999881e-01, 9.99999940e-01],
       [0.00000000e+00, 9.53674316e-07]])

Notes on the formats
--------------------
The IBM single-precision format has a precision of 6 hexadecimal digits, which
in practice translates to a precision of 21-24 bits, depending on the binade
that the relevant value belongs to. IEEE 754 single precision (properly, the
binary32 type) has precision 24. Thus all not-too-small, not-too-large IBM
single-precision values can be translated to IEEE single-precision values with
no rounding or precision loss. However, the IBM single precision range is
larger than the corresponding IEEE 754 range, so some IBM single-precision
values will overflow to infinity or underflow to a zero or subnormal value when
converted to IEEE 754 single-precision.

For double-precision, the tradeoff works the other way: the IBM
double-precision format has an effective precision of 53-56 bits, while IEEE
754 double-precision has 53-bit precision. So most IBM values will be rounded
when converting to IEEE. However, the IEEE 754 double-precision range is larger
than that of IBM double-precision, so there's no danger of underflow or
overflow when converting IBM double-precision to IEEE double-precision.

Every IBM single-precision value can be exactly represented in IEEE 754
double-precision.

Note that the IBM formats do not allow representations of special values
like infinities and NaNs. However, signed zeros are representable, and
the sign of a zero is preserved under all conversions.
