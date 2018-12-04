The **ibm2ieee** package provides NumPy universal functions ("ufuncs") for
converting IBM single-precision and double-precision hexadecimal floats
to the IEEE 754-format floats used by Python and NumPy on almost all
current platforms.


Features
--------

- Fast: 200-400 million values converted per second on a typical modern
  machine, assuming normal inputs.
- Correct: converted results are correctly rounded, according to the default
  IEEE 754 round-ties-to-even rounding mode. Corner cases (overflow, underflow,
  subnormal results, signed zeros, non-normalised input) are all handled
  correctly. Where the rounded converted value is out of range for the target
  type, an appropriately-signed infinity is returned.
- Handles both single-precision and double-precision input and output formats.

Portability note: the conversion functions provided in this module assume that
``numpy.float32`` and ``numpy.float64`` are based on the standard IEEE 754
binary32 and binary64 floating-point formats. This is true on the overwhelming
majority of current platforms, but is not guaranteed by the relevant language
standards.


Usage
-----

The package provides two functions:

- ``ibm2float32`` converts IBM single- or double-precision data to
  IEEE 754 single-precision values, in ``numpy.float32`` format.

- ``ibm2float64`` converts IBM single- or double-precision data to
  IEEE 754 double-precision values, in ``numpy.float64`` format.

For both functions, IBM single-precision input data must be represented
using the ``numpy.uint32`` dtype, while IBM double-precision inputs must
be represented using ``numpy.uint64``.

Both functions assume that the IBM data have been converted to NumPy integer
format in such a way that the most significant bits of the floating-point
number become the most significant bits of the integer values. So when decoding
byte data representing IBM hexadecimal floating-point numbers, it's important
to take the endianness of the byte data into account. See the Examples section
below for an example of converting big-endian byte data.


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

When converting byte data read from a file, it's important to know the
endianness of that data (which is frequently big-endian in historical data
files using IBM hex floating-point). Here's an example of converting IBM
single-precision data stored in big-endian form to ``numpy.float32``. Note the
use of the ``'>u4'`` dtype when converting the bytestring to a NumPy ``uint32``
array. For little-endian input data, you would use ``'<u4'`` instead.

>>> input_data = b'\xc12C\xf7\xc1\x19!\xfb\x00\x00\x00\x00A\x19!\xfbA2C\xf7'
>>> input_as_uint32 = numpy.frombuffer(input_data, dtype='>u4')
>>> input_as_uint32
array([3241296887, 3239649787,          0, 1092166139, 1093813239],
      dtype=uint32)
>>> ibm2float32(input_as_uint32)
array([-3.141593, -1.570796,  0.      ,  1.570796,  3.141593],
      dtype=float32)


Notes on the formats
--------------------

The IBM single-precision format has a precision of 6 hexadecimal digits, which
in practice translates to a precision of 21-24 bits, depending on the binade
that the relevant value belongs to. IEEE 754 single-precision has a precision
of 24 bits. So all not-too-small, not-too-large IBM single-precision values can
be translated to IEEE 754 single-precision values with no loss of precision.
However, the IBM single precision range is larger than the corresponding IEEE
754 range, so extreme IBM single-precision values may overflow to infinity,
underflow to zero, or be rounded to a subnormal value when converted to IEEE
754 single-precision.

For double-precision conversions, the tradeoff works the other way: the IBM
double-precision format has an effective precision of 53-56 bits, while IEEE
754 double-precision has 53-bit precision. So most IBM values will be rounded
when converted to IEEE 754. However, the IEEE 754 double-precision range is
larger than that of IBM double-precision, so there's no danger of overflow,
underflow, or reduced-precision subnormal results when converting IBM
double-precision to IEEE 754 double-precision.

Every IBM single-precision value can be exactly represented in IEEE 754
double-precision, so if you want a lossless representation of IBM
single-precision data, use ``ibm2float64``.

Note that the IBM formats do not allow representations of special values like
infinities and NaNs. However, signed zeros are representable, and the sign of a
zero is preserved under all conversions.


Installation
------------

The latest release of ibm2ieee is available from the Python Package Index, at
https://pypi.org/project/ibm2ieee. It can be installed with ``pip`` in the
usual way::

    pip install ibm2ieee

Note that it includes a C extension, so you'll need a compiler on your system
to be able to install.


License
-------

The ibm2ieee package is copyright (c) 2018, Enthought, Inc.

The ibm2ieee package is licensed under a standard BSD 3-clause License. See the
LICENSE file for details.
