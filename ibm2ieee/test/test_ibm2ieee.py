# Copyright (c) 2018, Enthought, Inc.
# All rights reserved.

from __future__ import absolute_import, print_function, unicode_literals

import contextlib
from fractions import Fraction as F
import unittest

import numpy as np
import six

from ibm2ieee import ibm2float32, ibm2float64

# Two as a fraction, so that TWO**n is an exact operation for any n.
TWO = F(2)

# Format-related constants
IEEE32_MINEXP = -126
IEEE32_FRAC = 23
IEEE32_POSINF = 0x7f800000
IEEE32_SIGN = 0x80000000

IEEE64_MINEXP = -1022
IEEE64_FRAC = 52
IEEE64_POSINF = 0x7ff0000000000000
IEEE64_SIGN = 0x8000000000000000


# Simple and inefficient Python conversions, for testing purposes.

def round_ties_to_even(f):
    """
    Round a Fraction to the nearest integer, rounding ties to even.
    """
    if six.PY2:
        q, r = divmod(f, 1)
        round_up = 2 * r > 1 or (2 * r == 1 and q % 2 == 1)
        return q + round_up
    else:
        return round(f)


def ilog2_fraction(f):
    """
    floor(log2(f)) for a positive fraction f.
    """
    if f <= 0:
        raise ValueError("Requires a positive fraction.")
    e = f.numerator.bit_length() - f.denominator.bit_length()
    return e if F(2)**e <= f else e - 1


def fraction_from_ibm32(f):
    """
    Convert a bit-representation of an IBM single-precision hexadecimal
    float to a pair (sign, fraction).
    """
    # Fraction operations get confused by NumPy integers; make sure
    # we're using Python integers.
    f = int(f)
    if f != f & 0xffffffff:
        raise ValueError("Input should be an unsigned 32-bit integer.")
    sign = (f & 0x80000000) >> 31
    exponent = ((f & 0x7f000000) >> 24) - 64
    fraction = F(f & 0x00ffffff, 0x01000000)
    result = fraction * F(16)**exponent
    return sign, result


def fraction_from_ibm64(f):
    """
    Convert a bit-representation of an IBM double-precision hexadecimal
    float to a pair (sign, fraction)
    """
    # Fraction operations get confused by NumPy integers; make sure
    # we're using Python integers.
    f = int(f)
    if f != f & 0xffffffffffffffff:
        raise ValueError("Input should be an unsigned 32-bit integer.")
    sign = (f & 0x8000000000000000) >> 63
    exponent = ((f & 0x7f00000000000000) >> 56) - 64
    fraction = F(f & 0x00ffffffffffffff, 0x0100000000000000)
    result = fraction * F(16)**exponent
    return sign, result


def ieee32_from_fraction(s, f):
    """
    Convert sign and fraction to a bit-representation of an IEEE single.
    """
    s = IEEE32_SIGN if s else 0
    if not f:
        return s
    exponent = max(ilog2_fraction(f), IEEE32_MINEXP)
    ieee_frac = round_ties_to_even(f / TWO**(exponent - IEEE32_FRAC))
    expt_and_frac = ((exponent - IEEE32_MINEXP) << IEEE32_FRAC) + ieee_frac
    return min(expt_and_frac, IEEE32_POSINF) + s


def ieee64_from_fraction(s, f):
    """
    Convert sign and fraction to a bit-representation of an IEEE double.
    """
    s = IEEE64_SIGN if s else 0
    if not f:
        return s
    exponent = max(ilog2_fraction(f), IEEE64_MINEXP)
    ieee_frac = round_ties_to_even(f / TWO**(exponent - IEEE64_FRAC))
    expt_and_frac = ((exponent - IEEE64_MINEXP) << IEEE64_FRAC) + ieee_frac
    return min(expt_and_frac, IEEE64_POSINF) + s


def ibm32ieee32(ibm):
    ieee = ieee32_from_fraction(*fraction_from_ibm32(int(ibm)))
    return np.uint32(ieee).view(np.float32)


def ibm64ieee32(ibm):
    ieee = ieee32_from_fraction(*fraction_from_ibm64(int(ibm)))
    return np.uint32(ieee).view(np.float32)


def ibm32ieee64(ibm):
    ieee = ieee64_from_fraction(*fraction_from_ibm32(int(ibm)))
    return np.uint64(ieee).view(np.float64)


def ibm64ieee64(ibm):
    ieee = ieee64_from_fraction(*fraction_from_ibm64(int(ibm)))
    return np.uint64(ieee).view(np.float64)


single_to_single_pairs = [
    (0x00000000, 0.0),
    (0x00000001, 0.0),
    (0x3f000000, 0.0),
    (0x7f000000, 0.0),
    (0x1b100000, 0.0),
    (0x1b200000, 0.0),
    (0x1b400000, 0.0),
    (0x1b400001, float.fromhex("0x1p-149")),
    (0x1b800000, float.fromhex("0x1p-149")),
    (0x1bbfffff, float.fromhex("0x1p-149")),
    (0x1bc00000, float.fromhex("0x2p-149")),
    # Checking round-ties-to-even behaviour on a mid-range subnormal
    (0x1da7bfff, float.fromhex("0x14fp-149")),
    (0x1da7c000, float.fromhex("0x150p-149")),
    (0x1da84000, float.fromhex("0x150p-149")),
    (0x1da84001, float.fromhex("0x151p-149")),
    (0x1da8bfff, float.fromhex("0x151p-149")),
    (0x1da8c000, float.fromhex("0x152p-149")),
    (0x1da94000, float.fromhex("0x152p-149")),
    (0x1da94001, float.fromhex("0x153p-149")),
    (0x1da9bfff, float.fromhex("0x153p-149")),
    (0x1da9c000, float.fromhex("0x154p-149")),
    (0x1daa4000, float.fromhex("0x154p-149")),
    (0x1daa4001, float.fromhex("0x155p-149")),

    (0x1fffffff, float.fromhex("0x1p-132")),
    (0x20fffff4, float.fromhex("0x0.fffff0p-128")),
    (0x20fffff5, float.fromhex("0x0.fffff8p-128")),
    (0x20fffff6, float.fromhex("0x0.fffff8p-128")),
    (0x20fffff7, float.fromhex("0x0.fffff8p-128")),
    (0x20fffff8, float.fromhex("0x0.fffff8p-128")),
    (0x20fffff9, float.fromhex("0x0.fffff8p-128")),
    (0x20fffffa, float.fromhex("0x0.fffff8p-128")),
    (0x20fffffb, float.fromhex("0x0.fffff8p-128")),
    (0x20fffffc, float.fromhex("0x1p-128")),
    (0x20fffffd, float.fromhex("0x1p-128")),
    (0x20fffffe, float.fromhex("0x1p-128")),
    (0x20ffffff, float.fromhex("0x1p-128")),  # largest rounded case
    (0x21100000, float.fromhex("0x1p-128")),
    (0x21200000, float.fromhex("0x1p-127")),
    (0x213fffff, float.fromhex("0x0.fffffcp-126")),
    (0x21400000, float.fromhex("0x1p-126")),  # smallest positive normal
    (0x40800000, 0.5),
    (0x46000001, 1.0),
    (0x45000010, 1.0),
    (0x44000100, 1.0),
    (0x43001000, 1.0),
    (0x42010000, 1.0),
    (0x41100000, 1.0),
    (0x41200000, 2.0),
    (0x41300000, 3.0),
    (0x41400000, 4.0),
    (0x41800000, 8.0),

    # Test full range of possible leading zero counts.
    (0x48000001, float.fromhex("0x1p+8")),
    (0x48000002, float.fromhex("0x1p+9")),
    (0x48000004, float.fromhex("0x1p+10")),
    (0x48000008, float.fromhex("0x1p+11")),
    (0x48000010, float.fromhex("0x1p+12")),
    (0x48000020, float.fromhex("0x1p+13")),
    (0x48000040, float.fromhex("0x1p+14")),
    (0x48000080, float.fromhex("0x1p+15")),
    (0x48000100, float.fromhex("0x1p+16")),
    (0x48000200, float.fromhex("0x1p+17")),
    (0x48000400, float.fromhex("0x1p+18")),
    (0x48000800, float.fromhex("0x1p+19")),
    (0x48001000, float.fromhex("0x1p+20")),
    (0x48002000, float.fromhex("0x1p+21")),
    (0x48004000, float.fromhex("0x1p+22")),
    (0x48008000, float.fromhex("0x1p+23")),
    (0x48010000, float.fromhex("0x1p+24")),
    (0x48020000, float.fromhex("0x1p+25")),
    (0x48040000, float.fromhex("0x1p+26")),
    (0x48080000, float.fromhex("0x1p+27")),
    (0x48100000, float.fromhex("0x1p+28")),
    (0x48200000, float.fromhex("0x1p+29")),
    (0x48400000, float.fromhex("0x1p+30")),
    (0x48800000, float.fromhex("0x1p+31")),

    (0x60ffffff, float.fromhex("0x0.ffffffp+128")),
    (0x61100000, float("inf")),
    (0x61200000, float("inf")),
    (0x61400000, float("inf")),
    (0x62100000, float("inf")),
    (0x7fffffff, float("inf")),

    # From https://en.wikipedia.org/wiki/IBM_hexadecimal_floating_point
    (0b11000010011101101010000000000000, -118.625),
]

double_to_single_pairs = [
    (0x0000000000000001, 0.0),
    (0x1effffffffffffff, float.fromhex("0x1p-136")),
    (0x1fffffffffffffff, float.fromhex("0x1p-132")),
    (0x20ffffffffffffff, float.fromhex("0x1p-128")),
    (0x213fffffbfffffff, float.fromhex("0x0.3fffff8p-124")),
    (0x213fffffc0000000, float.fromhex("0x1p-126")),
    (0x213fffffffffffff, float.fromhex("0x1p-126")),
    (0x40ffffff7fffffff, float.fromhex("0x0.ffffffp+0")),
    (0x40ffffff80000000, 1.0),
    (0x411fffffefffffff, float.fromhex("0x0.ffffffp+1")),
    (0x411ffffff0000000, 2.0),
    (0x411fffffffffffff, 2.0),
    (0x60ffffff7fffffff, float.fromhex("0x0.ffffffp+128")),
    (0x60ffffff80000000, float("inf")),
    (0x60ffffffffffffff, float("inf")),
    (0x7fffffffffffffff, float("inf")),
    # Values that would produce the wrong answer under double rounding
    # (rounding first to 24-bit precision, then to subnormal).
    (0x1da7bfffffffffff, float.fromhex("0x14fp-149")),
    (0x1da8400000000001, float.fromhex("0x151p-149")),
    (0x1da8bfffffffffff, float.fromhex("0x151p-149")),
    (0x1da9400000000001, float.fromhex("0x153p-149")),
    (0x1da9bfffffffffff, float.fromhex("0x153p-149")),
    (0x1daa400000000001, float.fromhex("0x155p-149")),
] + [
    (x << 32, f)
    for x, f in single_to_single_pairs
]

single_to_double_pairs = [
    (0x0, 0.0),
    (0x00000001, float.fromhex("0x1p-280")),
    (0x00ffffff, float.fromhex("0x0.ffffffp-256")),
    (0x41100000, 1.0),
    (0x7fffffff, float.fromhex("0x0.ffffffp+252")),
]

double_to_double_pairs = [
    (0x0000000000000001, float.fromhex("0x1p-312")),
    (0x0000000000000002, float.fromhex("0x2p-312")),
    (0x0000000000000003, float.fromhex("0x3p-312")),

    # Rounding boundaries near powers of two.
    (0x400ffffffffffffe, float.fromhex("0x1.ffffffffffffcp-5")),
    (0x400fffffffffffff, float.fromhex("0x1.ffffffffffffep-5")),
    (0x4010000000000000, float.fromhex("0x1.0000000000000p-4")),
    (0x4010000000000001, float.fromhex("0x1.0000000000001p-4")),
    (0x4010000000000002, float.fromhex("0x1.0000000000002p-4")),

    (0x401ffffffffffffe, float.fromhex("0x1.ffffffffffffep-4")),
    (0x401fffffffffffff, float.fromhex("0x1.fffffffffffffp-4")),
    (0x4020000000000000, float.fromhex("0x1.0000000000000p-3")),
    (0x4020000000000001, float.fromhex("0x1.0000000000000p-3")),
    (0x4020000000000002, float.fromhex("0x1.0000000000001p-3")),
    (0x4020000000000003, float.fromhex("0x1.0000000000002p-3")),

    (0x403ffffffffffffd, float.fromhex("0x1.ffffffffffffep-3")),
    (0x403ffffffffffffe, float.fromhex("0x1.fffffffffffffp-3")),
    (0x403fffffffffffff, float.fromhex("0x1.0000000000000p-2")),
    (0x4040000000000002, float.fromhex("0x1.0000000000000p-2")),
    (0x4040000000000003, float.fromhex("0x1.0000000000001p-2")),
    (0x4040000000000005, float.fromhex("0x1.0000000000001p-2")),
    (0x4040000000000006, float.fromhex("0x1.0000000000002p-2")),

    (0x407ffffffffffffa, float.fromhex("0x1.ffffffffffffep-2")),
    (0x407ffffffffffffb, float.fromhex("0x1.fffffffffffffp-2")),
    (0x407ffffffffffffd, float.fromhex("0x1.fffffffffffffp-2")),
    (0x407ffffffffffffe, float.fromhex("0x1.0000000000000p-1")),
    (0x4080000000000004, float.fromhex("0x1.0000000000000p-1")),
    (0x4080000000000005, float.fromhex("0x1.0000000000001p-1")),
    (0x408000000000000b, float.fromhex("0x1.0000000000001p-1")),
    (0x408000000000000c, float.fromhex("0x1.0000000000002p-1")),

    (0x40fffffffffffff4, float.fromhex("0x1.ffffffffffffep-1")),
    (0x40fffffffffffff5, float.fromhex("0x1.fffffffffffffp-1")),
    (0x40fffffffffffffb, float.fromhex("0x1.fffffffffffffp-1")),
    (0x40fffffffffffffc, float.fromhex("0x1.0000000000000p+0")),

    (0x4110000000000000, 1.0),
    (0x4110000000000001, float.fromhex("0x1.0000000000001p+0")),
    (0x4110000000000002, float.fromhex("0x1.0000000000002p+0")),
    (0x411fffffffffffff, float.fromhex("0x1.fffffffffffffp+0")),
    (0x4120000000000000, 2.0),
    (0x4120000000000001, 2.0),
    (0x4120000000000002, float.fromhex("0x1.0000000000001p+1")),

    # Full range of possible leading zero counts.
    (0x4800000000000001, float.fromhex("0x1p-24")),
    (0x4800000000000002, float.fromhex("0x1p-23")),
    (0x4800000000000004, float.fromhex("0x1p-22")),
    (0x4800000000000008, float.fromhex("0x1p-21")),
    (0x4800000000000010, float.fromhex("0x1p-20")),
    (0x4800000000000020, float.fromhex("0x1p-19")),
    (0x4800000000000040, float.fromhex("0x1p-18")),
    (0x4800000000000080, float.fromhex("0x1p-17")),
    (0x4800000000000100, float.fromhex("0x1p-16")),
    (0x4800000000000200, float.fromhex("0x1p-15")),
    (0x4800000000000400, float.fromhex("0x1p-14")),
    (0x4800000000000800, float.fromhex("0x1p-13")),
    (0x4800000000001000, float.fromhex("0x1p-12")),
    (0x4800000000002000, float.fromhex("0x1p-11")),
    (0x4800000000004000, float.fromhex("0x1p-10")),
    (0x4800000000008000, float.fromhex("0x1p-9")),
    (0x4800000000010000, float.fromhex("0x1p-8")),
    (0x4800000000020000, float.fromhex("0x1p-7")),
    (0x4800000000040000, float.fromhex("0x1p-6")),
    (0x4800000000080000, float.fromhex("0x1p-5")),
    (0x4800000000100000, float.fromhex("0x1p-4")),
    (0x4800000000200000, float.fromhex("0x1p-3")),
    (0x4800000000400000, float.fromhex("0x1p-2")),
    (0x4800000000800000, float.fromhex("0x1p-1")),
    (0x4800000001000000, float.fromhex("0x1p+0")),
    (0x4800000002000000, float.fromhex("0x1p+1")),
    (0x4800000004000000, float.fromhex("0x1p+2")),
    (0x4800000008000000, float.fromhex("0x1p+3")),
    (0x4800000010000000, float.fromhex("0x1p+4")),
    (0x4800000020000000, float.fromhex("0x1p+5")),
    (0x4800000040000000, float.fromhex("0x1p+6")),
    (0x4800000080000000, float.fromhex("0x1p+7")),
    (0x4800000100000000, float.fromhex("0x1p+8")),
    (0x4800000200000000, float.fromhex("0x1p+9")),
    (0x4800000400000000, float.fromhex("0x1p+10")),
    (0x4800000800000000, float.fromhex("0x1p+11")),
    (0x4800001000000000, float.fromhex("0x1p+12")),
    (0x4800002000000000, float.fromhex("0x1p+13")),
    (0x4800004000000000, float.fromhex("0x1p+14")),
    (0x4800008000000000, float.fromhex("0x1p+15")),
    (0x4800010000000000, float.fromhex("0x1p+16")),
    (0x4800020000000000, float.fromhex("0x1p+17")),
    (0x4800040000000000, float.fromhex("0x1p+18")),
    (0x4800080000000000, float.fromhex("0x1p+19")),
    (0x4800100000000000, float.fromhex("0x1p+20")),
    (0x4800200000000000, float.fromhex("0x1p+21")),
    (0x4800400000000000, float.fromhex("0x1p+22")),
    (0x4800800000000000, float.fromhex("0x1p+23")),
    (0x4801000000000000, float.fromhex("0x1p+24")),
    (0x4802000000000000, float.fromhex("0x1p+25")),
    (0x4804000000000000, float.fromhex("0x1p+26")),
    (0x4808000000000000, float.fromhex("0x1p+27")),
    (0x4810000000000000, float.fromhex("0x1p+28")),
    (0x4820000000000000, float.fromhex("0x1p+29")),
    (0x4840000000000000, float.fromhex("0x1p+30")),
    (0x4880000000000000, float.fromhex("0x1p+31")),

    (0x567faef3ff3dc282, float.fromhex("0x1.febbcffcf70a0p+86")),

    (0x7ffffffffffffff4, float.fromhex("0x1.ffffffffffffep+251")),
    (0x7ffffffffffffff5, float.fromhex("0x1.fffffffffffffp+251")),
    (0x7ffffffffffffffb, float.fromhex("0x1.fffffffffffffp+251")),
    (0x7ffffffffffffffc, float.fromhex("0x1p+252")),
    (0x7ffffffffffffffd, float.fromhex("0x1p+252")),
    (0x7ffffffffffffffe, float.fromhex("0x1p+252")),
    (0x7fffffffffffffff, float.fromhex("0x1p+252")),
] + [
    (x << 32, f)
    for x, f in single_to_double_pairs
]


class TestIBM2IEEE(unittest.TestCase):
    def setUp(self):
        self.random = np.random.RandomState(seed=616692)

    def test_single_to_single(self):
        # Inputs with known outputs.
        for input, expected in single_to_single_pairs:
            pos_input = np.uint32(input)
            pos_expected = np.float32(expected)
            self.assertFloatsIdentical(ibm2float32(pos_input), pos_expected)

            neg_input = np.uint32(input ^ 0x80000000)
            neg_expected = -np.float32(expected)
            self.assertFloatsIdentical(ibm2float32(neg_input), neg_expected)

    def test_double_to_single(self):
        for input, expected in double_to_single_pairs:
            pos_input = np.uint64(input)
            pos_expected = np.float32(expected)
            self.assertFloatsIdentical(ibm2float32(pos_input), pos_expected)

            neg_input = np.uint64(input ^ 0x8000000000000000)
            neg_expected = -np.float32(expected)
            self.assertFloatsIdentical(ibm2float32(neg_input), neg_expected)

    def test_single_to_double(self):
        for input, expected in single_to_double_pairs:
            pos_input = np.uint32(input)
            pos_expected = np.float64(expected)
            self.assertFloatsIdentical(ibm2float64(pos_input), pos_expected)

            neg_input = np.uint32(input ^ 0x80000000)
            neg_expected = -np.float64(expected)
            self.assertFloatsIdentical(ibm2float64(neg_input), neg_expected)

    def test_double_to_double(self):
        for input, expected in double_to_double_pairs:
            pos_input = np.uint64(input)
            pos_expected = np.float64(expected)
            self.assertFloatsIdentical(ibm2float64(pos_input), pos_expected)

            neg_input = np.uint64(input ^ 0x8000000000000000)
            neg_expected = -np.float64(expected)
            self.assertFloatsIdentical(ibm2float64(neg_input), neg_expected)

    def test_single_to_single_random_inputs(self):
        # Random inputs
        inputs = self.random.randint(2**32, size=32768, dtype=np.uint32)
        for input in inputs:
            actual = ibm2float32(input)
            expected = ibm32ieee32(input)
            self.assertFloatsIdentical(actual, expected)

    def test_double_to_single_random_inputs(self):
        # Random inputs
        inputs = self.random.randint(2**64, size=32768, dtype=np.uint64)
        for input in inputs:
            actual = ibm2float32(input)
            expected = ibm64ieee32(input)
            self.assertFloatsIdentical(actual, expected)

    def test_single_to_double_random_inputs(self):
        # Random inputs
        inputs = self.random.randint(2**32, size=32768, dtype=np.uint32)
        for input in inputs:
            actual = ibm2float64(input)
            expected = ibm32ieee64(input)
            self.assertFloatsIdentical(actual, expected)

    def test_double_to_double_random_inputs(self):
        # Random inputs
        inputs = self.random.randint(2**64, size=32768, dtype=np.uint64)
        for input in inputs:
            actual = ibm2float64(input)
            expected = ibm64ieee64(input)
            self.assertFloatsIdentical(actual, expected)

    def test_array_inputs(self):
        shapes = [(), (1,), (2,), (3, 2), (4, 1, 3), (0,), (2, 0), (0, 2)]
        converters = [(ibm2float32, np.float32), (ibm2float64, np.float64)]
        in_types = [(2**32, np.uint32), (2**64, np.uint64)]

        # Check variety of shape, input type and output type.
        for converter, out_type in converters:
            for shape in shapes:
                for bound, in_type in in_types:
                    inputs = self.random.randint(
                        bound, size=shape, dtype=in_type)
                    self.assertEqual(inputs.dtype, in_type)
                    self.assertEqual(inputs.shape, shape)
                    outputs = converter(inputs)
                    self.assertEqual(outputs.dtype, out_type)
                    self.assertEqual(outputs.shape, shape)
                    for input, output in zip(inputs.flat, outputs.flat):
                        self.assertFloatsIdentical(converter(input), output)

    def test_import_star(self):
        locals = {}
        exec("from ibm2ieee import *", locals, locals)
        self.assertIn("ibm2float32", locals)
        self.assertIn("ibm2float64", locals)
        self.assertIn("__version__", locals)

    def test_np_info(self):
        output = six.moves.StringIO()
        with contextlib.closing(output):
            np.info(ibm2float32, output=output)
            self.assertIn("Examples", output.getvalue())

        output = six.moves.StringIO()
        with contextlib.closing(output):
            np.info(ibm2float64, output=output)
            self.assertIn("Examples", output.getvalue())

    def assertFloatsIdentical(self, a, b):
        # Assert that float instances are equal, and that if they're zero,
        # then the signs match. N.B. we don't care about NaNs.
        self.assertEqual(
            (a, np.signbit(a)),
            (b, np.signbit(b)),
        )


if __name__ == "__main__":
    unittest.main()
