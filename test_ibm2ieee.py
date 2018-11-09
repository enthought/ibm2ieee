import unittest

import numpy as np

from npufunc_directory.ibm2ieee import ibm2float32

test_pairs = [
    (0x00000000, 0.0),
    (0x1b100000, 0.0),
    (0x1b200000, 0.0),
    (0x1b400000, 0.0),
    (0x1b400001, float.fromhex("0x1p-149")),
    (0x1b800000, float.fromhex("0x1p-149")),
    (0x1bbfffff, float.fromhex("0x1p-149")),
    (0x1bc00000, float.fromhex("0x2p-149")),

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
    # All values with IBM exponent 0x21 are exactly representable in IEEE.
    (0x21100000, float.fromhex("0x1p-128")),
    (0x21200000, float.fromhex("0x1p-127")),
    (0x213fffff, float.fromhex("0x0.fffffcp-126")),
    (0x21400000, float.fromhex("0x1p-126")),  # smallest positive normal
    (0x7f000000, 0.0),
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
    (0x60ffffff, float.fromhex("0x0.ffffffp+128")),
    (0x61100000, float("inf")),
]


class TestIBM2IEEE(unittest.TestCase):
    def test_single_precision(self):
        for input, expected in test_pairs:
            pos_input = np.uint32(input)
            pos_expected = np.float32(expected)
            self.assertEqual(ibm2float32(pos_input), pos_expected)

            neg_input = np.uint32(input + 0x80000000)
            neg_expected = -np.float32(expected)
            self.assertEqual(ibm2float32(neg_input), neg_expected)





if __name__ == '__main__':
    unittest.main()
