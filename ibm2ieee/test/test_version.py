# Copyright (c) 2018, Enthought, Inc.
# All rights reserved.

import unittest

import pkg_resources

import ibm2ieee.version


class TestVersion(unittest.TestCase):
    def test_version_string(self):
        version = ibm2ieee.version.version
        self.assertIsInstance(version, str)

        # Check that version number is normalised and complies with PEP 440.
        version_object = pkg_resources.parse_version(version)
        self.assertEqual(str(version_object), version)

    def test_top_level_package_version(self):
        self.assertEqual(
            ibm2ieee.__version__,
            ibm2ieee.version.version,
        )
