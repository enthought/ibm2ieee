# Copyright (c) 2018, Enthought, Inc.
# All rights reserved.

from __future__ import absolute_import, print_function, unicode_literals

import unittest

import packaging.version
import six

import ibm2ieee.version


class TestVersion(unittest.TestCase):
    def test_version_string(self):
        version = ibm2ieee.version.version
        self.assertIsInstance(version, six.text_type)

        # Check that version number is normalised and complies with PEP 440.
        version_object = packaging.version.Version(version)
        self.assertEqual(six.text_type(version_object), version)

    def test_top_level_package_version(self):
        self.assertEqual(
            ibm2ieee.__version__,
            ibm2ieee.version.version,
        )
