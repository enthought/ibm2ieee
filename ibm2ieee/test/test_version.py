# (C) Copyright 2018-2022 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

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
