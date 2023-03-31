# (C) Copyright 2018-2023 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

import numpy
import setuptools

setuptools.setup(
    ext_modules=[
        setuptools.Extension(
            name="ibm2ieee._ibm2ieee",
            sources=["ibm2ieee/_ibm2ieee.c"],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            include_dirs=[numpy.get_include()],
        )
    ],
)
