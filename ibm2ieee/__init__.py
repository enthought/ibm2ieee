# (C) Copyright 2018-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

# Not importing Unicode literals, because the elements of __all__
# need to be bytestrings in Python 2.
from __future__ import absolute_import, print_function

from .version import version as __version__
from ._ibm2ieee import ibm2float32, ibm2float64


__all__ = [
    "__version__", "ibm2float32", "ibm2float64",
]
