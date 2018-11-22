# Copyright (c) 2018, Enthought, Inc.
# All rights reserved.

import os
from setuptools import Extension, find_packages, setup

import numpy


def get_version_info():
    """ Extract version information as a dictionary from version.py. """
    version_info = {}
    version_filename = os.path.join("ibm2ieee", "version.py")
    with open(version_filename, "r") as version_module:
        version_code = compile(version_module.read(), "version.py", "exec")
        exec(version_code, version_info)
    return version_info


version = get_version_info()["version"]

ibm2ieee_extension = Extension(
    name="ibm2ieee._ibm2ieee",
    sources=[
        "ibm2ieee/_ibm2ieee.c",
    ],
    define_macros=[
        ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
    ],
    include_dirs=[numpy.get_include()],
)

SHORT_DESCRIPTION = """\
Conversions from IBM hexadecimal floating-point to IEEE 754 floating-point.
"""

if __name__ == "__main__":
    setup(
        name="ibm2ieee",
        version=version,
        author="Enthought",
        description=SHORT_DESCRIPTION,
        install_requires=["numpy"],
        packages=find_packages(),
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: BSD License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 2.7",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
        ],
        ext_modules=[ibm2ieee_extension],
    )
