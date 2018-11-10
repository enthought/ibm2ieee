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
    include_dirs=[numpy.get_include()],
)

# XXX Fix documentation of ufuncs!
# XXX Add README
# XXX Add description, long description, author, classifiers, dependencies, etc.
# XXX Get rid of deprecated NumPy API warning...
# XXX Check signs of zeros properly in tests.

if __name__ == "__main__":
    setup(
        name="ibm2ieee",
        packages=find_packages(),
        ext_modules=[ibm2ieee_extension],
    )
