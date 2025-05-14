# MIT License

# Copyright (c) 2019 HuguesTHOMAS


from setuptools import setup, Extension
import numpy as np

SOURCES = [
    "../cpp_utils/cloud/cloud.cpp",
    "neighbors/neighbors.cpp",
    "wrapper.cpp"
]

module = Extension(
    name="radius_neighbors",
    sources=SOURCES,
    include_dirs=[np.get_include()],
    extra_compile_args=[
        "-std=c++11",
        "-D_GLIBCXX_USE_CXX11_ABI=0"
    ]
)

# Setup call
setup(
    name="radius_neighbors",
    ext_modules=[module]
)

