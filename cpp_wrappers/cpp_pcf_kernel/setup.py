# PCF CUDA Kernel:
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022-2023 Apple Inc. All Rights Reserved.
#
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

src_files = [
    'pcf_cuda.cpp',
    'src/pcf.cu',
    'src/common.cu',
    'src/knn.cu',
    'src/pcf_ops.cu',
    'src/pconv_ops.cu',
]

setup(
    name='PCFcuda',
    version='1.1',
    author='Fuxin Li',
    author_email='fli26@apple.com',
    description='PointConvFormer CUDA Kernel',
    ext_modules=[
        CUDAExtension('pcf_cuda', src_files,
        include_dirs=[
            '/nfs/stak/users/sivakuml/hpc-memory/cutlass/ml-pointconvformer/cpp_wrappers/cpp_pcf_kernel/include',
            '/nfs/stak/users/sivakuml/hpc-memory/cutlass/cutlass/include',
            '/nfs/stak/users/sivakuml/hpc-memory/cutlass/cutlass/tools/util/include',
            ],
        extra_compile_args={'nvcc': ['-L/usr/local/cuda/lib64 -lcudadevrt -lcudart']})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
