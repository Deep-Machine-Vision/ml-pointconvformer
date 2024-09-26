# PConv CUDA Kernel:
# For licensing see accompanying LICENSE file.
#
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='PConvcuda',
    version='1.2',
    author='Stefan Lee, Skand',
    description='Faster PointConv CUDA Kernel',
    ext_modules=[
        CUDAExtension('pconv_cuda', [
            'faster_pconv_cuda.cpp',
            'faster_pconv_kernel.cu',
        ], extra_compile_args={'nvcc': ['-L/usr/local/cuda/lib64 -lcudadevrt -lcudart']})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
