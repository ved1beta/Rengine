"""
Setup script to build the CUDA sampling kernel as a PyTorch extension.

Usage:
    python setup.py install
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get CUDA architecture from environment or use default
cuda_arch = os.environ.get('CUDA_ARCH', 'sm_86')

setup(
    name='rengine_kernels',
    ext_modules=[
        CUDAExtension(
            name='rengine_kernels.sample_ops',
            sources=[
                'sample_binding.cu',
                'sample.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    f'-arch={cuda_arch}',
                    '--use_fast_math',
                    '-lineinfo',
                ]
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    python_requires='>=3.8',
)
