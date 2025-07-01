import os
import shutil
from setuptools import setup, Command
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

_root = os.path.dirname(os.path.abspath(__file__))
_src_path = os.path.join(_root, "modules", "functional", "src")
_torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")


class Clean(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for dir_path in ['build', 'dist', 'pvcnn_backend.egg-info']:
            if os.path.isdir(dir_path):
                print(f'Removing directory: {dir_path}')
                shutil.rmtree(dir_path)


setup(
    name="pvcnn_backend",
    ext_modules=[
        CUDAExtension(
            name="_pvcnn_backend",
            sources=[
                os.path.join(_src_path, f)
                for f in [
                    "ball_query/ball_query.cpp",
                    "ball_query/ball_query_cuda.cu",
                    "grouping/grouping.cpp",
                    "grouping/grouping_cuda.cu",
                    "interpolate/neighbor_interpolate.cpp",
                    "interpolate/neighbor_interpolate_cuda.cu",
                    "interpolate/trilinear_devox.cpp",
                    "interpolate/trilinear_devox_cuda.cu",
                    "sampling/sampling.cpp",
                    "sampling/sampling_cuda.cu",
                    "voxelization/vox.cpp",
                    "voxelization/vox_cuda.cu",
                    "bindings.cpp",
                ]
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"], # -O0 for debugging, -O3 for production
                "nvcc": ["--compiler-bindir=/usr/bin/gcc-9"], # ! for gcc-9, change to gcc-10 for gcc-10
            },
            extra_link_args=[f'-Wl,-rpath,{_torch_lib_dir}'],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension,
        "clean": Clean,
    },
)
