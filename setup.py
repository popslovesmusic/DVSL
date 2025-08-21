from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
from setuptools import setup, Extension
import platform

# Determine compiler flags based on platform
extra_compile_args = []
extra_link_args = []

if platform.system() == "Windows":
    extra_compile_args = [
        "/O2",           # Optimization level 2
        "/arch:AVX2",    # Enable AVX2 instructions
        "/std:c++17",    # C++17 standard
        "/DNOMINMAX",    # Prevent Windows min/max macros
    ]
elif platform.system() in ["Linux", "Darwin"]:  # Linux or macOS
    extra_compile_args = [
        "-O3",           # High optimization
        "-mavx2",        # Enable AVX2 instructions
        "-mfma",         # Enable FMA instructions
        "-std=c++17",    # C++17 standard
        "-fopenmp",      # OpenMP support
    ]
    extra_link_args = ["-fopenmp"]

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "dase_engine",
        sources=[
            "analog_universal_node_engine_avx2.cpp",
            "python_bindings.cpp"
        ],
        include_dirs=[
            pybind11.get_include(),
            ".",  # Current directory for headers
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        cxx_std=17,
    ),
]

setup(
    name="dase_engine",
    version="1.0.0",
    author="DASE Development Team",
    description="High-performance analog signal processing engine with AVX2 optimization",
    long_description="Digital Analog Signal Engine (DASE) for real-time analog circuit simulation",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
    ],
    zip_safe=False,
)