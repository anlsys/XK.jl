[![docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://anlsys.github.io/XK.jl/)
[![docs](https://img.shields.io/badge/docs-dev-purple.svg)](https://anlsys.github.io/XK.jl/dev)

# XK.jl - Julia bindings for the XKRT and XKBlas softwares

XKRT is a macro-dataflow and tasking runtime system for automating memory management alongside parallel execution.
It provides portable abstraction for C, C++, Julia and partial support for BLAS and OpenMP.
Please see the [XKRT](https://github.com/rpereira-dev/xkrt/) and [XKBlas](https://gitlab.inria.fr/xkblas/dev/-/tree/v2.0) repositories for more information.

This package is the Julia API of the XKRT environment.
It currently exposes the `XK.BLAS` module: a composable, performant and portable multi-GPU BLAS library.
Other modules are experimentals.

To use XK.jl, please refer to the [documentation](https://anlsys.github.io/XK.jl/) and [examples](https://github.com/anlsys/XK.jl/tree/main/examples)

# Installation
XK.jl must currently be installed manually, please use the following steps.

## Prerequisities
- Julia >= 1.11
- CMake >= 3.17
- A C++20 compatible compiler (LLVM/clang >= 20.x recommended)
- hwloc library

## Steps

- Edit the header of `deps/build_xkblas.jl` to your convenience
- Be sure to be using a C++20 compatible compiler (i.e., maybe `export CC=clang CXX=clang++`)
- Run `julia --project=./deps -e 'using Pkg; Pkg.instantiate(); include("./deps/build.jl")'`

## Testing
If the installation succeeded, this should run:
```bash
julia examples/blas/gemm.jl
```

You can run exhaustive tests with
```bash
julia --project -e 'using Pkg; Pkg.test()'
```

# Support
If you are look for any support, feel free to contact us directly.    
Romain Pereira (rpereira@anl.gov), Michel Schanen (mschanen@anl.gov), Swann Perarnau (swann@anl.gov)

# References
[1] Multi-GPU memory coherence for BLAS matrices. Romain Pereira, Pierre Etienne Polet, Swann Perarnau, Thierry Gautier. 40th IEEE International Parallel & Distributed Processing Symposium Workshops (IPDPSW, HIPS), New Orleans, USA.

[2] [UNDER REVIEW] XKRT: a Runtime System for Macro-Dataflow Programming on Multi-Devices Architectures. Romain Pereira, Swann Perarnau, Pierre-Etienne Polet, Brice Videau, Thomas Applencourt, Thierry Gautier.

[3] [UNDER REVIEW]: XK.jl: Composable and Portable Multi-GPU BLAS in Julia. Romain Pereira, Alexis Montoison, Michel Schanen, Swann Perarnau
