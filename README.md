[![docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://anlsys.github.io/XK.jl/)
[![docs](https://img.shields.io/badge/docs-dev-purple.svg)](https://anlsys.github.io/XK.jl/dev)

# XK.jl - Julia bindings for the XKRT and XKBlas softwares

XKRT is a macro-dataflow and tasking runtime system for automating memory management alongside parallel execution.
It provides portable abstraction for C, C++, Julia and partial support for BLAS and OpenMP.
Please see the [XKRT Repository](https://github.com/rpereira-dev/xkrt/) for more information.

This package is the Julia API of the XKRT environment.
It currently exposes the `XK.BLAS` module: a composable, performant and portable multi-GPU BLAS library.
Other modules are experimental.

# Installation
XK.jl must currently be installed manually, please use the following steps.

## Prerequisities
- Julia >= 1.11
- CMake >= 3.17
- LLVM/Clang >= 20 (required — earlier versions lack the C++20 support XKRT needs)
- hwloc library

## Steps

- Edit the header of `deps/build_xkblas.jl` to your convenience
- Make sure CMake will use Clang >= 20, e.g. `export CC=clang-20 CXX=clang++-20`
  (the build aborts early with a clear message if the compiler is too old)
- Run `julia --project=./deps -e 'using Pkg; Pkg.instantiate(); include("./deps/build.jl")'`

## Testing
If the installation succeeded, this should run (the `--project=.` is required so
Julia can find the `XK` package):
```bash
julia --project=. examples/blas/gemm.jl
```

You can run exhaustive tests with
```bash
julia --project -e 'using Pkg; Pkg.test()'
```

# Usage
See `examples/` for more examples.

# See Also
- [XKBlas Repository](https://gitlab.inria.fr/xkblas/dev/-/tree/v2.0)
- [XKRT Repository](https://github.com/rpereira-dev/xkrt/)
