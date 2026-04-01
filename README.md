[![docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://anlsys.github.io/XK.jl/)
[![docs](https://img.shields.io/badge/docs-dev-purple.svg)](https://anlsys.github.io/XK.jl/dev)

# XK.jl - Julia bindings for the XKRT software stack.
It currently exposes the `XK.BLAS` module, as bindings to the XKBlas library: a composable, performant and portable multi-GPU BLAS library.
Other modules are experimental.

# Installation
Installing XK.jl is currently manual.

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
If th einstallation succeeded, this should run:
```bash
julia examples/blas/gemm.jl
```

# Usage
See `examples/` for more examples.

# See Also
- [XKBlas Repository](https://gitlab.inria.fr/xkblas/dev/-/tree/v2.0)
- [XKRT Repository](https://github.com/rpereira-dev/xkrt/)
