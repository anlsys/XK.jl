# Contributing to XK.jl

We welcome any form of contributions to XK.jl, and more generally to the [XKRT](https://github.com/anlsys/xkrt) parallel programming environment.
Note that `XK.jl` is primarly a research software and may be unstable.

## Bug reports, Feature Requests

Please open issues at [https://github.com/anlsys/XK.jl/issues](https://github.com/anlsys/XK.jl/issues)

## Extending XK.jl

We strongly recommend any external contributors to reach-out with the team first ([Contact](#contact)).

Regarding the `XK.BLAS` package specifically:
- The list of routine supported is listed in the [XKBlas repository](https://gitlab.inria.fr/xkblas/dev/-/blob/v2.0/README.md#list-of-routines-supported).
- The Julia API in `XK.BLAS` is generated automatically from the C API of XKBlas.

## Testing
The repository includes a few minimal [tests](https://github.com/anlsys/XK.jl/blob/main/test/runtests.jl) that can be extended, and run via `julia --project -e 'using Pkg; Pkg.test()'`

## Contact
Romain Pereira (rpereira@anl.gov), Michel Schanen (mschanen@anl.gov), Swann Perarnau (swann@anl.gov)
