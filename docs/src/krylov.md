# Krylov.jl Integration

This example shows how to use [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl)
with `XK.BLAS` as the computational backend, so that iterative solvers
run transparently on multiple GPUs.

## Overview

Krylov.jl defines a set of internal kernel functions (`kdot`, `kaxpy!`,
`kaxpby!`, `kmul!`, `kscal!`, etc.) that it calls during every solver
iteration. By overriding these functions to call `XK.BLAS` routines,
Krylov solvers automatically offload all vector and matrix operations to
multi-GPU execution.

Two integration strategies are provided:

| Strategy | File | Description |
|:---------|:-----|:------------|
| **XK-types** | `krylov-utils-xk.jl` | Defines dispatch rules on `XKVector` / `XKSparseMatrixCSR` wrapper types. The original CPU code-paths are untouched; you opt-in by wrapping your data. |
| **Drop-in overrides** | `krylov-utils-overrides.jl` | Overrides Krylov's dispatch directly on plain `Vector` / `SparseMatrixCSR`. Every Krylov solve on CPU arrays is silently redirected to XKBlas. |

## Usage

```bash
julia examples/Krylov/krylov.jl [solver] [n] [tile_size] [use_xkblas]
```

| Argument | Default | Description |
|:---------|:--------|:------------|
| `solver` | `cg` | Name of the Krylov solver (e.g. `cg`, `minres`, `gmres`) |
| `n` | `4` | Problem size |
| `tile_size` | `2` | XKBlas tile parameter |
| `use_xkblas` | `false` | Set to `true` to use the XK.BLAS backend |

For example, running a Conjugate-Gradient solve of size 1024 on GPUs
with a tile size of 256:

```bash
julia examples/Krylov/krylov.jl cg 1024 256 true
```

## Source files

### Main script — `krylov.jl`

Sets up a symmetric positive-definite tridiagonal system, selects the
solver, and optionally wraps the data in XK types when `use_xkblas` is
`true`.  After the solve, it writes results back to the host with
`XK.memory_coherent_sync` and checks the residual.

```@eval
using Markdown
Markdown.parse("```julia\n" * read(joinpath(@__DIR__, "..", "..", "examples", "Krylov", "krylov.jl"), String) * "\n```")
```

### XK-types dispatch — `krylov-utils-xk.jl`

Implements Krylov's kernel API for `XKVector` and `XKSparseMatrixCSR`.
Each override delegates to the corresponding `XK.BLAS` synchronous
routine (e.g. `kaxpy!` calls `XK.BLAS.axpy_sync`, `kmul!` calls
`XK.BLAS.spmv_sync`).

```@eval
using Markdown
Markdown.parse("```julia\n" * read(joinpath(@__DIR__, "..", "..", "examples", "Krylov", "krylov-utils-xk.jl"), String) * "\n```")
```

### Drop-in overrides — `krylov-utils-overrides.jl`

Overrides Krylov's kernel API directly for standard Julia `Vector` and
`SparseMatrixCSR` types so that XKBlas is used without wrapping data.

```@eval
using Markdown
Markdown.parse("```julia\n" * read(joinpath(@__DIR__, "..", "..", "examples", "Krylov", "krylov-utils-overrides.jl"), String) * "\n```")
```
