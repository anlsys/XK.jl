# XK.BLAS

```@docs
XK.BLAS
```

## Execution flavors

Every routine below is available in three variants:

- **`routine`** — default (drop-in): submits the operation, ensures host–device coherence on outputs, and blocks.
- **`routine_async`** — asynchronous: submits the operation and returns immediately. Use [`XK.sync()`](@ref) to wait.
- **`routine_sync`** — synchronous: submits the operation and blocks until it completes.

## Constants

### Storage order
```@docs
XK.BLAS.ROW_MAJOR
XK.BLAS.COL_MAJOR
```

### Transpose
```@docs
XK.BLAS.NO_TRANS
XK.BLAS.TRANS
XK.BLAS.CONJ_TRANS
```

### Triangle
```@docs
XK.BLAS.UPPER
XK.BLAS.LOWER
```

### Diagonal
```@docs
XK.BLAS.UNIT
XK.BLAS.NON_UNIT
```

### Side
```@docs
XK.BLAS.LEFT
XK.BLAS.RIGHT
```

### Sparse formats
```@docs
XK.BLAS.SPARSE_CSR
XK.BLAS.SPARSE_CSC
XK.BLAS.SPARSE_COO
XK.BLAS.SPARSE_BSR
XK.BLAS.SPARSE_ELL
XK.BLAS.SPARSE_DIA
```

## Tile size

```@docs
XK.BLAS.set_tile_size
```

## Memory helpers

```@docs
XK.BLAS.memory_register
XK.BLAS.memory_unregister
XK.BLAS.memory_coherent_async
XK.BLAS.memory_coherent_sync
```

## Level 1 — Vector operations

```@docs
XK.BLAS.axpy
XK.BLAS.dot
```

## Level 2 — Matrix-vector operations

```@docs
XK.BLAS.gemv
```

## Level 3 — Matrix-matrix operations

```@docs
XK.BLAS.gemm
XK.BLAS.trmm
XK.BLAS.symm
XK.BLAS.herk
XK.BLAS.syrk
XK.BLAS.syr2k
```

## Sparse operations

```@docs
XK.BLAS.spmv
```

## Extended routines (`XK.BLAS.ext`)

```@docs
XK.BLAS.ext
XK.BLAS.ext.axpby
XK.BLAS.ext.scal
XK.BLAS.ext.gemmt
```

### `copy` / `copy_async` / `copy_sync`

    copy(n, x::AbstractVector{T}, incx, y::AbstractVector{T}, incy)

Copy vector `x` into vector `y`: `y := x`.
Both vectors must have the same element type.

Supported types: `Float32`, `Float64`, `ComplexF32`, `ComplexF64`.

### `fill` / `fill_async` / `fill_sync`

    fill(n, x, value::T)

Fill the first `n` elements of vector `x` with `value`: `x[1:n] .= value`.

Supported types: `Float32`, `Float64`, `ComplexF32`, `ComplexF64`.
