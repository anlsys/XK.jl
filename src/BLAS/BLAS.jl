# Kernels (see `xkblas/xkblas.hpp` and convert C++ prototypes)

"""
Multi-GPU BLAS module for XK.jl.

Provides BLAS Level 1, 2, and 3 routines that are automatically tiled and
distributed across all available GPUs by the XKRT runtime. Matrices and
vectors are stored in host memory; the runtime replicates and manages device
copies transparently.

## Execution flavors

Every routine is available in three flavors:

| Suffix     | Behavior |
|:-----------|:---------|
| `_async`   | Submits the operation as an asynchronous task and returns immediately. A subsequent [`XK.sync()`](@ref) (or a data-dependent task) is required to guarantee completion. |
| `_sync`    | Submits the operation and blocks until it completes. |
| *(none)*   | Default flavor: submits the operation, makes the **output** memory coherent between host and devices, and blocks until completion. This is the simplest ("drop-in") usage. |

## Supported element types

`Float32`, `Float64`, `ComplexF32`, `ComplexF64` — dispatched automatically
from the type of the scalar arguments (`alpha`, `beta`, etc.).

## Memory coherence

After asynchronous operations, device results are **not** automatically
written back to host memory.  Use [`XK.memory_coherent_async`](@ref) /
[`XK.memory_coherent_sync`](@ref) (or the `BLAS.memory_coherent_*` helpers)
to request a write-back, then [`XK.sync()`](@ref) to wait.

## Extended routines

Additional BLAS-like routines (`axpby`, `copy`, `fill`, `scal`, `gemmt`) are
available under the [`XK.BLAS.ext`](@ref) sub-module.

See also: [Examples](@ref)
"""
module BLAS

    import ..XK

    # ──────────────────────────────────────────────
    # Tile size
    # ──────────────────────────────────────────────

    """
        set_tile_size(ts::Int)

    Set the tile size (in elements) used by XKBlas to partition matrices and
    vectors across devices.  A common choice is `div(n, XK.get_ngpus())` so
    that each GPU receives exactly one tile.

    This is a convenience wrapper around [`XK.set_tile_parameter`](@ref).
    """
    set_tile_size(ts::Int) = XK.set_tile_parameter(ts)

    # ──────────────────────────────────────────────
    # Memory helpers
    # ──────────────────────────────────────────────

    """
        memory_register(x::AbstractVector [, n])
        memory_register(A::AbstractMatrix)

    Register host memory with the XKRT runtime so that it can be
    replicated to and from devices.  For vectors, `n` elements are
    registered (defaults to `length(x)`).
    """
    memory_register(x::AbstractVector, n)   = XK.register_memory(x, n*sizeof(eltype(x)))
    memory_register(x::AbstractVector)      = XK.BLAS.memory_register(x, length(x))
    memory_register(A::AbstractMatrix)      = XK.register_memory(pointer(A), sizeof(A))

    """
        memory_unregister(x::AbstractVector [, n])
        memory_unregister(A::AbstractMatrix)

    Unregister host memory previously registered with [`memory_register`](@ref).
    """
    memory_unregister(x::AbstractVector)    = unregister_memory(x, length(x)*sizeof(eltype(x)))
    memory_unregister(x::AbstractVector, n) = unregister_memory(x, n*sizeof(eltype(x)))
    memory_unregister(x::AbstractMatrix)    = XK.register_memory(pointer(A), sizeof(A))

    """
        memory_coherent_async(x::AbstractVector [, n])
        memory_coherent_async(A::AbstractMatrix)

    Asynchronously request that device data be written back to host
    memory.  Must be followed by [`XK.sync()`](@ref) to guarantee the
    write-back has completed.
    """
    memory_coherent_async(x::AbstractVector)    = XK.memory_segment_coherent_async(x, length(x)*sizeof(eltype(x)))
    memory_coherent_async(x::AbstractVector, n) = XK.memory_segment_coherent_async(x, n*sizeof(eltype(x)))
    memory_coherent_async(A::AbstractMatrix)    = XK.BLAS.memory_coherent_async(pointer(A), stride(A, 2), size(A, 1), size(A, 2))

    """
        memory_coherent_sync(x::AbstractVector [, n])
        memory_coherent_sync(A::AbstractMatrix)

    Synchronously write back device data to host memory.  Blocks until
    the write-back has completed.
    """
    memory_coherent_sync(x::AbstractVector)     = XK.memory_segment_coherent_async(x, length(x)*sizeof(eltype(x)))
    memory_coherent_sync(x::AbstractVector, n)  = XK.memory_segment_coherent_async(x, n*sizeof(eltype(x)))
    memory_coherent_sync(A::AbstractMatrix)     = XK.BLAS.memory_coherent_async(pointer(A), stride(A, 2), size(A, 1), size(A, 2))

    # ──────────────────────────────────────────────
    # CBLAS enum constants
    # ──────────────────────────────────────────────

    """
    Row-major storage order (CBLAS enum).
    """
    const ROW_MAJOR = XK.CblasRowMajor

    """
    Column-major storage order (CBLAS enum).  This is the default layout
    used by Julia arrays.
    """
    const COL_MAJOR = XK.CblasColMajor

    """
    Do not transpose the matrix (CBLAS enum).
    """
    const NO_TRANS   = XK.CblasNoTrans

    """
    Transpose the matrix (CBLAS enum).
    """
    const TRANS      = XK.CblasTrans

    """
    Conjugate-transpose the matrix (CBLAS enum).
    """
    const CONJ_TRANS = XK.CblasConjTrans

    """
    Refer to the upper-triangular part of the matrix (CBLAS enum).
    """
    const UPPER = XK.CblasUpper

    """
    Refer to the lower-triangular part of the matrix (CBLAS enum).
    """
    const LOWER = XK.CblasLower

    """
    The triangular matrix has a non-unit diagonal (CBLAS enum).
    """
    const NON_UNIT = XK.CblasNonUnit

    """
    The triangular matrix has a unit diagonal (CBLAS enum).
    """
    const UNIT     = XK.CblasUnit

    """
    Apply the operator from the left side (CBLAS enum).
    """
    const LEFT  = XK.CblasLeft

    """
    Apply the operator from the right side (CBLAS enum).
    """
    const RIGHT = XK.CblasRight

    """
    Compressed Sparse Row format identifier.
    """
    const SPARSE_CSR = XK.CblasSparseCSR

    """
    Compressed Sparse Column format identifier.
    """
    const SPARSE_CSC = XK.CblasSparseCSC

    """
    Coordinate (COO) sparse format identifier.
    """
    const SPARSE_COO = XK.CblasSparseCOO

    """
    Block Sparse Row format identifier.
    """
    const SPARSE_BSR = XK.CblasSparseBSR

    """
    ELLPACK sparse format identifier.
    """
    const SPARSE_ELL = XK.CblasSparseELL

    """
    Diagonal sparse format identifier.
    """
    const SPARSE_DIA = XK.CblasSparseDIA

    # ──────────────────────────────────────────────
    # Level 1
    # ──────────────────────────────────────────────

    """
        axpy(n, alpha, x, incx, y, incy)
        axpy_async(n, alpha, x, incx, y, incy)
        axpy_sync(n, alpha, x, incx, y, incy)

    Compute the vector operation

        y := alpha * x + y

    where `x` and `y` are vectors of length `n` with strides `incx` and
    `incy`, and `alpha` is a scalar.

    Supported types: `Float32`, `Float64`, `ComplexF32`, `ComplexF64`.
    """
    axpy, axpy_async, axpy_sync

    axpy(      n, alpha::ComplexF32, x, incx, y, incy)  = XK.caxpy(      n, Ref(alpha), x, incx, y, incy)
    axpy(      n, alpha::ComplexF64, x, incx, y, incy)  = XK.zaxpy(      n, Ref(alpha), x, incx, y, incy)
    axpy(      n, alpha::Float32,    x, incx, y, incy)  = XK.saxpy(      n, Ref(alpha), x, incx, y, incy)
    axpy(      n, alpha::Float64,    x, incx, y, incy)  = XK.daxpy(      n, Ref(alpha), x, incx, y, incy)
    axpy_async(n, alpha::ComplexF32, x, incx, y, incy)  = XK.caxpy_async(n, Ref(alpha), x, incx, y, incy)
    axpy_async(n, alpha::ComplexF64, x, incx, y, incy)  = XK.zaxpy_async(n, Ref(alpha), x, incx, y, incy)
    axpy_async(n, alpha::Float32,    x, incx, y, incy)  = XK.saxpy_async(n, Ref(alpha), x, incx, y, incy)
    axpy_async(n, alpha::Float64,    x, incx, y, incy)  = XK.daxpy_async(n, Ref(alpha), x, incx, y, incy)
    axpy_sync( n, alpha::ComplexF32, x, incx, y, incy)  = XK.caxpy_sync( n, Ref(alpha), x, incx, y, incy)
    axpy_sync( n, alpha::ComplexF64, x, incx, y, incy)  = XK.zaxpy_sync( n, Ref(alpha), x, incx, y, incy)
    axpy_sync( n, alpha::Float32,    x, incx, y, incy)  = XK.saxpy_sync( n, Ref(alpha), x, incx, y, incy)
    axpy_sync( n, alpha::Float64,    x, incx, y, incy)  = XK.daxpy_sync( n, Ref(alpha), x, incx, y, incy)

    """
        dot(n, x, incx, y, incy, result::Ref)
        dot_async(n, x, incx, y, incy, result::Ref)
        dot_sync(n, x, incx, y, incy, result::Ref)

    Compute the dot product

        result[] = x' * y

    where `x` and `y` are vectors of length `n`.  The result is written
    into `result`, which must be a `Ref{Float32}` or `Ref{Float64}`.

    !!! note
        Complex dot (`dotc`/`dotu`) is not yet supported.
    """
    dot, dot_async, dot_sync

    # TODO: complex dot (dotc) not supported yet.
    # Add it to that dispatcher once supported
    dot(      n, x, incx, y, incy, result::Ref{Float32}) = XK.sdot(      n, x, incx, y, incy, result)
    dot(      n, x, incx, y, incy, result::Ref{Float64}) = XK.ddot(      n, x, incx, y, incy, result)
    dot_async(n, x, incx, y, incy, result::Ref{Float32}) = XK.sdot_async(n, x, incx, y, incy, result)
    dot_async(n, x, incx, y, incy, result::Ref{Float64}) = XK.ddot_async(n, x, incx, y, incy, result)
    dot_sync( n, x, incx, y, incy, result::Ref{Float32}) = XK.sdot_sync( n, x, incx, y, incy, result)
    dot_sync( n, x, incx, y, incy, result::Ref{Float64}) = XK.ddot_sync( n, x, incx, y, incy, result)

    # ──────────────────────────────────────────────
    # Level 2
    # ──────────────────────────────────────────────

    """
        gemv(transA, m, n, alpha, A, lda, x, incx, beta, y, incy)
        gemv_async(transA, m, n, alpha, A, lda, x, incx, beta, y, incy)
        gemv_sync(transA, m, n, alpha, A, lda, x, incx, beta, y, incy)

    General matrix-vector multiply:

        y := alpha * op(A) * x + beta * y

    where `op(A)` is `A`, `A'`, or `conj(A')` depending on `transA`
    ([`NO_TRANS`](@ref), [`TRANS`](@ref), [`CONJ_TRANS`](@ref)).
    `A` is an `m x n` matrix stored in column-major order with leading
    dimension `lda`.

    Supported types: `Float32`, `Float64`.

    A convenience method accepting `AbstractMatrix` and `AbstractVector`
    arguments is also available:

        gemv(transA, alpha, A::AbstractMatrix, x, beta, y)
    """
    gemv, gemv_async, gemv_sync

    gemv(      transA, m, n, alpha::Float32, A, lda, x, incx, beta, y, incy) = XK.sgemv(      transA, m, n, Ref(alpha), A, lda, x, incx, Ref(beta), y, incy)
    gemv(      transA, m, n, alpha::Float64, A, lda, x, incx, beta, y, incy) = XK.dgemv(      transA, m, n, Ref(alpha), A, lda, x, incx, Ref(beta), y, incy)
    gemv_async(transA, m, n, alpha::Float32, A, lda, x, incx, beta, y, incy) = XK.sgemv_async(transA, m, n, Ref(alpha), A, lda, x, incx, Ref(beta), y, incy)
    gemv_async(transA, m, n, alpha::Float64, A, lda, x, incx, beta, y, incy) = XK.dgemv_async(transA, m, n, Ref(alpha), A, lda, x, incx, Ref(beta), y, incy)
    gemv_sync( transA, m, n, alpha::Float32, A, lda, x, incx, beta, y, incy) = XK.sgemv_sync( transA, m, n, Ref(alpha), A, lda, x, incx, Ref(beta), y, incy)
    gemv_sync( transA, m, n, alpha::Float64, A, lda, x, incx, beta, y, incy) = XK.dgemv_sync( transA, m, n, Ref(alpha), A, lda, x, incx, Ref(beta), y, incy)

    gemv(      transA, alpha, A::AbstractMatrix, x, beta, y) = XK.gemv(transA, size(A, 1), size(A, 2), size(A, 2), alpha, pointer(A), stride(A, 2), pointer(x), stride(x, 1), beta, pointer(y), stride(y, 1))
    gemv_async(transA, alpha, A::AbstractMatrix, x, beta, y) = XK.gemv(transA, size(A, 1), size(A, 2), size(A, 2), alpha, pointer(A), stride(A, 2), pointer(x), stride(x, 1), beta, pointer(y), stride(y, 1))
    gemv_sync( transA, alpha, A::AbstractMatrix, x, beta, y) = XK.gemv(transA, size(A, 1), size(A, 2), size(A, 2), alpha, pointer(A), stride(A, 2), pointer(x), stride(x, 1), beta, pointer(y), stride(y, 1))

    # ──────────────────────────────────────────────
    # Level 3
    # ──────────────────────────────────────────────

    """
        gemm(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
        gemm_async(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
        gemm_sync(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)

    General matrix-matrix multiply:

        C := alpha * op(A) * op(B) + beta * C

    where `op(X)` is `X`, `X'`, or `conj(X')` depending on `transA`/`transB`.
    `A` is `m x k`, `B` is `k x n`, `C` is `m x n`, all column-major with
    leading dimensions `lda`, `ldb`, `ldc`.

    Supported types: `Float32`, `Float64`, `ComplexF32`, `ComplexF64`.

    A convenience method accepting `AbstractMatrix` arguments is also available:

        gemm(transA, transB, alpha, A::AbstractMatrix, B::AbstractMatrix, beta, C::AbstractMatrix)
    """
    gemm, gemm_async, gemm_sync

    gemm(      transA, transB, m, n, k, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = XK.cgemm(      transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    gemm(      transA, transB, m, n, k, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = XK.zgemm(      transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    gemm(      transA, transB, m, n, k, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = XK.sgemm(      transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    gemm(      transA, transB, m, n, k, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = XK.dgemm(      transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    gemm_async(transA, transB, m, n, k, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = XK.cgemm_async(transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    gemm_async(transA, transB, m, n, k, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = XK.zgemm_async(transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    gemm_async(transA, transB, m, n, k, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = XK.sgemm_async(transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    gemm_async(transA, transB, m, n, k, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = XK.dgemm_async(transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    gemm_sync( transA, transB, m, n, k, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = XK.cgemm_sync( transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    gemm_sync( transA, transB, m, n, k, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = XK.zgemm_sync( transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    gemm_sync( transA, transB, m, n, k, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = XK.sgemm_sync( transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    gemm_sync( transA, transB, m, n, k, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = XK.dgemm_sync( transA, transB, m, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)

    gemm(      transA, transB, alpha, A::AbstractMatrix, B::AbstractMatrix, beta, C::AbstractMatrix)  = XK.BLAS.gemm(transA, transB, size(C, 1), size(C, 2), size(A, 2), alpha, pointer(A), stride(A, 2), pointer(B), stride(B, 2), beta, pointer(C), stride(C, 2))
    gemm_async(transA, transB, alpha, A::AbstractMatrix, B::AbstractMatrix, beta, C::AbstractMatrix)  = XK.BLAS.gemm(transA, transB, size(C, 1), size(C, 2), size(A, 2), alpha, pointer(A), stride(A, 2), pointer(B), stride(B, 2), beta, pointer(C), stride(C, 2))
    gemm_sync( transA, transB, alpha, A::AbstractMatrix, B::AbstractMatrix, beta, C::AbstractMatrix)  = XK.BLAS.gemm(transA, transB, size(C, 1), size(C, 2), size(A, 2), alpha, pointer(A), stride(A, 2), pointer(B), stride(B, 2), beta, pointer(C), stride(C, 2))

    """
        herk(uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
        herk_async(uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
        herk_sync(uplo, trans, n, k, alpha, A, lda, beta, C, ldc)

    Hermitian rank-k update:

        C := alpha * A * A^H + beta * C     (trans = NO_TRANS)
        C := alpha * A^H * A + beta * C     (trans = CONJ_TRANS)

    Only the `uplo` ([`UPPER`](@ref) or [`LOWER`](@ref)) triangle of `C`
    is updated.  `C` is `n x n`, `A` is `n x k` (or `k x n`).

    Supported types: `Float32`, `Float64`, `ComplexF32`, `ComplexF64`.
    """
    herk, herk_async, herk_sync

    herk(      uplo, trans, n, k, alpha::ComplexF32, A, lda, beta::ComplexF32, C, ldc)  = XK.cherk(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    herk(      uplo, trans, n, k, alpha::ComplexF64, A, lda, beta::ComplexF64, C, ldc)  = XK.zherk(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    herk(      uplo, trans, n, k, alpha::Float32,    A, lda, beta::Float32,    C, ldc)  = XK.sherk(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    herk(      uplo, trans, n, k, alpha::Float64,    A, lda, beta::Float64,    C, ldc)  = XK.dherk(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    herk_async(uplo, trans, n, k, alpha::ComplexF32, A, lda, beta::ComplexF32, C, ldc)  = XK.cherk_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    herk_async(uplo, trans, n, k, alpha::ComplexF64, A, lda, beta::ComplexF64, C, ldc)  = XK.zherk_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    herk_async(uplo, trans, n, k, alpha::Float32,    A, lda, beta::Float32,    C, ldc)  = XK.sherk_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    herk_async(uplo, trans, n, k, alpha::Float64,    A, lda, beta::Float64,    C, ldc)  = XK.dherk_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    herk_sync( uplo, trans, n, k, alpha::ComplexF32, A, lda, beta::ComplexF32, C, ldc)  = XK.cherk_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    herk_sync( uplo, trans, n, k, alpha::ComplexF64, A, lda, beta::ComplexF64, C, ldc)  = XK.zherk_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    herk_sync( uplo, trans, n, k, alpha::Float32,    A, lda, beta::Float32,    C, ldc)  = XK.sherk_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    herk_sync( uplo, trans, n, k, alpha::Float64,    A, lda, beta::Float64,    C, ldc)  = XK.dherk_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)

    """
        symm(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
        symm_async(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
        symm_sync(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)

    Symmetric matrix-matrix multiply:

        C := alpha * A * B + beta * C     (side = LEFT)
        C := alpha * B * A + beta * C     (side = RIGHT)

    where `A` is symmetric, stored in the `uplo` triangle.

    Supported types: `Float32`, `Float64`, `ComplexF32`, `ComplexF64`.
    """
    symm, symm_async, symm_sync

    symm(      side, uplo, m, n, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = XK.csymm(      uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    symm(      side, uplo, m, n, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = XK.zsymm(      uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    symm(      side, uplo, m, n, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = XK.ssymm(      uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    symm(      side, uplo, m, n, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = XK.dsymm(      uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    symm_async(side, uplo, m, n, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = XK.csymm_async(uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    symm_async(side, uplo, m, n, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = XK.zsymm_async(uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    symm_async(side, uplo, m, n, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = XK.ssymm_async(uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    symm_async(side, uplo, m, n, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = XK.dsymm_async(uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    symm_sync( side, uplo, m, n, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = XK.csymm_sync( uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    symm_sync( side, uplo, m, n, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = XK.zsymm_sync( uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    symm_sync( side, uplo, m, n, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = XK.ssymm_sync( uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    symm_sync( side, uplo, m, n, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = XK.dsymm_sync( uplo, side, m, n, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)

    """
        syr2k(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
        syr2k_async(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
        syr2k_sync(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)

    Symmetric rank-2k update:

        C := alpha * A * B' + alpha * B * A' + beta * C     (trans = NO_TRANS)
        C := alpha * A' * B + alpha * B' * A + beta * C     (trans = TRANS)

    Only the `uplo` triangle of `C` is updated.

    Supported types: `Float32`, `Float64`, `ComplexF32`, `ComplexF64`.
    """
    syr2k, syr2k_async, syr2k_sync

    syr2k(      uplo, trans, n, k, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = XK.csyr2k(      uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syr2k(      uplo, trans, n, k, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = XK.zsyr2k(      uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syr2k(      uplo, trans, n, k, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = XK.ssyr2k(      uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syr2k(      uplo, trans, n, k, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = XK.dsyr2k(      uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syr2k_async(uplo, trans, n, k, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = XK.csyr2k_async(uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syr2k_async(uplo, trans, n, k, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = XK.zsyr2k_async(uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syr2k_async(uplo, trans, n, k, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = XK.ssyr2k_async(uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syr2k_async(uplo, trans, n, k, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = XK.dsyr2k_async(uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syr2k_sync( uplo, trans, n, k, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = XK.csyr2k_sync( uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syr2k_sync( uplo, trans, n, k, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = XK.zsyr2k_sync( uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syr2k_sync( uplo, trans, n, k, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = XK.ssyr2k_sync( uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syr2k_sync( uplo, trans, n, k, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = XK.dsyr2k_sync( uplo, trans, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)

    """
        syrk(uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
        syrk_async(uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
        syrk_sync(uplo, trans, n, k, alpha, A, lda, beta, C, ldc)

    Symmetric rank-k update:

        C := alpha * A * A' + beta * C     (trans = NO_TRANS)
        C := alpha * A' * A + beta * C     (trans = TRANS)

    Only the `uplo` triangle of `C` is updated.

    Supported types: `Float32`, `Float64`, `ComplexF32`, `ComplexF64`.
    """
    syrk, syrk_async, syrk_sync

    syrk(      uplo, trans, n, k, alpha::ComplexF32, A, lda, beta::ComplexF32, C, ldc)  = XK.csyrk(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syrk(      uplo, trans, n, k, alpha::ComplexF64, A, lda, beta::ComplexF64, C, ldc)  = XK.zsyrk(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syrk(      uplo, trans, n, k, alpha::Float32,    A, lda, beta::Float32,    C, ldc)  = XK.ssyrk(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syrk(      uplo, trans, n, k, alpha::Float64,    A, lda, beta::Float64,    C, ldc)  = XK.dsyrk(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syrk_async(uplo, trans, n, k, alpha::ComplexF32, A, lda, beta::ComplexF32, C, ldc)  = XK.csyrk_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syrk_async(uplo, trans, n, k, alpha::ComplexF64, A, lda, beta::ComplexF64, C, ldc)  = XK.zsyrk_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syrk_async(uplo, trans, n, k, alpha::Float32,    A, lda, beta::Float32,    C, ldc)  = XK.ssyrk_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syrk_async(uplo, trans, n, k, alpha::Float64,    A, lda, beta::Float64,    C, ldc)  = XK.dsyrk_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syrk_sync( uplo, trans, n, k, alpha::ComplexF32, A, lda, beta::ComplexF32, C, ldc)  = XK.csyrk_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syrk_sync( uplo, trans, n, k, alpha::ComplexF64, A, lda, beta::ComplexF64, C, ldc)  = XK.zsyrk_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syrk_sync( uplo, trans, n, k, alpha::Float32,    A, lda, beta::Float32,    C, ldc)  = XK.ssyrk_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
    syrk_sync( uplo, trans, n, k, alpha::Float64,    A, lda, beta::Float64,    C, ldc)  = XK.dsyrk_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)

    """
        trmm(side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb)
        trmm_async(side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb)
        trmm_sync(side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb)

    Triangular matrix-matrix multiply:

        B := alpha * op(A) * B     (side = LEFT)
        B := alpha * B * op(A)     (side = RIGHT)

    where `A` is a triangular matrix.  `uplo` selects the triangle
    ([`UPPER`](@ref) / [`LOWER`](@ref)), `diag` indicates whether the
    diagonal is unit ([`UNIT`](@ref)) or non-unit ([`NON_UNIT`](@ref)),
    and `transA` controls transposition.

    Supported types: `Float32`, `Float64`, `ComplexF32`, `ComplexF64`.
    """
    trmm, trmm_async, trmm_sync

    trmm(      side, uplo, transA, diag, m, n, alpha::ComplexF32, A, lda, B, ldb)  = XK.ctrmm(      side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
    trmm(      side, uplo, transA, diag, m, n, alpha::ComplexF64, A, lda, B, ldb)  = XK.ztrmm(      side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
    trmm(      side, uplo, transA, diag, m, n, alpha::Float32,    A, lda, B, ldb)  = XK.strmm(      side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
    trmm(      side, uplo, transA, diag, m, n, alpha::Float64,    A, lda, B, ldb)  = XK.dtrmm(      side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
    trmm_async(side, uplo, transA, diag, m, n, alpha::ComplexF32, A, lda, B, ldb)  = XK.ctrmm_async(side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
    trmm_async(side, uplo, transA, diag, m, n, alpha::ComplexF64, A, lda, B, ldb)  = XK.ztrmm_async(side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
    trmm_async(side, uplo, transA, diag, m, n, alpha::Float32,    A, lda, B, ldb)  = XK.strmm_async(side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
    trmm_async(side, uplo, transA, diag, m, n, alpha::Float64,    A, lda, B, ldb)  = XK.dtrmm_async(side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
    trmm_sync( side, uplo, transA, diag, m, n, alpha::ComplexF32, A, lda, B, ldb)  = XK.ctrmm_sync( side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
    trmm_sync( side, uplo, transA, diag, m, n, alpha::ComplexF64, A, lda, B, ldb)  = XK.ztrmm_sync( side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
    trmm_sync( side, uplo, transA, diag, m, n, alpha::Float32,    A, lda, B, ldb)  = XK.strmm_sync( side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)
    trmm_sync( side, uplo, transA, diag, m, n, alpha::Float64,    A, lda, B, ldb)  = XK.dtrmm_sync( side, uplo, transA, diag, m, n, Ref(alpha), A, lda, B, ldb)

    # ──────────────────────────────────────────────
    # Sparse
    # ──────────────────────────────────────────────

    """
        spmv(alpha, transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y)
        spmv_async(alpha, transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y)
        spmv_sync(alpha, transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y)

    Sparse matrix-vector multiply:

        Y := alpha * op(A) * X + beta * Y

    The sparse matrix `A` is described by its `rows`, `cols`, and `values`
    arrays in the given `format` (e.g. [`SPARSE_CSR`](@ref)).
    `nrows`, `ncols`, and `nnz` give the matrix dimensions and number of
    non-zeros.

    Supported types: `Float32`, `Float64`, `ComplexF32`, `ComplexF64`.
    """
    spmv, spmv_async, spmv_sync

    spmv(      alpha::ComplexF32, transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = XK.cspmv(      Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)
    spmv(      alpha::ComplexF64, transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = XK.zspmv(      Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)
    spmv(      alpha::Float32,    transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = XK.sspmv(      Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)
    spmv(      alpha::Float64,    transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = XK.dspmv(      Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)
    spmv_async(alpha::ComplexF32, transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = XK.cspmv_async(Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)
    spmv_async(alpha::ComplexF64, transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = XK.zspmv_async(Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)
    spmv_async(alpha::Float32,    transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = XK.sspmv_async(Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)
    spmv_async(alpha::Float64,    transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = XK.dspmv_async(Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)
    spmv_sync( alpha::ComplexF32, transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = XK.cspmv_sync( Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)
    spmv_sync( alpha::ComplexF64, transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = XK.zspmv_sync( Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)
    spmv_sync( alpha::Float32,    transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = XK.sspmv_sync( Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)
    spmv_sync( alpha::Float64,    transA, nrows, ncols, nnz, format, rows, cols, values, X, beta, Y) = XK.dspmv_sync( Ref(alpha), transA, 1, 8*sizeof(eltype(rows)), nrows, ncols, nnz, format, rows, cols, values, X, Ref(beta), Y)

    # ──────────────────────────────────────────────
    # Extended routines (ext sub-module)
    # ──────────────────────────────────────────────

    """
    Sub-module containing extended BLAS-like routines not part of the
    standard BLAS specification:

    - **Level 1:** `axpby`, `copy`, `fill`, `scal`
    - **Level 3:** `gemmt`

    Each routine is available in the three execution flavors (`_async`,
    `_sync`, and default).
    """
    module ext

        import ..XK

        ### Level 1 ###

        """
            axpby(n, alpha, x, incx, beta, y, incy)
            axpby_async(n, alpha, x, incx, beta, y, incy)
            axpby_sync(n, alpha, x, incx, beta, y, incy)

        Compute the scaled vector addition:

            y := alpha * x + beta * y

        Supported types: `Float32`, `Float64`, `ComplexF32`, `ComplexF64`.
        """
        axpby, axpby_async, axpby_sync

        axpby(      n, alpha::ComplexF32, x, incx, beta, y, incy) = XK.caxpby(      n, Ref(alpha), x, incx, Ref(beta), y, incy)
        axpby(      n, alpha::ComplexF64, x, incx, beta, y, incy) = XK.zaxpby(      n, Ref(alpha), x, incx, Ref(beta), y, incy)
        axpby(      n, alpha::Float32,    x, incx, beta, y, incy) = XK.saxpby(      n, Ref(alpha), x, incx, Ref(beta), y, incy)
        axpby(      n, alpha::Float64,    x, incx, beta, y, incy) = XK.daxpby(      n, Ref(alpha), x, incx, Ref(beta), y, incy)
        axpby_async(n, alpha::ComplexF32, x, incx, beta, y, incy) = XK.caxpby_async(n, Ref(alpha), x, incx, Ref(beta), y, incy)
        axpby_async(n, alpha::ComplexF64, x, incx, beta, y, incy) = XK.zaxpby_async(n, Ref(alpha), x, incx, Ref(beta), y, incy)
        axpby_async(n, alpha::Float32,    x, incx, beta, y, incy) = XK.saxpby_async(n, Ref(alpha), x, incx, Ref(beta), y, incy)
        axpby_async(n, alpha::Float64,    x, incx, beta, y, incy) = XK.daxpby_async(n, Ref(alpha), x, incx, Ref(beta), y, incy)
        axpby_sync( n, alpha::ComplexF32, x, incx, beta, y, incy) = XK.caxpby_sync( n, Ref(alpha), x, incx, Ref(beta), y, incy)
        axpby_sync( n, alpha::ComplexF64, x, incx, beta, y, incy) = XK.zaxpby_sync( n, Ref(alpha), x, incx, Ref(beta), y, incy)
        axpby_sync( n, alpha::Float32,    x, incx, beta, y, incy) = XK.saxpby_sync( n, Ref(alpha), x, incx, Ref(beta), y, incy)
        axpby_sync( n, alpha::Float64,    x, incx, beta, y, incy) = XK.daxpby_sync( n, Ref(alpha), x, incx, Ref(beta), y, incy)

        # copy: copies vector x into vector y (y := x).
        # Both vectors must have the same element type.
        # Available as copy / copy_async / copy_sync.
        # Supported types: Float32, Float64, ComplexF32, ComplexF64.

        copy(      n, x::AbstractVector{ComplexF32}, incx, y::AbstractVector{ComplexF32}, incy)  = XK.ccopy(      n, x, incx, y, incy)
        copy(      n, x::AbstractVector{ComplexF64}, incx, y::AbstractVector{ComplexF64}, incy)  = XK.zcopy(      n, x, incx, y, incy)
        copy(      n, x::AbstractVector{Float32},    incx, y::AbstractVector{Float32}   , incy)  = XK.scopy(      n, x, incx, y, incy)
        copy(      n, x::AbstractVector{Float64},    incx, y::AbstractVector{Float64}   , incy)  = XK.dcopy(      n, x, incx, y, incy)
        copy_async(n, x::AbstractVector{ComplexF32}, incx, y::AbstractVector{ComplexF32}, incy)  = XK.ccopy_async(n, x, incx, y, incy)
        copy_async(n, x::AbstractVector{ComplexF64}, incx, y::AbstractVector{ComplexF64}, incy)  = XK.zcopy_async(n, x, incx, y, incy)
        copy_async(n, x::AbstractVector{Float32},    incx, y::AbstractVector{Float32}   , incy)  = XK.scopy_async(n, x, incx, y, incy)
        copy_async(n, x::AbstractVector{Float64},    incx, y::AbstractVector{Float64}   , incy)  = XK.dcopy_async(n, x, incx, y, incy)
        copy_sync( n, x::AbstractVector{ComplexF32}, incx, y::AbstractVector{ComplexF32}, incy)  = XK.ccopy_sync( n, x, incx, y, incy)
        copy_sync( n, x::AbstractVector{ComplexF64}, incx, y::AbstractVector{ComplexF64}, incy)  = XK.zcopy_sync( n, x, incx, y, incy)
        copy_sync( n, x::AbstractVector{Float32},    incx, y::AbstractVector{Float32}   , incy)  = XK.scopy_sync( n, x, incx, y, incy)
        copy_sync( n, x::AbstractVector{Float64},    incx, y::AbstractVector{Float64}   , incy)  = XK.dcopy_sync( n, x, incx, y, incy)

        # fill: fills the first n elements of vector x with value (x[1:n] .= value).
        # Available as fill / fill_async / fill_sync.
        # Supported types: Float32, Float64, ComplexF32, ComplexF64.

        fill(      n, x, value::ComplexF32) = XK.cfill(      n, x, value)
        fill(      n, x, value::ComplexF64) = XK.zfill(      n, x, value)
        fill(      n, x, value::Float32   ) = XK.sfill(      n, x, value)
        fill(      n, x, value::Float64   ) = XK.dfill(      n, x, value)
        fill_async(n, x, value::ComplexF32) = XK.cfill_async(n, x, value)
        fill_async(n, x, value::ComplexF64) = XK.zfill_async(n, x, value)
        fill_async(n, x, value::Float32   ) = XK.sfill_async(n, x, value)
        fill_async(n, x, value::Float64   ) = XK.dfill_async(n, x, value)
        fill_sync( n, x, value::ComplexF32) = XK.cfill_sync( n, x, value)
        fill_sync( n, x, value::ComplexF64) = XK.zfill_sync( n, x, value)
        fill_sync( n, x, value::Float32   ) = XK.sfill_sync( n, x, value)
        fill_sync( n, x, value::Float64   ) = XK.dfill_sync( n, x, value)

        """
            scal(n, alpha, x, incx)
            scal_async(n, alpha, x, incx)
            scal_sync(n, alpha, x, incx)

        Scale vector `x` by scalar `alpha`:

            x := alpha * x

        Supported types: `Float32`, `Float64`.

        !!! note
            Complex variants are not yet supported.
        """
        scal, scal_async, scal_sync

        # TODO: complex version not supported yet, but they will need to change the dispatcher
        scal(      n, alpha::Float32, x, incx) = XK.sscal(      n, Ref(alpha), x, incx)
        scal(      n, alpha::Float64, x, incx) = XK.dscal(      n, Ref(alpha), x, incx)
        scal_async(n, alpha::Float32, x, incx) = XK.sscal_async(n, Ref(alpha), x, incx)
        scal_async(n, alpha::Float64, x, incx) = XK.dscal_async(n, Ref(alpha), x, incx)
        scal_sync( n, alpha::Float32, x, incx) = XK.sscal_sync( n, Ref(alpha), x, incx)
        scal_sync( n, alpha::Float64, x, incx) = XK.dscal_sync( n, Ref(alpha), x, incx)

        ### Level 3 ###

        """
            gemmt(uplo, transA, transB, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
            gemmt_async(uplo, transA, transB, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
            gemmt_sync(uplo, transA, transB, n, k, alpha, A, lda, B, ldb, beta, C, ldc)

        General matrix-matrix multiply, updating only one triangle of `C`:

            C_uplo := alpha * op(A) * op(B) + beta * C_uplo

        This is like `gemm` but only the `uplo` (`UPPER` /
        `LOWER`) triangle of `C` is read and written.

        Supported types: `Float32`, `Float64`, `ComplexF32`, `ComplexF64`.
        """
        gemmt, gemmt_async, gemmt_sync

        gemmt(      uplo, transA, transB, n, k, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = XK.cgemmt(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
        gemmt(      uplo, transA, transB, n, k, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = XK.zgemmt(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
        gemmt(      uplo, transA, transB, n, k, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = XK.sgemmt(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
        gemmt(      uplo, transA, transB, n, k, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = XK.dgemmt(      uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
        gemmt_async(uplo, transA, transB, n, k, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = XK.cgemmt_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
        gemmt_async(uplo, transA, transB, n, k, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = XK.zgemmt_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
        gemmt_async(uplo, transA, transB, n, k, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = XK.sgemmt_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
        gemmt_async(uplo, transA, transB, n, k, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = XK.dgemmt_async(uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
        gemmt_sync( uplo, transA, transB, n, k, alpha::ComplexF32, A, lda, B, ldb, beta::ComplexF32, C, ldc)  = XK.cgemmt_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
        gemmt_sync( uplo, transA, transB, n, k, alpha::ComplexF64, A, lda, B, ldb, beta::ComplexF64, C, ldc)  = XK.zgemmt_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
        gemmt_sync( uplo, transA, transB, n, k, alpha::Float32,    A, lda, B, ldb, beta::Float32,    C, ldc)  = XK.sgemmt_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)
        gemmt_sync( uplo, transA, transB, n, k, alpha::Float64,    A, lda, B, ldb, beta::Float64,    C, ldc)  = XK.dgemmt_sync( uplo, transA, transB, n, k, Ref(alpha), A, lda, B, ldb, Ref(beta), C, ldc)

    end

end
