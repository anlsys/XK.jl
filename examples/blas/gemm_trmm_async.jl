## # Combining asynchronous BLAS routines: GEMM + TRMM
##
## This example demonstrates how to chain multiple asynchronous BLAS
## operations using `XK.BLAS`, with a final write-back to host memory.
##
## The computation performed is:
##
## ```
## C = alpha * A * B           (GEMM)
## C = alpha_trmm * L * C      (TRMM: triangular matrix multiply)
## ```
##
## where `L` is a lower-triangular matrix with unit diagonal.
##
## All operations are launched asynchronously with `_async` variants.
## The runtime automatically tracks data dependencies between tasks,
## so the TRMM will only execute after the GEMM that produces `C`
## has completed. A final `memory_coherent_async` call requests
## a write-back of `C` from device memory to host memory, and
## `XK.sync()` blocks until every task has finished.

using LinearAlgebra, XK

## ## Problem setup

const T = Float64
n = 4

## Allocate host memory. XKRT will replicate data to devices automatically.
A = T[rand() for _ in 1:(n*n)]
B = T[rand() for _ in 1:(n*n)]
C = T[0.0    for _ in 1:(n*n)]

## Build a lower-triangular matrix with unit diagonal for TRMM
L = T[0.0 for _ in 1:(n*n)]
for j in 1:n, i in 1:n
    if i == j
        L[(j-1)*n + i] = T(1.0)
    elseif i > j
        L[(j-1)*n + i] = T(rand())
    end
end

## BLAS parameters
alpha      = T(1.0)
beta       = T(0.0)
alpha_trmm = T(1.0)
transA, transB = XK.BLAS.NO_TRANS, XK.BLAS.NO_TRANS

## ## Asynchronous execution

@time begin
    ## Step 1: C = alpha * A * B  (GEMM)
    XK.BLAS.gemm_async(
        transA, transB,
        n, n, n,
        alpha,
        A, n,
        B, n,
        beta,
        C, n,
    )

    ## Step 2: C = alpha_trmm * L * C  (TRMM, left side, lower triangular, unit diagonal)
    XK.BLAS.trmm_async(
        XK.BLAS.LEFT,
        XK.BLAS.LOWER,
        XK.BLAS.NO_TRANS,
        XK.BLAS.UNIT,
        n, n,
        alpha_trmm,
        L, n,
        C, n,
    )

    ## Step 3: Write C back from device to host memory
    XK.memory_matrix_coherent_async(C, n, n, n, sizeof(T))

    ## Step 4: Wait for all asynchronous tasks to complete
    XK.sync()
end

## ## Verification
## Compare with Julia's built-in linear algebra

A_mat = reshape(A, n, n)
B_mat = reshape(B, n, n)
L_mat = reshape(L, n, n)

C_ref = L_mat * (A_mat * B_mat)
C_xk  = reshape(C, n, n)

println("XK    C = ", C_xk)
println("Julia C = ", C_ref)
println("Max error: ", maximum(abs.(C_xk .- C_ref)))
