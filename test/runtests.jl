using Test
using XK
using LinearAlgebra

# ──────────────────────────────────────────────────────────────────────────────
# Helper: compute a reference GEMM result using Julia
# C_ref = alpha * op(A) * op(B) + beta * C_orig
# where A, B, C are stored column-major as flat vectors.
# ──────────────────────────────────────────────────────────────────────────────
function ref_gemm(transA, transB, m, n, k, alpha::T, A_flat, lda, B_flat, ldb, beta::T, C_flat, ldc) where T
    A_mat = reshape(copy(A_flat), lda, :)
    B_mat = reshape(copy(B_flat), ldb, :)
    C_mat = reshape(copy(C_flat), ldc, :)

    opA = if transA == XK.BLAS.NO_TRANS
        A_mat[1:m, 1:k]
    else
        transpose(A_mat[1:k, 1:m])
    end

    opB = if transB == XK.BLAS.NO_TRANS
        B_mat[1:k, 1:n]
    else
        transpose(B_mat[1:n, 1:k])
    end

    C_ref = alpha .* (opA * opB) .+ beta .* C_mat[1:m, 1:n]
    return vec(C_ref)
end

# A helper function to synchronize XKRT and invalidate memory cache, so future
# run have a fresh MCC
function sync()
    XK.sync()
    XK.memory_invalidate_caches()
end

# ──────────────────────────────────────────────────────────────────────────────
# XKArray / XKVector / XKMatrix type tests
# (these are pure Julia, no GPU runtime needed)
# ──────────────────────────────────────────────────────────────────────────────

@testset "XK.jl" begin
    @testset "XKArray types" begin
        @testset "XKVector construction and interface" begin
            v = XKVector(Float64[1.0, 2.0, 3.0])
            @test eltype(v) == Float64
            @test length(v) == 3
            @test size(v) == (3,)
            @test v[1] == 1.0
            @test v[3] == 3.0
            v[2] = 42.0
            @test v[2] == 42.0
        end

        @testset "XKMatrix construction and interface" begin
            m = XKMatrix(Float32[1.0 2.0; 3.0 4.0])
            @test eltype(m) == Float32
            @test length(m) == 4
            @test size(m) == (2, 2)
            @test m[1, 1] == 1.0f0
            @test m[2, 1] == 3.0f0
            @test m[1, 2] == 2.0f0
            @test m[2, 2] == 4.0f0
            m[1, 2] = 99.0f0
            @test m[1, 2] == 99.0f0
        end

        @testset "XKArray construction and interface" begin
            a = XKArray(rand(Float64, 2, 3, 4))
            @test eltype(a) == Float64
            @test size(a) == (2, 3, 4)
            @test length(a) == 24
        end

        @testset "XKObject generic constructor" begin
            data = [1.0, 2.0, 3.0]
            obj = XK.XKObject(data)
            @test obj isa XKVector{Float64}
            @test length(obj) == 3

            data2d = rand(Float32, 4, 5)
            obj2 = XK.XKObject(data2d)
            @test obj2 isa XKMatrix{Float32}
            @test size(obj2) == (4, 5)
        end

        @testset "pointer and cconvert" begin
            v = XKVector(Float64[1.0, 2.0, 3.0])
            @test pointer(v) == pointer(v.data)
        end
    end

    # ──────────────────────────────────────────────────────────────────────────────
    # BLAS operations (require XK runtime)
    # ──────────────────────────────────────────────────────────────────────────────

    @testset "BLAS" begin
        @testset "axpy_async" begin

            @testset "axpy_async Float32 — basic" begin
                n = 128
                alpha = Float32(2.0)
                x = Float32[Float32(i) for i in 1:n]
                y = Float32[Float32(10.0) for _ in 1:n]
                y_orig = copy(y)

                expected = alpha .* x .+ y_orig

                XK.BLAS.axpy_async(n, alpha, x, 1, y, 1)
                XK.memory_coherent_async(y)
                sync()

                # Values scale with n, so use rtol for robustness
                @test isapprox(y, expected, rtol=1e-5, atol=1e-5)
            end

            @testset "axpy_async Float64 — basic" begin
                n = 256
                alpha = 0.5
                x = Float64[Float64(i) for i in 1:n]
                y = Float64[100.0 for _ in 1:n]
                y_orig = copy(y)

                expected = alpha .* x .+ y_orig

                XK.BLAS.axpy_async(n, alpha, x, 1, y, 1)
                XK.memory_coherent_async(y)
                sync()

                @test isapprox(y, expected, rtol=1e-12, atol=1e-10)
            end

            @testset "axpy_async Float32 — alpha=0 (y unchanged)" begin
                n = 64
                alpha = Float32(0.0)
                x = Float32.(rand(n))
                y = Float32.(rand(n))
                y_orig = copy(y)

                XK.BLAS.axpy_async(n, alpha, x, 1, y, 1)
                XK.memory_coherent_async(y)
                sync()

                @test isapprox(y, y_orig, rtol=1e-6, atol=1e-6)
            end

            @testset "axpy_async Float64 — alpha=1" begin
                n = 128
                alpha = 1.0
                x = Float64.(rand(n))
                y = Float64.(rand(n))
                y_orig = copy(y)

                expected = x .+ y_orig

                XK.BLAS.axpy_async(n, alpha, x, 1, y, 1)
                XK.memory_coherent_async(y)
                sync()

                @test isapprox(y, expected, rtol=1e-12, atol=1e-12)
            end

            @testset "axpy_async Float32 — negative alpha" begin
                n = 64
                alpha = Float32(-3.0)
                x = Float32.(rand(n))
                y = Float32.(rand(n))
                y_orig = copy(y)

                expected = alpha .* x .+ y_orig

                XK.BLAS.axpy_async(n, alpha, x, 1, y, 1)
                XK.memory_coherent_async(y)
                sync()

                @test isapprox(y, expected, rtol=1e-5, atol=1e-5)
            end

            @testset "axpy_async Float64 — partial vector (n < length)" begin
                n_alloc = 256
                n_compute = 128
                alpha = 2.0
                x = Float64.(rand(n_alloc))
                y = Float64.(rand(n_alloc))
                y_orig = copy(y)

                expected = copy(y_orig)
                expected[1:n_compute] .= alpha .* x[1:n_compute] .+ y_orig[1:n_compute]

                XK.BLAS.axpy_async(n_compute, alpha, x, 1, y, 1)
                XK.memory_coherent_async(y, n_compute)
                sync()

                @test isapprox(y[1:n_compute], expected[1:n_compute], rtol=1e-12, atol=1e-12)
            end

            @testset "axpy_async Float32 — large vector" begin
                n = 4096
                alpha = Float32(0.7)
                x = Float32.(rand(n))
                y = Float32.(rand(n))
                y_orig = copy(y)

                expected = alpha .* x .+ y_orig

                XK.BLAS.axpy_async(n, alpha, x, 1, y, 1)
                XK.memory_coherent_async(y)
                sync()

                # Large Float32 vector: use rtol to scale with magnitude
                @test isapprox(y, expected, rtol=1e-4, atol=1e-5)
            end
        end

        @testset "gemm_async" begin

            @testset "gemm_async Float32 — square identity-like" begin
                n = 64
                m, k = n, n
                T = Float32

                A = T.(rand(m * k))
                B = zeros(T, k * n)
                for i in 1:min(k, n)
                    B[(i - 1) * k + i] = one(T)  # column-major identity
                end
                C = zeros(T, m * n)

                alpha, beta = T(1.0), T(0.0)
                lda, ldb, ldc = m, k, m
                transA = XK.BLAS.NO_TRANS
                transB = XK.BLAS.NO_TRANS

                XK.BLAS.gemm_async(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
                XK.memory_matrix_coherent_async(C, ldc, m, n, sizeof(T))
                sync()

                A_mat = reshape(A, m, k)
                C_mat = reshape(C, m, n)
                @test isapprox(C_mat, A_mat, rtol=1e-5, atol=1e-5)
            end

            @testset "gemm_async Float32 — correctness vs Julia" begin
                m, n, k = 64, 48, 32
                T = Float32

                A = T.(rand(m * k))
                B = T.(rand(k * n))
                C = T.(rand(m * n))
                C_orig = copy(C)

                alpha = T(1.5)
                beta = T(0.5)
                lda, ldb, ldc = m, k, m
                transA = XK.BLAS.NO_TRANS
                transB = XK.BLAS.NO_TRANS

                C_ref = ref_gemm(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C_orig, ldc)

                XK.BLAS.gemm_async(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
                XK.memory_matrix_coherent_async(C, ldc, m, n, sizeof(T))
                sync()

                @test isapprox(C, C_ref, rtol=1e-3, atol=1e-5)
            end

            @testset "gemm_async Float64 — correctness vs Julia" begin
                m, n, k = 48, 64, 32
                T = Float64

                A = T.(rand(m * k))
                B = T.(rand(k * n))
                C = T.(rand(m * n))
                C_orig = copy(C)

                alpha = T(2.0)
                beta = T(-1.0)
                lda, ldb, ldc = m, k, m
                transA = XK.BLAS.NO_TRANS
                transB = XK.BLAS.NO_TRANS

                C_ref = ref_gemm(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C_orig, ldc)

                XK.BLAS.gemm_async(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
                XK.memory_matrix_coherent_async(C, ldc, m, n, sizeof(T))
                sync()

                @test isapprox(C, C_ref, rtol=1e-10, atol=1e-10)
            end

            @testset "gemm_async Float32 — beta=0 (ignore C initial)" begin
                m, n, k = 32, 32, 32
                T = Float32

                A = T.(rand(m * k))
                B = T.(rand(k * n))
                C = T.(fill(999.0, m * n))

                alpha = T(1.0)
                beta = T(0.0)
                lda, ldb, ldc = m, k, m
                transA = XK.BLAS.NO_TRANS
                transB = XK.BLAS.NO_TRANS

                C_ref = ref_gemm(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)

                XK.BLAS.gemm_async(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
                XK.memory_matrix_coherent_async(C, ldc, m, n, sizeof(T))
                sync()

                @test isapprox(C, C_ref, rtol=1e-3, atol=1e-5)
            end

            @testset "gemm_async Float64 — rectangular" begin
                m, n, k = 128, 64, 96
                T = Float64

                A = T.(rand(m * k))
                B = T.(rand(k * n))
                C = zeros(T, m * n)

                alpha = T(1.0)
                beta = T(0.0)
                lda, ldb, ldc = m, k, m
                transA = XK.BLAS.NO_TRANS
                transB = XK.BLAS.NO_TRANS

                C_ref = ref_gemm(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)

                XK.BLAS.gemm_async(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
                XK.memory_matrix_coherent_async(C, ldc, m, n, sizeof(T))
                sync()

                @test isapprox(C, C_ref, rtol=1e-10, atol=1e-10)
            end

            @testset "gemm_async Float32 — large matrix" begin
                m, n, k = 256, 256, 256
                T = Float32

                A = T.(rand(m * k))
                B = T.(rand(k * n))
                C = zeros(T, m * n)

                alpha = T(1.0)
                beta = T(0.0)
                lda, ldb, ldc = m, k, m
                transA = XK.BLAS.NO_TRANS
                transB = XK.BLAS.NO_TRANS

                C_ref = ref_gemm(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)

                XK.BLAS.gemm_async(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
                XK.memory_matrix_coherent_async(C, ldc, m, n, sizeof(T))
                sync()

                @test isapprox(C, C_ref, rtol=1e-3, atol=1e-5)
            end
        end

        @testset "memory_coherent_async" begin

            @testset "vector coherence — Float64 full vector" begin
                n = 256
                T = Float64
                fill_val = T(7.0)

                x = Vector{T}(undef, n)

                XK.BLAS.ext.fill_async(n, x, fill_val)
                XK.memory_coherent_async(x)
                sync()

                @test all(v -> isapprox(v, fill_val, rtol=1e-12, atol=1e-12), x)
            end

            @testset "vector coherence — Float32 full vector" begin
                n = 512
                T = Float32
                fill_val = T(3.14)

                x = Vector{T}(undef, n)

                XK.BLAS.ext.fill_async(n, x, fill_val)
                XK.memory_coherent_async(x)
                sync()

                @test all(v -> isapprox(v, fill_val, rtol=1e-5, atol=1e-6), x)
            end

            @testset "vector coherence — partial (n < length)" begin
                n_alloc = 256
                n_fill = 128
                T = Float64
                fill_val = T(42.0)

                x = zeros(T, n_alloc)

                XK.BLAS.ext.fill_async(n_fill, x, fill_val)
                XK.memory_coherent_async(x, n_fill)
                sync()

                @test all(v -> isapprox(v, fill_val, rtol=1e-12, atol=1e-12), x[1:n_fill])
            end

            @testset "matrix coherence — Float32 via gemm" begin
                m, n, k = 64, 64, 64
                T = Float32

                A = T.(rand(m * k))
                B = T.(rand(k * n))
                C = zeros(T, m * n)

                alpha = T(1.0)
                beta = T(0.0)
                lda, ldb, ldc = m, k, m
                transA = XK.BLAS.NO_TRANS
                transB = XK.BLAS.NO_TRANS

                C_ref = ref_gemm(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)

                XK.BLAS.gemm_async(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)

                XK.memory_coherent_async(C, ldc, m, n)
                sync()

                @test isapprox(C, C_ref, rtol=1e-3, atol=1e-5)
            end

            @testset "matrix coherence — Float64 via gemm" begin
                m, n, k = 48, 48, 48
                T = Float64

                A = T.(rand(m * k))
                B = T.(rand(k * n))
                C = zeros(T, m * n)

                alpha = T(1.0)
                beta = T(0.0)
                lda, ldb, ldc = m, k, m
                transA = XK.BLAS.NO_TRANS
                transB = XK.BLAS.NO_TRANS

                C_ref = ref_gemm(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)

                XK.BLAS.gemm_async(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)

                XK.memory_coherent_async(C, ldc, m, n)
                sync()

                @test isapprox(C, C_ref, rtol=1e-10, atol=1e-10)
            end

            @testset "vector coherence after axpy — verify write-back" begin
                n = 128
                T = Float64
                alpha = T(1.0)

                x = Vector{T}(undef, n)
                y = Vector{T}(undef, n)

                XK.BLAS.ext.fill_async(n, x, T(2.0))
                XK.BLAS.ext.fill_async(n, y, T(3.0))

                # axpy: y = 1.0 * x + y = 2.0 + 3.0 = 5.0
                XK.BLAS.axpy_async(n, alpha, x, 1, y, 1)

                XK.memory_coherent_async(y)
                sync()

                @test all(v -> isapprox(v, T(5.0), rtol=1e-12, atol=1e-12), y)
            end

            @testset "multiple coherence calls in sequence" begin
                n = 128
                T = Float64

                x = Vector{T}(undef, n)
                y = Vector{T}(undef, n)

                XK.BLAS.ext.fill_async(n, x, T(10.0))
                XK.BLAS.ext.fill_async(n, y, T(20.0))

                XK.memory_coherent_async(x)
                XK.memory_coherent_async(y)
                sync()

                @test all(v -> isapprox(v, T(10.0), rtol=1e-12, atol=1e-12), x)
                @test all(v -> isapprox(v, T(20.0), rtol=1e-12, atol=1e-12), y)
            end
        end

        @testset "fill_async" begin
            @testset "fill_async Float32" begin
                n = 256
                T = Float32
                val = T(3.14)
                x = Vector{T}(undef, n)

                XK.BLAS.ext.fill_async(n, x, val)
                XK.memory_coherent_async(x)
                sync()

                @test all(v -> isapprox(v, val, rtol=1e-5, atol=1e-6), x)
            end

            @testset "fill_async Float64" begin
                n = 512
                T = Float64
                val = T(2.718281828)
                x = Vector{T}(undef, n)

                XK.BLAS.ext.fill_async(n, x, val)
                XK.memory_coherent_async(x)
                sync()

                @test all(v -> isapprox(v, val, rtol=1e-12, atol=1e-12), x)
            end
        end

        @testset "copy_async" begin
            @testset "copy_async Float64" begin
                n = 128
                T = Float64
                x = T.(collect(1:n) .* 1.5)
                y = zeros(T, n)

                XK.BLAS.ext.copy_async(n, x, 1, y, 1)
                XK.memory_coherent_async(y)
                sync()

                @test isapprox(y, x, rtol=1e-12, atol=1e-12)
            end

            @testset "copy_async Float32" begin
                n = 256
                T = Float32
                x = T.(rand(n))
                y = zeros(T, n)

                XK.BLAS.ext.copy_async(n, x, 1, y, 1)
                XK.memory_coherent_async(y)
                sync()

                @test isapprox(y, x, rtol=1e-6, atol=1e-6)
            end
        end

        @testset "scal_async" begin
            @testset "scal_async Float64" begin
                n = 128
                T = Float64
                alpha = T(3.0)
                x = T.(collect(1:n))
                x_orig = copy(x)

                XK.BLAS.ext.scal_async(n, alpha, x, 1)
                XK.memory_coherent_async(x)
                sync()

                @test isapprox(x, alpha .* x_orig, rtol=1e-12, atol=1e-10)
            end

            @testset "scal_async Float32" begin
                n = 256
                T = Float32
                alpha = T(0.5)
                x = T.(rand(n))
                x_orig = copy(x)

                XK.BLAS.ext.scal_async(n, alpha, x, 1)
                XK.memory_coherent_async(x)
                sync()

                @test isapprox(x, alpha .* x_orig, rtol=1e-5, atol=1e-6)
            end
        end

        @testset "dot_async" begin
            @testset "dot_async Float64" begin
                n = 128
                T = Float64
                x = ones(T, n)
                y = T.(collect(1:n))
                result = Ref(T(0.0))

                XK.BLAS.dot_async(n, x, 1, y, 1, result)
                sync()

                expected = T(n * (n + 1) / 2)
                @test isapprox(result[], expected, rtol=1e-10, atol=1e-8)
            end

            @testset "dot_async Float32" begin
                n = 64
                T = Float32
                x = ones(T, n)
                y = ones(T, n)
                result = Ref(T(0.0))

                XK.BLAS.dot_async(n, x, 1, y, 1, result)
                sync()

                # dot(ones, ones) = n, use rtol to scale with n
                @test isapprox(result[], T(n), rtol=1e-4, atol=1e-4)
            end
        end

        @testset "gemv_async" begin
            @testset "gemv_async Float64 — matrix-vector multiply" begin
                m, n = 64, 48
                T = Float64

                A = T.(rand(m * n))
                x = T.(rand(n))
                y = zeros(T, m)

                alpha = T(1.0)
                beta = T(0.0)
                lda = m
                transA = XK.BLAS.NO_TRANS

                A_mat = reshape(copy(A), m, n)
                y_ref = alpha .* (A_mat * x)

                XK.BLAS.gemv_async(transA, m, n, alpha, A, lda, x, 1, beta, y, 1)
                XK.memory_coherent_async(y)
                sync()

                @test isapprox(y, y_ref, rtol=1e-10, atol=1e-10)
            end

            @testset "gemv_async Float32 — with beta" begin
                m, n = 32, 32
                T = Float32

                A = T.(rand(m * n))
                x = T.(rand(n))
                y = T.(rand(m))
                y_orig = copy(y)

                alpha = T(2.0)
                beta = T(0.5)
                lda = m
                transA = XK.BLAS.NO_TRANS

                A_mat = reshape(copy(A), m, n)
                y_ref = alpha .* (A_mat * x) .+ beta .* y_orig

                XK.BLAS.gemv_async(transA, m, n, alpha, A, lda, x, 1, beta, y, 1)
                XK.memory_coherent_async(y)
                sync()

                @test isapprox(y, y_ref, rtol=1e-3, atol=1e-5)
            end
        end

        @testset "chained async operations" begin
            @testset "fill -> scal -> axpy -> coherent pipeline" begin
                n = 256
                T = Float64

                x = Vector{T}(undef, n)
                y = Vector{T}(undef, n)

                XK.BLAS.ext.fill_async(n, x, T(1.0))
                XK.BLAS.ext.fill_async(n, y, T(2.0))

                # scal: x = 3 * x = 3
                XK.BLAS.ext.scal_async(n, T(3.0), x, 1)

                # axpy: y = 2 * x + y = 6 + 2 = 8
                XK.BLAS.axpy_async(n, T(2.0), x, 1, y, 1)

                XK.memory_coherent_async(y)
                sync()

                @test all(v -> isapprox(v, T(8.0), rtol=1e-12, atol=1e-12), y)
            end

            @testset "multiple gemm_async in sequence" begin
                n = 64
                T = Float32

                A = T.(rand(n * n))
                B = T.(rand(n * n))
                C = zeros(T, n * n)

                alpha = T(1.0)
                beta = T(0.0)
                lda, ldb, ldc = n, n, n
                transA = XK.BLAS.NO_TRANS
                transB = XK.BLAS.NO_TRANS

                # First gemm: C = A * B
                XK.BLAS.gemm_async(transA, transB, n, n, n, alpha, A, lda, B, ldb, beta, C, ldc)

                # Second gemm: C = A * B + C  (beta=1 accumulates)
                XK.BLAS.gemm_async(transA, transB, n, n, n, alpha, A, lda, B, ldb, T(1.0), C, ldc)

                XK.memory_matrix_coherent_async(C, ldc, n, n, sizeof(T))
                sync()

                A_mat = reshape(A, n, n)
                B_mat = reshape(B, n, n)
                C_expected = vec(T(2.0) .* (A_mat * B_mat))

                @test isapprox(C, C_expected, rtol=1e-3, atol=1e-5)
            end
        end
    end # BLAS
end # XK
