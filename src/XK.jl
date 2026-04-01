module XK

    using Libdl
    using Scratch

    const libxkblas = joinpath(get_scratch!(Base.UUID("8d3f9e88-0651-4e8b-8f79-7d9d5f5f9e88"), "xkblas"), "lib/libxkblas.so")
    const size_t = Csize_t
    include("bindings.jl")

    include("xkarrays.jl")
    include("wrappers.jl")
    include("logger.jl")
    include("threading.jl")

    include("BLAS/BLAS.jl")
    include("KA/KA.jl")

    function __init__()
        if get(ENV, "XK_DOCS_BUILD", "false") == "true"
            return
        end

        if !isfile(libxkblas)
            error("libxkblas not found at $libxkblas. Run `deps/build.jl` first.")
        end

        XK.init()
        XK.KA.init()
        XK.Threading.init()

        function cleanup()
            XK.KA.deinit()
            XK.Threading.deinit()
            XK.deinit()
        end
        atexit(cleanup)
    end

end
