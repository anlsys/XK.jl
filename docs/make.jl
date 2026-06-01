using Documenter
using Literate
using XK

# list of modules
const MODULES = [XK, XK.BLAS, XK.KA, XK.Logger, XK.Threading]

# Generate literate markdown strings for examples
const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR   = joinpath(@__DIR__, "src", "examples")
const example_files = [
    "blas/axpy_async.jl",
    "blas/axpy_sync.jl",
    "blas/axpy_dropin.jl",
    "blas/gemm_trmm_async.jl",
]
for f in example_files
    Literate.markdown(
        joinpath(EXAMPLES_DIR, f),
        OUTPUT_DIR;
        documenter = false,
        execute    = false
    )
end

# Add a stub docstring to symbols without any
for mod in MODULES
    for sym in names(mod; all = true, imported = false)
        isdefined(mod, sym)           || continue
        startswith(string(sym), '#')  && continue
        b = Base.Docs.Binding(mod, sym)
        if !haskey(Base.Docs.meta(mod), b)
            Base.eval(mod, :(@doc "No documentation provided." $sym))
        end
    end
end

# Make the docs
makedocs(
    sitename = "XK.jl",
    modules  = MODULES,
    format = Documenter.HTML(
        prettyurls      = get(ENV, "CI", nothing) == "true",
        size_threshold  = nothing,
    ),
    pages    = [
        "Home" => "index.md",
        "API" => [
            "XK"           => "api/xk.md",
            "XK.BLAS"      => "api/blas.md",
            "XK.KA"        => "api/ka.md",
            "XK.Logger"    => "api/logger.md",
            "XK.Threading" => "api/threading.md",
        ],
        "Examples" => [
            "BLAS" => [
                "AXPY" => [
                    "Asynchronous flavor"   => "examples/axpy_async.md",
                    "Synchronous flavor"    => "examples/axpy_sync.md",
                    "Dropin flavor"         => "examples/axpy_dropin.md",
               ],
               "GEMM + TRMM (async)"       => "examples/gemm_trmm_async.md",
            ],
            "Krylov.jl integration"         => "krylov.md",
        ],
    ],
    warnonly = [:missing_docs, :cross_references, :example_block]
)

deploydocs(
    repo      = "github.com/anlsys/XK.jl.git",
    devbranch = "paper",
)
