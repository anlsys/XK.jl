using Documenter
using Literate
using XK

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR   = joinpath(@__DIR__, "src", "examples")
const example_files = [
    "blas/axpy_async.jl",
    "blas/axpy_sync.jl",
    "blas/axpy_dropin.jl",
]

for f in example_files
    Literate.markdown(
        joinpath(EXAMPLES_DIR, f),
        OUTPUT_DIR;
        documenter = false,
        execute    = false
    )
end

makedocs(
    sitename = "XK.jl",
    modules  = [XK],
    format   = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    pages    = [
        "Home"     => "index.md",
        "API"      => "api.md",
        "Examples" => [
            "BLAS" => [
                "AXPY" => [
                    "Asynchronous flavor"   => "examples/axpy_async.md",
                    "Synchronous flavor"    => "examples/axpy_sync.md",
                    "Dropin flavor"         => "examples/axpy_dropin.md"
               ],
            ],
        ],
    ],
    warnonly = [:missing_docs, :cross_references, :example_block]
)

deploydocs(
    repo      = "github.com/anlsys/XK.jl.git",
    devbranch = "docs",
)
