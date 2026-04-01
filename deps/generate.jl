using Clang.Generators
using Clang_jll
using Scratch

XK_pkg = Base.UUID("8d3f9e88-0651-4e8b-8f79-7d9d5f5f9e88")
xkblas_dir = get_scratch!(XK_pkg, "xkblas")
xkblas_include_dir = joinpath(xkblas_dir, "include")

xkrt_dir = get_scratch!(XK_pkg, "xkrt")
xkrt_include_dir = joinpath(xkrt_dir, "include")

println("Using XK headers in $xkblas_include_dir")

# Load generator options (must include type_map and rename_functions)
options = load_options(joinpath(@__DIR__, "generator.toml"))
output_file_path = abspath(@__DIR__, "..", "src", "bindings.jl")
options["general"]["output_file_path"] = output_file_path

# Collect all headers
headers = [
    joinpath(xkblas_include_dir, "xkblas/xkblas.h"),
    joinpath(xkblas_include_dir, "xkblas/routines.h"),
    # joinpath(xkblas_include_dir, "xkblas/flops.h"), # TODO: almost work in "aggressive" mode, only "_Complex float" are missing
]
@show headers

# Default compiler flags
args = get_default_args()
push!(args, "-I$xkrt_include_dir")
push!(args, "-I$xkblas_include_dir")

# Create context: only two positional arguments are supported in this version
ctx = create_context(headers, args, options)

# -------------------------------------------------------------------------
# Skip macros that Clang.jl cannot generate
# -------------------------------------------------------------------------
# ctx.options[:macro_filter] = name -> !(startswith(name, "FMULS_") || startswith(name, "FADDS_"))
# -------------------------------------------------------------------------

# Generate bindings
build!(ctx)

#----------------------------
# Post-process
#----------------------------

println("bindings.jl generated successfully in src/ -- Post processing...")

# 1. fix C-style wrapping casts on unsigned typedefs
src = read(output_file_path, String)
src = replace(src, r"(\w+)\((-\d+)\)" => s"\2 % \1")    # Rewrite SomeType(-N) with -N % SomeType
write(output_file_path, src)
println("Fixed negative number casts to unsigned types")

# 2. Remove xkblas_ prefix from function names (but keep @ccall names unchanged)
content = read(output_file_path, String)
content = replace(content, r"^function xkblas_(\w+)\("m => s"function \1(")
write(output_file_path, content)
println("Removed xkblas_ prefix from function names")

