# Examples

## Asynchronous flavor
@eval Markdown.parse("""
```julia
$(read("examples/blas/axpy_async.jl", String))
```
""")

## Synchronous flavor
@eval Markdown.parse("""
```julia
$(read("examples/blas/axpy_sync.jl", String))
```
""")

## Drop-in flavor
@eval Markdown.parse("""
```julia
$(read("examples/blas/axpy_dropin.jl", String))
```
""")
