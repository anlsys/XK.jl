## This program example distribute an axpy over all available GPUs

using XK
const T = Float64

## Create empty vectors
n = 1024
alpha = T(0.2)

## Retrieve the number of GPUs
ngpus = XK.get_ngpus()

## Set tiling parameter, so that there is one tile per GPU
tile_size = div(n, ngpus)
XK.set_tile_parameter(tile_size)

## Initialize empty host memory
x = Vector{T}(undef, n)
y = Vector{T}(undef, n)

## Spawn tasks to replicate and fill device vectors
XK.BLAS.ext.fill_async(n, x, 1.0)
XK.BLAS.ext.fill_async(n, y, 0.5)

## Spawn tasks to perform the axpy
XK.BLAS.axpy_async(n, alpha, x, 1, y, 1)

## Spawn tasks to write back memory from devices to host
XK.memory_coherent_async(y)

## Wait for tasks completion
## After this point,
##   - 'x' is still 'undef' on the host
##   - 'y' contains values computed by GPUs
XK.sync()

## Print result
println(y)
