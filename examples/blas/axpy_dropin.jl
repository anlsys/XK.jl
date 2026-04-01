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

## Replicate and fill memory on the device.
## This flavor makes both host and device memory coherent after returning
XK.BLAS.ext.fill(n, x, 1.0)
XK.BLAS.ext.fill(n, y, 0.5)

## Run the axpy
XK.BLAS.axpy(n, alpha, x, 1, y, 1)

## Print result
println(y)
