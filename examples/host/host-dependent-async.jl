using XK

# spawn two tasks
x = 0

# first one sets 'x' to 42
XK.host_async(
    (accesses) -> begin
        push!(accesses, Access(ACCESS_MODE_VW, xkrt_segment_t(0, 64)))
    end,
    () -> begin
        global x = 42
    end
)

# second one sets 'x' to 43 - with an intersecting access to set a dependency
XK.host_async(
    set_accesses = (accesses) -> begin
        push!(accesses, Access(ACCESS_MODE_VR, xkrt_segment_t(16, 48)))
    end,
    () -> begin
        @assert x == 42
        global x = 43
    end
)

# wait for task execution
XK.sync()
println("x is $x")
@assert x == 43
