####################
# HOST ASYNC TASKS #
####################

# global table to keep Refs alive
const _host_async_refs = IdDict{Ptr{Cvoid}, Any}()

function _async_trampoline(fptr::Ptr{Cvoid})
    args = unsafe_pointer_to_objref(fptr)
    GC.enable(false)    # disable GC to avoid deadlocks if 'args[]()' ends-up calling the Julia runtime
    args[]()
    GC.enable(true)
    delete!(_host_async_refs, fptr)
    return
end

#
#   Generic API to spawn an XKRT task
#   The 'may_run_julia_runtime' indicates whether the task may enter the Julia runtime.
#   In such case,
#       - it is made detachable
#       - it will be executed by a Julia thread.
#   see https://docs.julialang.org/en/v1/manual/calling-c-and-fortran-code/#Thread-safety
#
function device_async(
    device_unique_id::xkrt_device_unique_id_t,
    fmt_or_func::Union{xkrt_task_format_id_t, Function};
    set_accesses::Union{Function,Nothing}=nothing,
    args=C_NULL,
    args_size=0,
    detach_ctr_initial=nothing
)
    accesses = xkrt_access_t[]
    if set_accesses !== nothing
        set_accesses(accesses)
    end
    local AC = length(accesses)
    local ocr_access = UNSPECIFIED_TASK_ACCESS

    local flags = TASK_FLAG_ZERO
    detach_ctr_initial != nothing  && (flags |= TASK_FLAG_DETACHABLE)
    (AC > 0)                       && (flags |= TASK_FLAG_DEPENDENT)
    (true)                         && (flags |= TASK_FLAG_DEVICE)

    runtime = XK.xkrt_runtime_get()

    if fmt_or_func isa xkrt_task_format_id_t
        XK.xkrt_task_spawn_generic(
            runtime,
            device_unique_id,
            flags,
            fmt_or_func::xkrt_task_format_id_t,
            args, args_size,
            pointer(accesses), Cint(AC),
            ocr_access,
            detach_ctr_initial
        )
    elseif fmt_or_func isa Function
        fptr = @cfunction(_async_trampoline, Cvoid, (Ptr{Cvoid},))
        args = Ref(fmt_or_func::Function)
        _host_async_refs[fptr] = args  # preserve Ref until trampoline executed
        XK.Logger.fatal("TODO: 2 args here, the function body and user parameter")
        # XK.async_generic_with_format(
        #     device_unique_id,
        #     fmt_or_func::xkrt_task_format_id_t,
        #     flags,
        #     args, args_size,
        #     pointer(accesses), Cint(AC),
        #     detach_ctr_initial
        # )
    end
end

function host_async(body::Function; set_accesses::Union{Function,Nothing}=nothing)
    return device_async(HOST_DEVICE_GLOBAL_ID, body, set_accesses=set_accesses)
end

function host_async(set_accesses::Function, body::Function)
    return device_async(HOST_DEVICE_GLOBAL_ID, body, set_accesses=set_accesses)
end

@enum ArgumentPassMode begin
    Value
end

# Helper constructor for xkrt_access_t
function Access(
    mode::xkrt_access_mode_t,
    region::Union{xkrt_handle_t, xkrt_segment_t, xkrt_matrix_t};
    scope::Union{xkrt_access_scope_t, Nothing}=nothing,
    concurrency::Union{xkrt_access_concurrency_t, Nothing}=nothing
)
    access     = xkrt_access_t(ntuple(_ -> 0x00, 80))
    access_ref = Ref(access)
    access_ptr = Base.unsafe_convert(Ptr{xkrt_access_t}, access_ref)

    # set mode
    @assert mode != nothing
    access_ptr.mode = mode

    # set type
    if region isa xkrt_handle_t
        access_ptr.type = ACCESS_TYPE_HANDLE
        access_ptr.region.handle = region
    elseif region isa xkrt_segment_t
        access_ptr.type = ACCESS_TYPE_SEGMENT
        access_ptr.region.segment = region
    elseif region isa xkrt_matrix_t
        access_ptr.type = ACCESS_TYPE_MATRIX
        access_ptr.region.matrix = region
    else
        error("Unknown region type: $(typeof(region))")
    end

    # set scope
    if scope == nothing
        scope = ACCESS_SCOPE_NONUNIFIED
    end
    access_ptr.scope = scope

    # set concurrency
    if concurrency == nothing
        concurrency = ACCESS_CONCURRENCY_SEQUENTIAL
    end
    access_ptr.concurrency = concurrency

    return access_ref[]
end

# # Access Types and regions
# const Handle  = xkrt_handle_t;
# const Segment = xkrt_segment_t;
# const Matrix  = xkrt_matrix_t;

# Wrappers
function Access(
    mode::xkrt_access_mode_t,
    vec::AbstractVector;
    scope::Union{xkrt_access_scope_t, Nothing}=nothing,
    concurrency::Union{xkrt_access_concurrency_t, Nothing}=nothing
)
    return Access(mode, xkrt_segment_t(pointer(vec), pointer(vec) + length(vec) * Base.elsize(vec)))
end

# Set tile parameters for all kernels
function set_tile_parameter(ts)
    XK.set_param(ts, 0)
end

########################
# Dispatcher for types #
########################

# Memory routines

memory_coherent_async(A, lda, m, n) = memory_matrix_coherent_async(A, lda, m, n, sizeof(eltype(A)))
memory_coherent_sync(A, lda, m, n)  = memory_matrix_coherent_async(A, lda, m, n, sizeof(eltype(A)))
memory_coherent_sync(x)             = memory_segment_coherent_async(x, length(x)*sizeof(eltype(x)))
memory_coherent_async(x)            = memory_segment_coherent_async(x, length(x)*sizeof(eltype(x)))
memory_coherent_async(x, n)         = memory_segment_coherent_async(x, n*sizeof(eltype(x)))
memory_coherent_sync(x, n)          = memory_segment_coherent_async(x, n*sizeof(eltype(x)))

# Export symbols
for name in names(@__MODULE__; all=true)
    @eval export $name
end
