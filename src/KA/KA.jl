#
#   Kernel abstractions built-in XK
#   Main differences with KernelAbstractions.jl and others, is:
#       - passed parameters are raw pointers, not Julia types
#       - there no bounds check: @inbounds is ignored
#
#   Currently only supports CUDA.
#   TODO: support other and add macros for block dim and shared memory
#

module KA

    using CUDA

    import ..XK

    # Julia CU context is 16 bytes, ignore them
    const JL_CU_CONTEXT_SIZE = 16

    # GPU block size magic number
    const BLOCK_SIZE = 256

    # Define getindex and setindex for Ptr{T}
    @inline Base.getindex( p::Ptr{T},      i::Integer) where T = unsafe_load(p, i)
    @inline Base.setindex!(p::Ptr{T}, val, i::Integer) where T = unsafe_store!(p, val, i)

    #############################################################
    # A disk cache to avoid recompiling kernels on every launch #
    #############################################################

    module Cache

        import ...XK

        using Serialization, SHA

        # Cache directory setup
        const BYTECODE_CACHE_DIR = joinpath(homedir(), ".julia_xkrt_bytecode_cache")

        """
            compute_cache_key(kernel::Function, kernel_args_type::Type)

        Compute a unique hash for the kernel function and its argument types.
        This hash is used as the cache key.
        """
        function compute_cache_key(kernel::Function, kernel_args_type::Type)
            # Get the method signature
            methods_list = methods(kernel, kernel_args_type.parameters)

            if isempty(methods_list)
                error("No methods found for kernel function with type signature $kernel_args_type")
            end

            method = first(methods_list)

            # Get code info - but hash it directly instead of converting to string
            code_info = code_lowered(kernel, kernel_args_type.parameters)

            # Hash based on:
            # 1. Function name
            # 2. Method signature
            # 3. Argument types
            # 4. Code info (detects all code changes)
            h = hash(nameof(kernel))
            h = hash(method.sig, h)
            h = hash(kernel_args_type, h)
            for ci in code_info
                code_str = sprint(show, ci.code)    # that is slow, but makes it constant across invocation for the same code
                h = hash(code_str, h)
            end

            # Convert to hex string for filename compatibility
            return string(h, base=16, pad=16)
        end

        """
            get_cache_path(cache_key::String)

        Get the file path for a given cache key.
        """
        function get_cache_path(cache_key::String)
            mkpath(BYTECODE_CACHE_DIR)
            return joinpath(BYTECODE_CACHE_DIR, "$(cache_key).jls")
        end

        """
            load_cached_bytecode(cache_key::String)

        Load cached BYTECODE data from disk if it exists.
        Returns `nothing` if cache miss.
        """
        function load_cached_bytecode(cache_key::String)
            path = get_cache_path(cache_key)
            if isfile(path)
                try
                    XK.Logger.debug("BYTECODE cache hit! Loading from disk... $(path)")
                    r = deserialize(path)
                    XK.Logger.debug("Loaded from disk")
                    return r
                catch e
                    # If deserialization fails, remove the corrupted cache file
                    rm(path; force=true)
                end
            end
            XK.Logger.debug("BYTECODE cache miss.")
            return nothing
        end

        """
            save_cached_bytecode(cache_key::String, bytecode::String, bytecode_size::Int, bytecode_fn_name::String)

        Save BYTECODE compilation results to disk cache.
        """
        function save_cached_bytecode(cache_key::String, bytecode::String, bytecode_size::Int, bytecode_fn_name::String)
            path = get_cache_path(cache_key)
            try
                data = (bytecode=bytecode, bytecode_size=bytecode_size, bytecode_fn_name=bytecode_fn_name)
                serialize(path, data)
                XK.Logger.debug("Saved BYTECODE to cache: $path")
            catch e
                @warn "Failed to save BYTECODE cache to $path: $e"
            end
        end

        """
            clear_bytecode_cache!()

        Clear all cached BYTECODE files from disk.
        """
        function clear_bytecode_cache!()
            if isdir(BYTECODE_CACHE_DIR)
                for file in readdir(BYTECODE_CACHE_DIR)
                    filepath = joinpath(BYTECODE_CACHE_DIR, file)
                    rm(filepath; force=true)
                end
                @info "Cleared BYTECODE cache directory: $BYTECODE_CACHE_DIR"
            end
        end

    end

    # Called on XK init once
    function init()
    end

    # Called on XK deinit
    function deinit()
    end

    # Mapping types to kernel bytecode
    mutable struct VersionStruct

        # 1. arguments input types
        arguments_input_type::Union{Type,Nothing}

        # 2. arguments return types
        arguments_return_type::Union{Type,Nothing}

        # 3. function BYTECODE
        bytecode::Union{String,Nothing}

        # 4. function BYTECODE size
        bytecode_size::Int

        # 5. function BYTECODE name
        bytecode_fn_name::Union{String,Nothing}

        # 6. Cached module
        moodule::XK.xkrt_driver_module_t

        # 7. Cached function
        fn::XK.xkrt_driver_module_fn_t

        # 8. Kernel cache key
        cache_key::Union{String,Nothing}

    end

    # offset in the format struct
    const VERSION_STRUCT_MOODULE_OFFSET = fieldoffset(XK.KA.VersionStruct, 6)
    const VERSION_STRUCT_FN_OFFSET      = fieldoffset(XK.KA.VersionStruct, 7)

    # Mutable so it is passed by reference
    mutable struct FormatStruct

        # 1. XKRT task format id
        fmtid::XK.xkrt_task_format_id_t

        # 2. The KA function annotated with @kernel
        kernel::Function

        # 3. functions telling how arguments of the kernels are accessed
        arguments::Function

        # 4. function to get the grid launch dimensions
        threads::Union{Function, Nothing}

        # 5. function to get the amount of shared memory to use
        shared_memory::Union{Function, Nothing}

        # 6. List of versions
        versions::Dict{Type,VersionStruct}  # Dict are mutable, so should be passed by ref
    end

    # kernel launcher routine - run by a julia task
    function task_ka_launcher(
        runtime::Ptr{XK.xkrt_runtime_t},
        device::Ptr{XK.xkrt_device_t},
        task::Ptr{XK.xkrt_task_t},
        queue::Ptr{XK.xkrt_command_queue_t},
        command::Ptr{XK.command_t},
        index::XK.xkrt_command_queue_list_counter_t
    )
        # TODO: doing a lot of illegal stuff here (we are executing within a foreign thread)
        # But it seems to work, by avoiding any Julia runtime calls, but only XKRT calls

        # retrieve task arguments
        task_args::Ptr{Int8} = XK.xkrt_task_args(task)

        # 1. retrieve task format and version
        # 2. Get the raw C pointer (void*) pointing to the location of fmt_ptr
        fmt_ptr_ptr::Ptr{Ptr{FormatStruct}} = Ptr{Ptr{FormatStruct}}(task_args)
        fmt_ptr::Ptr{FormatStruct} = unsafe_load(fmt_ptr_ptr)
        fmt::FormatStruct = unsafe_load(fmt_ptr)

        version_ptr_ptr::Ptr{Ptr{VersionStruct}} = Ptr{Ptr{VersionStruct}}(task_args + sizeof(Ptr{Cvoid}))
        version_ptr::Ptr{VersionStruct} = unsafe_load(version_ptr_ptr)
        version::VersionStruct = unsafe_load(version_ptr)

        # TODO: need to protect that with a mutex in case multiple tasks uses the same format
        # cached cuda function
        if version.moodule == C_NULL

            @assert version.fn == C_NULL

            # compile the bytecode
            driver           = XK.xkrt_device_driver_get(runtime, device)
            device_driver_id = XK.xkrt_device_driver_id_get(runtime, device)
            bin              = version.bytecode
            binsize          = version.bytecode_size
            format           = XK.XKRT_DRIVER_MODULE_FORMAT_NATIVE
            moodule          = XK.xkrt_driver_module_load(driver, device_driver_id, bin, binsize, format)

            # get executable function
            fn = XK.xkrt_driver_module_get_fn(driver, moodule, version.bytecode_fn_name)

            # save module and fn
            unsafe_store!(Ptr{XK.xkrt_driver_module_t   }(version_ptr + VERSION_STRUCT_MOODULE_OFFSET), moodule)
            unsafe_store!(Ptr{XK.xkrt_driver_module_fn_t}(version_ptr + VERSION_STRUCT_FN_OFFSET),      fn)

            version=unsafe_load(version_ptr)
            @assert version.moodule != C_NULL
            @assert version.fn != C_NULL

        end

        ###############
        # Grid launch #
        ###############

        threads::Ptr{Int8} = task_args + 2*sizeof(Ptr{Cvoid})
        tx::Int = unsafe_load(Ptr{Int}(threads + 0 * sizeof(Int)))
        ty::Int = unsafe_load(Ptr{Int}(threads + 1 * sizeof(Int)))
        tz::Int = unsafe_load(Ptr{Int}(threads + 2 * sizeof(Int)))
        bx = BLOCK_SIZE
        by = 1
        bz = 1
        gx = ceil(Int, tx / bx)
        gy = ceil(Int, ty / by)
        gz = ceil(Int, tz / bz)

        XK.Logger.debug("threads, blocks, grid: $(tx) $(ty) $(tz) $(bx) $(by) $(bz) $(gx) $(gy) $(gz)")

        #################
        # Shared memory #
        #################

        shared_memory_size_ptr::Ptr{Int8} = task_args + 2*sizeof(Ptr{Cvoid}) + 3 * sizeof(Int)
        shared_memory_size::Int = unsafe_load(Ptr{Int}(shared_memory_size_ptr))

        ###################
        # Build arguments #
        ###################

        # retrieve args buffer, that is right after the format pointer in task arguments
        kernel_args::Ptr{Int8} = task_args + 2*sizeof(Ptr{Cvoid}) + 3*sizeof(Int) + 1*sizeof(Int)

        # parse each accesses, and write replica address
        access_id = 0
        offset = JL_CU_CONTEXT_SIZE

        for i in 1:length(version.arguments_return_type.parameters)
            T = version.arguments_return_type.parameters[i]
            if T <: XK.xkrt_access_t
                device_ptr::Ptr{Cvoid} = XK.xkrt_task_access_replica(task, access_id)
                kernel_arg::Ptr{Int8}  = kernel_args + offset
                unsafe_store!(Ptr{Ptr{Cvoid}}(kernel_arg), device_ptr)
                access_id += 1
                offset += sizeof(Ptr{Cvoid})
            else
                # nothing to do, the producer thread already copied by value
                offset += sizeof(T)
            end
        end

        kernel_args_size = offset

        #####################
        # launch the kernel #
        #####################

        XK.xkrt_device_kernel_launch(
            runtime, device,
            queue, index,
            version.fn,
            gx, gy, gz,
            bx, by, bz,
            shared_memory_size,
            Ptr{Cvoid}(kernel_args), kernel_args_size
        )

        return nothing
    end

    # task routine
    function task_ka_main(
       runtime::Ptr{XK.xkrt_runtime_t},
       device::Ptr{XK.xkrt_device_t},
       task::Ptr{XK.xkrt_task_t}
    )
        # TODO: this is illegal but seems to be working

        # retrieve function pointer
        fptr_task_ka_launcher = @cfunction(
            task_ka_launcher,
            Cvoid,                                  # The C function returns void
            (Ptr{XK.xkrt_runtime_t},                # void * runtime
             Ptr{XK.xkrt_device_t},                 # void * device
             Ptr{XK.xkrt_task_t},                   # void * task
             Ptr{XK.xkrt_command_queue_t},          # void * queue
             Ptr{XK.command_t},                     # void * cmd
             XK.xkrt_command_queue_list_counter_t   # xkrt_command_queue_list_counter_t idx (Non-pointer type)
            )
        )

        # kernel launch is synchronous, XKRT should not expect pending events
        synchronous = Cint(0)

        # launch a synchronous kernel command
        XK.xkrt_task_kernel_launch(runtime, device, task, synchronous, fptr_task_ka_launcher)

        return nothing
    end

    ####################
    # Helper functions #
    ####################

    function get_arguments_tuple_size(f::Function)
        return length(methods(f).ms[1].sig.parameters) - 1
    end

    function get_returned_tuple_size(f::Function)
        param_methods = methods(f)
        if isempty(param_methods)
            XK.Logger.fatal("Function has no methods defined.")
        end
        method = param_methods.ms[1]
        arg_count = method.nargs - 1
        input_types = ntuple(_ -> Any, arg_count)
        return_types = Base.return_types(f, input_types)
        if isempty(return_types) || return_types[1] === Any
            XK.Logger.fatal("Could not infer a concrete return type for the function.")
        end
        return_type = return_types[1]
        if return_type <: Tuple
            return length(return_type.parameters)
        else
            return return_type === Nothing ? 0 : 1
        end
    end

    # Create an XK.KA.FormatStruct
    function Format(
        kernel::Function,
        arguments::Function;
        threads::Function,
        shared_memory::Function
    )
        # task main entry point
        task_main = @cfunction(task_ka_main, Cvoid, (Ptr{XK.xkrt_runtime_t}, Ptr{XK.xkrt_device_t}, Ptr{XK.xkrt_task_t}))

        ###################
        # check arguments #
        ###################

        nargs = get_arguments_tuple_size(arguments)
        if nargs != get_returned_tuple_size(arguments)
            XK.Logger.fatal("Format `arguments` is malformed")
        end
        if nargs != get_arguments_tuple_size(threads) || get_returned_tuple_size(threads) != 3
            XK.Logger.fatal("Format `threads` is malformed")
        end
        if nargs != get_arguments_tuple_size(shared_memory) || get_returned_tuple_size(shared_memory) != 1
            XK.Logger.fatal("Format `shared_memory` is malformed")
        end

        ##############################################
        # Extract constant information against types #
        ##############################################

        name = nameof(kernel)
        fmtid = XK.task_format_put("KA.$name")
        for target in instances(XK.xkrt_task_format_target_t)
            XK.task_format_set(fmtid, target, task_main)
        end

        # return the format
        return  FormatStruct(
                    fmtid,                              # XKRT task format id
                    kernel,                             # The Julia function
                    arguments,                          # Parameters
                    threads,                            # threads
                    shared_memory,                      # shared memory
                    Dict{Type, VersionStruct}()   # versions
               )
    end

    # Spawn a task to the given device, with the given XK.KA format and kernel arguments
    function device_async(
        device_unique_id::XK.xkrt_device_unique_id_t,
        fmt::FormatStruct,
        kernel_args...
    )
        # TODO: check that argument matches the format arguments more precisely
        if get_arguments_tuple_size(fmt.arguments) !== length(kernel_args)
            XK.Logger.fatal("Arguments do not match the task format accesses")
        end

        #########################################
        # Compile to bytecode to target devices #
        #########################################

        # Set arguments type tuple
        kernel_args_type = Tuple{map(arg -> (typeof(arg) <: AbstractVector ? Ptr{eltype(typeof(arg))} : typeof(arg)), kernel_args)...}
        XK.Logger.debug("$(kernel_args_type)")

        # retrieve version for these types
        version = get!(fmt.versions, kernel_args_type) do
            VersionStruct(
                nothing,    # arguments input type
                nothing,    # arguments return type
                nothing,    # bytecode
                0,          # bytecode size
                nothing,    # function name
                C_NULL,     # module
                C_NULL,     # function
                nothing     # cache key
            )
        end

        # Try to load from cache
        if version.cache_key == nothing
            XK.Logger.debug("Computing cache key...")
            version.cache_key = XK.KA.Cache.compute_cache_key(fmt.kernel, kernel_args_type)
            XK.Logger.debug("Cache key: $(version.cache_key)")
        end
        cached_data = XK.KA.Cache.load_cached_bytecode(version.cache_key)
        if cached_data !== nothing
            version.bytecode         = cached_data.bytecode
            version.bytecode_size    = cached_data.bytecode_size
            version.bytecode_fn_name = cached_data.bytecode_fn_name
        else
            # Compile to BYTECODE - TODO: do that portably, not only ptx, targetting the passed device type
            buf = IOBuffer()
            XK.Logger.debug("Compiling to BYTECODE")
            CUDA.code_ptx(buf, fmt.kernel, kernel_args_type; raw=false, kernel=true)
            version.bytecode      = String(take!(buf))
            version.bytecode_size = length(version.bytecode)

            XK.Logger.debug("Compiled to BYTECODE")
            XK.Logger.debug(version.bytecode)

            # Regex to find function names
            regex_func = r"\.entry\s+([a-zA-Z_0-9]+)\("

            # Find all matches and extract the captured name
            function_names = [m.captures[1] for m in eachmatch(regex_func, version.bytecode)]
            XK.Logger.debug("Functions in the BYTECODE: $(function_names)")
            @assert length(function_names) == 1
            version.bytecode_fn_name = String(function_names[1])

            # Save to cache
            XK.KA.Cache.save_cached_bytecode(version.cache_key, version.bytecode, version.bytecode_size, version.bytecode_fn_name)
        end
        XK.Logger.debug("BYTECODE of name $(version.bytecode_fn_name) and size $(version.bytecode_size)")

        ######################################################
        # Set the args buffer for launching the kernel later #
        ######################################################

        #
        # Task arguments are
        #   [pointer_to_format | pointer_to_version | tx | ty | tz | shared_memory_size | julia_context | kernel_args...]
        # with
        #   pointer_to_format  -> a 'void *' to the FormatStruct
        #   pointer_to_version -> a 'void *' to the VersionStruct
        #   tx, ty, tz         -> the number of threads to launch the kernel
        #   shared_memory_size -> amount of shared memory
        #   julia_context      -> opaque structure of `JL_CU_CONTEXT_SIZE` bytes
        #   kernel_args        -> the kernel arguments, prefilled
        #                           - empty spaces of sizeof(void *) bytes per access
        #                           - values, for values passed by copy
        #

        # 0. reflect on the 'parameters' function to known which parameters are passed by access or copy
        version.arguments_input_type = Tuple{map(arg -> typeof(arg), kernel_args)...}
        version.arguments_return_type = Base.return_types(fmt.arguments, version.arguments_input_type)[1]
        @assert (version.arguments_return_type <: Tuple)

        # 1. compute sizes for each kernel argument
        kernel_args_size = Vector{Int}(undef, length(kernel_args))
        for i in 1:length(version.arguments_return_type.parameters)
            T = version.arguments_return_type.parameters[i]
            if T <: XK.xkrt_access_t
                # we will store a raw pointer (machine pointer size)
                kernel_args_size[i] = sizeof(Ptr{Cvoid})
            else
                # store the raw bytes of the value
                # sizeof should work for typical scalar isbitstype arguments (Int, Float64, etc.)
                kernel_args_size[i] = sizeof(kernel_args[i])
            end
        end

        # 2. allocate contiguous byte buffer
        total_size = 2*sizeof(Ptr{Cvoid}) + 3*sizeof(Int) + 1*sizeof(Int) + JL_CU_CONTEXT_SIZE + sum(kernel_args_size)
        task_args_buf = Vector{UInt8}(undef, total_size)
        task_args_buf_ptr = Ptr{Int8}(pointer(task_args_buf))

        # 3. copy format and version ptr
        unsafe_store!(Ptr{Ptr{Cvoid}}(task_args_buf_ptr + 0*sizeof(Ptr{Cvoid})), Base.unsafe_convert(Ptr{Cvoid}, Ref(fmt)))
        unsafe_store!(Ptr{Ptr{Cvoid}}(task_args_buf_ptr + 1*sizeof(Ptr{Cvoid})), Base.unsafe_convert(Ptr{Cvoid}, Ref(version)))

        # 4. set the number of threads
        if fmt.threads != nothing
            # given explicitly by the programmer
            tx, ty, tz = fmt.threads(kernel_args...)
        else
            # not given by the programmer, guess it from accesses
            XK.Logger.fatal("Default launch grid size is not implemented. Please specify a launcher with `threads = (args...) -> (tx, ty, tz) --- the number of threads to use`")
        end
        threads_ptr = task_args_buf_ptr + 2*sizeof(Ptr{Cvoid})
        unsafe_store!(Ptr{Int}(threads_ptr + 0*sizeof(Int)), tx)
        unsafe_store!(Ptr{Int}(threads_ptr + 1*sizeof(Int)), ty)
        unsafe_store!(Ptr{Int}(threads_ptr + 2*sizeof(Int)), tz)

        # 5. copy shared memory size
        if fmt.shared_memory != nothing
            shared_memory_size = fmt.shared_memory(kernel_args...)
        else
            shared_memory_size = 0
        end
        shared_memory_size_ptr = task_args_buf_ptr + 2*sizeof(Ptr{Cvoid})+ 3*sizeof(Int)
        unsafe_store!(Ptr{Int}(shared_memory_size_ptr), shared_memory_size)

        # 6. fill rest of buffer with arguments
        offset = sizeof(Ptr{Cvoid}) + 3*sizeof(Int) + 1*sizeof(Int) + JL_CU_CONTEXT_SIZE
        for i in 1:length(version.arguments_return_type.parameters)
            T = version.arguments_return_type.parameters[i]
            kernel_arg_dst   = Ptr{Int8}(task_args_buf_ptr + offset)
            kernel_arg_value = kernel_args[i]
            kernel_arg_size  = kernel_args_size[i]
            if T <: XK.xkrt_access_t
                @assert isa(kernel_arg_value, AbstractArray)
                @assert kernel_arg_size == sizeof(Ptr{Cvoid})
                # nothing to do, this space will be filled when the kernel is
                # scheduled with the replicated device memory pointer
            else
                # copy raw bytes of the scalar argument value
                kernel_arg_value_ref = Ref(kernel_arg_value)
                kernel_arg_value_struct = Base.unsafe_convert(Ptr{typeof(kernel_arg_value)}, kernel_arg_value_ref)
                kernel_arg_value_i8 = Ptr{Int8}(kernel_arg_value_struct)
                unsafe_copyto!(kernel_arg_dst, kernel_arg_value_i8, kernel_arg_size)
            end
            offset += kernel_arg_size
        end

        # set args pointer and size to the buffer
        task_args = Ptr{Cvoid}(task_args_buf_ptr)
        task_args_size = total_size

        ############################################
        # create a lambda to set the task accesses #
        ############################################

        set_accesses = (accesses) -> begin
            arguments = fmt.arguments(kernel_args...)
            for i in 1:length(version.arguments_return_type.parameters)
                T = version.arguments_return_type.parameters[i]
                if T <: XK.xkrt_access_t
                    push!(accesses, arguments[i])
                end
            end
        end

        ##################
        # spawn the task #
        ##################

        XK.device_async(
            device_unique_id,
            fmt.fmtid,
            set_accesses=set_accesses,
            args=task_args, args_size=task_args_size,
            detach_ctr_initial=0
        )
    end

    # device_async wrapper to automatically target device 1
    function device_async(fmt::FormatStruct, kernel_args...)
        device_unique_id = XK.xkrt_device_unique_id_t(1)
        return XK.KA.device_async(device_unique_id, fmt, kernel_args...)
    end

    # Note: No 'using MacroTools' needed now!

    """
    @KA.tid
    A macro that expands to the backend-specific index calculation.
    """
    macro tid()
        # The definition for @tid is simple and remains the same
        return :(
            (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
        )
    end

    """
    @KA.kernel function my_kernel(...) ... end
    A macro that wraps a function definition, adds 'return nothing',
    and works by manually inspecting the Abstract Syntax Tree (AST).
    """
    macro kernel(ex)
        # The 'ex' argument is the expression for the function definition.

        # 1. Basic check: Ensure it is a function definition
        if ex.head !== :function
            error("@kernel must be applied to a function definition.")
        end

        # The structure of a function expression is typically:
        # ex.args[1]: The function signature (e.g., :(vector_add(a, b, c, n)))
        # ex.args[2]: The function body (a Block expression)

        # 2. Get the function body (which is often a :block expression)
        func_body = ex.args[2]

        # The function body, 'func_body', is typically an Expr with head :block.
        # Its contents are in func_body.args.
        if func_body.head !== :block
            # This handles very unusual function bodies, but usually, it is :block
            error("Function body is not a block expression.")
        end

        # 3. Add the required 'return nothing' to the function body's arguments.
        # This is done by pushing the expression :(return nothing) onto the array
        # of expressions that form the function body.
        push!(func_body.args, :(return nothing))

        # 4. Reconstruct and return the modified expression.
        # We use 'esc' to prevent the macro hygiene system from renaming variables.
        return esc(ex)
    end

end
