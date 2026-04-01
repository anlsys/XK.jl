####################
# Build parameters #
####################

# You may want to modify it for your specific system

const XKRT_BRANCH   = "release/v1.0.2"
const XKBLAS_BRANCH = "release/v2.0.2"

# Example Release build
# const build_type="Release"
# const use_stats="OFF"
# const use_shut_up="ON"

# Example Debug build
build_type="Debug"
use_stats="ON"
use_shut_up="OFF"

use_cuda = "ON"
cmake_prefix_path = get(ENV, "CUDA_PATH", "")

use_sycl = "OFF"
use_ze = "OFF"
use_ze_sycl_interop = "OFF"
#cmake_prefix_path = "/usr/include/level_zero"

use_hip = "OFF"
cmake_prefix_path = get(ENV, "ROCM_PATH", "")

############################################################################

# If building for documentation, skip all dependencies
const is_docs_build = get(ENV, "XK_DOCS_BUILD", "false") == "true"
if is_docs_build
    build_type = "Release"
    use_stats = "OFF"
    use_shut_up = "ON"
    use_cuda = "OFF"
    use_ze   = "OFF"
    use_sycl = "OFF"
    use_hip  = "OFF"
    use_ze_sycl_interop = "OFF"
    cmake_prefix_path = ""
end

# build XKBlas with CMake (including XKRT dependency)

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Scratch, Preferences, CMake_jll, LibGit2

XK_pkg = Base.UUID("8d3f9e88-0651-4e8b-8f79-7d9d5f5f9e88")

# Configuration from install.sh
const XKRT_URL = "https://gitlab.inria.fr/xkaapi/dev-v2.git"
const XKBLAS_URL = "https://gitlab.inria.fr/xkblas/dev.git"

# get scratch directories for installations
xkrt_install_dir = get_scratch!(XK_pkg, "xkrt")
xkblas_install_dir = get_scratch!(XK_pkg, "xkblas")
rm(xkrt_install_dir; recursive=true, force=true)
rm(xkblas_install_dir; recursive=true, force=true)

# get scratch directories for source code
xkrt_source_dir = get_scratch!(XK_pkg, "xkrt_src")
xkblas_source_dir = get_scratch!(XK_pkg, "xkblas_src")

# get build directories
xkrt_build_dir = if length(ARGS) >= 1
    joinpath(ARGS[1], "xkrt")
else
    mktempdir(prefix="xkrt_build_")
end
mkpath(xkrt_build_dir)

xkblas_build_dir = if length(ARGS) >= 1
    joinpath(ARGS[1], "xkblas")
else
    mktempdir(prefix="xkblas_build_")
end
mkpath(xkblas_build_dir)

#######################
# Clone XKRT source #
#######################
@info "="^60
@info "Cloning/updating XKRT source (branch: $XKRT_BRANCH)..."
@info "="^60

# Check if directory exists and is a valid git repository
if isdir(xkrt_source_dir) && isdir(joinpath(xkrt_source_dir, ".git"))
    @info "XKRT source directory exists, updating..."
    try
        repo = LibGit2.GitRepo(xkrt_source_dir)
        try
            LibGit2.fetch(repo)

            # Get current HEAD
            head_ref = LibGit2.head(repo)
            head_branch = LibGit2.shortname(head_ref)

            # If we're not on the right branch, switch to it
            if head_branch != XKRT_BRANCH
                remote_ref = "refs/remotes/origin/$XKRT_BRANCH"
                LibGit2.branch!(repo, XKRT_BRANCH, remote_ref, force=true, track="origin/$XKRT_BRANCH")
                LibGit2.head!(repo, "refs/heads/$XKRT_BRANCH")
            end

            # Pull latest changes (fast-forward merge)
            LibGit2.merge!(repo, fastforward=true)
        finally
            close(repo)
        end
    catch e
        @warn "Failed to update existing repository: $e"
        @info "Removing and re-cloning..."
        rm(xkrt_source_dir; recursive=true, force=true)
        LibGit2.clone(XKRT_URL, xkrt_source_dir; branch=XKRT_BRANCH)
    end
else
    # Remove if directory exists but is not a git repo
    if isdir(xkrt_source_dir)
        @warn "Directory exists but is not a git repository. Removing..."
        rm(xkrt_source_dir; recursive=true, force=true)
    end
    @info "Cloning XKRT from $XKRT_URL"
    LibGit2.clone(XKRT_URL, xkrt_source_dir; branch=XKRT_BRANCH)
end

# Get commit hash for logging
xkrt_hash = let repo = LibGit2.GitRepo(xkrt_source_dir)
    hash = string(LibGit2.head_oid(repo))[1:12]
    close(repo)
    hash
end
@info "Using XKRT commit: $xkrt_hash"

#######################
# Clone XKBlas source #
#######################
@info "="^60
@info "Cloning/updating XKBlas source (branch: $XKBLAS_BRANCH)..."
@info "="^60

if isdir(xkblas_source_dir) && isdir(joinpath(xkblas_source_dir, ".git"))
    @info "XKBlas source directory exists, updating..."
    try
        repo = LibGit2.GitRepo(xkblas_source_dir)
        try
            LibGit2.fetch(repo)

            # Get current HEAD
            head_ref = LibGit2.head(repo)
            head_branch = LibGit2.shortname(head_ref)

            # If we're not on the right branch, switch to it
            if head_branch != XKBLAS_BRANCH
                remote_ref = "refs/remotes/origin/$XKBLAS_BRANCH"
                LibGit2.branch!(repo, XKBLAS_BRANCH, remote_ref, force=true, track="origin/$XKBLAS_BRANCH")
                LibGit2.head!(repo, "refs/heads/$XKBLAS_BRANCH")
            end

            # Pull latest changes (fast-forward merge)
            LibGit2.merge!(repo, fastforward=true)
        finally
            close(repo)
        end
    catch e
        @warn "Failed to update existing repository: $e"
        @info "Removing and re-cloning..."
        rm(xkblas_source_dir; recursive=true, force=true)
        LibGit2.clone(XKBLAS_URL, xkblas_source_dir; branch=XKBLAS_BRANCH)
    end
else
    # Remove if directory exists but is not a git repo
    if isdir(xkblas_source_dir)
        @warn "Directory exists but is not a git repository. Removing..."
        rm(xkblas_source_dir; recursive=true, force=true)
    end
    @info "Cloning XKBLAS from $XKBLAS_URL"
    LibGit2.clone(XKBLAS_URL, xkblas_source_dir; branch=XKBLAS_BRANCH)
end

# Get commit hash for logging
xkblas_hash = let repo = LibGit2.GitRepo(xkblas_source_dir)
    hash = string(LibGit2.head_oid(repo))[1:12]
    close(repo)
    hash
end
@info "Using XKBlas commit: $xkblas_hash"

#######################
# Build XKRT (xkrt) #
#######################
@info "="^60
@info "Building XKRT (xkrt) dependency..."
@info "="^60

xkrt_cmake_options = String[
    "-DCMAKE_BUILD_TYPE=$build_type",
    "-DCMAKE_INSTALL_PREFIX=$xkrt_install_dir",
    "-DCMAKE_PREFIX_PATH=$cmake_prefix_path",
    "-DSTRICT=OFF",
    "-DUSE_STATS=$use_stats",
    "-DUSE_JULIA=ON",
    "-DUSE_CUDA=$use_cuda",
    "-DUSE_HIP=$use_hip",
    "-DUSE_SYCL=$use_sycl",
    "-DUSE_ZE=$use_ze",
    "-DUSE_ZE_SYCL_INTEROP=$use_ze_sycl_interop",
    "-DUSE_SHUT_UP=$use_shut_up"
]

# build and install XKRT
cmake() do cmake_path
    make = "make"  # path to `make`, or find it via `which("make")`

    # Configure
    cmd = `$cmake_path -G "Unix Makefiles" $xkrt_cmake_options -S $xkrt_source_dir -B $xkrt_build_dir`
    @info "Configuring XKRT with Unix Makefiles..." cmd
    run(cmd)

    # Build and install
    cmd = `$make -j -C $xkrt_build_dir install`
    @info "Building and installing XKRT..." cmd
    run(cmd)
end

# Verify XKRT installation
xkrt_lib = joinpath(xkrt_install_dir, "lib", "libxkrt.so")
if !ispath(xkrt_lib)
    xkrt_lib = joinpath(xkrt_install_dir, "lib64", "libxkrt.so")
end
@assert ispath(xkrt_lib) "Could not find libxkrt.so at $xkrt_lib"
@info "XKRT installed successfully at: $xkrt_install_dir"

##################
# Build XKBlas   #
##################
@info "="^60
@info "Building XKBlas..."
@info "="^60

# Update CMAKE_PREFIX_PATH to include XKRT
cmake_prefix_path = "$cmake_prefix_path;$xkrt_install_dir"

xkblas_cmake_options = String[
    "-DCMAKE_C_FLAGS=-Wno-error",
    "-DCMAKE_CXX_FLAGS=-Wno-error",
    "-DCMAKE_BUILD_TYPE=$build_type",
    "-DCMAKE_INSTALL_PREFIX=$xkblas_install_dir",
    "-DCMAKE_PREFIX_PATH=$cmake_prefix_path",
    "-DXKRT_DIR=$(joinpath(xkrt_install_dir, "lib", "cmake", "XKRT"))",
    "-DSTRICT=OFF",
    "-DUSE_CUDA=$use_cuda",
    "-DUSE_CUBLAS=$use_cuda",
    "-DUSE_HIP=$use_hip",
    "-DUSE_HIPBLAS=$use_hip",
    "-DUSE_SYCL=$use_sycl",
    "-DUSE_ZE=$use_ze",
    "-DUSE_CBLAS=off",    # OpenBLAS_jll doesn't include lapacke.h needed for tests
    "-DUSE_TESTS=off",    # Disable tests - they require LAPACK headers
    "-DUSE_OPENBLAS=off"
]

# build and install XKBlas
cmake() do cmake_path
    make = "make"  # path to `make`, or find it via `which("make")`

    # Configure
    cmd = `$cmake_path -G "Unix Makefiles" $xkblas_cmake_options -S $xkblas_source_dir -B $xkblas_build_dir`
    @info "Configuring XKBlas with Unix Makefiles..." cmd
    run(cmd)

    # Build and install
    cmd = `$make -j -C $xkblas_build_dir install`
    @info "Building and installing XKBlas..." cmd
    run(cmd)
end

# Find the installed libraries
# TODO: adapt when we support more platforms
xkblas_lib_path = joinpath(xkblas_install_dir, "lib", "libxkblas.so")
if !ispath(xkblas_lib_path)
    # Try lib64
    xkblas_lib_path = joinpath(xkblas_install_dir, "lib64", "libxkblas.so")
end
@assert ispath(xkblas_lib_path) "Could not find libxkblas.so at $xkblas_lib_path"

xkrt_lib_path = joinpath(xkrt_install_dir, "lib", "libxkrt.so")
if !ispath(xkrt_lib_path)
    xkrt_lib_path = joinpath(xkrt_install_dir, "lib64", "libxkrt.so")
end
@assert ispath(xkrt_lib_path) "Could not find libxkrt.so at $xkrt_lib_path"

@info "="^60
@info "Build complete!"
@info "XKBlas library: $xkblas_lib_path"
@info "XKRT library:   $xkrt_lib_path"
@info "="^60

# tell XKBlas_jll and XKRT_jll to load our libraries instead of the default artifact ones
set_preferences!(
    joinpath(dirname(@__DIR__), "LocalPreferences.toml"),
    "XKBlas_jll",
    "libxkblas_path" => xkblas_lib_path;
    force=true,
)

set_preferences!(
    joinpath(dirname(@__DIR__), "LocalPreferences.toml"),
    "XKRT_jll",
    "libxkrt_path" => xkrt_lib_path;
    force=true,
)

# copy the preferences to `test/` as well to work around Pkg.jl#2500
cp(joinpath(dirname(@__DIR__), "LocalPreferences.toml"),
   joinpath(dirname(@__DIR__), "test", "LocalPreferences.toml"); force=true)

@info "Library paths saved to LocalPreferences.toml"
