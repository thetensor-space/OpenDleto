# BLAS forwarding is process-wide, not specific to Dleto.
const _accelerate_library = "/System/Library/Frameworks/Accelerate.framework/Accelerate"

"""
    blas_backend()

Return Julia's current libblastrampoline configuration. The listed libraries
include fallbacks; other packages can change this process-wide configuration.
"""
blas_backend() = LinearAlgebra.BLAS.get_config()

function _init_blas_backend()
    choice = lowercase(get(ENV, "DLETO_BLAS", "auto"))
    if choice == "openblas"
        return # Leave the existing process-wide BLAS configuration alone.
    elseif choice != "auto"
        @warn "Unknown DLETO_BLAS value; leaving BLAS unchanged" choice
        return
    end
    Sys.isapple() || return
    # Accelerate's NEWLAPACK ILP64 interface needs macOS 13.4 or later;
    # earlier releases contain a LAPACK bug. Check before changing LBT.
    version = try
        VersionNumber(strip(read(`/usr/bin/sw_vers -productVersion`, String)))
    catch
        return
    end
    version >= v"13.4" || return
    handle = Libdl.dlopen_e(_accelerate_library)
    handle == C_NULL && return
    try
        Libdl.dlsym_e(handle, "dgemm\$NEWLAPACK\$ILP64") == C_NULL && return
        # Keep existing forwarding as a fallback for missing symbols (e.g. gemmt).
        LinearAlgebra.BLAS.lbt_forward(_accelerate_library; suffix_hint="\x1a\$NEWLAPACK")
        LinearAlgebra.BLAS.lbt_forward(_accelerate_library; suffix_hint="\x1a\$NEWLAPACK\$ILP64")
    catch err
        @warn "Could not enable Apple Accelerate BLAS; retaining available BLAS forwarding" exception=(err, catch_backtrace())
    finally
        Libdl.dlclose(handle)
    end
end
