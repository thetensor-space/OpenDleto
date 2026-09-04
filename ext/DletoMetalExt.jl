#
# DletoMetalExt -- Apple-GPU backend for the dense kernels, via Metal.jl.
#
# Loaded automatically by `using Metal` alongside `using Dleto`.  Metal has no
# Float64, so `to_gpu` converts to Float32; results come back as Float32 host
# arrays.  Measured on an M4 Max (40 GPU cores) against 8 CPU threads:
# fp32 GEMM 12 TFlop/s vs 0.8, mode products (TTM) 3-5x, Gram formation 9x.
#
# Layout: this file owns the device hooks; kernel-specific GPU methods live in
# the files it includes so that the SylverLining and QuickDer work can proceed
# in parallel without editing the same file.
#
module DletoMetalExt

using Dleto
using Metal
using LinearAlgebra

Dleto.gpu_available() = Metal.functional()

Dleto.to_gpu(x::AbstractArray{Float64}) = MtlArray{Float32}(x)
Dleto.to_gpu(x::AbstractArray{Float32}) = MtlArray(x)
Dleto.to_gpu(x::AbstractArray{<:Complex}) = MtlArray{ComplexF32}(x)
Dleto.to_gpu(x::MtlArray) = x

Dleto.to_cpu(x::MtlArray) = Array(x)

Dleto.gpu_sync(f) = Metal.@sync f()

# SylverLining array kernel on the GPU (owned by the sylvesterLM work)
isfile(joinpath(@__DIR__, "DletoMetalSylver.jl")) && include("DletoMetalSylver.jl")
# QuickDer mode products / Gram on the GPU (owned by the QuickDer work)
isfile(joinpath(@__DIR__, "DletoMetalQuickDer.jl")) && include("DletoMetalQuickDer.jl")

function __init__()
    Metal.functional() && @info "Loading Dleto Metal Extension ($(Metal.device().name))"
end

end # module
