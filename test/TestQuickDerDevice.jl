#
# TestQuickDerDevice.jl -- the `device = :gpu` path of :QuickDer, and the cost
# model that decides how a mode product is run on it.
#
# TWO HALVES, and only the second needs hardware.
#
#   * The cost model (`_qdn_slab_is_cheap`, `_qdn_mode_cost`, `_qdn_mode_order`)
#     is pure arithmetic on shapes.  It runs everywhere, and it is what guards
#     the property the CPU depends on: a HOST array keeps the natural axis
#     order, so nothing about an existing Float32/Float64 CPU answer moves.
#
#   * The device path itself is guarded on `Dleto.gpu_available()`, which is
#     false unless a backend extension has been loaded (`using Metal`).  The
#     suite stays green on a machine without one; on a machine with one these
#     compare the device answer against the CPU answer on the same tensor.
#
# Video-shaped throughout (`H x W x F x 3`), because that is the regime the
# device path exists for and the one where the axis order matters: a movie has
# two big edge-ish axes, one middle axis whose trailing block is 3 (the frame
# axis) and one whose trailing block is 3F (the column axis), and the whole
# point of `_qdn_mode_order` is to tell those two apart.
#
using Test
using Dleto
using ITensors
using LinearAlgebra
using Random

# Loaded opportunistically: `using Metal` on a machine without Metal.jl (or
# without a functional GPU) must not fail the suite, and `gpu_available()`
# stays false in that case because the extension's `__init__` never runs.
const DEVICE_TESTS = try
    @eval using Metal
    Dleto.gpu_available()
catch
    false
end

if !isdefined(@__MODULE__, :der_residual)
    function der_residual(Γ::ITensor, D::Vector{ITensor}, P::AbstractMatrix)
        C = Chisel(P, collect(inds(Γ)))
        R = applyDerivation(Γ, D, C)
        scale = norm(Γ) * maximum(norm.(D))
        return norm(R) / max(scale, eps())
    end
end

"""
    principal_angle(A, B) -> Float64

`sin` of the largest principal angle between the column spans of `A` and `B`
(0 when they span the same subspace, 1 when one has a direction orthogonal to
the other).  The comparison the device tests want: the two paths solve the same
problem in different arithmetic, so their bases agree as SUBSPACES, not
entry by entry.
"""
function principal_angle(A::AbstractMatrix, B::AbstractMatrix)
    (size(A, 2) == 0 || size(B, 2) == 0) && return size(A, 2) == size(B, 2) ? 0.0 : 1.0
    QA = Matrix(qr(Matrix{Float64}(A)).Q)[:, 1:size(A, 2)]
    QB = Matrix(qr(Matrix{Float64}(B)).Q)[:, 1:size(B, 2)]
    s = svdvals(transpose(QA) * QB)
    return sqrt(max(0.0, 1 - minimum(s)^2))
end

"""A video-shaped random tensor and the pieces `derTrOpsReduced` wants."""
function video_input(H, W, F, T)
    Random.seed!(20260904)
    fr = [Index(H, "h"), Index(W, "w"), Index(F, "f"), Index(3, "c")]
    Γ = ITensor(Array{T}(randn(H, W, F, 3)), fr...)
    return (Ω = IndTransverseOps(fr, UniversalOp()), ch = UniversalChisel(4), Γ = Γ)
end

@testset "QuickDer device cost model" begin
    # The two middle axes of a movie, told apart.  Column axis: 3F slabs, so a
    # dispatch each is dear against one permuted copy of a 331 MB tensor -- but
    # still cheaper than it, which is the finding the slab route rests on.
    # Frame axis: 3 slabs, cheaper by two orders of magnitude.
    @test Dleto._qdn_slab_is_cheap((640, 480, 90, 3), 3, 15, Float32)
    @test Dleto._qdn_slab_is_cheap((640, 480, 90, 3), 2, 25, Float32)
    # Small tensor, many slabs: the copy is microseconds and a dispatch is not.
    @test !Dleto._qdn_slab_is_cheap((10, 10, 10, 3), 2, 5, Float32)

    rng = MersenneTwister(20260904)
    movie = [Dleto._qdn_axis(Float32, d, r, :random, rng)
             for (d, r) in zip((640, 480, 90, 3), (33, 25, 15, 3))]
    cost = [Dleto._qdn_mode_cost((640, 480, 90, 3), movie[a], a, Float32) for a in 1:4]
    # Axis 1 is an edge (one GEMM), axis 3 a middle axis with 3 slabs, axis 2 a
    # middle axis with 270, axis 4 saturated and so infinitely bad value.
    @test cost[1] < cost[3] < cost[2] < cost[4]
    @test isinf(cost[4])

    # HOST ARRAYS KEEP 1:N.  This is the CPU-invariance guarantee: every
    # Float32/Float64 CPU result in the rest of the suite depends on it.
    Gh = randn(Float32, 8, 7, 6, 3)
    axs = [Dleto._qdn_axis(Float32, d, r, :random, rng)
           for (d, r) in zip((8, 7, 6, 3), (4, 4, 4, 3))]
    @test Dleto._qdn_mode_order(Gh, axs, 1:4) == [1, 2, 3, 4]
    @test Dleto._qdn_mode_order(Gh, axs, (c for c in 1:4 if c != 2)) == [1, 3, 4]

    # The chain is a permutation of what it was asked for, whatever the order.
    S = Dleto._qdn_cross_sketches(Gh, axs, [true, true, true, true])
    @test size(S[1]) == (8, 4, 4, 3)
    @test size(S[3]) == (4, 4, 6, 3)
    @test size(Dleto._qdn_pair_tensor(Gh, axs, 1, 3)) == (4, 4, 6, 3)
end

if !DEVICE_TESTS
    @info "QuickDer device tests skipped: Dleto.gpu_available() is false " *
          "(load a backend, e.g. `using Metal`, on hardware that has one)"
else
@testset "QuickDer device = :gpu" begin
    # Small enough that the CPU oracle is instant and the device tensor is
    # kilobytes, video-shaped so the axis order is the movie's.
    (H, W, F) = (20, 20, 10)

    # Both budgets zeroed: the matrix-free restricted branch is the one the
    # movie regime takes, and it is host-side on both devices, so this compares
    # exactly the tensor stages.
    saved = (Dleto.QDN_DENSE_BUDGET_BYTES[], Dleto.QDN_GPU_DENSE_BUDGET_BYTES[])
    try
        Dleto.QDN_DENSE_BUDGET_BYTES[] = 0.0
        Dleto.QDN_GPU_DENSE_BUDGET_BYTES[] = 0.0

        @testset "same answer as the CPU, $T" for T in (Float32, Float16)
            inp = video_input(H, W, F, T)
            solve_on(device) = begin
                m = Dleto.get_derivation_method(:QuickDer; whiten = true,
                                                solver = :AutoSolver, device = device,
                                                verify = :random, seed = 20260904)
                (_, _, ders) = Dleto.derTrOpsReduced(m, inp.Ω, inp.ch, inp.Γ; tol = 1e-6)
                ders
            end
            host = solve_on(:cpu)
            dev = solve_on(:gpu)
            @test size(dev, 2) == size(host, 2)
            # HOW CLOSE THE TWO SPANS MAY BE ASKED TO BE is set by the solve,
            # not by the arithmetic.  Both paths run Float32 (Apple GPUs have no
            # Float64, and `compute_eltype` promotes Float16 to Float32 on the
            # host too), and they differ only in the ORDER their mode products
            # were applied in -- a Float32 rounding difference in the restricted
            # operator.  But the matrix-free branch then solves that operator
            # ITERATIVELY to `_qd_tolerance(T, tol)`, which is floored on the
            # STORED type: 3.4e-4 for Float32 and 3.1e-2 for Float16.  So the
            # answers can only agree to the tolerance they were asked for, and
            # that is the bound to hold them to.  Measured today, both an order
            # or two inside it: 4.3e-6 (Float32), 1.5e-4 (Float16).
            @test principal_angle(host, dev) < Dleto._qd_tolerance(T, 1e-6)
            # Coordinates come back in the STORED type on both paths.
            @test eltype(dev) === T
        end

        # A Float16 tensor is never certified past what its data resolves.
        # `_qd_tolerance` floors on the stored type, so the device path -- which
        # promotes the ARITHMETIC to Float32 -- must not move the floor.
        @test Dleto._qd_tolerance(Float16, 1e-12) >= sqrt(Dleto.data_floor(Float16))
        @test Dleto._qd_tolerance(Float16, 1e-12) > Dleto._qd_tolerance(Float32, 1e-12)

        # NOTHING THE SIZE OF THE TENSOR IS FLOAT64 ON THE DEVICE PATH.  Apple
        # GPUs have no Float64 at all, so a Float64 array of `d^n` would mean
        # the pass silently ran on the host; the mode products keep the device
        # array's own element type.
        @testset "no host-width copy of the tensor" begin
            G = Dleto.to_gpu(Array{Float32}(randn(H, W, F, 3)))
            rng = MersenneTwister(20260904)
            axs = Dleto._qdn_axes_device(
                [Dleto._qdn_axis(Float32, d, r, :random, rng)
                 for (d, r) in zip((H, W, F, 3), (6, 6, 5, 3))])
            S = Dleto._qdn_cross_sketches(G, axs, [true, true, true, true])
            for a in 1:4
                @test eltype(S[a]) === Float32
                @test !(S[a] isa Array)          # never came back to the host
            end
            @test size(S[1]) == (H, 6, 5, 3)
            @test size(S[3]) == (6, 6, F, 3)

            # The slab route and the permute route are the same arithmetic.
            # Axis 2 of this shape takes the permute route (small tensor, 30
            # slabs); forcing the comparison against the host says both agree.
            Gh = Array(G)
            for a in 1:4
                M = randn(Float32, size(Gh, a), 4)
                dev = Array(Dleto._qdn_ttm(G, Dleto.to_gpu(M), a))
                @test isapprox(dev, Dleto._qdn_ttm(Gh, M, a); rtol = 1e-4)
            end
        end
    finally
        (Dleto.QDN_DENSE_BUDGET_BYTES[], Dleto.QDN_GPU_DENSE_BUDGET_BYTES[]) = saved
    end
end
end
