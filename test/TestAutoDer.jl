#
# :Auto -- QuickDer where it applies, SylverLining otherwise.  The contract is
# that whichever path answers, the answer satisfies the Z-law and agrees with
# SylverLining's nullity.
#
using Test
using Dleto
using ITensors
using LinearAlgebra
using Random

if !isdefined(@__MODULE__, :der_residual)
    function der_residual(Γ::ITensor, D::Vector{ITensor}, P::AbstractMatrix)
        C = Chisel(P, collect(inds(Γ)))
        R = applyDerivation(Γ, D, C)
        return norm(R) / max(norm(Γ) * maximum(norm.(D)), eps())
    end
end

@testset "AutoDer: routing and correctness" begin
    Random.seed!(20260904)
    m = Dleto.get_derivation_method(:Auto)
    @test m isa AutoDerMethod
    @test m.quick isa QuickDerMethod
    @test m.fallback isa SylverLiningMethod

    @testset "routing" begin
        small = ITensor(randn(4, 5, 3), [Index(4, "x"), Index(5, "y"), Index(3, "z")]...)
        big = ITensor(randn(10, 10, 10, 10), [Index(10, "b$i") for i in 1:4]...)
        Ωs = IndTransverseOps(collect(inds(small)), UniversalOp())
        Ωb = IndTransverseOps(collect(inds(big)), UniversalOp())
        @test !Dleto.autoder_applicable(m, Ωs, UniversalChisel(3), small)
        @test Dleto.autoder_applicable(m, Ωb, UniversalChisel(4), big)
        # a chisel with no engaged axis never goes to QuickDer
        @test !Dleto.autoder_applicable(m, Ωb, zeros(1, 4), big)
    end

    @testset "agrees with SylverLining, valence $(length(dims))" for dims in ((9, 8, 7), (7, 6, 5, 4))
        fr = [Index(dims[i], "a$i") for i in 1:length(dims)]
        Γ = ITensor(randn(dims...), fr...)
        P = UniversalChisel(length(dims))
        auto = der(:Auto, Γ; tol = 1e-8)
        ref = der(:SylverLining, Γ; tol = 1e-8)
        @test length(auto) == length(ref) == length(dims) - 1
        for D in auto
            @test der_residual(Γ, D, P) < 1e-10
        end
    end

    @testset "stratify defaults to :Auto and recovers the valence-4 sphere" begin
        include(joinpath(@__DIR__, "..", "bench", "SphereHarness.jl"))
        inp = build_sphere(12; valence = 4)
        r = run_stratify(inp)                # default method
        @test r.status == "ok"
        @test r.nullity == 4
        @test r.lsq_err < 1e-8
    end
end
