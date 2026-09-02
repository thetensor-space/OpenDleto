#
# Test SylverLining
#
# These tests pin two properties of the sylve/ester pair built by `sylvesterLM`:
#
#   1. `densor_map` and its adjoint are genuinely transpose to one another,
#      i.e. <a, f'(B)> == <f(a), B>.  This is the property the transpose
#      convention rests on (see docs/review/Refactor-Plan.md section 1.3).
#   2. `derdensor_map` is the composition f' . f of that pair -- the closure
#      object of the Galois connection.
#
# Ported from the pre-rename API: Local*Ops -> *Op and
# TransverseOpsIndependant -> IndTransverseOps.  The file had been commented
# out of runtests.jl while it was stale.
#
# KNOWN GAP (found by re-enabling this file, 2026-09-02):
#   TransverseOpsSymmetries does not implement `unsafe_embedITensorsSwapped`.
#   IndTransverseOps does (src/ops/TransverseOpsIndependant.jl:102); the
#   symmetries version is simply absent, so the abstract placeholder in
#   TransverseOperators.jl:143 asserts.  `sylvesterLM` applies that embedding
#   inside `ester`, so *no derivation can be solved with symmetry-restricted
#   operators at all* -- construction and plain embedding work, solving does
#   not.  The symmetry test sets below are therefore marked broken rather than
#   deleted, so the gap stays visible until the method is implemented.
#
using Dleto
using ITensors
using LinearAlgebra
using LinearMaps

# ScalarOp and EmptyOp are excluded: they drive the reduced operator space to
# dimension < 3, where these identities are vacuous.
LΩs = [UniversalOp(), DiagonalOp(), SymmetricOp(), AntiSymmetricOp()]

syms = [[1, 1, 1, 1], [1, 2, 1, 2, 2, 1], [1, 2, 3, -1, 2, 3],
        [1, 2, -2, -2, 5, -5, -1, -1], [1, -1, 3, 3, -3, 1, 7, 8]]

# Sample count scaled to the size of the map, so that high-valence frames
# (the symmetry sweep below reaches valency 10) stay affordable.
adaptive_samples(f::LinearMap) = clamp(1_000_000 ÷ max(1, size(f, 1)), 5, 100)

function testTranspose(f::LinearMap, num::Integer)
    n, m = size(f, 2), size(f, 1)
    for _ in 1:num
        a = rand(n)
        B = rand(m)
        A = f(a)
        b = f'(B)
        @assert isapprox(LinearAlgebra.dot(a, b), LinearAlgebra.dot(A, B)) "Failed Transpose dims $n $m"
    end
    return true
end

function testComposition(f::LinearMap, g::LinearMap, num::Integer)
    n, m = size(f, 2), size(f, 1)
    for _ in 1:num
        a = rand(n)
        A = f(a)
        AA = f'(A)
        C = g(a)
        @assert isapprox(C, AA) "Failed Composition dims $n $m"
    end
    return true
end

function testGlobalOpChisel(Ω::TransverseOps, ch::Matrix, num::Integer; broken::Bool=false)
    @testset "Testing with $ch" begin
        for _ in 1:num
            Γ = random_itensor(frames(Ω))
            derdensor_map, densor_map = sylvesterLM(Ω, ch, Γ)
            nsamp = adaptive_samples(densor_map)
            if broken
                # See the KNOWN GAP note at the top of this file.
                @test_broken testTranspose(densor_map, nsamp)
                @test_broken testComposition(densor_map, derdensor_map, nsamp)
            else
                @test testTranspose(densor_map, nsamp)
                @test testComposition(densor_map, derdensor_map, nsamp)
            end
        end
    end
end

function testGlobalOp(Ω::TransverseOps; ntimes::Integer=5, broken::Bool=false)
    d = globalDim(Ω)
    @testset "Testing Ω with dim $d" begin
        for (name, build) in (
            ("Tucker", TuckerChisel),
            ("Derivation", UniversalChisel),
            ("Centroid", CentroidChisel),
        )
            @testset "Random $name chisels" begin
                for _ in 1:ntimes
                    eng = [rand(1:10) % 2 == 1 for _ in 1:valency(Ω)]
                    # CentroidChisel needs at least two engaged axes to be nonempty.
                    needed = name == "Centroid" ? 2 : 1
                    if sum(eng) >= needed
                        testGlobalOpChisel(Ω, build(eng), ntimes; broken=broken)
                    end
                end
            end
        end
    end
end

@testset "SylverLining Independent Tests" begin
    for val in 3:5
        @testset "Valency $val Tests" begin
            for _ in 1:5
                axisdim = rand(2:10, val)
                frame = axisdim .|> (i -> Index(i, "dim $i"))
                localops = rand(1:length(LΩs), val) .|> (i -> LΩs[i])
                testGlobalOp(IndTransverseOps(frame, localops))
            end
        end
    end
end

@testset "SylverLining Trivial Symmetry Tests" begin
    for val in 3:5
        @testset "Valency $val Tests" begin
            for _ in 1:5
                axisdim = rand(2:10, val)
                frame = axisdim .|> (i -> Index(i, "dim $i"))
                localops = rand(1:length(LΩs), val) .|> (i -> LΩs[i])
                testGlobalOp(TransverseOpsSymmetries(frame, localops, [i for i in 1:val]);
                             broken=true)   # KNOWN GAP, see header
            end
        end
    end
end

@testset "SylverLining Symmetry Tests" begin
    val = 10   # must be at least maximum(abs, sym) over the syms above
    for sym in syms
        @testset "Symmetry $sym" begin
            for _ in 1:2
                axisdim = rand(2:3, val)
                frame = axisdim .|> (i -> Index(i, "dim $i"))
                localops = rand(1:length(LΩs), val) .|> (i -> LΩs[i])
                Ω = TransverseOpsSymmetries(
                    [prime(frame[abs(sym[i])], i) for i in 1:length(sym)],
                    [localops[abs(sym[i])] for i in 1:length(sym)],
                    sym)
                testGlobalOp(Ω; ntimes=2, broken=true)   # KNOWN GAP, see header
            end
        end
    end
end
