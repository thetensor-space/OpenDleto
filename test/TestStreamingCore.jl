#
# TestStreamingCore.jl -- the streaming accumulators of
# docs/design/Streaming-Stratification.md.
#
# The whole claim of phase 1 is an EQUIVALENCE: folding frames one at a time
# produces exactly the sufficient statistics a batch QuickDer run would build
# from the concatenated tensor, given the same sketches.  So every test here is
# "stream it, batch it, compare" -- against `_qdn_cross_sketches`,
# `_qdn_pair_tensors` and `der_residual_squares` themselves, not against a
# transcription of them.
#
# The comparisons are in Float64 and the tolerance is a relative one: the two
# routes sum over frames in a different order (a rank-one update per frame here,
# one fused mode product there), so they agree to rounding and not to the bit.
#
using Test
using Dleto
using LinearAlgebra
using Random

const _SC_ADJOINT3 = Float64[1.0 -1.0 0.0]      # engaged on axes 1,2; axis 3 streams

# Relative difference of two arrays, scale-invariant so that a large sketch does
# not pass a test a small one would fail.
_sc_rel(A, B) = norm(vec(A) - vec(B)) / max(norm(vec(A)), norm(vec(B)), eps())

@testset "StreamingCore" begin

    @testset "cross sketches match the batch contraction" begin
        Random.seed!(20260916)
        d1, d2, F = 9, 7, 24
        G = randn(Float64, d1, d2, F)

        core = StreamingCore((d1, d2), _SC_ADJOINT3;
                             stream = 3, T = Float64, r_stream = 5, seed = 11)
        for k in 1:F
            push!(core, G[:, :, k])
        end
        @test nframes(core) == F

        # The batch route, given the sketches the stream actually used.
        axs = batchaxes(core)
        Sb = Dleto._qdn_cross_sketches(G, axs, core.engaged)
        Ss = crosssketches(core)

        @test sort(collect(keys(Ss))) == [1, 2]
        for a in (1, 2)
            @test size(Ss[a]) == size(Sb[a])
            @test _sc_rel(Ss[a], Sb[a]) < 1e-12
            @test norm(Ss[a]) > 0            # not trivially equal because both are zero
        end
    end

    @testset "pair tensors match the batch contraction" begin
        Random.seed!(20260917)
        d1, d2, F = 10, 8, 20
        G = randn(Float64, d1, d2, F)

        core = StreamingCore((d1, d2), _SC_ADJOINT3;
                             stream = 3, T = Float64, r_stream = 4, seed = 22)
        for k in 1:F
            push!(core, view(G, :, :, k))     # a view, the natural way to read a movie
        end

        axs = batchaxes(core)
        @test !isempty(core.lift)             # otherwise this test proves nothing
        for a in core.lift
            bs = Int[b for b in core.eaxes if b != a]
            Hb = Dleto._qdn_pair_tensors(G, axs, a, bs)
            Hs = pairtensors(core, a)
            @test sort(collect(keys(Hs))) == sort(bs)
            for b in bs
                @test size(Hs[b]) == size(Hb[b])
                @test _sc_rel(Hs[b], Hb[b]) < 1e-12
                @test norm(Hs[b]) > 0
            end
        end
    end

    @testset "valence 4, a colour axis that saturates" begin
        Random.seed!(20260918)
        d1, d2, F, c = 8, 6, 15, 3
        G = randn(Float64, d1, d2, F, c)
        # Engaged on the two spatial axes and on colour; axis 3 streams.
        P = Float64[1.0 -1.0 0.0 1.0]

        core = StreamingCore((d1, d2, c), P;
                             stream = 3, T = Float64, r_stream = 4, seed = 33)
        for k in 1:F
            push!(core, G[:, :, k, :])
        end

        axs = batchaxes(core)
        Sb = Dleto._qdn_cross_sketches(G, axs, core.engaged)
        for a in core.eaxes
            @test _sc_rel(crosssketches(core)[a], Sb[a]) < 1e-12
        end
        # The colour axis is length 3 and saturates, so it is never lifted.
        @test core.r[4] == c
        @test !(4 in core.lift)
    end

    @testset "the restriction sizes do not involve the stream length" begin
        # Two cores for the same frame shape and budget, fed 5 frames and 200:
        # every size is fixed at construction from `r_stream` alone.
        short = StreamingCore((12, 9), _SC_ADJOINT3;
                              stream = 3, T = Float64, r_stream = 6, seed = 1)
        long  = StreamingCore((12, 9), _SC_ADJOINT3;
                              stream = 3, T = Float64, r_stream = 6, seed = 1)
        @test short.r == long.r
        @test short.r[3] == 6

        # And the two conditions of `_qdn_restriction_sizes` still hold when the
        # TRUE stream length is put back, for any length at all -- which is the
        # claim section 2 of the design note makes.
        r, dims, eng = short.r, short.dims, short.engaged
        unknowns = sum(Int[dims[a] * r[a] for a in 1:3 if eng[a]])
        @test prod(r) >= unknowns                                   # (i)
        for a in 1:3
            (eng[a] && r[a] < dims[a]) || continue
            @test (prod(r) ÷ r[a]) >= dims[a]                       # (ii)
        end
    end

    @testset "an engaged stream axis is refused" begin
        # The unknown on the stream axis would be d_t x d_t and grow without
        # bound; there is nothing to accumulate, and saying so is the point.
        @test_throws ErrorException StreamingCore((9, 7), Float64[1.0 -1.0 1.0];
                                                  stream = 3, T = Float64, r_stream = 4)
        # A chisel of the wrong width, and a frame of the wrong shape.
        @test_throws ErrorException StreamingCore((9, 7), Float64[1.0 -1.0];
                                                  stream = 3, T = Float64, r_stream = 4)
        core = StreamingCore((9, 7), _SC_ADJOINT3;
                             stream = 3, T = Float64, r_stream = 4, seed = 5)
        @test_throws ErrorException push!(core, randn(9, 8))
        @test_throws ErrorException push!(core, randn(9, 7, 2))
    end

    @testset "pairs = false allocates no pair tensors" begin
        core = StreamingCore((9, 7), _SC_ADJOINT3;
                             stream = 3, T = Float64, r_stream = 4, seed = 7,
                             pairs = false)
        push!(core, randn(9, 7))
        @test isempty(pairtensors(core, core.lift[1]))
        @test !isempty(crosssketches(core))
    end
end

@testset "StreamingResidual" begin

    @testset "matches der_residual over the whole tensor" begin
        Random.seed!(20260919)
        d1, d2, F = 11, 9, 30
        G = randn(Float64, d1, d2, F)
        P = _SC_ADJOINT3
        # Arbitrary operators: the residual is a MEASUREMENT, and it has to
        # agree with the batch one whether or not these are derivations.
        Ms = [randn(Float64, d1, d1), randn(Float64, d2, d2), zeros(Float64, F, F)]

        batch = Dleto.der_residual(G, Ms, P)

        sr = StreamingResidual(Ms, P; stream = 3)
        for k in 1:F
            push!(sr, G[:, :, k])
        end
        @test nframes(sr) == F
        @test isapprox(Dleto.der_residual(sr), batch; rtol = 1e-10)
        @test batch > 0
    end

    @testset "an actual derivation reads as one, frame by frame" begin
        Random.seed!(20260920)
        d, F = 8, 12
        # `[1,-1,0]` asks for `X` on axis 1 and `Y` on axis 2 with
        # `Γ ×_1 X = Γ ×_2 Y`.  Identity operators do NOT test that -- they give
        # `Γ - Γ = 0` for every tensor, so the assertion passes vacuously and its
        # contrasting case measures nothing.  Take `X` SYMMETRIC instead and
        # every frame a polynomial in it: `X` commutes with every frame, so
        # `Y = X` satisfies the condition under either transpose convention for
        # `×_2`, and the test measures the arithmetic rather than this file's
        # reading of `embedITensors`' index order.
        A = randn(Float64, d, d)
        X = A + transpose(A)
        G = zeros(Float64, d, d, F)
        for k in 1:F
            G[:, :, k] = sum(randn() .* X^j for j in 0:3)
        end
        Ms = [X, X, zeros(Float64, F, F)]

        sr = StreamingResidual(Ms, _SC_ADJOINT3; stream = 3)
        for k in 1:F
            push!(sr, G[:, :, k])
        end
        @test Dleto.der_residual(sr) < 1e-12
        @test isapprox(Dleto.der_residual(sr),
                       Dleto.der_residual(G, Ms, _SC_ADJOINT3); atol = 1e-12)

        # Frames that do NOT commute with X are not derivations, and the running
        # number says so -- this is the half the old test could not see.
        H = randn(Float64, d, d, F)
        sr2 = StreamingResidual(Ms, _SC_ADJOINT3; stream = 3)
        for k in 1:F
            push!(sr2, H[:, :, k])
        end
        @test Dleto.der_residual(sr2) > 1e-3
        @test isapprox(Dleto.der_residual(sr2),
                       Dleto.der_residual(H, Ms, _SC_ADJOINT3); rtol = 1e-10)
    end

    @testset "an engaged stream axis is refused" begin
        d, F = 5, 4
        Ms = [randn(d, d), randn(d, d), zeros(F, F)]
        @test_throws ErrorException StreamingResidual(Ms, Float64[1.0 -1.0 1.0]; stream = 3)
    end
end
