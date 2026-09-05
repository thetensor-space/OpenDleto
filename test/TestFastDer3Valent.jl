using LinearAlgebra
using Random

@testset "FastDer3Valent Derivation Method" begin
    Random.seed!(7)

    dims = (4, 4, 4)
    Γ_array = randn(Float64, dims...)
    frame = [Index(dims[i], "a_$i") for i in 1:3]
    Γ = ITensor(Γ_array, frame...)

    Ω = IndTransverseOps(frame, UniversalOp())
    ch = UniversalChisel(3)

    @testset "derTrOpsReduced returns basis coordinates" begin
        method = FastDer3ValentMethod()
        rΩ, expand_map, ders = Dleto.derTrOpsReduced(method, Ω, ch, Γ; tol=1e-6)

        @test rΩ === Ω
        @test size(ders, 1) == globalDim(Ω)
        @test size(ders, 2) >= 1

        # Ensure expanded vectors embed as three derivation matrices.
        v = expand_map(ders[:, 1])
        δ = embedITensors(Ω, v)
        @test length(δ) == 3
    end

    # --- the cut inside `_fastder_restrict_to_ops`, on a built spectrum -----
    #
    # The Float32 undercount lived here.  `_fastder_tall_nullspace` used to keep
    # exactly the singular values `<= atol`, and `atol` is `qd_tolerance(T)` --
    # `sqrt(eps(Float32))` = 3.45e-4 -- which is also the noise level of the
    # Float32 basis whose residual this matrix is.  So the cutoff sat inside the
    # answer.  Measured on the scrambled sphere v3 at d = 48 in Float32: the
    # residual spectrum was
    #   [1.71, 1.29, ..., 0.244 | 4.486e-4, 2.011e-4, 1.157e-4]
    # -- three genuine directions separated from the rest by 544x, but the
    # first of them 1.30 times `atol`, so 13 lifted derivations went in and 2
    # came out of a true 3.  The values below are those, rounded.
    @testset "restrict_to_ops cuts on a gap, not on atol" begin
        atol32 = Dleto.qd_tolerance(Float32)          # sqrt(eps(Float32))
        @test atol32 ≈ sqrt(eps(Float32))

        # A tall matrix with a prescribed singular spectrum: `Q * S * V'` for
        # orthonormal `Q` (300 x 13) and `V` (13 x 13), so the singular values
        # are exactly the diagonal and the right singular vectors are `V`.
        function tall_with_svals(svals::Vector{Float64}; m::Int = 300, seed::Int = 3)
            rng = MersenneTwister(seed)
            k = length(svals)
            Q = Matrix(qr(randn(rng, m, k)).Q)
            V = Matrix(qr(randn(rng, k, k)).Q)
            return Q * Diagonal(svals) * transpose(V)
        end

        measured = vcat([1.709, 1.291, 0.7792, 0.5897, 0.5593, 0.4716, 0.4201,
                         0.3289, 0.2683, 0.244],
                        [4.486e-4, 2.011e-4, 1.157e-4])
        A = tall_with_svals(measured)
        # THE FIX: the gap at 544x decides, and all three genuine directions
        # come back even though the largest of them is 1.3 times `atol`.
        @test size(Dleto._fastder_tall_nullspace(A, atol32), 2) == 3
        # ... and the old rule is what it replaces.
        @test count(<=(atol32), measured) == 2

        # NO GAP -> the old absolute count, unchanged.  This is the raw sphere's
        # nullity-13 shape: every direction lies in Ω, so the whole residual
        # matrix is roundoff and there is nothing to split.
        flat = [1.7e-6, 1.4e-6, 1.1e-6, 9e-7, 8e-7, 6e-7, 5e-7, 4e-7,
                3e-7, 2.5e-7, 2e-7, 1.5e-7, 1e-7]
        @test size(Dleto._fastder_tall_nullspace(tall_with_svals(flat), atol32), 2) ==
              length(flat)

        # Float64 is untouched: the null cluster sits nine decades under
        # `atol` = 1e-6 and both rules give the same answer.
        atol64 = Dleto.qd_tolerance(Float64)
        @test atol64 == 1e-6
        f64 = vcat([1.38, 1.05, 0.879, 0.82, 0.704, 0.677, 0.51, 0.464, 0.262, 0.177],
                   [9.99e-13, 1.25e-13])
        @test size(Dleto._fastder_tall_nullspace(tall_with_svals(f64), atol64), 2) ==
              count(<=(atol64), f64) == 2

        # NO GAP CLEARS -> the OLD absolute count, not the count below the
        # ceiling.  Here 1e-3 sits between `atol` (3.45e-4) and the ceiling
        # (1.1e-2), so the eligible cuts run to 3, but the largest ratio among
        # them is 20x -- under `GAP_RATIO`.  Without the fallback the looser
        # ceiling would quietly widen every uncertain answer by one; with it,
        # the gap can only refine and never invent.
        nogap = vcat([1.0, 0.9, 0.5, 0.02], [1e-3], [2e-7, 1e-7])
        @test count(<=(atol32), nogap) == 2                     # the old count
        @test count(<(32 * atol32), nogap) == 3                 # the ceiling's
        @test size(Dleto._fastder_tall_nullspace(tall_with_svals(nogap), atol32), 2) == 2
    end

    @testset "stratify supports solver selection" begin
        fast = stratify(Γ; method=:FastDer3Valent, tol=1e-6)
        base = stratify(Γ; method=:SylverLining, tol=1e-6)

        @test ndims(Array(fast.Σ, inds(fast.Σ)...)) == 3
        @test ndims(Array(base.Σ, inds(base.Σ)...)) == 3
        @test length(fast.Xs) == 3
        @test length(base.Xs) == 3
    end
end
