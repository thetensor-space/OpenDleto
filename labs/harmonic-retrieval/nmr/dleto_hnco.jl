# Dleto on a true 3-way NMR record: the (15N, 13C', 1H) sub-band of the BMRB 15063 HNCO.
# Run: julia -t 4 --project=. labs/harmonic-retrieval/nmr/dleto_hnco.jl <subband.h5> <NC list> <M list>
using Dleto, ITensors, HDF5, LinearAlgebra, Statistics, Printf, Random
using Plots; gr()
include(joinpath(@__DIR__, "..", "HarmonicRetrieval.jl")); using .HarmonicRetrieval
path = ARGS[1]; NCs = parse.(Int, split(ARGS[2], ",")); Ms = parse.(Int, split(ARGS[3], ","))
fid, ref, swN, swC, swH, obsN, obsC, obsH, fcH = h5open(path, "r") do h
    read(h["fid"]), read(h["ref_peaks"]), (read_attribute(h, k) for k in ("swN", "swC", "swH", "obsN", "obsC", "obsH", "fcH"))...
end
fid = permutedims(fid, (3, 2, 1)); ref = permutedims(ref)          # h5py row-major -> (15N, 13C', 1H)
nN, nC, nH = size(fid)
println("sub-band record (15N, 13C', 1H) = ", size(fid), "; FFT bins: 15N ", round(swN / nN / obsN; digits = 2), " ppm, 13C' ", round(swC / nC / obsC; digits = 2), " ppm at ", nC, " increments")
strong = ref[ref[:, 4] .> 100, :]
println("reference peaks (15N, 13C', 1H ppm from carriers): ", [(round(r[1] / obsN; digits = 2), round(r[2] / obsC; digits = 2), round(r[3] / obsH; digits = 3)) for r in eachrow(strong)])
tohz(f, sw) = (f - round(f)) * sw
Random.seed!(1)
for NC in NCs, M in Ms
    T = fid[:, 1:NC, :]
    try
        Tnorm = T / norm(T)
        fr = [Index(size(Tnorm, a), ("a", "b", "c")[a]) for a in 1:3]
        Γ = ITensor(Tnorm, fr...)
        Δit, Es = nondeg(Γ; tol = 0.05)
        ch = UniversalChisel(length(inds(Δit)))
        Ω = IndTransverseOps(collect(inds(Δit)), UniversalOp())
        ders = der(Ω, ch, Δit; nd = 2M, tol = 0.3)

        best = nothing
        for _ in 1:10
            coefs = randn(length(ders))
            δ = [sum(coefs[j] * ders[j][a] for j in eachindex(ders)) for a in 1:3]
            St_try = stratify(Δit, δ)
            Σ_try = Array(St_try.Σ, inds(St_try.Σ)...)
            nz_try = count(x -> abs(x) > 0.1 * maximum(abs, Σ_try), Σ_try)
            if best === nothing || nz_try < best.nonzeros
                best = (; nonzeros = nz_try, St = St_try)
            end
        end
        St = best.St

        Ems = Matrix{ComplexF64}[]
        Xms = Matrix{ComplexF64}[]
        e2s = Index[]
        for a in 1:3
            E = only(filter(t -> hasind(t, fr[a]), Es))
            e2 = only(filter(i -> i != fr[a], inds(E)))
            X = only(filter(t -> hasind(t, e2), St.Xs))
            x2 = only(filter(i -> i != e2, inds(X)))
            push!(Ems, Array(E, fr[a], e2))
            push!(Xms, Array(X, e2, x2))
            push!(e2s, e2)
        end

        Δ = Array(Δit, inds(Δit)...)
        Σ = Array(St.Σ, e2s...)
        nonzeros = count(x -> abs(x) > 0.1 * maximum(abs, Σ), Σ)
        record_nonzeros = count(x -> abs(x) > 0.1 * maximum(abs, T), T)
        top = CartesianIndices(Σ)[partialsortperm(vec(abs2.(Σ)), 1:M; rev = true)]
        factors = [[(conj(Ems[a]) * transpose(inv(Xms[a])))[:, Tuple(ci)[a]] for a in 1:3] for ci in top]

        @printf("\n13C' increments %2d (bin %.2f ppm), M = %d: nondeg %s | strata nonzeros (>10%% of max) %d of %d vs record %d of %d\n", NC, swC / NC / obsC, M, size(Δ), nonzeros, length(Σ), record_nonzeros, length(T))
        for f in factors
            fN = tohz(fit_exponential(f[1])[2], swN) / obsN; fC = tohz(fit_exponential(f[2])[2], swC) / obsC; fH = (tohz(fit_exponential(f[3])[2], swH) + fcH) / obsH
            d = [hypot((fN - s[1] / obsN) / 0.86, (fC - s[2] / obsC) / 0.62, (fH - s[3] / obsH) / 0.05) for s in eachrow(strong)]; m = argmin(d)
            @printf("   component: 15N %6.2f  13C' %6.2f  1H %6.3f ppm | nearest reference 15N %6.2f 13C' %6.2f 1H %6.3f | Δ13C' %+.2f ppm (%.0f Hz)\n", fN, fC, fH, strong[m, 1] / obsN, strong[m, 2] / obsC, strong[m, 3] / obsH, fC - strong[m, 2] / obsC, (fC - strong[m, 2] / obsC) * obsC)
        end
        if NC == NCs[end] && M == Ms[1]
            A = abs.(T) ./ maximum(abs, T); S = abs.(Σ) ./ maximum(abs, Σ)
            p = plot(plot_tensor(A, 0.1; title = "record |T| (15N × 13C' × 1H), $(nC)×$(NC)×$(nH)", xlabel = "15N", ylabel = "13C'", zlabel = "1H"),
                     plot_tensor(S, 0.1; title = "strata |Σ| after nondeg + stratify", xlabel = "a", ylabel = "b", zlabel = "c"); layout = (1, 2), size = (1200, 520))
            savefig(p, joinpath(@__DIR__, "..", "figures", "hnco-pre-post-NC$(NC).png"))
        end
    catch e; println("NC=$NC M=$M failed: ", first(sprint(showerror, e), 160)); end
end
