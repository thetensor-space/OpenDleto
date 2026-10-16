# Dleto harmonic retrieval on the decimated NMR sub-band: Hankel third mode.
#   T[t1, j, k] = fid[t1, j + k],  j = 1..J, k = 1..K,  J + K - 1 = n2
# Run: julia -t 4 --project=. labs/harmonic-retrieval/nmr/dleto_subband.jl
using Dleto, ITensors, HDF5, LinearAlgebra, Statistics, Printf, Random
path = ARGS[1]; Kgiven = parse(Int, ARGS[2]); N1s = parse.(Int, split(ARGS[3], ",")); Ms = parse.(Int, split(ARGS[4], ","))
fid, ref, sw1, sw2, f1c, f2c = h5open(path, "r") do h
    read(h["fid"]), read(h["ref_peaks"]), read_attribute(h, "sw1"), read_attribute(h, "sw2"), read_attribute(h, "f1_center"), read_attribute(h, "f2_center")
end
fid = permutedims(fid)                     # h5py wrote (n1, n2) row-major → Julia reads (n2, n1); transpose back
ref = permutedims(ref)                     # rows: F1 Hz, F2 Hz, height
n1full, n2 = size(fid); K = Kgiven; J = n2 - K + 1
println("sub-band FID ", size(fid), "  sw1 ", sw1, " Hz  sw2 ", sw2, " Hz;  reference peaks (F1, F2 Hz): ", [(round(ref[r, 1]), round(ref[r, 2])) for r in 1:size(ref, 1)])
fit_exponential(v) = (r = [v[i+1] / v[i] for i in 1:length(v)-1]; w = abs.(v[1:end-1]).^2; z = sum(w .* log.(r)) / sum(w); (-real(z), imag(z) / 2π))
tohz(f, sw, c) = (fw = f - round(f); fw * sw + c)       # cycles/sample → Hz offset (wrapped to ±sw/2) + band centre
function run(N1, M; tol_nd, tol_der, ntries = 12)
    T = [fid[t, j + k - 1] for t in 1:N1, j in 1:J, k in 1:K]; T ./= norm(T)
    fr = [Index(N1, "t1"), Index(J, "j"), Index(K, "k")]; Γ = ITensor(T, fr...)
    Δ, Es = nondeg(Γ; tol = tol_nd); r = [ITensors.dim(i) for i in inds(Δ)]
    # several random derivation combinations; keep the one with the sparsest strata (fewest entries above 10% of max)
    best = nothing
    for trial in 1:ntries
        St_t = stratify(Δ; nd = 2M, tol = tol_der); Σ_t = Array(St_t.Σ, inds(St_t.Σ)...)
        nz_t = count(x -> abs(x) > 0.1 * maximum(abs, Σ_t), Σ_t)
        (best === nothing || nz_t < best[1]) && (best = (nz_t, St_t, Σ_t))
    end
    St = best[2]
    # nondeg orders axes by size, so read Σ back in the frame order (t1, j, k)
    Ems = Matrix{ComplexF64}[]; Xms = Matrix{ComplexF64}[]; e2s = Index[]
    for a in 1:3
        E = only(filter(t -> hasind(t, fr[a]), Es)); e2 = only(filter(i -> i != fr[a], inds(E)))
        X = only(filter(t -> hasind(t, e2), St.Xs)); x2 = only(filter(i -> i != e2, inds(X)))
        push!(Ems, Array(E, fr[a], e2)); push!(Xms, Array(X, e2, x2)); push!(e2s, e2)
    end
    Σ = Array(St.Σ, e2s...)
    nz_orig = count(x -> abs(x) > 0.1 * maximum(abs, T), T); nz_strat = count(x -> abs(x) > 0.1 * maximum(abs, Σ), Σ)
    v = sort(vec(abs2.(Σ)); rev = true); conc = sum(v[1:M]) / sum(v)
    top = partialsortperm(vec(abs2.(Σ)), 1:M; rev = true); cis = CartesianIndices(Σ)[top]
    factor(a, p) = (conj(Ems[a]) * transpose(inv(Xms[a])))[:, p]
    @printf("N1=%2d M=%d: nondeg dims %s, strata nonzeros (>10%% of max) %d of %d (original %d of %d), top-%d mass %.3f\n", N1, M, r, nz_strat, length(Σ), nz_orig, length(T), M, conc)
    for ci in cis
        p = Tuple(ci)
        d1, f1 = fit_exponential(factor(1, p[1])); dj, fj = fit_exponential(factor(2, p[2])); dk, fk = fit_exponential(factor(3, p[3]))
        F1 = tohz(f1, sw1, f1c); F2k = tohz(fk, sw2, f2c); F2j = tohz(fj, sw2, f2c)
        d = [hypot((F1 - ref[m, 1]) / sw1, (F2k - ref[m, 2]) / sw2) for m in 1:size(ref, 1)]; m = argmin(d)
        @printf("   component at Σ%s: F1 %7.1f Hz, F2 %7.1f Hz (j-axis check %7.1f), damping (t1 %.3f, t2 %.3f) | nearest reference (%7.1f, %7.1f): ΔF1 %6.1f ΔF2 %6.1f Hz\n", p, F1, F2k, F2j, d1, dk, ref[m, 1], ref[m, 2], F1 - ref[m, 1], F2k - ref[m, 2])
    end
end
# singular values of the unfoldings, to choose the rank
T = [fid[t, j + k - 1] for t in 1:n1full, j in 1:J, k in 1:K]; T ./= norm(T)
for (a, name) in enumerate(("t1", "j", "k"))
    U = reshape(permutedims(T, (a, setdiff(1:3, a)...)), size(T, a), :); s = svdvals(U); s ./= s[1]
    println("axis $name singular values: ", join([@sprintf("%.3f", x) for x in s[1:min(8, length(s))]], " "))
end
Random.seed!(1)
for N1 in N1s, M in Ms
    try; run(N1, M; tol_nd = 0.05, tol_der = 0.3); catch e; println("N1=$N1 M=$M failed: ", first(sprint(showerror, e), 120)); end
end
println("FFT resolution of the short records: F1 ", round.(sw1 ./ N1s; digits = 0), " Hz; F2 ", round(sw2 / n2; digits = 0), " Hz")
