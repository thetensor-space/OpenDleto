# Synthetic sanity check: can Dleto's derivation path recover non-uniform,
# clustered frequencies from a short uniformly sampled 3-way record where a
# 16-point FFT cannot resolve them?
#
#   T[i,j,k] = Σ_m c_m · exp((−α_m + 2πi f^x_m) i) · exp((−β_m + 2πi f^y_m) j) · exp((−γ_m + 2πi f^z_m) k)
#
# Run: julia -t 4 --project=. labs/harmonic-retrieval/synthetic_check.jl
using Dleto, ITensors, LinearAlgebra, Statistics, Printf, Random, FFTW
using Plots; gr()
const OUT = joinpath(@__DIR__, "results"); const FIG = joinpath(@__DIR__, "figures")
n = isempty(ARGS) ? 16 : parse(Int, ARGS[1]); M = 5
sep = length(ARGS) < 2 ? 0.02 : parse(Float64, ARGS[2])      # spacing of the clustered pairs, cycles/sample
println("n = ", n, "  cluster spacing = ", sep, "  (", round(sep * n; digits = 2), " FFT bins)")
# frequencies in cycles per sample; FFT resolution at n = 16 is 1/16 = 0.0625
fx = [0.100, 0.100 + sep, 0.300, 0.300 + sep, 0.450]     # two clustered pairs, sep apart
fy = [0.210, 0.210 + sep, 0.070, 0.390, 0.390 + sep / 4]  # pairs (1,2) and (4,5) clustered
fz = [0.150, 0.330, 0.150 + sep / 2, 0.050, 0.250]       # pair (1,3) clustered
damp = fill(0.02, M)
amps = [1.0, 0.8, 0.9, 0.7, 1.0] .* exp.((2π * im) .* [0.1, 0.7, 0.3, 0.9, 0.5])
expo(f, a) = [exp((-a + (2π * im) * f) * t) for t in 0:n-1]
A = hcat([expo(fx[m], damp[m]) for m in 1:M]...); B = hcat([expo(fy[m], damp[m]) for m in 1:M]...); C = hcat([expo(fz[m], damp[m]) for m in 1:M]...)
T0 = zeros(ComplexF64, n, n, n)
for m in 1:M, i in 1:n, j in 1:n, k in 1:n; T0[i, j, k] += amps[m] * A[i, m] * B[j, m] * C[k, m]; end

"Estimate (damping, frequency) of a sampled exponential by least squares on log ratios."
function fit_exponential(v::AbstractVector)
    r = [v[i+1] / v[i] for i in 1:length(v)-1]
    w = abs.(v[1:end-1]).^2                         # weight by amplitude
    z = sum(w .* log.(r)) / sum(w)
    return (-real(z), mod(imag(z) / 2π, 1.0))
end
"Frequencies the FFT resolves along one axis: local maxima of the 16-point spectrum of the dominant mode of the unfolding."
function fft_peaks(T, axis)
    U = reshape(permutedims(T, (axis, setdiff(1:3, axis)...)), n, :)
    u = svd(U).U[:, 1]                               # dominant mode along this axis (the generous choice for the FFT)
    S = abs.(fft(u)); S = S[1:n]
    pk = [i for i in 1:n if S[i] > S[mod1(i-1, n)] && S[i] >= S[mod1(i+1, n)] && S[i] > 0.1 * maximum(S)]
    return sort((pk .- 1) ./ n), S
end
"Dleto path: nondeg → stratify along all derivations → factor vectors and their frequencies."
function dleto_recover(T; tol_nd, nd, label, io, tol_der = 1e-6)
    fr = [Index(n, s) for s in ("x", "y", "z")]
    Γ = ITensor(T ./ norm(T), fr...)
    Δ, Es = nondeg(Γ; tol = tol_nd)
    r = [ITensors.dim(i) for i in inds(Δ)]
    minimum(r) < 2 && error("nondeg truncated an axis to dimension < 2")
    nd == 0 && (nd = 2 * minimum(r))          # CP model with M terms: 2M derivations including the two scalars
    ders = der(Δ; nd = nd, tol = tol_der)
    St = stratify(Δ; nd = nd, tol = tol_der)
    Σ = Array(St.Σ, inds(St.Σ)...)
    # per-axis: nondeg basis E (d × r, enters conjugated: Δ = Γ * E = Wᴴ Γ with W = conj(E)) and stratify transform X (r × r)
    Ems = Matrix{ComplexF64}[]; Xms = Matrix{ComplexF64}[]
    for a in 1:3
        E = only(filter(t -> hasind(t, fr[a]), Es)); e2 = only(filter(i -> i != fr[a], inds(E)))
        X = only(filter(t -> hasind(t, e2), St.Xs)); x2 = only(filter(i -> i != e2, inds(X)))
        push!(Ems, Array(E, fr[a], e2)); push!(Xms, Array(X, e2, x2))
    end
    mats = [Ems[a] * Xms[a] for a in 1:3]
    # dominant entries of Σ: do they form a permutation pattern (one per index on each axis)?
    v = sort(vec(abs2.(Σ)); rev = true); conc = sum(v[1:M]) / sum(v)
    top = partialsortperm(vec(abs2.(Σ)), 1:M; rev = true); cis = CartesianIndices(Σ)[top]
    perm_ok = all(allunique(getindex.(Tuple.(cis), a)) for a in 1:3)
    # factor vectors: Σ = T ×ₐ (E X) bilinearly and the factors lie in range(W) = range(conj E), so with
    # Eᵀ conj(E) = I the factor for stratified index p on axis a is column p of conj(E) · X⁻ᵀ
    recovered = [[fit_exponential((conj(Ems[a]) * transpose(inv(Xms[a])))[:, Tuple(ci)[a]]) for ci in cis] for a in 1:3]
    @printf(io, "%-22s nondeg dims %s | derivations found %d | mass of top-%d entries of Σ %.3f, permutation pattern %s | cond %s\n",
            label, r, length(ders), M, conc, perm_ok, join([@sprintf("%.1e", cond(m)) for m in mats], " "))
    return (; recovered, conc, perm_ok, r, nders = length(ders))
end
function match_error(rec, truth)   # best assignment of recovered to true frequencies (greedy on circular distance)
    d(a, b) = min(abs(a - b), 1 - abs(a - b))
    errs = Float64[]; used = Set{Int}()
    for f in rec
        j = argmin([i in used ? Inf : d(f, truth[i]) for i in eachindex(truth)]); push!(used, j); push!(errs, d(f, truth[j]))
    end
    return maximum(errs)
end
io = open(joinpath(OUT, "synthetic_check-n$(n)-sep$(sep).log"), "w")
println(io, "true frequencies (cycles/sample): x ", fx, " y ", fy, " z ", fz, "; damping ", damp[1], "; n = ", n, "; FFT resolution ", 1/n)
for (a, truth) in zip(("x", "y", "z"), (fx, fy, fz))
    pk, _ = fft_peaks(T0, a == "x" ? 1 : a == "y" ? 2 : 3)
    @printf(io, "FFT along %s (noise-free, dominant mode): %d peaks at %s  (truth has %d distinct, clustered pairs unresolved)\n", a, length(pk), join([@sprintf("%.3f", p) for p in pk], " "), length(unique(truth)))
end
Random.seed!(1)
rows = []
for snr_db in (Inf, 50, 40, 30, 20, 10)
    T = snr_db == Inf ? copy(T0) : T0 .+ (norm(T0) / sqrt(length(T0)) * 10^(-snr_db / 20)) .* (randn(ComplexF64, size(T0)...) ./ sqrt(2))
    noise = snr_db == Inf ? 0.0 : 10^(-snr_db / 20)
    for (tol_nd, nd, tol_der) in ((1e-6, -1, 1e-6), (max(1e-6, 3noise), 0, max(1e-6, 10noise)), (max(1e-6, noise), 0, max(1e-6, 10noise)))
        snr_db == Inf && tol_nd != 1e-6 && continue
        label = @sprintf("SNR %s dB, nondeg %.0e, nd %d, tol %.0e", snr_db == Inf ? "inf" : string(snr_db), tol_nd, nd, tol_der)
        try
            res = dleto_recover(T; tol_nd, nd, label, io, tol_der)
            errs = [match_error([r[2] for r in res.recovered[a]], truth) for (a, truth) in enumerate((fx, fy, fz))]
            println(io, @sprintf("   max frequency error (cycles/sample): x %.4f  y %.4f  z %.4f", errs...), "   (cluster spacing ", sep, ", FFT bin ", round(1/n; digits=4), ")")
            push!(rows, (; snr_db, tol_nd, nd, errs, res))
        catch e
            @printf(io, "%-22s FAILED: %s\n", label, first(sprint(showerror, e), 160))
        end
    end
end
close(io); println(read(joinpath(OUT, "synthetic_check-n$(n)-sep$(sep).log"), String))
# figure: FFT spectrum along x vs truth vs Dleto (noise-free and 20 dB)
_, S = fft_peaks(T0, 1)
p = plot((0:n-1) ./ n, S ./ maximum(S); seriestype = :sticks, lw = 3, label = "$(n)-point FFT (dominant x-mode)", xlabel = "frequency (cycles/sample)", ylabel = "normalised", xlims = (0, 0.5), size = (900, 380), title = "Clustered frequencies, n = $(n): FFT vs Dleto")
vline!(p, fx; label = "true frequencies", ls = :dash, lc = :black)
for (lab, i) in (("Dleto, noise-free", 1), ("Dleto, 40 dB", findfirst(r -> r.snr_db == 40 && r.nd == 0 && r.tol_nd > 1e-6, rows)))
    i === nothing && continue
    scatter!(p, [r[2] for r in rows[i].res.recovered[1]], fill(0.05 + 0.1 * (i == 1 ? 1 : 2), M); label = lab, markersize = 6)
end
savefig(p, joinpath(FIG, "synthetic-x-axis-n$(n)-sep$(sep).png")); println("figure saved")
