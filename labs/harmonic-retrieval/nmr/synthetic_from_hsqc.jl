# Synthetic HSQC sub-band with the cyclosporin three-peak parameters: does Dleto
# resolve the 81 Hz 13C pair once the indirect record is long enough, and at
# what SNR?  Long axis t1 is Hankelized (j + k = t1), short decimated t2 is the
# third mode.  Run: julia -t 4 --project=. labs/harmonic-retrieval/nmr/synthetic_from_hsqc.jl
using Dleto, ITensors, LinearAlgebra, Statistics, Printf, Random
sw1 = 20833.33; sw2 = 100.16; n2 = 8
F1 = [-1546.2, -1464.8, -2685.5]; F2 = [373.1 - 350, 325.6 - 350, 345.9 - 350]      # Hz, F2 relative to the band centre
amp = [124.5, 105.4, 9.8] .* exp.((2π * im) .* [0.1, 0.6, 0.3]); lw1 = 5.0; lw2 = 3.0  # linewidths (Hz) -> damping π·lw
fit_exponential(v) = (r = [v[i+1] / v[i] for i in 1:length(v)-1]; w = abs.(v[1:end-1]).^2; z = sum(w .* log.(r)) / sum(w); (-real(z), imag(z) / 2π))
tohz(f, sw) = (f - round(f)) * sw
function record(N1)
    t1 = (0:N1-1) ./ sw1; t2 = (0:n2-1) ./ sw2
    sum(amp[m] .* exp.((-π * lw1 + 2π * im * F1[m]) .* t1) * transpose(exp.((-π * lw2 + 2π * im * F2[m]) .* t2)) for m in 1:3)
end
function dleto(Y, M; tol_nd, tol_der, ntries = 8)
    N1 = size(Y, 1); K = N1 ÷ 2; J = N1 - K + 1
    T = [Y[j + k - 1, t] for j in 1:J, k in 1:K, t in 1:n2]; T ./= norm(T)
    fr = [Index(J, "j"), Index(K, "k"), Index(n2, "t2")]; Γ = ITensor(T, fr...)
    Δ, Es = nondeg(Γ; tol = tol_nd); r = [ITensors.dim(i) for i in inds(Δ)]
    best = nothing
    for _ in 1:ntries
        St = stratify(Δ; nd = 2M, tol = tol_der); Σ = Array(St.Σ, inds(St.Σ)...)
        v = sort(vec(abs2.(Σ)); rev = true); conc = sum(v[1:M]) / sum(v)
        (best === nothing || conc > best[1]) && (best = (conc, St, Σ))
    end
    conc, St, Σ = best
    top = CartesianIndices(Σ)[partialsortperm(vec(abs2.(Σ)), 1:M; rev = true)]
    Ems = Matrix{ComplexF64}[]; Xms = Matrix{ComplexF64}[]
    for a in 1:3
        E = only(filter(t -> hasind(t, fr[a]), Es)); e2 = only(filter(i -> i != fr[a], inds(E)))
        X = only(filter(t -> hasind(t, e2), St.Xs)); x2 = only(filter(i -> i != e2, inds(X)))
        push!(Ems, Array(E, fr[a], e2)); push!(Xms, Array(X, e2, x2))
    end
    factor(a, p) = (conj(Ems[a]) * transpose(inv(Xms[a])))[:, p]
    est = [(tohz(fit_exponential(factor(2, Tuple(ci)[2]))[2], sw1), tohz(fit_exponential(factor(3, Tuple(ci)[3]))[2], sw2)) for ci in top]
    return r, conc, est
end
function score(est)   # max error in F1 after best assignment; and whether the 81 Hz pair came out as two distinct components
    d(a, b) = abs(a - b); used = Set{Int}(); errs = Float64[]
    for (f1, f2) in est
        m = argmin([i in used ? Inf : d(f1, F1[i]) + d(f2, F2[i]) for i in 1:3]); push!(used, m); push!(errs, d(f1, F1[m]))
    end
    return maximum(errs)
end
Random.seed!(2)
println("pair separation 81 Hz; FFT bin at N1 = 128/256/512: ", round.(sw1 ./ (128, 256, 512); digits = 0), " Hz; cycles of relative phase over the record: ", round.(81 .* (128, 256, 512) ./ sw1; digits = 2))
for N1 in (128, 256, 512), snr in (Inf, 40, 30, 20, 15)
    Y0 = record(N1); noise = snr == Inf ? 0.0 : 10^(-snr / 20)
    Y = Y0 .+ (norm(Y0) / sqrt(length(Y0)) * noise) .* (randn(ComplexF64, size(Y0)...) ./ sqrt(2))
    try
        r, conc, est = dleto(Y, 3; tol_nd = max(1e-6, noise), tol_der = max(1e-6, 10noise))
        @printf("N1=%3d SNR %4s dB: nondeg %s top-3 mass %.3f | 13C estimates %s | max 13C error %6.1f Hz\n", N1, string(snr), r, conc, join([@sprintf("%8.1f", e[1]) for e in sort(est; by = first)], " "), score(est))
    catch e; println("N1=$N1 SNR $snr: ", first(sprint(showerror, e), 100)); end
end
