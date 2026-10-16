# Shared analysis code for the measured-grid stratification pilot.
#
# Method: Dleto's universal linear chisel (the all-plus relation D1 ⊕ D2 ⊕ D3
# annihilating the tensor) with the universal operator space.  The derivation
# operator is built by Dleto.sylvesterLM; here we form it densely (the pilots
# are small) so the whole singular spectrum is available, and we follow Dleto's
# own `stratify(Γ, δ)` to turn a chosen derivation into per-axis transforms.
#
# Run with:  julia -t 4 --project=. labs/measured-grids/run_<dataset>.jl
using Dleto, ITensors, LinearAlgebra, Statistics, Printf, Random, FFTW, Dates
using Plots; gr()

const RESULTS = joinpath(@__DIR__, "results")
const FIGURES = joinpath(@__DIR__, "figures")

"Build the dense derivation operator for the universal chisel. Returns the
symmetric form L = AᵀA (unknowns × unknowns), the operator space, and the map
from reduced coordinates to full coordinates (identity for the universal chisel)."
function derivation_operator(Γ::ITensor; ops = UniversalOp())
    ch = UniversalChisel(ndims(Γ))
    fr = collect(inds(Γ))
    Ω = IndTransverseOps(fr, ops)
    eng = engaged(ch)
    Ωr, expand_map = reduceByEngaged(Ω, eng, eltype(Γ))
    Lmap, _ = Dleto.sylvesterLM(Ωr, Matrix{eltype(Γ)}(ch[:, eng]), Γ)
    L = Symmetric(Matrix(Lmap))
    return (; L, Ω, Ωr, expand_map, ch)
end

"Eigen-decomposition of L; singular values of the derivation operator A are
sqrt of the eigenvalues. Returned ascending, relative to the largest."
function derivation_spectrum(L::Symmetric)
    E = eigen(L)
    λ = max.(E.values, 0.0)
    σ = sqrt.(λ)
    return (; σ = σ ./ σ[end], σabs = σ, vecs = E.vectors)
end

"Embed a coordinate vector as per-axis matrices (ITensors)."
embed(op, v::AbstractVector) = embedITensors(op.Ω, op.expand_map(v))

"How close a derivation is to a scalar triple: 1.0 means each axis matrix is a
multiple of the identity."
function scalar_score(δ::Vector{ITensor})
    s = 0.0
    for X in δ
        M = Array(X, inds(X)...); n = size(M, 1)
        c = tr(M) / n
        s += norm(M - c * I)^2 / max(norm(M)^2, eps())
    end
    return 1 - s / length(δ)
end

"Per-axis eigenvalue summary of a derivation: multiplicity structure matters
for the canonical form (repeated eigenvalues leave within-subspace freedom)."
function eigen_summary(δ::Vector{ITensor}; rtol = 1e-3)
    out = String[]
    for X in δ
        M = Array(X, inds(X)...)
        ev = eigvals(M)
        imagfrac = maximum(abs.(imag.(ev))) / max(maximum(abs.(ev)), eps())
        r = sort(real.(ev)); scale = max(maximum(abs.(r)), eps())
        # count clusters of nearly equal real parts
        clusters = 1
        for i in 2:length(r); (r[i] - r[i-1]) > rtol * scale && (clusters += 1); end
        push!(out, @sprintf("%d distinct of %d (imag/|λ| max %.1e)", clusters, length(r), imagfrac))
    end
    return out
end

"Stratify Γ along the derivation δ using Dleto's own routine; return the
result, the per-axis transforms as matrices, their inverses as ITensors, the
condition numbers, and the relative reconstruction error in original coordinates."
function stratify_along(Γ::ITensor, δ::Vector{ITensor}; orthogonal::Bool = false)
    if orthogonal
        # symmetric derivations: use the orthogonal eigenbasis directly, so the
        # learned coordinates are an orthonormal change of basis on each axis
        Xs = ITensor[]
        for X in δ
            M = Array(X, inds(X)...); asym = norm(M - M') / max(norm(M), eps())
            asym > 1e-8 && @warn "derivation matrix not symmetric (rel. asymmetry $asym); symmetrizing"
            Q = eigen(Symmetric((M + M') / 2)).vectors
            orig = only(filter(i -> hasind(Γ, i), inds(X))); temp = only(filter(i -> !hasind(Γ, i), inds(X)))
            push!(Xs, ITensor(Q, orig, temp))
        end
        Σ = Γ * Xs
        for X in Xs
            orig = only(filter(i -> hasind(Γ, i), inds(X))); temp = only(filter(i -> !hasind(Γ, i), inds(X)))
            Σ = replaceind(Σ, temp, orig)
        end
    else
        St = stratify(Γ, δ)
        Σ, Xs = St.Σ, St.Xs
    end
    mats = [Array(X, inds(X)...) for X in Xs]
    conds = [cond(M) for M in mats]
    invs = [ITensor(inv(mats[a]), reverse(collect(inds(Xs[a])))...) for a in 1:length(Xs)]
    # Σ lives on Γ's frame (stratify retags), the inverses map temp -> frame;
    # retag Σ back to the temporary frame to contract with the inverses.
    Σt = Σ
    for X in Xs
        orig = only(filter(i -> hasind(Γ, i), inds(X))); temp = only(filter(i -> !hasind(Γ, i), inds(X)))
        Σt = replaceind(Σt, orig, temp)
    end
    rec = Σt * invs
    relerr = norm(Array(rec, inds(Γ)...) - Array(Γ, inds(Γ)...)) / norm(Array(Γ, inds(Γ)...))
    return (; Σ, Σt, Xs, mats, invs, conds, relerr)
end

"Keep the k largest-magnitude entries of A."
function hardthreshold(A::AbstractArray, k::Integer)
    B = zero(A); idx = partialsortperm(vec(abs.(A)), 1:min(k, length(A)); rev = true)
    B[idx] .= vec(A)[idx]; return B
end

"Mode-n unfolding."
unfold(T::AbstractArray, n) = reshape(permutedims(T, (n, setdiff(1:ndims(T), n)...)), size(T, n), :)

"Per-axis orthogonal HOSVD bases (left singular vectors of each unfolding) and
the core; a standard data-adapted comparator for any learned orthogonal basis."
function hosvd(T::AbstractArray)
    Us = [svd(unfold(T, n)).U for n in 1:ndims(T)]
    core = copy(T)
    for n in 1:ndims(T); core = modeprod(core, Us[n]', n); end
    return Us, core
end
"Mode-n product T ×ₙ M."
function modeprod(T::AbstractArray, M::AbstractMatrix, n)
    perm = (n, setdiff(1:ndims(T), n)...)
    Y = M * reshape(permutedims(T, perm), size(T, n), :)
    sz = collect(size(T)); sz[n] = size(M, 1)
    return permutedims(reshape(Y, sz[n], sz[setdiff(1:end, n)]...), invperm(perm))
end

"Relative reconstruction error of Γ from the k largest coefficients in four
representations: physical coordinates, learned coordinates (through the
inverse transforms), a frozen orthogonal DCT basis, and the HOSVD bases."
function coefficient_curves(Γ::ITensor, S::ITensor, invs::Vector{ITensor}, ks)
    T = Array(Γ, inds(Γ)...); nT = norm(T)
    Sarr = Array(S, inds(S)...)
    C = dct(T)                                     # orthonormal DCT-II along every axis
    Us, core = hosvd(T)
    fromcore(G) = (X = G; for n in 1:ndims(T); X = modeprod(X, Us[n], n); end; X)
    phys = Float64[]; learned = Float64[]; four = Float64[]; hos = Float64[]
    for k in ks
        push!(phys, norm(hardthreshold(T, k) - T) / nT)
        Sk = ITensor(hardthreshold(Sarr, k), inds(S)...)
        push!(learned, norm(Array(Sk * invs, inds(Γ)...) - T) / nT)
        push!(four, norm(idct(hardthreshold(C, k)) - T) / nT)
        push!(hos, norm(fromcore(hardthreshold(core, k)) - T) / nT)
    end
    return (; ks = collect(ks), phys, learned, fourier = four, hosvd = hos)
end

"Principal angles (degrees) between the r learned basis vectors carrying the
most energy of S along axis a and the leading-r HOSVD subspace of that axis,
with r the HOSVD rank that captures 99% of the energy."
function subspace_overlap(T::AbstractArray, Sarr::AbstractArray, mats::Vector{<:AbstractMatrix})
    out = String[]
    for a in 1:ndims(T)
        U, sv = (F = svd(unfold(T, a)); (F.U, F.S))
        r = findfirst(>=(0.99), cumsum(sv.^2) ./ sum(sv.^2))
        energy = vec(sum(abs2, unfold(Sarr, a); dims = 2))
        top = partialsortperm(energy, 1:r; rev = true)
        Q = Matrix(qr(mats[a][:, top]).Q)[:, 1:r]        # learned vectors are columns of the transform
        angles = acosd.(clamp.(svdvals(U[:, 1:r]' * Q), 0, 1))
        push!(out, @sprintf("axis %d: r=%d, max angle %.1f°, mean %.1f°", a, r, maximum(angles), mean(angles)))
    end
    return out
end

"Spectrum of controls that destroy any axis relation while keeping the values:
entries shuffled over positions, and iid Gaussian of the same size."
function control_spectra(Γ::ITensor; rng = MersenneTwister(0), ops = UniversalOp())
    A = Array(Γ, inds(Γ)...)
    sh = reshape(shuffle(rng, vec(A)), size(A))
    g = randn(rng, size(A)...) .* std(A)
    s1 = derivation_spectrum(derivation_operator(ITensor(sh, inds(Γ)...); ops).L).σ
    s2 = derivation_spectrum(derivation_operator(ITensor(g, inds(Γ)...); ops).L).σ
    return (; shuffled = s1, gaussian = s2)
end

fmt(v) = join([@sprintf("%.1e", x) for x in v], " ")

"Log to stdout and a file."
mutable struct Logger; io::IO; end
function logln(lg::Logger, args...); s = join(string.(args)); println(s); println(lg.io, s); flush(lg.io); end

"Full pipeline on one real 3-way tensor. `name` labels outputs. Returns a
NamedTuple of diagnostics."
function analyze(T::AbstractArray{<:Real}, name::String, lg::Logger;
                 ops = UniversalOp(), ndirs = (1, 2, 3), noise_levels = (0.01, 0.05), rng = MersenneTwister(1))
    orthogonal = ops isa SymmetricOp
    dims = size(T); N = ndims(T)
    ntriv = N - 1            # scalar tuples (c₁I, …, c_N I) with Σc = 0: the exact, trivial null space
    fr = [Index(dims[a], ("x", "y", "z", "t")[a]) for a in 1:N]
    Γ = ITensor(Float64.(T), fr...)
    logln(lg, "\n==== ", name, "  [", typeof(ops), "]  size ", dims, "  ‖T‖=", @sprintf("%.4g", norm(T)), "  mean/std ", @sprintf("%.4g / %.4g", mean(T), std(T)))
    t0 = time(); op = derivation_operator(Γ; ops); sp = derivation_spectrum(op.L); tL = time() - t0
    n = length(sp.σ)
    logln(lg, "derivation operator: ", size(op.L), " unknowns, built+decomposed in ", @sprintf("%.2fs", tL), "; memory ", @sprintf("%.1f MB", sizeof(op.L) / 2^20))
    logln(lg, "smallest 12 relative singular values: ", fmt(sp.σ[1:min(12, n)]))
    logln(lg, "largest 3: ", fmt(sp.σ[max(1, n-2):n]))
    # trivial scalar pair
    sc = [scalar_score(embed(op, sp.vecs[:, i])) for i in 1:min(ntriv + 2, n)]
    logln(lg, "scalar score of the ", ntriv + 2, " smallest directions (1 = pure scalar tuple; the first ", ntriv, " must be trivial): ", fmt(sc))
    # gaps after the trivial directions
    ratios = [sp.σ[i+1] / sp.σ[i] for i in (ntriv + 1):min(ntriv + 8, n-1)]
    logln(lg, "successive ratios σ[i+1]/σ[i] for i=", ntriv + 1, "..", ntriv + 8, ": ", fmt(ratios))
    ctrl = control_spectra(Γ; ops)
    logln(lg, "control (values shuffled) smallest 8: ", fmt(ctrl.shuffled[1:8]))
    logln(lg, "control (iid Gaussian)   smallest 8: ", fmt(ctrl.gaussian[1:8]))
    # candidate nontrivial directions: indices 3.. (eigenvectors are orthogonal
    # to the scalar pair, which is the quotient for a symmetric operator)
    results = []
    for k in ndirs
        v = sp.vecs[:, ntriv + k]; δ = embed(op, v)
        es = eigen_summary(δ)
        st = stratify_along(Γ, δ; orthogonal)
        ks = unique(round.(Int, exp10.(range(0, log10(length(T)), length = 25))))
        cc = coefficient_curves(Γ, st.Σt, st.invs, ks)
        # error at a fixed budget: 5% and 20% of the entries
        at(frac) = (i = argmin(abs.(cc.ks .- frac * length(T))); (cc.ks[i], cc.phys[i], cc.learned[i], cc.fourier[i]))
        at(frac) = (i = argmin(abs.(cc.ks .- frac * length(T))); (cc.ks[i], cc.phys[i], cc.learned[i], cc.fourier[i], cc.hosvd[i]))
        a5, a20 = at(0.05), at(0.20)
        logln(lg, @sprintf("direction %d (σ=%.2e): eigen %s | cond %s | reconstruction relerr %.1e", k, sp.σ[ntriv + k], join(es, "; "), fmt(st.conds), st.relerr))
        logln(lg, @sprintf("   coefficients: at k=%d  phys %.3f  learned %.3f  dct %.3f  hosvd %.3f | at k=%d  phys %.3f  learned %.3f  dct %.3f  hosvd %.3f", a5..., a20...))
        logln(lg, @sprintf("   representation size: learned basis entries %d (+k), tensor entries %d", sum(d -> d^2, dims), length(T)))
        orthogonal && logln(lg, "   learned vs HOSVD subspaces: ", join(subspace_overlap(T, Array(st.Σ, fr...), st.mats), " | "))
        push!(results, (; k, σ = sp.σ[ntriv + k], es, st, cc))
    end
    # random combination of the first three nontrivial directions (Dleto's default behaviour with nd=5)
    v = sp.vecs[:, ntriv+1:ntriv+3] * randn(rng, 3); δ = embed(op, v); st = stratify_along(Γ, δ; orthogonal)
    cc = coefficient_curves(Γ, st.Σt, st.invs, results[1].cc.ks)
    i = argmin(abs.(cc.ks .- 0.05 * length(T)))
    logln(lg, @sprintf("random combination of directions 1-3: cond %s | relerr %.1e | at k=%d phys %.3f learned %.3f dct %.3f hosvd %.3f", fmt(st.conds), st.relerr, cc.ks[i], cc.phys[i], cc.learned[i], cc.fourier[i], cc.hosvd[i]))
    # sensitivity: noise perturbation of the measurements
    for lvl in noise_levels
        Tn = T .+ lvl * std(T) .* randn(rng, size(T)...)
        spn = derivation_spectrum(derivation_operator(ITensor(Float64.(Tn), fr...); ops).L)
        logln(lg, @sprintf("perturbed by %.0f%% of std: smallest 8 = %s", 100lvl, fmt(spn.σ[1:8])))
    end
    # figures: spectrum with controls; coefficient curves; learned representation
    nshow = min(40, n)
    rng_ = (ntriv + 1):nshow      # the quotient: everything after the exact trivial directions
    p1 = plot(rng_, sp.σ[rng_]; yscale = :log10, marker = :circle, markersize = 3, label = "measured", xlabel = "index (after the $ntriv trivial scalar directions)", ylabel = "relative singular value", title = "Derivation operator spectrum: $name", legend = :bottomright, ylims = (min(1e-6, 0.5 * minimum(sp.σ[rng_])), 1.5))
    plot!(p1, rng_, ctrl.shuffled[rng_]; marker = :x, markersize = 3, label = "control: values shuffled")
    plot!(p1, rng_, ctrl.gaussian[rng_]; marker = :+, markersize = 3, label = "control: iid Gaussian")
    savefig(p1, joinpath(FIGURES, "$name-spectrum.png"))
    r = results[1]
    p2 = plot(r.cc.ks, r.cc.phys; xscale = :log10, label = "physical coordinates", xlabel = "coefficients kept", ylabel = "relative error", title = "Sparse reconstruction: $name", legend = :bottomleft, marker = :circle, markersize = 2)
    plot!(p2, r.cc.ks, r.cc.learned; label = "learned coordinates (direction 1)", marker = :square, markersize = 2)
    plot!(p2, r.cc.ks, r.cc.fourier; label = "DCT baseline (frozen)", marker = :diamond, markersize = 2)
    plot!(p2, r.cc.ks, r.cc.hosvd; label = "HOSVD bases", marker = :utriangle, markersize = 2)
    savefig(p2, joinpath(FIGURES, "$name-coefficients.png"))
    # learned basis vectors: the columns of each axis transform (first 4), against physical index
    pb = [plot(r.st.mats[a][:, 1:min(4, dims[a])]; xlabel = "physical index, axis $a", ylabel = "basis vector", title = "learned axis-$a basis (direction 1)", legend = false) for a in 1:N]
    savefig(plot(pb...; layout = (1, N), size = (420N, 320)), joinpath(FIGURES, "$name-basis.png"))
    # representation pictures: middle slice in physical and learned coordinates
    Sarr = Array(r.st.Σ, fr...)
    mid = Tuple(dims[a] ÷ 2 + 1 for a in 3:N)
    Tsl = T[:, :, mid...]; Ssl = Sarr[:, :, mid...]
    p3 = plot(heatmap(Tsl'; title = "physical coords, slice $mid", xlabel = "x index", ylabel = "y index", aspect_ratio = 1),
              heatmap(abs.(Ssl)'; title = "learned coords, slice $mid (|S|)", xlabel = "learned a", ylabel = "learned b", aspect_ratio = 1); size = (1000, 450))
    savefig(p3, joinpath(FIGURES, "$name-representation.png"))
    # save transforms and spectrum
    open(joinpath(RESULTS, "$name-spectrum.csv"), "w") do io
        println(io, "index,measured,shuffled,gaussian")
        for i in 1:n; println(io, i, ",", sp.σ[i], ",", ctrl.shuffled[i], ",", ctrl.gaussian[i]); end
    end
    open(joinpath(RESULTS, "$name-coefficients.csv"), "w") do io
        println(io, "k,phys,learned_dir1,dct,hosvd")
        for i in eachindex(r.cc.ks); println(io, r.cc.ks[i], ",", r.cc.phys[i], ",", r.cc.learned[i], ",", r.cc.fourier[i], ",", r.cc.hosvd[i]); end
    end
    for (a, M) in enumerate(r.st.mats)
        open(joinpath(RESULTS, "$name-transform-axis$a.csv"), "w") do io
            for row in eachrow(M); println(io, join(row, ",")); end
        end
    end
    return (; sp, ctrl, results, op)
end
