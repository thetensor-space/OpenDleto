# Probe of the step at index ~24 in the Haneda stride-2 symmetric spectrum at t0:
# are the 22 small directions rescalings of near-empty slices?
include(joinpath(@__DIR__, "common.jl"))
using HDF5
lg = Logger(open(joinpath(RESULTS, "haneda-step-probe.log"), "w"))
h5 = h5open(joinpath(dirname(@__DIR__), "data", "meshgrid-ir-raw", "Raw_A_ir_48k_sp1.h5"), "r")
fulldims = Tuple(Int.(read_attribute(h5, "grid_dimensions"))); rir = h5["rir"]
t0 = 381; sel = [1:2:fulldims[a] for a in 1:3]
T = Float64.(reshape(rir[t0, :], fulldims...))[sel...]; T ./= maximum(abs, T)
N = 3; fr = [Index(size(T, a), ("x", "y", "z")[a]) for a in 1:N]; Γ = ITensor(T, fr...)
for a in 1:N
    e = vec(sum(abs2, T; dims = setdiff(1:N, a))); e ./= maximum(e)
    logln(lg, "axis $a slice energies (rel.): ", join([@sprintf("%.2f", x) for x in e], " "), "   below 1%: ", count(<(0.01), e), "  below 5%: ", count(<(0.05), e))
end
op = derivation_operator(Γ; ops = SymmetricOp()); sp = derivation_spectrum(op.L)
logln(lg, "symmetric spectrum σ[3..30]: ", fmt(sp.σ[3:30]))
for k in (1, 2, 3, 10, 20, 22, 23, 24, 25)
    δ = embed(op, sp.vecs[:, 2 + k]); parts = String[]
    for (a, X) in enumerate(δ)
        M = Array(X, inds(X)...); M = (M + M') / 2; E = eigen(Symmetric(M)); λ = E.values ./ maximum(abs.(E.values))
        groups = []; cur = [1]
        for i in 2:length(λ); abs(λ[i] - λ[i-1]) < 2e-2 ? push!(cur, i) : (push!(groups, cur); cur = [i]); end; push!(groups, cur)
        desc = map(groups) do g
            w = vec(sum(abs2, E.vectors[:, g]; dims = 2)); supp = findall(>(0.5 * maximum(w)), w)
            @sprintf("%.2f×%d@%s", mean(λ[g]), length(g), length(supp) == size(M, 1) ? "all" : string(first(supp), "..", last(supp)))
        end
        push!(parts, "ax$a: " * join(desc, ";"))
    end
    logln(lg, @sprintf("dir %2d (σ=%.1e): ", k, sp.σ[2 + k]), join(parts, " | "))
end
close(h5); close(lg.io); println("done")

# ---- second pass: are the post-step directions diagonal with position-ordered eigenvalues?
lg = Logger(open(joinpath(RESULTS, "haneda-step-probe.log"), "a"))
h5 = h5open(joinpath(dirname(@__DIR__), "data", "meshgrid-ir-raw", "Raw_A_ir_48k_sp1.h5"), "r")
G = read(h5["grid"]); sp_rel = read_attribute(h5, "sp_pos") .- read_attribute(h5, "mic_array_pos")
xs = sort(unique(G[:, 1]))[sel[1]]; ys = sort(unique(G[:, 2]))[sel[2]]; zs = sort(unique(G[:, 3]))[sel[3]]
pred = [(xs .- sp_rel[1]).^2, (ys .- sp_rel[2]).^2, (zs .- sp_rel[3]).^2]     # the surface-law profiles, known physics (for comparison only)
spearman(a, b) = cor(sortperm(sortperm(a)), sortperm(sortperm(b)))
logln(lg, "\npost-step directions: diagonal share of ‖D_a‖², Spearman(diag, index), Spearman(diag, (coord − speaker)²)")
for k in 15:32
    δ = embed(op, sp.vecs[:, 2 + k]); parts = String[]
    for (a, X) in enumerate(δ)
        M = Array(X, inds(X)...); M = (M + M') / 2; dg = diag(M)
        push!(parts, @sprintf("ax%d diag %.2f ρ(idx) %+.2f ρ(pred) %+.2f", a, sum(abs2, dg) / sum(abs2, M), spearman(dg, 1:length(dg)), spearman(dg, pred[a])))
    end
    logln(lg, @sprintf("dir %2d (σ=%.1e): ", k, sp.σ[2 + k]), join(parts, " | "))
end
close(h5); close(lg.io); println("done2")
