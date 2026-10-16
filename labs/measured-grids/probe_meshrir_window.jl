# Probe of the four-way direct-sound window of MeshRIR S1-M3969: is the gapped
# symmetric spectrum a new relation or degenerate (near-empty) slices?
include(joinpath(@__DIR__, "common.jl"))
using NPZ
lg = Logger(open(joinpath(RESULTS, "meshrir-window-probe.log"), "w"))
d = joinpath(dirname(@__DIR__), "data", "meshrir", "S1-M3969_npy")
pos = npzread(joinpath(d, "pos_mic.npy")); nmic = size(pos, 1)
levels = [sort(unique(round.(pos[:, a]; digits = 4))) for a in 1:3]; dims = Tuple(length.(levels))
idx = [findfirst(l -> isapprox(l, pos[m, a]; atol = 1e-4), levels[a]) for m in 1:nmic, a in 1:3]
t0 = 8538; tr = t0-40:t0+40
R = zeros(Float64, dims..., length(tr))
for m in 0:nmic-1
    ir = npzread(joinpath(d, "ir_$m.npy")); R[idx[m+1, 1], idx[m+1, 2], idx[m+1, 3], :] = ir[1, tr]
end
win(c, h) = (lo = c - h - first(tr) + 1; W = R[:, :, :, lo:(lo + 2h - 1)]; W ./ maximum(abs, W))
function probe(T, label; ndirs = 3)
    N = ndims(T); fr = [Index(size(T, a), ("x", "y", "z", "t")[a]) for a in 1:N]
    Γ = ITensor(T, fr...)
    logln(lg, "\n### ", label, " size ", size(T))
    for a in 1:N
        e = vec(sum(abs2, T; dims = setdiff(1:N, a))); e ./= maximum(e)
        logln(lg, "  axis $a slice energies (rel.): ", join([@sprintf("%.2f", x) for x in e], " "))
    end
    op = derivation_operator(Γ; ops = SymmetricOp()); sp = derivation_spectrum(op.L)
    logln(lg, "  symmetric spectrum σ[N..N+9]: ", fmt(sp.σ[N:N+9]), "  ratios ", fmt([sp.σ[i+1]/sp.σ[i] for i in N:N+6]))
    for k in 1:ndirs
        δ = embed(op, sp.vecs[:, N - 1 + k])
        parts = String[]
        for (a, X) in enumerate(δ)
            M = Array(X, inds(X)...); M = (M + M') / 2; E = eigen(Symmetric(M))
            λ = E.values ./ maximum(abs.(E.values))
            # cluster eigenvalues and report each cluster's multiplicity and the index support of its eigenspace
            groups = []; cur = [1]
            for i in 2:length(λ); abs(λ[i] - λ[i-1]) < 1e-2 ? push!(cur, i) : (push!(groups, cur); cur = [i]); end; push!(groups, cur)
            desc = map(groups) do g
                P = E.vectors[:, g]; w = vec(sum(abs2, P; dims = 2))        # weight of the eigenspace on each physical index
                supp = findall(>(0.5 * maximum(w)), w)
                @sprintf("λ=%.2f×%d on idx %s", mean(λ[g]), length(g), length(supp) == size(M,1) ? "all" : string(first(supp), "..", last(supp), length(supp) == last(supp)-first(supp)+1 ? "" : "(gappy)"))
            end
            push!(parts, "axis $a: " * join(desc, "; "))
        end
        logln(lg, @sprintf("  direction %d (σ=%.1e): ", k, sp.σ[N - 1 + k]), join(parts, " | "))
    end
    return sp
end
probe(win(t0, 8), "direct window t0-8..t0+7 (as in the run)")
for (c, h) in ((t0 - 8, 8), (t0 + 8, 8), (t0 - 16, 8), (t0 + 16, 8), (t0, 4), (t0, 16))
    probe(win(c, h), "window centre $(c) halfwidth $(h)"; ndirs = 1)
end
# nondeg first: drop near-empty slices (relative singular-value tolerance), then re-probe
T = win(t0, 8); fr = [Index(size(T, a), ("x", "y", "z", "t")[a]) for a in 1:4]; Γ = ITensor(T ./ norm(T), fr...)
for tol in (1e-2, 3e-2)
    Δ, Es = nondeg(Γ; tol = tol)
    logln(lg, "\nnondeg tol=$tol: dims ", [ITensors.dim(i) for i in inds(Δ)], " (truncation keeps the dominant subspace on each axis)")
    probe(Array(Δ, inds(Δ)...), "direct window after nondeg(tol=$tol)"; ndirs = 2)
end
close(lg.io); println("done")
