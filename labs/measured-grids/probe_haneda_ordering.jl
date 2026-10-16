# Robustness of the position-ordered diagonal directions found after the
# empty-slice band in the Haneda stride-2 symmetric spectrum.
include(joinpath(@__DIR__, "common.jl"))
using HDF5
lg = Logger(open(joinpath(RESULTS, "haneda-ordering-probe.log"), "w"))
h5 = h5open(joinpath(dirname(@__DIR__), "data", "meshgrid-ir-raw", "Raw_A_ir_48k_sp1.h5"), "r")
fulldims = Tuple(Int.(read_attribute(h5, "grid_dimensions"))); rir = h5["rir"]
sel = [1:2:fulldims[a] for a in 1:3]
spearman(a, b) = cor(sortperm(sortperm(a)), sortperm(sortperm(b)))
rng = MersenneTwister(7)
function ordering(T, label)
    N = 3; fr = [Index(size(T, a), ("x", "y", "z")[a]) for a in 1:N]; Γ = ITensor(T, fr...)
    e2 = vec(sum(abs2, T; dims = (1, 3))); nempty = count(<(0.05 * maximum(e2)), e2)
    op = derivation_operator(Γ; ops = SymmetricOp()); sp = derivation_spectrum(op.L)
    ratios = [sp.σ[i+1] / sp.σ[i] for i in 3:60]; step = 2 + argmax(ratios)      # index of the largest jump
    dirs = (step + 1):(step + 10)
    stats = zeros(N, 2)
    for k in dirs
        δ = embed(op, sp.vecs[:, k])
        for (a, X) in enumerate(δ)
            M = Array(X, inds(X)...); M = (M + M') / 2; dg = diag(M)
            stats[a, 1] += sum(abs2, dg) / sum(abs2, M) / length(dirs); stats[a, 2] += abs(spearman(dg, 1:length(dg))) / length(dirs)
        end
    end
    logln(lg, @sprintf("%-34s y-slices <5%%: %2d | largest jump x%.1f at index %d | 10 directions after it: diag share x %.2f y %.2f z %.2f ; |Spearman(diag, position)| x %.2f y %.2f z %.2f",
        label, nempty, maximum(ratios), step, stats[1,1], stats[2,1], stats[3,1], stats[1,2], stats[2,2], stats[3,2]))
end
for t in (309, 345, 381, 417, 453)
    T = Float64.(reshape(rir[t, :], fulldims...))[sel...]; T ./= maximum(abs, T)
    ordering(T, "t=$t")
    t == 381 && ordering(T .+ 0.01 * std(T) .* randn(rng, size(T)...), "t=381 + 1% noise")
    t == 381 && ordering(reshape(shuffle(rng, vec(T)), size(T)), "t=381 values shuffled (control)")
end
close(h5); close(lg.io); println("done")
