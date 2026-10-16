# Pilot 2: MeshRIR S1-M3969 (Koyama et al.), v1.2, Zenodo 10.5281/zenodo.10852693,
# file S1-M3969_npy.zip, MD5 2cb598eb44bb9905560c545db7af3432, CC BY 4.0.
# 3969 measured microphone positions on a 21 x 21 x 9 grid (5 cm), one source,
# 32768-sample impulse responses at 48 kHz.
include(joinpath(@__DIR__, "common.jl"))
using NPZ
lg = Logger(open(joinpath(RESULTS, "meshrir.log"), "w"))
logln(lg, "MeshRIR pilot, ", now(), ", Julia ", VERSION, ", Dleto at git ", strip(read(`git -C $(dirname(@__DIR__)) rev-parse --short HEAD`, String)))
d = joinpath(dirname(@__DIR__), "data", "meshrir", "S1-M3969_npy")
pos = npzread(joinpath(d, "pos_mic.npy")); src = npzread(joinpath(d, "pos_src.npy"))
nmic = size(pos, 1)
logln(lg, "microphones: ", nmic, "  source (m): ", vec(src))
levels = [sort(unique(round.(pos[:, a]; digits = 4))) for a in 1:3]
dims = Tuple(length.(levels))
logln(lg, "grid levels: ", dims, "  x ", (levels[1][1], levels[1][end]), " y ", (levels[2][1], levels[2][end]), " z ", (levels[3][1], levels[3][end]))
# index map: measured position -> (i, j, k), as MeshRIR's sortIR3 does
idx = [findfirst(l -> isapprox(l, pos[m, a]; atol = 1e-4), levels[a]) for m in 1:nmic, a in 1:3]
@assert all(!isnothing, idx)
@assert length(unique(eachrow(idx))) == nmic == prod(dims) "positions do not form a complete cuboid"
logln(lg, "every grid point measured exactly once; index map saved")
open(joinpath(RESULTS, "meshrir-index-map.csv"), "w") do io
    println(io, "mic,x_m,y_m,z_m,i,j,k")
    for m in 1:nmic; println(io, m - 1, ",", join(pos[m, :], ","), ",", join(idx[m, :], ",")); end
end
# load all responses: R[x, y, z, t]
fs = 48000
ir0 = npzread(joinpath(d, "ir_0.npy")); nt = size(ir0, 2)
R = zeros(Float32, dims..., nt)
for m in 0:nmic-1
    ir = npzread(joinpath(d, "ir_$m.npy"))
    R[idx[m+1, 1], idx[m+1, 2], idx[m+1, 3], :] = ir[1, :]
end
logln(lg, "R[x,y,z,t] size ", size(R), " (", round(sizeof(R) / 2^20; digits = 1), " MB)")
# documented sample-selection rule, fixed before any chiseling result was seen:
# t0 = median over microphones of the sample of maximum |response| (direct sound);
# snapshots at fixed offsets covering direct sound, early reflections, late reverberation.
arrival = [argmax(abs.(R[i, j, k, :])) for i in 1:dims[1], j in 1:dims[2], k in 1:dims[3]]
t0 = round(Int, median(arrival))
offsets = (-24, 0, 24, 96, 480, 4800)
logln(lg, "arrival samples: min ", minimum(arrival), " median ", t0, " max ", maximum(arrival), " (", round(t0 / fs * 1000; digits = 2), " ms)")
logln(lg, "snapshot rule: t0 + ", offsets, " samples")
# measured-volume figure: energy over the direct-sound window
E = dropdims(sum(abs2, R[:, :, :, t0-24:t0+24]; dims = 4); dims = 4)
savefig(plot_tensor(E ./ maximum(E), 0.1; title = "MeshRIR energy, samples $(t0-24):$(t0+24)", xlabel = "x", ylabel = "y", zlabel = "z"), joinpath(FIGURES, "meshrir-measured.png"))
out = Dict()
for off in offsets
    t = t0 + off
    T = Float64.(R[:, :, :, t]); T ./= maximum(abs, T)
    for (opname, ops) in (("univ", UniversalOp()), ("sym", SymmetricOp()))
        out["t$(t)-$opname"] = analyze(T, "meshrir-t$(t)-$opname", lg; ops)
    end
end
# nearby-sample sensitivity: t0 ± 1 sample
for t in (t0 - 1, t0 + 1)
    T = Float64.(R[:, :, :, t]); T ./= maximum(abs, T)
    out["t$(t)-sym"] = analyze(T, "meshrir-t$(t)-sym", lg; ops = SymmetricOp())
end
# separate formulation: a time-valued tensor R[x, y, z, t] over a 16-sample window
for (label, win) in (("direct", t0-8:t0+7), ("late", t0+4800:t0+4815))
    T = Float64.(R[:, :, :, win]); T ./= maximum(abs, T)
    for (opname, ops) in (("univ", UniversalOp()), ("sym", SymmetricOp()))
        out["win-$label-$opname"] = analyze(T, "meshrir-win-$label-$opname", lg; ops, noise_levels = (0.01,))
    end
end
close(lg.io)
println("done")
