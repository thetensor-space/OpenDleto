# Pilot 3: Haneda-Ren 3D Meshgrid RIR dataset, Part 0 (raw, 48 kHz, frame
# reflections retained), Zenodo 10.5281/zenodo.19656813, CC BY 4.0.
# File Raw_A_ir_48k_sp1.h5, MD5 4a8c756cadac33326b5bcae02135df95 (Zenodo API).
# Layout per the release's example_read_visualization_Raw.m: /rir [time x N],
# /grid [N x 3], reshape to [T, X, Y, Z] with x varying fastest.
include(joinpath(@__DIR__, "common.jl"))
using HDF5
lg = Logger(open(joinpath(RESULTS, "haneda.log"), "w"))
logln(lg, "Haneda Part 0 pilot, ", now(), ", Julia ", VERSION, ", Dleto at git ", strip(read(`git -C $(dirname(@__DIR__)) rev-parse --short HEAD`, String)))
path = joinpath(dirname(@__DIR__), "data", "meshgrid-ir-raw", "Raw_A_ir_48k_sp1.h5")
h5 = h5open(path, "r")
fs = read_attribute(h5, "fs")[1]; fulldims = Tuple(Int.(read_attribute(h5, "grid_dimensions")))
G = read(h5["grid"]); rir = h5["rir"]
logln(lg, "root attributes: ", Dict(a => read_attribute(h5, a) for a in keys(attributes(h5))))
logln(lg, "rir ", size(rir), " ", eltype(rir), "; grid ", size(G))
levels = [sort(unique(G[:, a])) for a in 1:3]
@assert Tuple(length.(levels)) == fulldims
# verify the flattening order from the coordinates themselves: x fastest, then y, then z
expected = [levels[a][((n - 1) ÷ prod(fulldims[1:a-1])) % fulldims[a] + 1] for n in 1:size(G, 1), a in 1:3]
@assert isapprox(expected, G; atol = 1e-5) "grid is not x-fastest; do not reshape blindly"
logln(lg, "grid order verified from coordinates: x fastest, then y, then z; levels ", fulldims, " at ", levels[1][2] - levels[1][1], " m")
# direct-sound arrival rule, as in the other pilots, on a sample of positions
sample = 1:25:size(G, 1)
arrival = [argmax(abs.(rir[:, i])) for i in sample]
t0 = round(Int, median(arrival))
offsets = (-72, 0, 72, 288, 1440, 3600)       # 48 kHz: 1.5 ms, 6 ms, 30 ms, 75 ms
logln(lg, "arrival samples (every 25th position): min ", minimum(arrival), " median ", t0, " max ", maximum(arrival), " (", round(t0 / fs * 1000; digits = 2), " ms)")
logln(lg, "snapshot rule: t0 + ", offsets)
block = Float64.(rir[max(1, t0 - 100):min(size(rir, 1), t0 + 3700), :])
tb = max(1, t0 - 100)
field(t) = reshape(block[t - tb + 1, :], fulldims...)
# retained subgrid for the sweep: every second measured position on each axis
sel = [1:2:fulldims[a] for a in 1:3]
logln(lg, "stride-2 subgrid keeps positions ", [levels[a][sel[a]] for a in 1:3])
savefig(plot_tensor(abs.(field(t0)) ./ maximum(abs, field(t0)), 0.1; title = "Haneda Part 0, |p| at sample $t0", xlabel = "x", ylabel = "y", zlabel = "z"), joinpath(FIGURES, "haneda-measured.png"))
out = Dict()
for off in offsets
    t = t0 + off
    T = field(t)[sel...]; T ./= maximum(abs, T)
    for (opname, ops) in (("univ", UniversalOp()), ("sym", SymmetricOp()))
        out["t$(t)-$opname"] = analyze(T, "haneda-s2-t$(t)-$opname", lg; ops)
    end
end
# escalation: the full 48 x 51 x 26 grid at the direct-sound instant, both families
T = field(t0); T ./= maximum(abs, T)
for (opname, ops) in (("sym", SymmetricOp()), ("univ", UniversalOp()))
    out["full-$opname"] = analyze(T, "haneda-full-t$(t0)-$opname", lg; ops, ndirs = (1, 2), noise_levels = (0.01,))
end
close(h5); close(lg.io)
println("done")
