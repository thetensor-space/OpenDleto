# Pilot 1: 3DFM magnetic-field scan, 10 cm cube, 11 steps per axis.
# Source: https://raw.githubusercontent.com/mri4all/field_mapper_3DFM/main/measurements/10cmcube_11steps_per_axis_measurement.csv
# SHA-256 a39160b59f6516e1931ce5125886fa8e25f8c718350b2719a1c810d924e53f3f
include(joinpath(@__DIR__, "common.jl"))
using CSV, DataFrames
lg = Logger(open(joinpath(RESULTS, "3dfm.log"), "w"))
logln(lg, "3DFM pilot, ", now(), ", Julia ", VERSION, ", Dleto at git ", strip(read(`git -C $(dirname(@__DIR__)) rev-parse --short HEAD`, String)))
csv = joinpath(dirname(@__DIR__), "data", "3dfm", "10cmcube_11steps_per_axis_measurement.csv")
df = CSV.read(csv, DataFrame)
logln(lg, "rows: ", nrow(df), "  columns: ", names(df))
# positions are cumulative sums of the relative moves
pos = cumsum(Matrix(df[:, [:dx, :dy, :dz]]); dims = 1)
levels = [sort(unique(pos[:, a])) for a in 1:3]
logln(lg, "coordinate levels (mm): x=", levels[1], " y=", levels[2], " z=", levels[3])
@assert all(length.(levels) .== 11)
rows = 2:1332                     # 1-based: the brief's zero-based rows 1..1331
idx = [(findfirst(==(pos[r, a]), levels[a])) for r in rows, a in 1:3]
@assert length(unique(eachrow(idx))) == 1331 "incomplete Cartesian coverage"
logln(lg, "rows 2..1332 cover the full 11×11×11 product exactly once")
center_rows = findall(r -> all(pos[r, :] .== 0), 1:nrow(df))
logln(lg, "center checks at rows ", center_rows, " (1-based): ", [Tuple(df[r, [:Bx, :By, :Bz, :Bmod]]) for r in center_rows])
B = Dict{String, Array{Float64,3}}()
for comp in ("Bx", "By", "Bz", "Bmod")
    A = fill(NaN, 11, 11, 11)
    for (i, r) in enumerate(rows); A[idx[i, 1], idx[i, 2], idx[i, 3]] = df[r, comp]; end
    @assert all(isfinite, A)
    B[comp] = A
    logln(lg, @sprintf("%s: min %.3f max %.3f mean %.3f std %.3f (instrument units)", comp, minimum(A), maximum(A), mean(A), std(A)))
end
drift = B["Bx"][6, 6, 6] .- [df[r, :Bx] for r in center_rows]
logln(lg, "center-point Bx from the grid minus the two center checks: ", drift, " (drift over the scan)")
# measured volume figure
p = plot([heatmap(B[c][:, :, 6]'; title = "$c, z = 0 mm slice", xlabel = "x index", ylabel = "y index", aspect_ratio = 1) for c in ("Bx", "By", "Bz", "Bmod")]...; layout = (2, 2), size = (900, 800))
savefig(p, joinpath(FIGURES, "3dfm-measured.png"))
out = Dict()
for (opname, ops) in (("univ", UniversalOp()), ("sym", SymmetricOp()))
    for comp in ("Bx", "By", "Bz", "Bmod")
        out["$comp-$opname"] = analyze(B[comp], "3dfm-$comp-$opname", lg; ops)
    end
    # transparent global normalisation: remove the mean of Bx (the dominant, nearly constant component)
    out["Bx-demeaned-$opname"] = analyze(B["Bx"] .- mean(B["Bx"]), "3dfm-Bx-demeaned-$opname", lg; ops)
end
close(lg.io)
println("done")
