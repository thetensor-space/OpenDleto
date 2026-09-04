# Script twin of labs/MovieRuntime.ipynb -- same cells, saves the figures next to the notebook.
# Run through the wrapper:  bench/jl labs/MovieRuntime.jl
const FIGDIR = @__DIR__

# ---- cell 1 ----
using CSV, DataFrames, Plots, Printf, Statistics
ENV["GKSwstype"] = "100"   # headless GR

const REPO = normpath(joinpath(@__DIR__, ".."))
const WCSV = joinpath(REPO, "bench", "reports", "2026-09-04", "whitened", "whitened.csv")

runs = CSV.read(WCSV, DataFrame; comment = "#")
parsedims(s) = parse.(Int, split(strip(String(s), ['[', ']']), ','))
runs.dimv    = parsedims.(runs.dims)
runs.entries = prod.(runs.dimv)
runs.gb      = runs.entries .* ifelse.(runs.eltype .== "Float64", 8, 4) ./ 2^30
runs.family  = [startswith(c, "video") ? "video-shaped" :
                startswith(c, "sphere") ? "sphere (valence $v)" :
                startswith(c, "random") ? "random dense" : "degenerate"
                for (c, v) in zip(runs.case, runs.valence)]

ok = runs[(runs.whiten .== 1) .& (runs.status .== "ok"), :]
tbl = select(ok, :case, :eltype, :entries, :gb, :applies, :seconds, :solve_seconds, :maxrss_GB,
       :nullity, :oracle_nullity, :uncertified)
show(stdout, MIME("text/plain"), tbl); println()

# ---- cell 2 ----
p1 = plot(xscale = :log10, yscale = :log10, xlabel = "tensor entries",
          ylabel = "wall seconds", legend = :topleft, size = (820, 480),
          title = "Whitened QuickDer, matrix-free branch, 5 CPU threads (measured 2026-09-04)")
for fam in unique(ok.family), T in ("Float64", "Float32")
    sub = ok[(ok.family .== fam) .& (ok.eltype .== T), :]
    isempty(sub) && continue
    scatter!(p1, sub.entries, sub.seconds, label = "$fam, $T",
             marker = T == "Float64" ? :circle : :diamond, ms = 6)
end
savefig(p1, joinpath(FIGDIR, "movie-runtime-all.png")); println("saved movie-runtime-all.png")

# ---- cell 3 ----
movie = ok[startswith.(ok.case, "video-640x480") .& (ok.eltype .== "Float32"), :]
movie.F = getindex.(movie.dimv, 3)
sort!(movie, :F)

affine(x, y) = ([ones(length(x)) x] \ y)            # least squares a + b x
(a_t, b_t) = affine(movie.F, movie.seconds)
(a_m, b_m) = affine(movie.F, movie.maxrss_GB)
F_min = 1800
t_min = a_t + b_t * F_min
m_min = a_m + b_m * F_min

println("measured (640 x 480 x F x 3, Float32):")
for r in eachrow(movie)
    @printf("  F = %4d (%5.1f s of video)  %8.1f s wall  (solve %6.1f s)  peak RSS %5.1f GB  nullity %d/%d %s\n",
            r.F, r.F / 30, r.seconds, r.solve_seconds, r.maxrss_GB, r.nullity, r.oracle_nullity,
            r.uncertified ? "(uncertified)" : "")
end
@printf("\nfit: t(F) = %.1f s + %.3f s/frame;   RSS(F) = %.2f GB + %.4f GB/frame\n", a_t, b_t, a_m, b_m)
@printf("\nPROJECTION, 1 minute at 30 fps (F = 1800, 6.6 GB Float32 tensor):\n")
@printf("  wall time  ~ %.0f s  (%.1f min)\n", t_min, t_min / 60)
@printf("  peak RSS   ~ %.0f GB  with today's copies (the tensor itself is 6.6 GB)\n", m_min)

Fs = range(0, 1900, length = 200)
p2 = plot(Fs, a_t .+ b_t .* Fs, ls = :dash, lw = 2, color = :gray, label = "affine fit / projection",
          xlabel = "frames F  (640 x 480 x F x 3, Float32)", ylabel = "wall seconds",
          legend = :topleft, size = (820, 480), title = "Stratifying a 640 x 480 movie")
scatter!(p2, movie.F, movie.seconds, ms = 8, color = :black, label = "measured")
scatter!(p2, [F_min], [t_min], ms = 10, marker = :star5, color = :red,
         label = @sprintf("1 min @ 30 fps: ~%.0f s (projected)", t_min))
vline!(p2, [F_min], ls = :dot, color = :red, label = "")
savefig(p2, joinpath(FIGDIR, "movie-runtime-projection.png")); println("saved movie-runtime-projection.png")

# ---- cell 4 ----
# Where the seconds go, per measured frame count: the eigensolve is the part that does not
# grow with F; sketches and lift are the parts that do.
other = max.(movie.seconds .- movie.solve_seconds .- movie.sketch_seconds .- movie.lift_seconds, 0)
stages = [("sketches", movie.sketch_seconds), ("eigensolve", movie.solve_seconds),
          ("lift", movie.lift_seconds), ("verify + other", other)]
xs  = string.(movie.F) .* " frames"
cum = zeros(nrow(movie))
p3 = plot(ylabel = "seconds", size = (820, 400), legend = :topleft,
          title = "Stage breakdown, 640 x 480 x F x 3 (Float32)")
for (lab, col) in stages                     # stacked bars without StatsPlots
    bar!(p3, xs, cum .+ col, fillrange = cum, bar_width = 0.6, label = lab)
    cum .+= col
end
savefig(p3, joinpath(FIGDIR, "movie-runtime-stages.png")); println("saved movie-runtime-stages.png")
