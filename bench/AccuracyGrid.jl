#
# AccuracyGrid -- the baseline accuracy sweep for the true-colors/precision-accuracy
# lever: no lost derivations, no false certificates, Float16/Float32/Float64.
#
# Grid: scrambled sphere octant (SymmetricOp, build_sphere) valence 3 at
# d in {48, 64, 100, 140} and valence 4 at d in {24, 40}; a smooth video-like
# block 40x40x20x3 (a ramp plus small texture, near-degenerate the way the
# Float16 luma-block repro in docs/CONTEXT.md is) and a random 60x60x30x3
# block (UniversalOp, UniversalChisel(4), generic -- only the 3 scalar
# derivations).  T in {Float64, Float32, Float16}.  Six seeds
# {20260904..20260909} so a 1-in-8 failure rate (docs/CONTEXT.md, "Session 4,
# part 3") is visible in six draws.
#
# Three method configurations per cell, all :QuickDer:
#   default     whiten = true (QuickDerMethod's own default), solver = :AutoSolver,
#               dense branch allowed (QDN_DENSE_BUDGET_BYTES left at its package default)
#   mf-arpack   forced matrix-free (QDN_DENSE_BUDGET_BYTES[] = 0.0), solver = :ArpackSolver
#   mf-krylov   forced matrix-free (QDN_DENSE_BUDGET_BYTES[] = 0.0), solver = :KrylovSolver
#
# Every cell is `derTrOpsReduced(m, Ω, ch, Γ; return_diagnostics = true)`; the
# report is what is recorded (`report.nullity` = the deciding/restricted
# solve's count before the lift filter and the Ω-intersection, `report.returned`
# = the final answer, `report.certified`, `report.status`, `report.undecidable`,
# `report.near_null`, `report.selected`, `report.next_value`,
# `maximum(report.residuals)` = the worst Z-law residual of a returned
# direction).  "expected" is the known truth: valence for the sphere
# (build_sphere's SymmetricOp intersection has dimension = valence, per
# docs/CONTEXT.md "Session 3"), 3 (valence - 1 scalars) for the video blocks --
# the smooth block may legitimately exceed 3 if it is degenerate, which is
# exactly what part of this sweep is checking (see the README).
#
# CSV rows are appended AS EACH CELL COMPLETES (one `open(..., "a")` per row),
# so a killed or interrupted run still leaves a valid, resumable partial table;
# `full` skips any (case,T,seed,config) already present in the CSV.
#
# Usage:
#   bench/jl bench/AccuracyGrid.jl list
#       Print every (case, T, seed, config) cell without running anything.
#
#   bench/jl bench/AccuracyGrid.jl full
#       Run the whole grid, appending to
#       bench/reports/2026-09-08/accuracy/accuracy-grid.csv as it goes.
#       Skips any (case,T,seed,config) already in the CSV, so an interrupted
#       run resumes for free.
#
#   bench/jl bench/AccuracyGrid.jl full <part> <nparts>
#       Same, but only the cells whose position in the grid is `part` mod
#       `nparts` -- run three of these (part = 0,1,2; nparts = 3) as separate
#       bench/jl processes to use all three JL_SLOTS at once; they append to
#       the same CSV and do not step on each other's rows.
#
#   bench/jl bench/AccuracyGrid.jl cell <case> <T> <seed> <config>
#       Run exactly one cell (case in sphere3-48/sphere3-64/sphere3-100/
#       sphere3-140/sphere4-24/sphere4-40/video-smooth/video-random; T in
#       Float64/Float32/Float16; config in default/mf-arpack/mf-krylov).
#
#   bench/jl bench/AccuracyGrid.jl spectrum <case> <T> <seed> <config>
#       Like `cell`, but also dumps the full restricted spectrum (report.spectrum)
#       around the cut to stdout, for failure diagnosis -- no CSV row written.
#
using Arpack
using Dleto
using ITensors
using IterativeSolvers
using LinearAlgebra
using LinearMaps
using Printf
using Random

include(joinpath(@__DIR__, "SphereHarness.jl"))

const REPORT_DIR = joinpath(@__DIR__, "reports", "2026-09-08", "accuracy")
mkpath(REPORT_DIR)
const CSV_PATH = joinpath(REPORT_DIR, "accuracy-grid.csv")

const QDN_DEFAULT_BUDGET = Dleto.QDN_DENSE_BUDGET_BYTES[]

const CSV_COLS = "family,case,dims,valence,expected_final,T,seed,config,solver," *
                  "restricted_nullity,final_nullity,certified,status,undecidable," *
                  "near_null,rule,gap,selected,next_value,max_zlaw_resid,seconds,notes"

csv_header!() = isfile(CSV_PATH) || open(io -> println(io, CSV_COLS), CSV_PATH, "w")

# No embedded commas in ANY field -- `existing_keys` below (and any plain
# `split(',')` a reader reaches for) depends on that, so `dims` joins with
# 'x' and `selected`/`notes` with ';' rather than the default array/comma
# rendering.
sanitize_field(x) = replace(replace(string(x), "," => ";"), "\n" => " ")

function append_row!(row::NamedTuple)
    csv_header!()
    open(CSV_PATH, "a") do io
        @printf(io, "%s,%s,%s,%d,%d,%s,%d,%s,%s,%d,%d,%s,%s,%d,%d,%s,%.6g,%s,%.6g,%.6g,%.6f,%s\n",
                row.family, row.case, row.dims, row.valence, row.expected_final,
                row.T, row.seed, row.config, row.solver, row.restricted_nullity,
                row.final_nullity, row.certified, row.status, row.undecidable, row.near_null,
                row.rule, row.gap, row.selected, row.next_value,
                row.max_zlaw_resid, row.seconds, sanitize_field(row.notes))
    end
    return nothing
end

existing_keys() = isfile(CSV_PATH) ?
    Set(begin
            f = split(strip(l), ',')
            (f[2], f[6], f[7], f[8])   # case, T, seed, config
        end for l in readlines(CSV_PATH)[2:end] if !isempty(strip(l))) :
    Set{NTuple{4,String}}()

# ------------------------------------------------------------------ the grid

const SEEDS = (20260904, 20260905, 20260906, 20260907, 20260908, 20260909)
const TYPES = (Float64, Float32, Float16)
const CONFIGS = (
    (; tag = "default",   force_mf = false, solver = :AutoSolver),
    (; tag = "mf-arpack",  force_mf = true,  solver = :ArpackSolver),
    (; tag = "mf-krylov",  force_mf = true,  solver = :KrylovSolver),
)

"""One tensor-family case: a name, a builder `(T, seed) -> (; Γ, Ω, ch, dims)`,
its valence and the known/expected FINAL nullity."""
struct Case
    name::String
    family::String
    valence::Int
    expected::Int
    build::Function
end

function video_smooth(dims, seed::Integer, T::Type; texture::Real = 1e-3)
    Random.seed!(seed)
    A = zeros(Float64, dims...)
    for I in CartesianIndices(A)
        A[I] = sum(Tuple(I) ./ dims)     # smooth ramp, linear in every axis
    end
    A .+= texture .* randn(dims...)
    fr = [Index(d, "a$i") for (i, d) in enumerate(dims)]
    Γ = ITensor(T === Float64 ? A : Array{T}(A), fr...)
    Ω = IndTransverseOps(fr, UniversalOp())
    return (; Γ, Ω, ch = UniversalChisel(length(dims)), dims = collect(dims))
end

function video_random(dims, seed::Integer, T::Type)
    Random.seed!(seed)
    A = randn(dims...)
    fr = [Index(d, "a$i") for (i, d) in enumerate(dims)]
    Γ = ITensor(T === Float64 ? A : Array{T}(A), fr...)
    Ω = IndTransverseOps(fr, UniversalOp())
    return (; Γ, Ω, ch = UniversalChisel(length(dims)), dims = collect(dims))
end

sphere_case(d::Integer, valence::Integer) =
    (T, seed) -> begin
        inp = build_sphere(d; valence, T, seed)
        (; Γ = inp.Γ, Ω = inp.Ω, ch = inp.ch, dims = collect(inp.dims))
    end

const CASES = [
    Case("sphere3-48",  "sphere", 3, 3, sphere_case(48, 3)),
    Case("sphere3-64",  "sphere", 3, 3, sphere_case(64, 3)),
    Case("sphere3-100", "sphere", 3, 3, sphere_case(100, 3)),
    Case("sphere3-140", "sphere", 3, 3, sphere_case(140, 3)),
    Case("sphere4-24",  "sphere", 4, 4, sphere_case(24, 4)),
    Case("sphere4-40",  "sphere", 4, 4, sphere_case(40, 4)),
    Case("video-smooth", "video", 4, 3, (T, seed) -> video_smooth((40, 40, 20, 3), seed, T)),
    Case("video-random", "video", 4, 3, (T, seed) -> video_random((60, 60, 30, 3), seed, T)),
]

case_by_name(name) = only(filter(c -> c.name == name, CASES))
config_by_tag(tag) = only(filter(c -> c.tag == tag, CONFIGS))

# ------------------------------------------------------------------ one cell

"""
    run_cell(case, T, seed, cfg) -> NamedTuple row

Builds the tensor once, sets the dense-budget lever the config asks for, runs
`derTrOpsReduced(:QuickDer, ...; return_diagnostics = true)`, and returns the
CSV row.  Never throws: an exception is caught and recorded as `status =
"error: ..."` with every count at -1, so a bad cell does not stop the sweep.
"""
function run_cell(case::Case, T::Type, seed::Integer, cfg)
    Dleto.QDN_DENSE_BUDGET_BYTES[] = cfg.force_mf ? 0.0 : QDN_DEFAULT_BUDGET
    row = nothing
    try
        inp = case.build(T, seed)
        m = Dleto.get_derivation_method(:QuickDer; seed = seed, solver = cfg.solver)
        local rep
        t = @elapsed begin
            (_, _, ders, rep) = Dleto.derTrOpsReduced(m, inp.Ω, inp.ch, inp.Γ;
                                                       return_diagnostics = true)
        end
        resid = isempty(rep.residuals === nothing ? Float64[] : rep.residuals) ? NaN :
                maximum(rep.residuals)
        row = (; family = case.family, case = case.name, dims = join(inp.dims, "x"),
                 valence = case.valence, expected_final = case.expected,
                 T = string(T), seed = Int(seed), config = cfg.tag,
                 solver = string(rep.solver === nothing ? cfg.solver : rep.solver),
                 restricted_nullity = rep.nullity, final_nullity = rep.returned,
                 certified = rep.certified, status = string(rep.status),
                 undecidable = rep.undecidable, near_null = rep.near_null,
                 rule = string(rep.rule), gap = rep.gap,
                 selected = join(round.(rep.selected; sigdigits = 4), ";"),
                 next_value = rep.next_value, max_zlaw_resid = resid,
                 seconds = t, notes = "")
    catch e
        msg = "error: " * first(split(sprint(showerror, e), '\n'))
        row = (; family = case.family, case = case.name, dims = "?",
                 valence = case.valence, expected_final = case.expected,
                 T = string(T), seed = Int(seed), config = cfg.tag,
                 solver = string(cfg.solver), restricted_nullity = -1, final_nullity = -1,
                 certified = false, status = "error", undecidable = -1, near_null = -1,
                 rule = "none", gap = NaN, selected = "", next_value = NaN,
                 max_zlaw_resid = NaN, seconds = NaN, notes = msg)
    finally
        Dleto.QDN_DENSE_BUDGET_BYTES[] = QDN_DEFAULT_BUDGET
    end
    return row
end

"""
    run_full!(; part = 0, nparts = 1)

Runs every (case, T, seed, config) cell not already in the CSV, appending as
each completes.  `part`/`nparts` let several `bench/jl` processes share the
grid: cell number `i` (1-based, in the fixed nested-loop order below) runs
here iff `(i - 1) % nparts == part` -- a round-robin split, so each partition
sees a mix of cheap and expensive cells rather than one partition getting all
the slow d = 140 rows.  `existing_keys` is still checked first, so a partition
that is re-run after an interruption skips whatever another partition (or an
earlier attempt) already wrote.
"""
function run_full!(; part::Integer = 0, nparts::Integer = 1)
    csv_header!()
    seen = existing_keys()
    n = length(CASES) * length(TYPES) * length(SEEDS) * length(CONFIGS)
    i = 0
    for case in CASES, T in TYPES, seed in SEEDS, cfg in CONFIGS
        i += 1
        (i - 1) % nparts == part || continue
        key = (case.name, string(T), string(seed), cfg.tag)
        if key in seen
            continue
        end
        row = run_cell(case, T, seed, cfg)
        append_row!(row)
        @printf("[%4d/%4d] %-14s %-8s seed=%d %-10s -> restricted=%d final=%d/%d \
certified=%s status=%s undecidable=%d resid=%.2e t=%.2fs %s\n",
                i, n, case.name, string(T), seed, cfg.tag, row.restricted_nullity,
                row.final_nullity, row.expected_final, row.certified, row.status,
                row.undecidable, row.max_zlaw_resid, row.seconds, row.notes)
        flush(stdout)
    end
    return nothing
end

function run_one(casename, Tname, seedstr, cfgtag)
    case = case_by_name(casename)
    T = Tname == "Float64" ? Float64 : Tname == "Float32" ? Float32 : Float16
    seed = parse(Int, seedstr)
    cfg = config_by_tag(cfgtag)
    row = run_cell(case, T, seed, cfg)
    append_row!(row)
    println(row)
    return row
end

function run_spectrum(casename, Tname, seedstr, cfgtag)
    case = case_by_name(casename)
    T = Tname == "Float64" ? Float64 : Tname == "Float32" ? Float32 : Float16
    seed = parse(Int, seedstr)
    cfg = config_by_tag(cfgtag)
    Dleto.QDN_DENSE_BUDGET_BYTES[] = cfg.force_mf ? 0.0 : QDN_DEFAULT_BUDGET
    inp = case.build(T, seed)
    m = Dleto.get_derivation_method(:QuickDer; seed = seed, solver = cfg.solver)
    (_, _, ders, rep) = Dleto.derTrOpsReduced(m, inp.Ω, inp.ch, inp.Γ; return_diagnostics = true)
    Dleto.QDN_DENSE_BUDGET_BYTES[] = QDN_DEFAULT_BUDGET
    v = rep.verdict
    println("case=$(case.name) T=$T seed=$seed config=$(cfg.tag) solver=$(rep.solver)")
    println("nullity(restricted)=$(rep.nullity) returned=$(rep.returned) expected=$(case.expected)")
    println("certified=$(rep.certified) status=$(rep.status) rule=$(rep.rule) gap=$(rep.gap)")
    println("undecidable=$(rep.undecidable) near_null=$(rep.near_null)")
    if v !== nothing
        println("floor=$(v.floor) data_floor=$(v.data_floor) threshold=$(v.threshold)")
        println("spectrum (ascending, relative), $(length(v.spectrum)) values:")
        for (i, x) in enumerate(v.spectrum)
            marker = i == v.nullity ? "  <-- cut" : ""
            println("  [$i] $x$marker")
        end
    end
    println("selected: $(rep.selected)")
    println("next_value: $(rep.next_value)")
    println("Z-law residuals: $(rep.residuals)")
    return rep
end

function main(args)
    isempty(args) && error("first argument must be list, full, cell, or spectrum")
    task = args[1]
    if task == "list"
        i = 0
        for case in CASES, T in TYPES, seed in SEEDS, cfg in CONFIGS
            i += 1
            println(i, "  ", case.name, "  ", T, "  ", seed, "  ", cfg.tag)
        end
    elseif task == "full"
        if length(args) >= 3
            run_full!(part = parse(Int, args[2]), nparts = parse(Int, args[3]))
        else
            run_full!()
        end
    elseif task == "cell"
        length(args) >= 5 || error("cell needs <case> <T> <seed> <config>")
        run_one(args[2], args[3], args[4], args[5])
    elseif task == "spectrum"
        length(args) >= 5 || error("spectrum needs <case> <T> <seed> <config>")
        run_spectrum(args[2], args[3], args[4], args[5])
    else
        error("unknown task $task; expected list, full, cell, or spectrum")
    end
    return nothing
end

main(ARGS)
