#
# Tuning the precision policy's constants by experiment.
#
# Exp. A (this file, stage A): the constants `FLOOR_EPS` (the `c` in
# `c·eps(T)·‖L‖`) and `tol` change only the VERDICT, not the solve.  So the
# solve happens once per (case, element type, solver), its spectrum is kept, and
# every candidate constant is scored against that spectrum offline through
# `gap_verdict`.  That buys a 9-point x 7-point sweep for the price of one run
# and, more importantly, makes the sweep exact: two constants are compared on
# the same numbers, not on two runs of a randomised solver.
#
# Exp. B (stage B): the constants that change the SOLVE -- LSMR's `lsmr_tol` and
# `rank_tol`, and `qd_tolerance`'s floor for QuickDer -- need real runs, so they
# get a coarser grid.
#
#   JL_PROJECT=$(pwd) bench/jl bench/reports/precision-tune.jl [A|B]
#
# Writes bench/reports/precision-tune-a.csv and -b.csv.
#
using Dleto, ITensors, LinearAlgebra, Random, Printf
using Arpack, KrylovKit, IterativeSolvers

# `LSMRSolver` is defined inside DletoIterativeSolversExt, not in Dleto, so
# stage B reaches it through the registry rather than by name.
const LSMRSolverT = typeof(Dleto.SOLVER_REGISTRY[:LSMRSolver])

include(joinpath(@__DIR__, "..", "SphereHarness.jl"))

const OUT = @__DIR__
const STAGE = isempty(ARGS) ? "AB" : uppercase(ARGS[1])

quiet(f) = redirect_stdout(f, devnull)

# ------------------------------------------------------------------ the cases
#
# Each case is (name, truth, Γ-builder).  `truth` is the nullity a Float64 run
# must return; for the near-degenerate pair it is the count that the Float64
# spectrum supports, recorded from the run rather than asserted.

"""Random dense valence-`n` tensor: only the `n-1` scalar derivations."""
function rand_case(dims; seed = 1234)
    Random.seed!(seed)
    fr = [Index(d, "a$i") for (i, d) in enumerate(dims)]
    return ITensor(randn(dims...), fr...)
end

"""Video-shaped box: rows x cols x frames x channels, smooth in space+time."""
function video_case(dims; seed = 99)
    Random.seed!(seed)
    A = randn(dims...)
    # A smooth spatial/temporal field plus noise -- the structure real video has,
    # and the reason the derivation spectrum has a soft shoulder.
    for I in CartesianIndices(A)
        t = sum(Tuple(I) ./ dims)
        A[I] = sinpi(t) + 0.1 * A[I]
    end
    fr = [Index(d, "a$i") for (i, d) in enumerate(dims)]
    return ITensor(A, fr...)
end

"""`ramp + delta*noise` at 6^3: the near-degenerate family of TestDerivationLaws."""
function neardeg_case(delta; n = 6, seed = 20260908)
    Random.seed!(seed)
    ramp = [i + j + k for i in 1:n, j in 1:n, k in 1:n] .+ 0.0
    fr = [Index(n, "a$i") for i in 1:3]
    return ITensor(ramp .+ delta .* randn(n, n, n), fr...)
end

const CASES = [
    # `ops` is the operator space the case is scored in.  The sphere harness
    # stratifies with `SymmetricOp` and its truth (nullity = valence) is stated
    # for that space; under `UniversalOp` the same tensor has nullity 13, which
    # is a different (and much harder) question.
    (; name = "sphere3-d10", truth = 3, ops = SymmetricOp(),
       build = () -> build_sphere(10; valence = 3).Γ),
    (; name = "sphere3-d16", truth = 3, ops = SymmetricOp(),
       build = () -> build_sphere(16; valence = 3).Γ),
    (; name = "sphere4-d6",  truth = 4, ops = SymmetricOp(),
       build = () -> build_sphere(6; valence = 4).Γ),
    (; name = "rand3-d12",   truth = 2, ops = UniversalOp(),
       build = () -> rand_case((12, 12, 12))),
    (; name = "rand4-d6",    truth = 3, ops = UniversalOp(),
       build = () -> rand_case((6, 6, 6, 6))),
    (; name = "video-20x20x10x3", truth = 3, ops = UniversalOp(),
       build = () -> video_case((20, 20, 10, 3))),
    (; name = "video-40x40x20x3", truth = 3, ops = UniversalOp(),
       build = () -> video_case((40, 40, 20, 3))),
    (; name = "neardeg-1e-3", truth = 2, ops = UniversalOp(),
       build = () -> neardeg_case(1e-3)),
    (; name = "neardeg-1e-8", truth = -1, ops = UniversalOp(),
       build = () -> neardeg_case(1e-8)),
]

const TYPES = (Float64, Float32, Float16)

"""
    sylvester_map(Γ) -> (L, scale, n)

The square derivation operator `AᵗA` that `SylverLining` hands a null solver,
in `compute_eltype(eltype(Γ))`, plus the operator-norm estimate the relative
spectrum divides by.  Mirrors `derTrOpsReduced(::SylverLiningMethod, ...)` up to
the `solve_nullspace` call.
"""
function sylvester_map(Γ::ITensor, ops::Operator = UniversalOp())
    fr = collect(inds(Γ))
    T = eltype(Γ)
    Tc = Dleto.compute_eltype(T)
    Γc = T === Tc ? Γ : ITensor(Array{Tc}(ITensors.array(Γ, fr...)), fr...)
    Ω = IndTransverseOps(fr, ops)
    P = UniversalChisel(length(fr))
    eng = engaged(P)
    (Ωr, _) = reduceByEngaged(Ω, eng, Tc)
    L, _ = Dleto.sylvesterLM(Ωr, Matrix{Tc}(P[:, eng]), Γc)
    # `solve_nullspace`'s own scale, so the relative spectra match what the
    # library sees.  `L` is already `AᵗA`, hence the extra square root.
    scale = max(sqrt(max(Dleto.opnorm_estimate(L' * L; iters = 10), 0.0)), eps(real(Tc)))
    return L, scale, size(L, 2)
end

# --------------------------------------------------------------- Exp. A: verdict

const A_SOLVERS = (:SVDSolver, :GramSolver, :ArpackSolver, :KrylovSolver)
const A_CS   = (1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 100.0, 300.0, 1e3, 1e4)
const A_TOLS = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9)

"""
    spectrum_of(L, scale, solver; nv) -> (rel, seconds)

The values `solver` returns on `L`, relative to `scale`, sorted.  `nv` is
generous: the verdict sweep needs values on BOTH sides of every candidate cut,
so ask for more than any candidate nullity.
"""
function spectrum_of(L, scale, solver::Symbol; nv::Integer = 24)
    t = @elapsed res = quiet() do
        Dleto.solve(Dleto.SOLVER_REGISTRY[solver], L; nv = min(nv, size(L, 2)))
    end
    rel = sort!(Float64[abs(v) / scale for v in res.vals])
    return rel, t
end

function stage_A()
    rows = String[]
    push!(rows, "case,truth,T,n,solver,knob,value,nullity,rule,certified,undecidable," *
                "floor_binding,gap,near_null,solve_s,spec1,spec2,spec3,spec4,spec5")
    for case in CASES, T in TYPES
        Γ0 = nothing
        try
            Γ0 = case.build()
        catch e
            @warn "build failed" case.name e
            continue
        end
        fr = collect(inds(Γ0))
        Γ = T === Float64 ? Γ0 :
            ITensor(Array{T}(ITensors.array(Γ0, fr...)), fr...)
        L, scale, n = try
            sylvester_map(Γ, case.ops)
        catch e
            @warn "map failed" case.name T e
            continue
        end
        Tc = Dleto.compute_eltype(T)
        @printf("%-22s %-8s n=%-6d scale=%.3g\n", case.name, string(T), n, scale)
        for solver in A_SOLVERS
            rel, ts = try
                spectrum_of(L, scale, solver)
            catch e
                println("    ", rpad(string(solver), 14), " THROWN ",
                        first(split(sprint(showerror, e), '\n')))
                continue
            end
            spec5 = [i <= length(rel) ? rel[i] : NaN for i in 1:5]
            @printf("    %-14s %.2fs  spectrum %s\n", string(solver), ts,
                    join(map(x -> @sprintf("%.2g", x), spec5), " "))
            # sweep c with tol fixed, then tol with c fixed
            for (knob, vals) in (("c", A_CS), ("tol", A_TOLS))
                for v in vals
                    c   = knob == "c"   ? v : Float64(Dleto.FLOOR_EPS)
                    tol = knob == "tol" ? v : Dleto.TOL_DEFAULT
                    # `solve_nullspace` squares the ceiling for this Gram map.
                    fl  = c * eps(real(Tc))
                    thr = max(tol^2, fl)
                    _, vd = Dleto.gap_verdict(rel, 1.0; threshold = thr, floor = fl,
                                              data_floor = Dleto.data_floor(T),
                                              gap_ratio = Dleto.GAP_RATIO)
                    push!(rows, join((case.name, case.truth, T, n, solver, knob, v,
                                      vd.nullity, vd.rule, vd.certified, vd.undecidable,
                                      vd.floor_binding,
                                      @sprintf("%.4g", vd.gap), vd.near_null,
                                      @sprintf("%.3f", ts),
                                      (@sprintf("%.4g", x) for x in spec5)...), ","))
                end
            end
        end
        L = nothing; Γ = nothing; GC.gc()
    end
    open(joinpath(OUT, "precision-tune-a.csv"), "w") do io
        for r in rows; println(io, r); end
    end
    println("wrote precision-tune-a.csv (", length(rows) - 1, " rows)")
end

# ------------------------------------------- Exp. B: constants that change the solve

const B_CASES = ("sphere3-d10", "sphere4-d6", "rand3-d12", "rand4-d6",
                 "video-20x20x10x3", "neardeg-1e-3", "neardeg-1e-8")

"""
    stage_B()

Real runs for the knobs a captured spectrum cannot answer.

1. `LSMRSolver`'s `lsmr_tol` (the projector's least-squares tolerance) and
   `rank_tol` (the pivot cut of its revealing QR).  Both are absolute Float64
   numbers below `eps(Float32)`, so unlike block Lanczos's unreachable
   tolerance -- where flooring was free -- these are a genuine accuracy/time
   trade and have to be measured.  Scored on the map directly: nullity, and the
   worst relative residual `‖L v‖ / (‖L‖ ‖v‖)` over the returned vectors, which
   is the quantity a caller cares about.

2. `tol` end to end through `QuickDer`, swept across and below
   `qd_tolerance`'s floor.  Values under the floor are clamped to it, so they
   all agree -- which is the floor demonstrating its purpose -- and the sweep
   locates the upper edge of the plateau.
"""
function stage_B()
    rows = String[]
    push!(rows, "case,truth,T,route,knob,value,nullity,max_rel_resid,seconds,status")
    byname = Dict(c.name => c for c in CASES)

    for cname in B_CASES, T in TYPES
        case = byname[cname]
        Γ0 = case.build()
        fr = collect(inds(Γ0))
        Γ = T === Float64 ? Γ0 : ITensor(Array{T}(ITensors.array(Γ0, fr...)), fr...)
        Tc = Dleto.compute_eltype(T)
        RT = real(Tc)
        L, scale, n = sylvester_map(Γ, case.ops)
        @printf("%-22s %-8s n=%-6d\n", cname, string(T), n)

        # worst relative residual of a returned basis, on the SQUARED map, so
        # comparable to the relative spectrum the verdict sees
        function score(vecs)
            size(vecs, 2) == 0 && return NaN
            m = 0.0
            for j in 1:size(vecs, 2)
                v = vecs[:, j]
                nv = norm(v)
                nv == 0 && continue
                m = max(m, norm(L * v) / (scale * nv))
            end
            return m
        end

        # --- 1. LSMR
        for (knob, vals) in (
            ("rank_tol", (1e-8, Float64(eps(RT)), 10 * Float64(eps(RT)),
                          sqrt(n) * Float64(eps(RT)), n * Float64(eps(RT)))),
            ("lsmr_tol", (1e-12, Float64(eps(RT)), 10 * Float64(eps(RT)),
                          100 * Float64(eps(RT)), sqrt(Float64(eps(RT))))),
        )
            for v in vals
                base = Dleto.SOLVER_REGISTRY[:LSMRSolver]
                m = knob == "rank_tol" ?
                    LSMRSolverT(base.lsmr_tol, base.lsmr_maxiter, v, base.margin,
                                base.refine) :
                    LSMRSolverT(v, base.lsmr_maxiter, base.rank_tol, base.margin,
                                base.refine)
                nullity = -1; resid = NaN; status = "ok"; t = 0.0
                try
                    t = @elapsed out = quiet() do
                        Dleto.solve_nullspace(L, m; squared = true, store_eltype = real(T))
                    end
                    nullity = size(out.vecs, 2)
                    resid = score(out.vecs)
                catch e
                    status = first(split(sprint(showerror, e), '\n'))
                end
                push!(rows, join((cname, case.truth, T, "LSMRSolver", knob,
                                  @sprintf("%.3g", v), nullity,
                                  @sprintf("%.3g", resid), @sprintf("%.2f", t), status), ","))
            end
        end

        # --- 2. QuickDer tol sweep (only where QuickDer applies: valence >= 3
        #        with an engaged universal chisel, which is every case here)
        Ω = IndTransverseOps(fr, case.ops)
        P = UniversalChisel(length(fr))
        for v in (1e-9, 1e-7, Dleto.qd_tolerance(T), 3 * Dleto.qd_tolerance(T),
                  10 * Dleto.qd_tolerance(T), 30 * Dleto.qd_tolerance(T))
            nullity = -1; status = "ok"; t = 0.0
            try
                t = @elapsed out = quiet() do
                    Dleto.derTrOpsReduced(Dleto.QuickDerMethod(), Ω, P, Γ; tol = v)
                end
                nullity = size(out[3], 2)
            catch e
                status = first(split(sprint(showerror, e), '\n'))
            end
            push!(rows, join((cname, case.truth, T, "QuickDer", "tol",
                              @sprintf("%.3g", v), nullity, "", @sprintf("%.2f", t),
                              status), ","))
        end
        L = nothing; Γ = nothing; GC.gc()
    end
    open(joinpath(OUT, "precision-tune-b.csv"), "w") do io
        for r in rows; println(io, r); end
    end
    println("wrote precision-tune-b.csv (", length(rows) - 1, " rows)")
end

# Only when run as a script -- precision-tune-iter.jl includes this file for
# `CASES` and `sylvester_map` and must not trigger the sweeps.
if abspath(PROGRAM_FILE) == @__FILE__
    occursin("A", STAGE) && stage_A()
    occursin("B", STAGE) && stage_B()
end
