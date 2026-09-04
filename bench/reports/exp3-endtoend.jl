# Experiment 3: end-to-end through the modified extensions.
#   (a) resilience: run_stratify x 5 seeds x d in (32,36,40) for Arpack, Krylov
#   (b) request size: solve_nullspace nv0 in (4,8,16,32) with matvec counts, d in (32,40)
#   (c) scaling: d in (24,32,40,48,56): Arpack, Krylov, SVD (dense), CG (d<=40); true gap
using KrylovKit, IterativeSolvers, Arpack
using Dleto, LinearMaps, LinearAlgebra, Printf, Random
include(joinpath(@__DIR__, "..", "SphereHarness.jl"))
const R = joinpath(@__DIR__, "")
const PART = length(ARGS) >= 1 ? ARGS[1] : "abc"

function counted_op(inp)
    L = Dleto.sylvesterLM(inp.Ω, inp.ch, inp.Γ)[1]
    n = size(L, 2); cnt = Ref(0)
    Lc = LinearMap{Float64}(v -> (cnt[] += 1; L * v), n, n; issymmetric = true, isposdef = true)
    return L, Lc, cnt
end
fmt(x) = @sprintf("%.3e", x)
rss() = Sys.maxrss() / 2^30

function put(path, header, line)
    isfile(path) || open(path, "w") do io; println(io, header); end
    println(line); flush(stdout)
    open(path, "a") do io; println(io, line); end
end

# ---- warm-up / smoke test
begin
    inp = build_sphere(8)
    for s in (:ArpackSolver, :KrylovSolver, :SVDSolver, :CGSolver)
        r = run_stratify(inp; solver = s, tol = 1e-6)
        @printf("warmup %-14s nullity=%d lsq=%s status=%s\n", s, r.nullity, fmt(r.lsq_err), r.status)
        r.nullity == 3 || error("warm-up failed for $s: $(r.status)")
    end
    L, Lc, cnt = counted_op(inp)
    for s in (:ArpackSolver, :KrylovSolver)
        v = Dleto.solve_nullspace(Lc, s; tol = 1e-6, nv0 = 4)
        println("warmup solve_nullspace $s -> ", length(v[1]), " null values, ", cnt[], " matvecs"); cnt[] = 0
    end
end

# ---- (a) resilience over seeds, end to end
if occursin('a', PART)
    H = "d,seed,solver,seconds,bytes,rss_gb,nullity,lsq_err,support,status"
    for d in (32, 36, 40), seed in 1:5
        inp = build_sphere(d; seed)
        for s in (:ArpackSolver, :KrylovSolver)
            r = run_stratify(inp; solver = s, tol = 1e-6)
            put(R * "exp3-seeds.csv", H, @sprintf("%d,%d,%s,%.3f,%d,%.3f,%d,%s,%.6f,\"%s\"",
                d, seed, s, r.seconds, r.bytes, rss(), r.nullity, fmt(r.lsq_err), r.support, r.status))
        end
    end
end

# ---- (b) request size
if occursin('b', PART)
    H = "d,seed,solver,nv0,seconds,matvecs,nullity,max_null_resid_rel"
    for d in (32, 40), seed in (d, 5)
        inp = build_sphere(d; seed)
        L, Lc, cnt = counted_op(inp)
        lmax = Dleto.opnorm_estimate(Lc; iters = 30)
        for s in (:ArpackSolver, :KrylovSolver), nv0 in (4, 8, 16, 32)
            Random.seed!(7); cnt[] = 0
            t = @elapsed (vals, V) = quietly(() -> Dleto.solve_nullspace(Lc, s; tol = 1e-6, nv0))
            res = [norm(L * V[:, j]) / (lmax * norm(V[:, j])) for j in 1:size(V, 2)]
            put(R * "exp3-nv0.csv", H, @sprintf("%d,%d,%s,%d,%.3f,%d,%d,%s", d, seed, s, nv0, t, cnt[],
                length(vals), fmt(isempty(res) ? NaN : maximum(res))))
        end
    end
end

# ---- (c) scaling
if occursin('c', PART)
    H = "d,n,solver,seconds,bytes,rss_gb,nullity,lsq_err,matvecs,l4_over_lmax,status"
    dense_budget_ok = true
    for d in (24, 32, 40, 48, 56)
        inp = build_sphere(d)
        L, Lc, cnt = counted_op(inp)
        n = size(L, 2)
        gap = NaN
        if d <= 48
            tg = @elapsed lam = eigvals(Symmetric(Matrix(L)))
            gap = lam[4] / lam[end]
            @printf("d=%d n=%d dense eigvals %.1fs  l1..l5 = %s %s %s %s %s  lmax=%s gap=%.3e\n", d, n, tg,
                    fmt(lam[1]), fmt(lam[2]), fmt(lam[3]), fmt(lam[4]), fmt(lam[5]), fmt(lam[end]), gap)
        end
        solvers = Symbol[:ArpackSolver, :KrylovSolver]
        dense_budget_ok && push!(solvers, :SVDSolver)
        d <= 40 && push!(solvers, :CGSolver)
        for s in solvers
            # matvec count of the null solve alone, then the full timed stratification
            cnt[] = 0
            quietly(() -> Dleto.solve_nullspace(Lc, s; tol = 1e-6))
            mv = cnt[]
            r = run_stratify(inp; solver = s, tol = 1e-6)
            put(R * "exp3-scaling.csv", H, @sprintf("%d,%d,%s,%.3f,%d,%.3f,%d,%s,%d,%.3e,\"%s\"",
                d, n, s, r.seconds, r.bytes, rss(), r.nullity, fmt(r.lsq_err), mv, gap, r.status))
            s === :SVDSolver && r.seconds > 90 && (dense_budget_ok = false)
        end
    end
end
println("EXP3 DONE")
