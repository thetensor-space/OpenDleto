# Run: julia -t 2 --project=. bench/reports/exp1-grid.jl   (after `using Arpack` is resolvable)
# Experiment 1: parameter grid for Arpack and KrylovKit on the sphere derivation operator.
using KrylovKit, IterativeSolvers, Arpack
using Dleto, LinearMaps, LinearAlgebra, Printf, Random
include(joinpath(@__DIR__, "..", "SphereHarness.jl"))
const OUT = joinpath(@__DIR__, "exp1-grid.csv")
const TRUTH = joinpath(@__DIR__, "exp1-truth.csv")

function counted_op(inp)
    L = Dleto.sylvesterLM(inp.Ω, inp.ch, inp.Γ)[1]
    n = size(L, 2); cnt = Ref(0)
    Lc = LinearMap{Float64}(v -> (cnt[] += 1; L * v), n, n; issymmetric = true, isposdef = true)
    return L, Lc, cnt
end

fmt(x) = @sprintf("%.3e", x)

function row(io, d, solver, cfg, t, nmv, vals, thr, lam_true, conv)
    s = sort(vals)
    nbelow = count(<(thr), s)
    v4 = length(s) >= 4 ? s[4] : NaN
    line = @sprintf("%d,%s,%s,%.4f,%d,%d,%d,%s,%s,%s,%s,%s,%s", d, solver, cfg, t, nmv, conv, nbelow,
                    fmt(get(s,1,NaN)), fmt(get(s,2,NaN)), fmt(get(s,3,NaN)), fmt(v4), fmt(thr), fmt(lam_true[4]))
    println(line); flush(stdout)
    println(io, line); flush(io)
end

isfile(OUT) || open(OUT, "w") do io
    println(io, "d,solver,cfg,seconds,matvecs,converged,nbelow,l1,l2,l3,l4,threshold,l4_true")
end
isfile(TRUTH) || open(TRUTH, "w") do io
    println(io, "d,n,l1,l2,l3,l4,l5,lmax,gap_l4_over_lmax,thr_1e-6,dense_seconds,scale_est")
end

# ---- warm-up
let inp = build_sphere(8); L, Lc, cnt = counted_op(inp)
    Arpack.eigs(Lc; nev = 4, which = :SM, tol = 1e-8)
    eigsolve(Lc, randn(size(L,2)), 4, :SR; tol = 1e-8, krylovdim = 12, issymmetric = true)
    eigsolve(Lc, randn(size(L,2)), 4, :SR; tol = 1e-8, krylovdim = 12)
    eigen(Symmetric(Matrix(L)))
end

for d in (32, 36, 40)
    inp = build_sphere(d)
    L, Lc, cnt = counted_op(inp)
    n = size(L, 2)
    # ground truth
    td = @elapsed E = eigen(Symmetric(Matrix(L)))
    lam = E.values
    lmax = lam[end]
    thr = 1e-6 * lmax
    scale_est = sqrt(Dleto.opnorm_estimate(L' * L; iters = 10))
    @printf("d=%d n=%d dense %.2fs  lam1..5 = %s %s %s %s %s  lmax=%s  gap l4/lmax=%.3e  thr=%s  scale_est=%s\n",
            d, n, td, fmt(lam[1]), fmt(lam[2]), fmt(lam[3]), fmt(lam[4]), fmt(lam[5]), fmt(lmax), lam[4]/lmax, fmt(thr), fmt(scale_est))
    open(TRUTH, "a") do io
        @printf(io, "%d,%d,%s,%s,%s,%s,%s,%s,%.4e,%s,%.3f,%s\n", d, n, fmt(lam[1]), fmt(lam[2]), fmt(lam[3]), fmt(lam[4]), fmt(lam[5]), fmt(lmax), lam[4]/lmax, fmt(thr), td, fmt(scale_est))
    end
    open(OUT, "a") do io
        # ---- Arpack grid
        for nev in (4, 8, 16), ncvf in (:def, :x2), tol in (0.0, 1e-10, 1e-8, 1e-6)
            ncv = ncvf === :def ? max(20, 2nev + 1) : 2 * max(20, 2nev + 1)
            Random.seed!(1); cnt[] = 0
            t = @elapsed r = Arpack.eigs(Lc; nev, ncv, which = :SM, tol, maxiter = 300)
            vals = real.(r[1]); nconv = r[3]
            row(io, d, "Arpack", "nev=$nev;ncv=$ncv;tol=$tol", t, cnt[], vals, thr, lam, nconv)
        end
        # ---- KrylovKit grid
        for nev in (4, 8, 16), kdf in (2, 4, :k64), tol in (1e-12, 1e-8, 1e-6), herm in (false, true)
            kd = kdf === :k64 ? 64 : max(10, kdf * nev)
            Random.seed!(1); cnt[] = 0
            x0 = randn(n)
            t = @elapsed (va, ve, info) = eigsolve(Lc, x0, nev, :SR; tol, krylovdim = kd, maxiter = 100,
                                                   issymmetric = herm, verbosity = 0)
            row(io, d, herm ? "KK-Lanczos" : "KK-Arnoldi", "nev=$nev;kd=$kd;tol=$tol", t, cnt[], real.(va), thr, lam, info.converged)
        end
    end
end
println("EXP1 DONE")
