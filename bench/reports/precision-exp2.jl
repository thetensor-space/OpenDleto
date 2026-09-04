#
# Precision study, experiment 2: tolerances and the near-zero spectrum.
#
#   julia -t 2 --project=. bench/reports/precision-exp2.jl
#
# A. Float32 `tol` sweep through run_stratify (the RELATIVE null threshold in
#    solve_nullspace) for Arpack and Krylov at d = 30.
# B. Eigensolver tolerance sweep, calling Arpack.eigs / KrylovKit.eigsolve
#    directly on the derivation--densor map L, wrapped in a counting map:
#    time, map applications, iterations, converged count, the returned
#    eigenvalues, and whether exactly 3 sit below the relative threshold.
# C. The exact spectrum near zero by dense symmetric eigen, in Float64 and in
#    Float32, at d = 20, 30, 40, 50: the three "null" eigenvalues relative to
#    lambda_max and the gap to the 4th.
#
# Output: bench/reports/precision-exp2-{A,B,C}.csv plus the printed rows.
#
using KrylovKit, IterativeSolvers, Arpack
using LinearMaps
include(joinpath(@__DIR__, "..", "SphereHarness.jl"))
using Printf
LinearAlgebra.BLAS.set_num_threads(2)

const DIR = @__DIR__
function csvopen(name, header)
    path = joinpath(DIR, name)
    isfile(path) || open(path, "w") do io; println(io, header); end
    return path
end
function emit(path, line)
    println(line); flush(stdout)
    open(path, "a") do io; println(io, line); end
end

"The derivation--densor map L (square, symmetric) exactly as derTrOpsReduced builds it."
function sylmap(inp)
    T = eltype(inp.Γ)
    eng = Dleto.engaged(inp.ch)
    (Ωr, _) = Dleto.reduceByEngaged(inp.Ω, eng, T)
    P_eng = Matrix{T}(inp.ch[:, eng])
    return Dleto.sylvesterLM(Ωr, P_eng, inp.Γ)[1]
end

"Wrap L so applications are counted."
function counting(L)
    cnt = Ref(0)
    T = eltype(L); n = size(L, 2)
    Lc = LinearMap{T}(v -> (cnt[] += 1; L * v), n, n; issymmetric = true, ismutating = false)
    return Lc, cnt
end

# ------------------------------------------------------------------ Part A
println("== Part A: run_stratify tol sweep, Float32, d = 30"); flush(stdout)
pathA = csvopen("precision-exp2-A.csv", "d,T,solver,tol,seconds,nullity,lsq_err,status")
warmup!([(; solver = :ArpackSolver), (; solver = :KrylovSolver)]; T = Float32)
let inp = build_sphere(30; T = Float32)
    for s in (:ArpackSolver, :KrylovSolver), tol in (1e-3, 1e-4, 1e-5, 1e-6)
        r = run_stratify(inp; solver = s, tol)
        emit(pathA, @sprintf("30,Float32,%s,%.0e,%.3f,%d,%.3e,\"%s\"", s, tol, r.seconds, r.nullity, r.lsq_err, r.status))
    end
end

# ------------------------------------------------------------------ Part B
println("== Part B: eigensolver tolerance sweep on L directly"); flush(stdout)
pathB = csvopen("precision-exp2-B.csv",
    "d,T,solver,eigtol,nev,seconds,applications,iterations,converged,opnorm,n_below_1e-6,n_below_1e-4,lam1,lam2,lam3,lam4,lam5")

function report_vals(path, d, T, solver, eigtol, nev, secs, apps, iters, conv, nrm, vals)
    v = sort(abs.(vals))
    n6 = count(<(1e-6 * nrm), v); n4 = count(<(1e-4 * nrm), v)
    lam = [i <= length(v) ? v[i] / nrm : NaN for i in 1:5]
    emit(path, @sprintf("%d,%s,%s,%.0e,%d,%.3f,%d,%d,%d,%.3e,%d,%d,%.2e,%.2e,%.2e,%.2e,%.2e",
        d, T, solver, eigtol, nev, secs, apps, iters, conv, nrm, n6, n4, lam...))
end

for T in (Float32, Float64), d in (20, 30)
    inp = build_sphere(d; T)
    L = sylmap(inp)
    N = size(L, 2)
    nrm = Dleto.opnorm_estimate(L; iters = 30)
    println("-- d=$d T=$T N=$N  opnorm ~ $nrm  eps(T)*opnorm = $(eps(T)*nrm)"); flush(stdout)
    nev = 16
    # Arpack: tol = 0 means ARPACK's machine epsilon for the element type.
    tols_a = T === Float32 ? (0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2) : (0.0, 1e-8, 1e-6, 1e-4)
    for τ in tols_a
        Lc, cnt = counting(L)
        Random.seed!(1)
        t = @elapsed out = Arpack.eigs(Lc; nev, which = :SM, tol = τ, maxiter = 300)
        vals, _, nconv, niter, nmult = out
        report_vals(pathB, d, T, "Arpack", τ, nev, t, cnt[], niter, nconv, nrm, real.(vals))
    end
    # KrylovKit: tol is an ABSOLUTE residual norm.  Sweep absolute values and
    # values relative to the operator norm.
    tols_k = T === Float32 ? (1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1e-7 * nrm, 1e-6 * nrm, 1e-5 * nrm) :
                             (1e-8, 1e-12 * nrm, 1e-10 * nrm)
    for τ in tols_k
        Lc, cnt = counting(L)
        Random.seed!(1)
        x0 = randn(T, N)
        t = @elapsed vals, vecs, info = KrylovKit.eigsolve(Lc, x0, nev, :SR;
                                            tol = τ, krylovdim = 32, maxiter = 100, verbosity = 0)
        report_vals(pathB, d, T, "KrylovKit", τ, nev, t, cnt[], info.numiter, info.converged, nrm, real.(vals))
    end
end

# ------------------------------------------------------------------ Part C
println("== Part C: exact near-zero spectrum by dense eigen"); flush(stdout)
pathC = csvopen("precision-exp2-C.csv",
    "d,N,map_T,eigen_T,densify_s,eigen_s,lam_max,lam1_rel,lam2_rel,lam3_rel,lam4_rel,lam5_rel,gap_ratio_4_over_3,gap_rel")
for d in (20, 30, 40, 50)
    for T in (Float64, Float32)
        inp = build_sphere(d; T)
        L = sylmap(inp)
        N = size(L, 2)
        td = @elapsed M = Matrix(L)
        M = Symmetric((M + M') / 2)
        for ET in (T === Float32 ? (Float32, Float64) : (Float64,))
            ME = ET === T ? M : Symmetric(Matrix{ET}(M))
            te = @elapsed ev = eigvals(ME)
            ev = sort(abs.(Float64.(ev)))
            lmax = maximum(ev)
            rel = ev[1:5] ./ lmax
            emit(pathC, @sprintf("%d,%d,%s,%s,%.2f,%.2f,%.4e,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e",
                d, N, T, ET, td, te, lmax, rel..., ev[4] / max(ev[3], eps()), rel[4] - rel[3]))
        end
        GC.gc()
    end
end
println("EXP2 DONE")
