#
# Precision study, experiment 2b: the reported d = 20 Float32 Arpack slowdown,
# and whether CGSolver (LOBPCG) works in Float32 once its tolerance is
# reachable.
#
#   julia -t 2 --project=. bench/reports/precision-exp2b.jl
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
function sylmap(inp)
    T = eltype(inp.Γ)
    eng = Dleto.engaged(inp.ch)
    (Ωr, _) = Dleto.reduceByEngaged(inp.Ω, eng, T)
    P_eng = Matrix{T}(inp.ch[:, eng])
    return Dleto.sylvesterLM(Ωr, P_eng, inp.Γ)[1]
end
function counting(L)
    cnt = Ref(0)
    T = eltype(L); n = size(L, 2)
    Lc = LinearMap{T}(v -> (cnt[] += 1; L * v), n, n; issymmetric = true, ismutating = false)
    return Lc, cnt
end

# ---------------------------------------------------- d = 20, several seeds
println("== d = 20, Float64 vs Float32, Arpack and Krylov, 4 seeds"); flush(stdout)
pathA = csvopen("precision-exp2b-d20.csv", "d,T,solver,seed,seconds,bytes,nullity,lsq_err,status")
for T in (Float64, Float32)
    warmup!([(; solver = :ArpackSolver), (; solver = :KrylovSolver)]; T)
    for seed in (20, 1020, 2020, 3020), s in (:ArpackSolver, :KrylovSolver)
        inp = build_sphere(20; T, seed)
        r = run_stratify(inp; solver = s)
        emit(pathA, @sprintf("20,%s,%s,%d,%.3f,%d,%d,%.3e,\"%s\"", T, s, seed, r.seconds, r.bytes, r.nullity, r.lsq_err, r.status))
    end
end

# ---------------------------------------------------- LOBPCG tolerance in Float32
println("== CGSolver (LOBPCG) direct, Float32, d = 20 and 30, tol sweep"); flush(stdout)
pathB = csvopen("precision-exp2b-cg.csv", "d,T,tol,nv,seconds,applications,opnorm,n_below_1e-6,n_below_1e-4,lam1,lam2,lam3,lam4,lam5")
cg = Dleto.SOLVER_REGISTRY[:CGSolver]
for T in (Float32, Float64), d in (20, 30)
    inp = build_sphere(d; T)
    L = sylmap(inp)
    nrm = Dleto.opnorm_estimate(L; iters = 30)
    tols = T === Float32 ? (1e-10, 1e-6, 1e-4, 1e-3, 1e-2) : (1e-10, 1e-6)
    for τ in tols
        Lc, cnt = counting(L)
        Random.seed!(1)
        t = @elapsed res = try
            redirect_stdout(devnull) do; Dleto.solve(cg, Lc; nv = 16, tol = τ, maxiter = 500); end
        catch e
            (; vals = Float64[], vecs = zeros(T, size(L, 2), 0), err = first(split(sprint(showerror, e), '\n')))
        end
        v = sort(abs.(Float64.(res.vals)))
        n6 = count(<(1e-6 * nrm), v); n4 = count(<(1e-4 * nrm), v)
        lam = [i <= length(v) ? v[i] / nrm : NaN for i in 1:5]
        emit(pathB, @sprintf("%d,%s,%.0e,16,%.3f,%d,%.3e,%d,%d,%.2e,%.2e,%.2e,%.2e,%.2e%s",
            d, T, τ, t, cnt[], nrm, n6, n4, lam..., haskey(res, :err) ? "  # " * res.err : ""))
    end
end
println("EXP2B DONE")
