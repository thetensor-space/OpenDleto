# Experiment 2: resilience of candidate null-space strategies over seeds.
#   Arpack :SM (nev 4/16, ncv 20/64/128), KK Lanczos (large krylovdim), KK Lanczos + deflation
#   guard, KK BlockLanczos (block 2/4/8).  5 seeds x d in (32, 36, 40).
using KrylovKit, IterativeSolvers, Arpack
using Dleto, LinearMaps, LinearAlgebra, Printf, Random
include(joinpath(@__DIR__, "..", "SphereHarness.jl"))
const OUT = joinpath(@__DIR__, "exp2-seeds.csv")
const DS = length(ARGS) >= 1 ? parse.(Int, split(ARGS[1], ",")) : [32, 36, 40]
const SEEDS = length(ARGS) >= 2 ? parse.(Int, split(ARGS[2], ",")) : [1, 2, 3, 4, 5]

function counted_op(inp)
    L = Dleto.sylvesterLM(inp.Ω, inp.ch, inp.Γ)[1]
    n = size(L, 2); cnt = Ref(0)
    Lc = LinearMap{Float64}(v -> (cnt[] += 1; L * v), n, n; issymmetric = true, isposdef = true)
    return L, Lc, cnt
end
fmt(x) = @sprintf("%.2e", x)

# ---- strategies: each returns (vals, V) with V columns the vectors for vals
arpack(Lc, nev, ncv, tol) = begin
    r = Arpack.eigs(Lc; nev, ncv, which = :SM, tol, maxiter = 300)
    (real.(r[1]), real.(r[2]))
end

function kk_lanczos(Lc, nev, kd, tol)
    n = size(Lc, 2)
    va, ve, info = eigsolve(Lc, randn(n), nev, :SR; tol, krylovdim = kd, maxiter = 200,
                            issymmetric = true, verbosity = 0)
    (real.(va), reduce(hcat, ve))
end

# Lanczos, then deflate the vectors found under `thr` and re-solve for the next smallest
# from a fresh random start, until the smallest found is above `thr` (bracket proven).
function kk_deflated(Lc, nev, kd, tol, thr, c)
    n = size(Lc, 2)
    vals = Float64[]; V = zeros(n, 0)
    rounds = 0
    while true
        rounds += 1
        Vk = V
        Ld = size(Vk, 2) == 0 ? Lc :
             LinearMap{Float64}(v -> Lc * v + c * (Vk * (Vk' * v)), n, n; issymmetric = true, isposdef = true)
        va, ve, info = eigsolve(Ld, randn(n), nev, :SR; tol, krylovdim = kd, maxiter = 200,
                                issymmetric = true, verbosity = 0)
        va = real.(va); W = reduce(hcat, ve)
        new = findall(<(thr), va)
        if isempty(new)
            # bracket: report the smallest non-null so the caller sees a bracket
            append!(vals, va[1:1]); V = hcat(V, W[:, 1:1])
            return vals, V, rounds
        end
        # orthogonalize new vectors against V (deflation makes them nearly orthogonal already)
        Wn = W[:, new]
        if size(V, 2) > 0
            Wn = Wn - V * (V' * Wn)
        end
        Q = Matrix(qr(Wn).Q)[:, 1:size(Wn, 2)]
        append!(vals, va[new]); V = hcat(V, Q)
        rounds > 12 && return vals, V, rounds
    end
end

function kk_block(Lc, nev, bs, kd, tol)
    n = size(Lc, 2)
    x0 = KrylovKit.Block([randn(n) for _ in 1:bs])
    va, ve, info = eigsolve(Lc, x0, nev, :SR, KrylovKit.BlockLanczos(; krylovdim = kd, maxiter = 200, tol, verbosity = 0))
    (real.(va), reduce(hcat, ve))
end

function score(io, d, seed, name, t, nmv, vals, V, L, thr, lmax, extra = "")
    ord = sortperm(vals); vals = vals[ord]; V = V[:, ord]
    nbelow = count(<(thr), vals)
    # eigenvector quality of the null vectors: relative residual ||Lv||/(||L|| ||v||)
    res = [norm(L * V[:, j]) / (lmax * norm(V[:, j])) for j in 1:nbelow]
    maxres = isempty(res) ? NaN : maximum(res)
    l4 = length(vals) >= nbelow + 1 ? vals[nbelow + 1] : NaN
    line = @sprintf("%d,%d,%s,%.3f,%d,%d,%s,%s,%s", d, seed, name, t, nmv, nbelow, fmt(maxres), fmt(l4 / lmax), extra)
    println(line); flush(stdout); println(io, line); flush(io)
end

isfile(OUT) || open(OUT, "w") do io
    println(io, "d,seed,strategy,seconds,matvecs,nbelow,max_null_resid_rel,next_over_lmax,extra")
end

# warm-up
let inp = build_sphere(8); L, Lc, cnt = counted_op(inp); thr = 1e-4; c = 100.0
    arpack(Lc, 4, 20, 1e-10); kk_lanczos(Lc, 4, 32, 1e-10); kk_deflated(Lc, 2, 20, 1e-10, thr, c); kk_block(Lc, 4, 4, 40, 1e-10)
end

for d in DS, seed in SEEDS
    inp = build_sphere(d; seed)
    L, Lc, cnt = counted_op(inp)
    n = size(L, 2)
    lmax = Dleto.opnorm_estimate(Lc; iters = 30)
    thr = 1e-6 * lmax
    cnt[] = 0
    open(OUT, "a") do io
        run(name, f) = begin
            Random.seed!(1000 * seed + d); cnt[] = 0
            t = @elapsed out = f()
            score(io, d, seed, name, t, cnt[], out[1], out[2], L, thr, lmax, length(out) > 2 ? string(out[3]) : "")
        end
        run("Arpack nev4 ncv20 tol1e-10", () -> arpack(Lc, 4, 20, 1e-10))
        run("Arpack nev4 ncv64 tol1e-10", () -> arpack(Lc, 4, 64, 1e-10))
        run("Arpack nev4 ncv128 tol1e-10", () -> arpack(Lc, 4, 128, 1e-10))
        run("Arpack nev16 ncv33 tol1e-10", () -> arpack(Lc, 16, 33, 1e-10))
        run("Arpack nev16 ncv128 tol1e-10", () -> arpack(Lc, 16, 128, 1e-10))
        run("KK-Lanczos nev4 kd32 tol1e-10", () -> kk_lanczos(Lc, 4, 32, 1e-10))
        run("KK-Lanczos nev4 kd128 tol1e-10", () -> kk_lanczos(Lc, 4, 128, 1e-10))
        run("KK-Lanczos nev4 kd256 tol1e-10", () -> kk_lanczos(Lc, 4, 256, 1e-10))
        run("KK-Lanczos nev16 kd64 tol1e-10", () -> kk_lanczos(Lc, 16, 64, 1e-10))
        run("KK-deflated nev2 kd32 tol1e-10", () -> kk_deflated(Lc, 2, 32, 1e-10, thr, lmax))
        run("KK-deflated nev4 kd64 tol1e-10", () -> kk_deflated(Lc, 4, 64, 1e-10, thr, lmax))
        run("KK-block b2 nev4 kd64 tol1e-10", () -> kk_block(Lc, 4, 2, 64, 1e-10))
        run("KK-block b4 nev4 kd64 tol1e-10", () -> kk_block(Lc, 4, 4, 64, 1e-10))
        run("KK-block b4 nev4 kd128 tol1e-10", () -> kk_block(Lc, 4, 4, 128, 1e-10))
        run("KK-block b8 nev8 kd128 tol1e-10", () -> kk_block(Lc, 8, 8, 128, 1e-10))
    end
end
println("EXP2 DONE")
