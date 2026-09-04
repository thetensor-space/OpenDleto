#
# Precision study, experiment 4: is the benchmark itself well posed at d = 45..50?
#
#   julia -t 2 --project=. bench/reports/precision-exp4.jl
#
# Experiment 2C found the 4th eigenvalue of the derivation--densor operator at
# d = 50 sitting at 1.2e-8 (relative) in Float64, versus 2.6e-3 .. 7e-3 for
# d <= 40.  Before blaming precision: check the support size of the sphere
# tensor against the exact lattice count (i + j + k = d - 1 has C(d+1, 2)
# points), and recompute the near-zero spectrum at d = 45 and 50 on three
# seeds, in Float64 only.
#
using KrylovKit, IterativeSolvers, Arpack
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

println("== support size vs exact lattice count"); flush(stdout)
p1 = csvopen("precision-exp4-support.csv", "d,nnz,expected_C(d+1,2),dims_after_nondeg")
for d in (20, 30, 40, 45, 48, 50, 52, 55, 60)
    inp = build_sphere(d)
    emit(p1, @sprintf("%d,%d,%d,\"%s\"", d, inp.nnz, binomial(d + 1, 2), string(inp.dims)))
end

println("== near-zero spectrum, Float64, d = 45 and 50, three seeds"); flush(stdout)
p2 = csvopen("precision-exp4-spectrum.csv", "d,seed,N,lam_max,lam1_rel,lam2_rel,lam3_rel,lam4_rel,lam5_rel,lam6_rel")
for d in (45, 50), seed in (d, d + 1000, d + 2000)
    inp = build_sphere(d; seed)
    L = sylmap(inp)
    M = Matrix(L); M = Symmetric((M + M') / 2)
    ev = sort(abs.(eigvals(M)))
    lmax = maximum(ev)
    emit(p2, @sprintf("%d,%d,%d,%.4e,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e", d, seed, size(L, 2), lmax, (ev[1:6] ./ lmax)...))
    GC.gc()
end
println("EXP4 DONE")
