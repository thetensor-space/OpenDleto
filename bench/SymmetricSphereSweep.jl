# Run from repository root: julia --project=. --threads=4 bench/SymmetricSphereSweep.jl [output-dir]
include(joinpath(@__DIR__, "..", "labs", "SymmetricSphereDemo.jl"))
BLAS.set_num_threads(4)
outdir = isempty(ARGS) ? joinpath(@__DIR__, "reports", "symmetric-sphere") : abspath(ARGS[1])
mkpath(outdir)
csv = joinpath(outdir, "comparison.csv")
failed_cases = Ref(0)
function experiment(label; kwargs...)
    println("START ",label); flush(stdout)
    open(joinpath(outdir,"current-case.txt"),"w") do io
        println(io,label)
    end
    t = time()
    r = run_case(; out=csv, kwargs...)
    startswith(r.status, "error:") && (failed_cases[] += 1)
    println("RESULT ", label," elapsed=",time()-t," ",r); flush(stdout)
    return r
end
# First runs pay Julia compilation; do not compare them to warm timings.
experiment("QuickDer warmup"; dim=10, seed=50, artifacts=false)
experiment("SymmetricGram warmup"; dim=6, seed=50, method=:SymmetricGram,
           nd=3, tol=Inf, artifacts=false)
for seed in (17,50,91), precision in (:mixed,:full)
    experiment("exact d50 seed $seed $precision";dim=50,seed,
               gram_precision=precision,artifacts=seed==50 && precision==:full)
end
for tol in (1e-8,1e-4)
    experiment("full precision tolerance $tol";dim=50,seed=50,tol,artifacts=false)
end
for noise in (1e-8,1e-6,1e-4,1e-3)
    experiment("automatic near-null noise $noise";dim=50,seed=50,noise,artifacts=false)
    experiment("direct symmetric noise $noise";dim=50,seed=50,noise,
               method=:SymmetricGram,nd=3,tol=Inf,artifacts=noise==1e-3)
end
for seed in (17,50,91)
    experiment("bounded amplitudes seed $seed";dim=50,seed,amplitudes=:bounded,artifacts=false)
end
experiment("literal lattice d50";dim=50,seed=50,sampling=:lattice,artifacts=true)
if get(ENV,"SPHERE_SLOW","0") == "1"
    experiment("FastDer d50";dim=50,seed=50,method=:QuickDer3,artifacts=false)
    experiment("SylverLining direct noisy d50";dim=50,seed=50,noise=1e-3,
               method=:SylverLining,nd=3,tol=Inf,artifacts=false)
end
println("COMPLETE ",csv,"; failed cases=",failed_cases[]); flush(stdout)
failed_cases[] == 0 || exit(1)
