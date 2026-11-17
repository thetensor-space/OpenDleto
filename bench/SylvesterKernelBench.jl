#
# KernelProfile3.jl -- all three backends in one process, so the before/after
# table is apples to apples.
#
using Dleto
using ITensors
using LinearAlgebra
using LinearMaps
using Printf
using Random

include("/Users/algeboy/CODE/OpenDleto/bench/SphereHarness.jl")

function measure(f!, y, x; reps::Integer = 7)
    f!(y, x)
    t = Inf
    for _ in 1:reps
        t = min(t, @elapsed f!(y, x))
    end
    return (t, @allocated f!(y, x))
end

function profile_case(name, Ω, ch, Γ)
    T = eltype(Γ)
    out = Dict{Symbol,NTuple{6,Float64}}()
    for bk in (:itensor, :array, :auto)
        der_map, den_map = sylvesterLM(Ω, ch, Γ; backend = bk)
        nrow, ncol = size(den_map)
        x = randn(T, ncol); y = randn(T, nrow); xo = similar(x)
        (te, be) = measure((yy, xx) -> mul!(yy, den_map, xx), y, x)
        (ts, bs) = measure((xx, yy) -> mul!(xx, den_map', yy), xo, y)
        (tq, bq) = measure((xx, x2) -> mul!(xx, der_map, x2), xo, x)
        out[bk] = (1e3te, 1e3ts, 1e3tq, be, bs, bq)
        der_map = nothing; den_map = nothing
        GC.gc()
    end
    i = out[:itensor]; a = out[:array]; u = out[:auto]
    @printf("%-18s | %7.3f %7.3f %7.3f %9.0f | %7.3f %7.3f %7.3f | %7.3f %7.3f %7.3f %7.0f | %5.2f %5.2f %5.2f\n",
            name, i[1], i[2], i[3], i[6],
            a[1], a[2], a[3],
            u[1], u[2], u[3], u[6],
            i[1] / u[1], i[2] / u[2], i[3] / u[3])
    flush(stdout)
end

dense_case(val, d) = begin
    fr = [Index(d, "a$a") for a in 1:val]
    (IndTransverseOps(fr, UniversalOp()), UniversalChisel(val),
     ITensor(randn(ntuple(_ -> d, val)), fr...))
end

function sphere_case(val, d)
    Γ = val == 3 ? sphere_octant(d) : sphere_octant(d; valence = val)
    fr = collect(inds(Γ))
    (IndTransverseOps(fr, UniversalOp()), UniversalChisel(val), Γ)
end

Random.seed!(20260903)
println("threads = $(Threads.nthreads()) ; BLAS = $(BLAS.get_num_threads())")
println("                   |            :itensor              |      :array (dense)     |         :auto                   | speedup auto/itensor")
@printf("%-18s | %7s %7s %7s %9s | %7s %7s %7s | %7s %7s %7s %7s | %5s %5s %5s\n",
        "case", "ester", "sylve", "sylv", "sylv B", "ester", "sylve", "sylv",
        "ester", "sylve", "sylv", "sylv B", "est", "syl", "sq")

for (val, d) in ((3, 40), (3, 80), (4, 16), (4, 24))
    profile_case("dense v$val d=$d", dense_case(val, d)...)
    GC.gc()
end
for (val, d) in ((3, 40), (3, 80), (4, 16), (4, 24))
    profile_case("sphere v$val d=$d", sphere_case(val, d)...)
    GC.gc()
end
