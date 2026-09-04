#
# Precision study, experiment 3: what Float16 can and cannot do.
#
#   julia -t 2 --project=. bench/reports/precision-exp3.jl
#
# 1. Pure Float16: build_sphere(d; T = Float16) and run each solver; report
#    exactly where it fails.
# 2. Map-application speed per element type at d = 20 (is Float16 arithmetic
#    even fast on this machine, or promoted/emulated?).
# 3. Mixed strategy: round Γ to Float16, convert back to Float32 (and to
#    Float64), stratify, and score.  This measures whether the DATA can be
#    coarse even if the arithmetic cannot.
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

"Replace Γ in an input by its rounding through `Tstore`, computed in `Tcompute`."
function requantize(inp, Tstore, Tcompute)
    fr = collect(inds(inp.Γ))
    A = Array(inp.Γ, fr...)
    Aq = Array{Tcompute}(Array{Tstore}(A))
    return merge(inp, (; Γ = ITensor(Aq, fr...), T = Tcompute))
end

# ------------------------------------------------------------- 1. pure Float16
println("== 1. Pure Float16"); flush(stdout)
path1 = csvopen("precision-exp3-float16.csv", "d,stage,solver,seconds,nullity,lsq_err,status")
for d in (10, 20)
    inp = try
        build_sphere(d; T = Float16)
    catch e
        emit(path1, @sprintf("%d,build_sphere,-,NaN,0,NaN,\"error: %s\"", d, first(split(sprint(showerror, e), '\n'))))
        continue
    end
    emit(path1, @sprintf("%d,build_sphere,-,NaN,0,NaN,\"ok eltype=%s\"", d, eltype(inp.Γ)))
    # Can the map even be applied?
    try
        L = sylmap(inp)
        v = randn(Float16, size(L, 2))
        t = @elapsed w = L * v
        t = @elapsed w = L * v
        emit(path1, @sprintf("%d,map_apply,-,%.4f,0,NaN,\"ok eltype(L*v)=%s\"", d, t, eltype(w)))
    catch e
        emit(path1, @sprintf("%d,map_apply,-,NaN,0,NaN,\"error: %s\"", d, first(split(sprint(showerror, e), '\n'))))
    end
    for s in (:SVDSolver, :ArpackSolver, :KrylovSolver, :CGSolver)
        r = run_stratify(inp; solver = s)
        emit(path1, @sprintf("%d,stratify,%s,%.3f,%d,%.3e,\"%s\"", d, s, r.seconds, r.nullity, r.lsq_err, r.status))
    end
end

# ------------------------------------------------- 2. map application speed
println("== 2. Map application time per element type, d = 20"); flush(stdout)
path2 = csvopen("precision-exp3-apply.csv", "d,T,N,apply_seconds_median,apply_bytes")
for T in (Float64, Float32, Float16)
    inp = try build_sphere(20; T) catch; nothing end
    inp === nothing && continue
    try
        L = sylmap(inp)
        N = size(L, 2)
        v = randn(T, N)
        L * v
        ts = Float64[]; bytes = 0
        for _ in 1:7
            st = @timed L * v
            push!(ts, st.time); bytes = st.bytes
        end
        emit(path2, @sprintf("20,%s,%d,%.5f,%d", T, N, sort(ts)[4], bytes))
    catch e
        emit(path2, @sprintf("20,%s,0,NaN,0  # error: %s", T, first(split(sprint(showerror, e), '\n'))))
    end
end

# ------------------------------------------------- 3. Float16 data, wider compute
println("== 3. Float16-rounded Γ, computed in Float32 / Float64"); flush(stdout)
path3 = csvopen("precision-exp3-mixed.csv", "d,store_T,compute_T,solver,seconds,nullity,lsq_err,support,status,gamma_rounding_relerr")
for Tc in (Float32, Float64)
    warmup!([(; solver = :SVDSolver), (; solver = :ArpackSolver)]; T = Tc)
end
for d in (20, 30, 40)
    base = build_sphere(d; T = Float64)
    A64 = Array(base.Γ, collect(inds(base.Γ))...)
    for (Ts, Tc) in ((Float16, Float32), (Float16, Float64), (Float32, Float32), (Float32, Float64), (Float64, Float64))
        inp = requantize(base, Ts, Tc)
        Aq = Array(inp.Γ, collect(inds(inp.Γ))...)
        rerr = norm(Float64.(Aq) - A64) / norm(A64)
        for s in (:SVDSolver, :ArpackSolver)
            r = run_stratify(inp; solver = s)
            emit(path3, @sprintf("%d,%s,%s,%s,%.3f,%d,%.3e,%.6f,\"%s\",%.3e",
                d, Ts, Tc, s, r.seconds, r.nullity, r.lsq_err, r.support, r.status, rerr))
        end
    end
end
println("EXP3 DONE")
