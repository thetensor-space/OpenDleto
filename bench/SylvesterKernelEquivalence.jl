#
# Equivalence.jl -- the array kernel against the ITensor reference.
#
using Dleto
using ITensors
using LinearAlgebra
using LinearMaps
using Printf
using Random

Random.seed!(20260903)

const LOPS = Dict(:Universal => UniversalOp(), :Symmetric => SymmetricOp(),
                  :Diagonal => DiagonalOp(), :AntiSymmetric => AntiSymmetricOp())

reldiff(a, b) = maximum(abs, a .- b) / max(eps(Float64), maximum(abs, b))

worst = 0.0
nfail = 0

function compare(name, Ω, ch, Γ; backends = (:auto, :array))
    global worst, nfail
    dref, nref = sylvesterLM(Ω, ch, Γ; backend = :itensor)
    T = eltype(Γ)
    rows, cols = size(nref)
    xs = [randn(T, cols) for _ in 1:3]
    ys = [randn(T, rows) for _ in 1:3]
    for bk in backends
        d, nmap = sylvesterLM(Ω, ch, Γ; backend = bk)
        e = 0.0
        for x in xs
            e = max(e, reldiff(nmap * x, nref * x))
            e = max(e, reldiff(d * x, dref * x))
        end
        for y in ys
            e = max(e, reldiff(nmap' * y, nref' * y))
        end
        tol = T === Float32 ? 1e-5 : 1e-12
        ok = e <= tol
        ok || (nfail += 1)
        worst = max(worst, e)
        @printf("%-58s %-7s %10.3e %s\n", name, bk, e, ok ? "" : "  <-- FAIL")
    end
end

# sparse sphere-octant support i_1 + ... + i_n = d-1, built directly
function octant(::Type{T}, d, n) where {T}
    A = zeros(T, ntuple(_ -> d, n))
    idx = Vector{Int}(undef, n)
    function rec(axis, rem)
        if axis == n
            idx[n] = rem
            A[(idx .+ 1)...] = randn(T)
            return
        end
        for v in max(0, rem - (d - 1) * (n - axis)):min(d - 1, rem)
            idx[axis] = v
            rec(axis + 1, rem - v)
        end
    end
    rec(1, d - 1)
    return A
end

@printf("%-58s %-7s %10s\n", "case", "backend", "rel diff")

for T in (Float64, Float32)
    for val in (3, 4)
        d = val == 3 ? 7 : 5
        fr = [Index(d, "a$a") for a in 1:val]
        dense = ITensor(randn(T, ntuple(_ -> d, val)), fr...)
        sparseT = ITensor(octant(T, d, val), fr...)
        for (tname, Γ) in (("dense", dense), ("sphere", sparseT))
            for opname in (:Universal, :Symmetric, :Diagonal, :AntiSymmetric)
                Ω = IndTransverseOps(fr, LOPS[opname])
                for (cname, ch) in (("universal", UniversalChisel(val)),
                                    ("tucker", TuckerChisel(val)),
                                    ("centroid", CentroidChisel(val)))
                    compare("$T v$val $tname $opname $cname", Ω, ch, Γ)
                end
            end
            # non-square frame + engagement-reduced Ω: operator a is NOT axis a
            eng = [a != 2 for a in 1:val]
            Ωr = IndTransverseOps(fr[eng], UniversalOp())
            compare("$T v$val $tname reduced-Universal universal",
                    Ωr, UniversalChisel(val)[:, eng], Γ)
            # symmetry-restricted operator space
            Ωs = TransverseOpsSymmetries(fr, [LOPS[:Symmetric] for _ in 1:val],
                                         [1 for _ in 1:val])
            compare("$T v$val $tname symmetries universal",
                    Ωs, UniversalChisel(val), Γ)
        end
    end
end

# ragged dims, mixed operator spaces, valence 5
for T in (Float64,)
    dims = (3, 4, 2, 5, 3)
    fr = [Index(dims[a], "b$a") for a in 1:5]
    Γ = ITensor(randn(T, dims), fr...)
    Ω = IndTransverseOps(fr, [UniversalOp(), SymmetricOp(), DiagonalOp(),
                              AntiSymmetricOp(), UniversalOp()])
    compare("$T v5 ragged mixed universal", Ω, UniversalChisel(5), Γ)
    compare("$T v5 ragged mixed centroid", Ω, CentroidChisel(5), Γ)
end

@printf("\nworst relative difference: %.3e ; failures: %d\n", worst, nfail)
exit(nfail == 0 ? 0 : 1)
