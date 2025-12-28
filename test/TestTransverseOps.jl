# 
# TransverseOps Tests
#


using ITensors
using Dleto
using Dleto.TransverseOperators

#
# Test UniversalOps with law 
#
# v = contains(Ops,transverse(Ops,v))
#
function testUnvOps(valence)
    # build random dimensions
    dims = [ rand(1:100) for a in 1:valence ]
    axes = collect([ Index(dims[a], "a_$a") for a in 1:valence ])
    Xs = [ ITensor(randn(dims[a],dims[a]), axes[a], prime(axes[a])) for a in 1:valence ]
    Ω = UniversalOps(axes)
    flat_x = member(Ω, Xs)
    Zs = transverse(Ω, flat_x)
    for a in 1:valence
        @assert Zs[a] ≈ Xs[a]
    end
end