#
# Strata Dleto: AutoDer
#   Pick the derivation method from the problem, verify, fall back.
#
# Two derivation methods exist and they have opposite profiles.
#
#   :QuickDer      solve-and-lift on a sketch of the tensor.  Its cost is set
#                  by the restriction sizes `r`, not by `d`: measured 65x
#                  faster than SylverLining at 30^3 and 150x at 16^4, and the
#                  gap widens with `d`.  It is only GENERICALLY correct at a
#                  given `r`, which is why it verifies its own answer against
#                  the defining equation and errors when the check fails.
#   :SylverLining  the derivation operator as a matrix-free LinearMap handed
#                  to a null solver.  Exact for every tensor, at a cost of
#                  ~500 applications of an operator that touches all of Γ.
#
# `AutoDerMethod` runs QuickDer whenever the setting allows it and lets
# SylverLining answer everything else -- including the cases QuickDer itself
# refuses, because a failed verification is exactly the signal that the tensor
# is not generic enough for the sketch.  The only policy here is that order;
# the null-solver policy stays where it belongs, in `solve_nullspace`.
#

"""
    AutoDerMethod(; quick = QuickDerMethod(), fallback = SylverLiningMethod(),
                    min_entries = AUTODER_MIN_ENTRIES)

The default derivation method: `:QuickDer` first when the setting supports it
(`IndTransverseOps`, a chisel with an engaged axis, at least `min_entries`
tensor entries), `:SylverLining` otherwise or when QuickDer's own verification
rejects its answer.  Construct through `get_derivation_method(:Auto; ...)`;
keywords not named here go to the QuickDer constructor (`restriction`,
`sizes`, `verify`, `seed`, `solver`).
"""
struct AutoDerMethod <: DerivationMethod
    quick::QuickDerMethod
    fallback::SylverLiningMethod
    min_entries::Int
end

"""
Below this many tensor entries the whole derivation operator is small enough
that SylverLining's dense SVD answers in milliseconds and is exact, so there
is nothing for a sketch to save.  A 12x12x12 tensor has 1728 entries and a
3x144 = 432-dimensional operator space.
"""
const AUTODER_MIN_ENTRIES = 2000

AutoDerMethod(; fallback_solver::Symbol = :AutoSolver,
                min_entries::Integer = AUTODER_MIN_ENTRIES, kwargs...) =
    AutoDerMethod(QuickDerMethod(; kwargs...), SylverLiningMethod(; solver = fallback_solver),
                  Int(min_entries))

"""
    autoder_applicable(m::AutoDerMethod, Ω, P, Γ) -> Bool

Whether QuickDer is worth trying: the operator space is axis-independent, the
chisel engages at least one axis, and the tensor is big enough to matter.
"""
function autoder_applicable(m::AutoDerMethod, Ω::TransverseOps, P::AbstractMatrix, Γ::ITensor)
    Ω isa IndTransverseOps || return false
    any(engaged(Matrix(P))) || return false
    return prod(ITensors.dim.(inds(Γ))) >= m.min_entries
end

function derTrOpsReduced(
    method::AutoDerMethod,
    Ω::TransverseOps,
    P::AbstractMatrix,
    Γ::ITensor;
    tol::Real = TOL_DEFAULT,
    nd = -1,
    progress = false,
    kwargs...,
)::Tuple{TransverseOps, LinearMaps.LinearMap, AbstractMatrix{<:Number}}
    if autoder_applicable(method, Ω, P, Γ)
        try
            return derTrOpsReduced(method.quick, Ω, P, Γ; tol = tol, nd = nd,
                                   progress = progress, kwargs...)
        catch err
            # QuickDer errors deliberately when its lift is infeasible or the
            # Z-law check fails: the tensor is not generic at these sizes.
            # That is information, not a crash -- SylverLining handles it.
            err isa InterruptException && rethrow()
            @info "AutoDer: QuickDer declined ($(first(split(sprint(showerror, err), '\n')))); " *
                  "falling back to SylverLining."
        end
    end
    return derTrOpsReduced(method.fallback, Ω, P, Γ; tol = tol, nd = nd,
                           progress = progress, kwargs...)
end
