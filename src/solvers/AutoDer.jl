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
# DECLINES (`QuickDerDeclined`), because a failed verification is exactly the
# signal that the tensor is not generic enough for the sketch.  Any other
# exception out of QuickDer is a bug or a resource limit and propagates.  The
# only policy here is that order; the null-solver policy stays where it
# belongs, in `solve_nullspace`.
#

"""
    AutoDerMethod(; quick = QuickDerMethod(), fallback = SylverLiningMethod(),
                    min_entries = AUTODER_MIN_ENTRIES)

The default derivation method: `:QuickDer` first when the setting supports it
(`IndTransverseOps`, a chisel with an engaged axis, at least `min_entries`
tensor entries), `:SylverLining` otherwise or when QuickDer's own verification
rejects its answer.  Construct through `get_derivation_method(:Auto; ...)`;
keywords not named here go to the QuickDer constructor (`restriction`,
`sizes`, `verify`, `seed`, `solver`, `whiten`) -- so `:Auto` inherits the
whitened restriction, which is what lets the matrix-free branch answer at all
on a structured tensor above d ~ 30 (see `QuickDerMethod`'s table).
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
    # Forwarded verbatim to whichever route answers, so the report says
    # `method = :QuickDer` or `:SylverLining` -- which is the question a caller
    # of `:Auto` most wants answered, and the reason the report never says
    # `:Auto`.
    return_diagnostics::Bool = false,
    # Per-call keywords are split BY DESTINATION, not forwarded as one splat:
    # `backend` is SylverLining's kernel choice and goes only to the fallback
    # (QuickDer has no such keyword; its options live on `QuickDerMethod`).
    # There is no `kwargs...` sink -- an option neither route implements is a
    # `MethodError` here rather than a silent no-op on whichever route answered.
    backend::Symbol = method.fallback.backend,
)
    # No return-type annotation: see the note on the QuickDer method.  The
    # three-tuple is unchanged when `return_diagnostics` is false.
    if autoder_applicable(method, Ω, P, Γ)
        try
            return derTrOpsReduced(method.quick, Ω, P, Γ; tol = tol, nd = nd,
                                   progress = progress,
                                   return_diagnostics = return_diagnostics)
        catch err
            # QuickDer DECLINES deliberately -- `QuickDerDeclined` -- when its
            # restricted solve found nothing without converging, its lift is
            # infeasible or the Z-law check fails: the tensor is not generic at
            # these sizes.  That is information, not a crash, and SylverLining
            # handles it.  Anything else is a bug or a resource limit
            # (`MethodError`, `OutOfMemoryError` on d^n, ...) and is rethrown:
            # the fallback's operator is n·d^{n+1}, strictly larger, so
            # swallowing an OOM here would retry with more memory, not less.
            err isa QuickDerDeclined || rethrow()
            # `@warn`, not `@info`: the fallback is exact but 65-150x slower at
            # this size, which a caller timing a loop of blocks wants to see.
            @warn "AutoDer: QuickDer declined; falling back to SylverLining " *
                  "(exact, and far slower at this size)." reason = err.msg
        end
    end
    return derTrOpsReduced(method.fallback, Ω, P, Γ; tol = tol, nd = nd,
                           progress = progress, backend = backend,
                           return_diagnostics = return_diagnostics)
end
