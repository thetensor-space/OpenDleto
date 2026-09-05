#
# Strata Dleto: DerivationReport
#   What a derivation solve decided, and on what evidence.
#
# `derTrOpsReduced` returns a basis and nothing else, and for a caller who
# trusts the count that is enough.  The callers who do not trust it -- the ones
# whose tensors are real data rather than a benchmark family -- have had to
# read the answer out of log lines: whether the nullity was certified, what the
# values around the cut were, how far the lift residual of each returned
# direction is from zero.  That is not an interface; `@warn` text moves, and
# `maxlog = 1` silences the second call in a loop over a thousand video blocks.
#
# So the same numbers are available as a value.  `return_diagnostics = true`
# adds a fourth element to the returned tuple and changes nothing else -- the
# default is `false` and the three-tuple is bit-for-bit what it was.
#
# ONE STRUCT, THREE ROUTES.  QuickDer, SylverLining and AutoDer fill what they
# have and leave the rest `nothing`, rather than each growing its own named
# tuple: a caller that switches method to compare should not have to switch
# the code that reads the answer.  A field that is `nothing` means "this route
# does not have that number", never "zero" -- `lift_residuals` is `nothing`
# from SylverLining, which has no lift, and `Float64[]` from QuickDer when the
# restriction was full and there was nothing to lift.
#

"""
    DerivationReport

The evidence behind one `derTrOpsReduced` answer: the null-space verdict, the
policy that placed the cut, and the residuals of the directions that came back.
Obtained with `return_diagnostics = true`; see `derTrOpsReduced`.

THE VERDICT AND ITS SUMMARY.  `verdict` is the `NullVerdict` of the solve that
decided the count -- the RESTRICTED solve for QuickDer, the derivation operator
itself for SylverLining -- carried whole, so nothing about it has to be
guessed.  `certified`, `rule`, `gap`, `threshold`, `precision_floor`,
`data_floor`, `undecidable` and `near_null` are copied out of it for the common
case where that is all a caller wants; they always agree with `verdict` when
`verdict !== nothing`.

- `method`      which kernel answered: `:QuickDer`, `:SylverLining`, and
                `:Auto` never appears -- `AutoDerMethod` reports the method
                that actually ran, which is the point of asking.
- `policy`      `:auto`, the verdict placed the cut, or `:fixed_nd`, the CALLER
                did (see `derTrOpsReduced`'s `nd`).  Under `:fixed_nd`
                `certified` is false unless the verdict's own cut agrees with
                the count asked for, and the directions returned are the `nd`
                smallest by singular value whether or not they are
                derivations -- `lift_residuals` and `residuals` are how far
                each one is from being one.  THE TWO ROUTES MEAN DIFFERENT
                THINGS BY IT, which is why `requested_nd` and `returned` are
                both reported: QuickDer solves without a ceiling and returns
                exactly `nd` (when `Ω` constrains nothing -- see there),
                reaching directions ABOVE the tolerance, while SylverLining's
                `nd` is the cap it has always been and can only return values
                the threshold already called null, so `returned <=
                requested_nd` there and it never reaches a near-derivation.
                The near-null hunt is QuickDer's.
- `requested_nd` the caller's `nd`; `-1` (or any non-positive value) for the
                automatic policy.
- `nullity`     the count the deciding solve returned, BEFORE the lift filter
                and the intersection with `Ω`.
- `returned`    `size(ders, 2)`, the count the caller actually got.
- `scalar_dim`  `dim ker P`, the dimension of the scalar derivations (`D_a =
                c_a·I`) that this chisel admits -- 2 for the 3-valent universal
                chisel, 3 at valence 4.  Every tensor has them when `Ω` carries
                the identity, so a `returned` equal to this is the generic
                answer "no structure here", and the interesting question is
                then what the values just above the cut are.
- `spectrum`    every relative value the deciding solve returned, sorted.
- `selected`    the relative values the CUT kept -- `nullity` of them, ascending.
                Under `:fixed_nd` these are the values of the directions the
                caller asked for, one per returned direction; under `:auto`
                they are the null cluster, and QuickDer's `returned` can be
                smaller than this, because the lift filter and the intersection
                with `Ω` shrink the count afterwards and what survives is a
                combination of all of them, not a prefix.
- `next_value`  the first relative value above the selection, `NaN` when the
                solve returned nothing above it.  With `selected` it is the
                gap the caller is being asked to believe.
- `status`      the solver's own word (`:ok`, `:unconverged`, `:capped`).
- `store_eltype`, `compute_eltype`  the type the tensor was STORED in and the
                type the arithmetic ran in; they differ exactly when the tensor
                was handed in as Float16 (see `Dleto.compute_eltype`), and that
                difference is why `data_floor` exists.
- `residuals`   the Z-law residual `Dleto.der_residual` of each RETURNED
                direction, in the order returned: the number that says whether
                the answer is a derivation, computed on the tensor itself
                rather than on the restricted system.  `nothing` when it was
                not computed.
- `lift_dim`, `lift_residuals`  QuickDer only: how many restricted directions
                the lift's consistency filter accepted, and the lift residual
                of each, relative to the size of the lift equations.  A true
                derivation's restriction lifts exactly; a near-null direction of
                the restricted system does not, and this is the first place
                that shows.  These count UNIVERSAL derivations -- the answer
                before the intersection with `Ω` -- so `lift_dim >= returned`,
                and on the scrambled sphere (13 universal, 3 in the symmetric
                `Ω`) the difference is the whole point of that intersection.
- `stage_times`  a copy of `QDN_STAGE_TIMES[]` when stage timing was on,
                `nothing` otherwise.

Route-specific fields (`nothing` where the route has no answer): `solver`,
`device`, `seed`, `whitened`, `restriction`, `restricted_size`, `lift_dim`,
`lift_residuals`.

# Example

```julia
julia> m = get_derivation_method(:QuickDer; seed = 4242);

julia> (Ω, expand, ders, rep) = derTrOpsReduced(m, Ω, ch, Γ; return_diagnostics = true);

julia> rep.returned, rep.scalar_dim, rep.certified
(2, 2, true)

julia> rep.selected, rep.next_value      # the null cluster, and what is next
([2.7e-7, 4.5e-7], 0.0067)

julia> maximum(rep.residuals)            # the Z-law, per returned direction
3.1e-7
```
"""
struct DerivationReport
    method::Symbol
    policy::Symbol
    verdict::Union{Nothing,NullVerdict}
    requested_nd::Int
    nullity::Int
    returned::Int
    scalar_dim::Int
    certified::Bool
    rule::Symbol
    gap::Float64
    threshold::Float64
    precision_floor::Float64
    data_floor::Float64
    undecidable::Int
    near_null::Int
    spectrum::Vector{Float64}
    selected::Vector{Float64}
    next_value::Float64
    status::Symbol
    store_eltype::Type
    compute_eltype::Type
    dims::Vector{Int}
    solver::Union{Nothing,Symbol}
    device::Union{Nothing,Symbol}
    seed::Union{Nothing,Int}
    whitened::Union{Nothing,Bool}
    restriction::Union{Nothing,Vector{Int}}
    restricted_size::Union{Nothing,Tuple{Int,Int}}
    lift_dim::Union{Nothing,Int}
    lift_residuals::Union{Nothing,Vector{Float64}}
    residuals::Union{Nothing,Vector{Float64}}
    stage_times::Union{Nothing,Dict{Symbol,Float64}}
end

"""
    DerivationReport(; method, store_eltype, dims, kwargs...)

Keyword constructor.  Everything a route cannot fill has a default -- `nothing`
for a route-specific field, and for the verdict summary the value a missing
verdict implies (`certified = false`, `rule = :none`, `NaN` gaps) -- so adding
a field here does not break the routes that do not know about it.  Pass
`verdict` and the summary fields are taken from it unless overridden, which is
the only place the two can disagree and the reason it is done once, here.
"""
function DerivationReport(; method::Symbol,
                          store_eltype::Type,
                          dims::AbstractVector{<:Integer},
                          verdict::Union{Nothing,NullVerdict} = nothing,
                          policy::Symbol = :auto,
                          requested_nd::Integer = -1,
                          nullity::Integer = verdict === nothing ? 0 : verdict.nullity,
                          returned::Integer = nullity,
                          scalar_dim::Integer = 0,
                          certified::Bool = verdict === nothing ? false : verdict.certified,
                          rule::Symbol = verdict === nothing ? :none : verdict.rule,
                          gap::Real = verdict === nothing ? NaN : verdict.gap,
                          threshold::Real = verdict === nothing ? NaN : verdict.threshold,
                          precision_floor::Real = Dleto.precision_floor(store_eltype),
                          data_floor::Real = Dleto.data_floor(store_eltype),
                          undecidable::Integer = verdict === nothing ? 0 : verdict.undecidable,
                          near_null::Integer = verdict === nothing ? 0 : verdict.near_null,
                          spectrum::AbstractVector{<:Real} =
                              verdict === nothing ? Float64[] : verdict.spectrum,
                          # The values the CUT kept, which is `nullity` of them
                          # and not `returned`: QuickDer's lift filter and the
                          # intersection with `Ω` shrink the count afterwards,
                          # and the directions that survive are combinations of
                          # all `nullity` of these, not the first few.
                          selected::AbstractVector{<:Real} =
                              Float64[spectrum[i] for i in 1:min(nullity, length(spectrum))],
                          next_value::Real =
                              length(spectrum) > length(selected) ?
                              spectrum[length(selected) + 1] : NaN,
                          status::Symbol = verdict === nothing ? :ok : verdict.status,
                          compute_eltype::Type = Dleto.compute_eltype(store_eltype),
                          solver = nothing, device = nothing, seed = nothing,
                          whitened = nothing, restriction = nothing,
                          restricted_size = nothing, lift_dim = nothing,
                          lift_residuals = nothing, residuals = nothing,
                          stage_times = nothing)
    return DerivationReport(method, policy, verdict, Int(requested_nd), Int(nullity),
                            Int(returned), Int(scalar_dim), certified, rule,
                            Float64(gap), Float64(threshold), Float64(precision_floor),
                            Float64(data_floor), Int(undecidable), Int(near_null),
                            Float64.(spectrum), Float64.(selected), Float64(next_value),
                            status, store_eltype, compute_eltype, Int[d for d in dims],
                            solver === nothing ? nothing : Symbol(solver),
                            device === nothing ? nothing : Symbol(device),
                            seed === nothing ? nothing : Int(seed),
                            whitened === nothing ? nothing : Bool(whitened),
                            restriction === nothing ? nothing : Int[r for r in restriction],
                            restricted_size === nothing ? nothing :
                            (Int(restricted_size[1]), Int(restricted_size[2])),
                            lift_dim === nothing ? nothing : Int(lift_dim),
                            lift_residuals === nothing ? nothing : Float64.(lift_residuals),
                            residuals === nothing ? nothing : Float64.(residuals),
                            stage_times)
end

function Base.show(io::IO, ::MIME"text/plain", r::DerivationReport)
    fmt(x) = x === nothing ? "-" :
             x isa Real ? (isfinite(x) ? string(round(Float64(x); sigdigits = 3)) :
                           string(x)) : string(x)
    fmtv(v) = v === nothing ? "-" : string(map(fmt, v))
    println(io, "DerivationReport(:", r.method, ", policy :", r.policy, ")")
    println(io, "  returned ", r.returned, " of nullity ", r.nullity,
            " (scalars: ", r.scalar_dim, ")",
            r.requested_nd > 0 ? ", requested $(r.requested_nd)" : "")
    println(io, "  ", r.certified ? "CERTIFIED" : "uncertified",
            " by :", r.rule, ", gap ", fmt(r.gap),
            r.status === :ok ? "" : ", solver :" * string(r.status),
            r.undecidable > 0 ? ", UNDECIDABLE: $(r.undecidable)" : "",
            r.near_null > 0 ? ", near-null above cut: $(r.near_null)" : "")
    println(io, "  selected ", fmtv(r.selected), " | next ", fmt(r.next_value))
    println(io, "  floors: precision ", fmt(r.precision_floor),
            ", data ", fmt(r.data_floor), ", threshold ", fmt(r.threshold),
            "  (", r.store_eltype, " stored, ", r.compute_eltype, " computed)")
    r.lift_residuals === nothing ||
        println(io, "  lift residuals ", fmtv(r.lift_residuals))
    r.residuals === nothing ||
        println(io, "  Z-law residuals ", fmtv(r.residuals))
    print(io, "  dims ", r.dims,
          r.restriction === nothing ? "" : ", restriction $(r.restriction)",
          r.restricted_size === nothing ? "" :
          ", restricted $(r.restricted_size[1])x$(r.restricted_size[2])",
          r.whitened === nothing ? "" : (r.whitened ? ", whitened" : ", unwhitened"),
          r.solver === nothing ? "" : ", solver :$(r.solver)",
          r.device === nothing ? "" : ", device :$(r.device)",
          r.seed === nothing ? "" : ", seed $(r.seed)")
end

Base.show(io::IO, r::DerivationReport) =
    print(io, "DerivationReport(:", r.method, ", :", r.policy, ", returned ", r.returned,
          r.certified ? ", certified)" : ", uncertified)")

"""
    _der_scalar_dim(P) -> Int

`dim ker P`: how many SCALAR derivations (`D_a = c_a·I`) the chisel `P` admits,
which is the count a generic tensor returns and therefore the baseline any
larger answer has to be read against.  Two for the 3-valent universal chisel,
three at valence 4.  It is an upper bound on what a given `Ω` realises -- the
scalars have to lie in `Ω`, and they do whenever `Ω` carries the identity on
every axis, which every `localOps` in this package does.
"""
_der_scalar_dim(P::AbstractMatrix) = size(nullspace(Matrix{Float64}(P)), 2)

"""
    _der_zlaw_residuals(Ω, expand_map, ders, G, P) -> Vector{Float64}

`Dleto.der_residual` of every column of `ders`, in the order returned.

The columns are `Ω`-coordinates, so each is embedded back into one matrix per
axis (`embedMatrices`, frame index first -- `der_residual`'s convention) and
checked against `G` in the COMPUTE type: a Float16 answer checked in Float16
would measure its own storage rounding rather than the equation.

COSTS ONE PASS OVER THE TENSOR PER DIRECTION, which is why it is behind
`return_diagnostics` and not computed on every call.  `der_residual` itself is
blocked, so the memory is bounded whatever the size of `G`; the time is not.
"""
function _der_zlaw_residuals(Ω::TransverseOps, expand_map, ders::AbstractMatrix,
                             G::AbstractArray, P::AbstractMatrix)
    Tc = eltype(G)
    n = size(ders, 2)
    out = Vector{Float64}(undef, n)
    for j in 1:n
        coords = Vector{Tc}(expand_map * ders[:, j])
        Ms = [Matrix{Tc}(M) for M in embedMatrices(Ω, coords)]
        out[j] = Float64(der_residual(G, Ms, P))
    end
    return out
end
