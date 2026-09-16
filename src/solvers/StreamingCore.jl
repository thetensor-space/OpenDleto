#
# Strata Dleto: StreamingCore
#   Single-pass accumulation of everything QuickDer's solve and lift consume,
#   for a tensor whose frames arrive one at a time.
#
# See docs/design/Streaming-Stratification.md for the derivation; the short
# version is the only thing that has to be remembered here.
#
# THE CONDITION.  The stream axis must be DISENGAGED in the chisel.  A
# derivation's unknown on an engaged axis `a` is `d_a x d_a`, so an engaged
# stream axis is an unknown that grows without bound and no amount of lifting
# repairs it.  With the stream axis disengaged, frames enter the restricted
# system as EQUATIONS -- `_qs_system_matrix` in QuickSylver.jl is already a
# `vcat` of one block per slice of the disengaged axis -- and the derivation
# space is monotonically non-increasing along the stream.
#
# WHAT ACCUMULATES.  Everything downstream of the tensor is a contraction of Γ
# against fixed matrices on every axis but one or two, and the stream axis `t`
# is never one of those two:
#
#   cross sketch   S_a    = Γ ×_{b≠a} W_b                     (the restricted solve)
#   pair tensor    H_{ab} = Γ ×_a W_a⊥ ×_{c ∉ {a,b}} W_c      (the lift)
#
# so each is `Σ_k (frame k, sketched on the other axes) ⊗ W_t[k, :]` -- one
# rank-one update per frame into a buffer whose size does not depend on how
# many frames there are.  Both are accumulated in the SAME pass, because a lift
# that had to re-read the stream would not be streaming at all.
#
# The stream axis's own sketch `W_t` cannot come from `_qdn_axis`, which needs a
# `d x d` QR and therefore a known `d`.  It does not need to: a disengaged axis
# carries no unknown, no `W⊥` and no lift, so `W_t` is only ever applied as a
# sketch inside another axis's contraction and a plain Gaussian suffices.  Its
# rows are drawn one per frame from the core's own RNG.  `:corner` would be
# WRONG here -- `I[:, 1:r_t]` looks only at the first `r_t` frames -- and is
# refused for the stream axis.
#
# HOST ARRAYS ONLY.  `_qdn_mode_order` returns the natural axis order on the
# host and a cost-driven one on a device; every frame has to walk the SAME
# chain for the accumulation to be a sum of like terms, so the device path is
# left for later rather than silently given a per-frame ordering.
#

using LinearAlgebra
using Random

# ---------------------------------------------------------------------------
# The core
# ---------------------------------------------------------------------------

"""
    StreamingCore(framedims, P; stream, T = Float32, r_stream,
                  sizes = nothing, seed = nothing,
                  restriction = :random, pairs = true)

Accumulator for the sufficient statistics of a QuickDer solve over a tensor
whose `stream` axis grows one frame at a time.

- `framedims`  the size of ONE frame: the full tensor's dimensions with the
  stream axis removed, in the remaining axes' order.  A `640x480xFx3` movie
  streamed along axis 3 is `framedims = (640, 480, 3)`, `stream = 3`.
- `P`          the chisel matrix.  Its `stream` column must be zero; see the
  file header for why.
- `T`          the compute/accumulation element type (`Float32` by default,
  which is the movie regime's type).
- `r_stream`   the restriction size on the stream axis.  This is a DECLARED
  BUDGET, not a derived quantity: neither condition in
  `_qdn_restriction_sizes` involves `d_t` when `t` is disengaged, and a larger
  `r_stream` only makes both easier.  The other `r_a` are derived from it.
- `sizes`      override for all `n` restriction sizes; `sizes[stream]` must
  equal `r_stream`.
- `pairs`      accumulate the lift's pair tensors as well as the cross
  sketches.  On by default -- without them the lift has to re-read the stream
  -- but they are the memory-heavy part, so a run that only wants the
  restricted solve can turn them off.

Feed it with `push!(core, frame)` and read it with `crosssketches`,
`pairtensors`, `batchaxes` and `nframes`.
"""
mutable struct StreamingCore{Tc, N}
    # `dims[stream]` is `r_stream`, NOT the number of frames: the stream axis is
    # presented to the sizing rule as already saturated at its budget, which is
    # exactly what makes the restriction sizes independent of the stream length.
    dims::Vector{Int}
    framedims::Vector{Int}
    stream::Int
    P::Matrix{Tc}
    engaged::Vector{Bool}
    r::Vector{Int}
    axs::Vector{_QDNAxis{Tc}}
    eaxes::Vector{Int}
    lift::Vector{Int}
    S::Dict{Int, Array{Tc, N}}
    H::Dict{Tuple{Int, Int}, Array{Tc, N}}
    Wt::Vector{Vector{Tc}}
    rng::Random.AbstractRNG
    nrm2::Float64
    nframes::Int
end

function StreamingCore(framedims, P::AbstractMatrix;
                       stream::Int,
                       T::Type = Float32,
                       r_stream::Integer = 16,
                       sizes = nothing,
                       seed = nothing,
                       restriction::Symbol = :random,
                       pairs::Bool = true)
    fd = Int[Int(d) for d in framedims]
    N = length(fd) + 1
    1 <= stream <= N ||
        error("StreamingCore: stream axis $stream is outside 1:$N for a " *
              "valence-$N tensor.")
    N >= 2 || error("StreamingCore needs valence at least 2, got $N.")
    restriction === :random || restriction === :corner ||
        error("StreamingCore: restriction must be :random or :corner, got :$restriction.")
    r_stream >= 1 || error("StreamingCore: r_stream must be at least 1.")

    size(P, 2) == N ||
        error("StreamingCore: the chisel has $(size(P,2)) columns but the full " *
              "tensor has $N axes (frame valence $(N-1) plus the stream axis).")
    eng = engaged(Matrix{Float64}(P))
    eng[stream] &&
        error("StreamingCore: the stream axis $stream is ENGAGED in this chisel. " *
              "Its derivation unknown is d_t x d_t and grows with the stream, so " *
              "there is nothing to accumulate. Zero column $stream of the chisel " *
              "(an adjoint-type chisel does this), or see " *
              "docs/design/Streaming-Stratification.md section 5.")
    any(eng) || error("StreamingCore: the chisel engages no axis.")

    # The stream axis enters the sizing rule as `d_t = r_t`: it is disengaged,
    # so it appears in neither condition except through `∏ r_a`, and presenting
    # it saturated is the statement "its restriction is the budget and its true
    # length is irrelevant".
    dims = Int[i == stream ? Int(r_stream) : fd[i < stream ? i : i - 1] for i in 1:N]
    r = sizes === nothing ? _qdn_restriction_sizes(dims, eng, N) :
                            _qdn_check_sizes(sizes, dims)
    r[stream] == r_stream ||
        error("StreamingCore: sizes[$stream] is $(r[stream]) but r_stream is " *
              "$r_stream; the stream axis's restriction is the budget and the two " *
              "must agree.")

    rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)
    # The stand-in for the stream axis: a saturated `ident` axis of length 1, so
    # that `_qdn_modeW` leaves the frame's singleton stream slot alone and the
    # per-frame chain contracts exactly the axes it should.  The real `W_t` is
    # applied by the rank-one accumulation instead.
    axs = _QDNAxis{T}[i == stream ?
                      _QDNAxis{T}(1, 1, true, Matrix{T}(undef, 1, 0), Matrix{T}(undef, 1, 0)) :
                      _qdn_axis(T, dims[i], r[i], restriction, rng)
                      for i in 1:N]

    eaxes = Int[a for a in 1:N if eng[a]]
    lift = Int[a for a in eaxes if r[a] < dims[a]]

    S = Dict{Int, Array{T, N}}(
        a => zeros(T, ntuple(i -> i == a ? dims[a] : r[i], N)) for a in eaxes)
    H = Dict{Tuple{Int, Int}, Array{T, N}}()
    if pairs
        for a in lift, b in eaxes
            b == a && continue
            H[(a, b)] = zeros(T, ntuple(i -> i == a ? dims[a] - r[a] :
                                             i == b ? dims[b] : r[i], N))
        end
    end

    bytes = sum(length, values(S); init = 0) + sum(length, values(H); init = 0)
    @debug "StreamingCore allocated" dims = dims r = r eaxes = eaxes lift = lift accumulator_MB = bytes * sizeof(T) / 2^20

    return StreamingCore{T, N}(dims, fd, stream, Matrix{T}(P), eng, r, axs,
                               eaxes, lift, S, H, Vector{Vector{T}}(), rng,
                               0.0, 0)
end

"""
    push!(core::StreamingCore, frame) -> core

Fold one frame into the accumulators.  `frame` has the tensor's axes with the
stream axis removed, in their original order.

The frame is reshaped -- no copy -- to carry a singleton at the stream slot, so
that `_qdn_cross_sketches` and `_qdn_pair_tensors` can be called on it
VERBATIM.  That is deliberate: the streaming path must not grow its own
transcription of those contractions, or the two will drift.  The stream axis's
stand-in is a saturated `ident` axis, which those functions pass through
untouched, and the actual `W_t[k, :]` is applied here as the rank-one factor of
the accumulation.
"""
function Base.push!(core::StreamingCore{Tc, N}, frame::AbstractArray) where {Tc, N}
    ndims(frame) == N - 1 ||
        error("StreamingCore: a frame of a valence-$N stream has $(N-1) axes, got " *
              "$(ndims(frame)).")
    collect(size(frame)) == core.framedims ||
        error("StreamingCore: frame size $(size(frame)) does not match the declared " *
              "$(Tuple(core.framedims)).")

    st = core.stream
    # A dense host `Array`, so that the singleton reshape is a view rather than
    # a `ReshapedArray` and `_qdn_mode_order` takes its HOST branch -- the
    # natural axis order, the same chain for every frame and the same one a
    # batch run would walk.  A `view` into a movie buffer is the common case and
    # is materialised here; a device array is copied back, which the file header
    # says is not the supported path.
    Gk = reshape(frame isa Array ? frame : Array(frame),
                 ntuple(i -> i == st ? 1 : core.dims[i], N))

    # One Gaussian row per frame, scaled so that the sketch does not drift in
    # magnitude with `r_stream`.  The scale is GLOBAL to every accumulator and
    # cancels out of a homogeneous null problem; it is here for conditioning,
    # not for correctness.
    rt = core.r[st]
    wk = randn(core.rng, Tc, rt) ./ sqrt(Tc(rt))
    push!(core.Wt, wk)
    wshape = reshape(wk, ntuple(i -> i == st ? rt : 1, N))

    Sf = _qdn_cross_sketches(Gk, core.axs, core.engaged)
    for a in core.eaxes
        core.S[a] .+= Sf[a] .* wshape
    end

    if !isempty(core.H)
        for a in core.lift
            bs = Int[b for b in core.eaxes if b != a]
            isempty(bs) && continue
            Hf = _qdn_pair_tensors(Gk, core.axs, a, bs)
            for b in bs
                core.H[(a, b)] .+= Hf[b] .* wshape
            end
        end
    end

    core.nrm2 += Float64(_qdn_safe_norm(frame))^2
    core.nframes += 1
    return core
end

"""
    nframes(core) -> Int

Frames folded in so far -- the true `d_t`, which nothing in the sizing or the
accumulators depends on.
"""
nframes(core::StreamingCore) = core.nframes

"""
    framenorm(core) -> Float64

`‖Γ‖` over the frames seen so far, accumulated in Float64 whatever the frames'
element type (a Float16 movie is the case this protects).
"""
framenorm(core::StreamingCore) = sqrt(core.nrm2)

"""
    crosssketches(core) -> Dict{Int, Array}

`S_a = Γ ×_{b≠a} W_b` for every engaged axis, exactly what
`_qdn_cross_sketches` would return for the concatenated tensor (to rounding:
the sum over frames is accumulated in a different order).
"""
crosssketches(core::StreamingCore) = core.S

"""
    pairtensors(core, a) -> Dict{Int, Array}

`H_{ab}` for the lift of axis `a`, keyed by `b` -- the shape
`_qdn_solve_and_lift` consumes.  Empty if the core was built with
`pairs = false`.
"""
pairtensors(core::StreamingCore, a::Integer) =
    Dict{Int, Array}(b => core.H[(a, b)] for b in core.eaxes if haskey(core.H, (a, b)))

"""
    streamsketch(core) -> Matrix

The stream axis's sketch `W_t`, `nframes(core) x r_stream`, materialised from
the rows drawn so far.  Small (frames by tens) and kept so that a streaming run
can be reproduced against a batch one.
"""
streamsketch(core::StreamingCore{Tc}) where {Tc} =
    isempty(core.Wt) ? Matrix{Tc}(undef, 0, core.r[core.stream]) :
                       Matrix(transpose(reduce(hcat, core.Wt)))

"""
    batchaxes(core) -> Vector{_QDNAxis}

The per-axis restriction data as the batch kernel wants it, with the stream
axis's singleton stand-in replaced by the `W_t` actually used.  Its `W⊥` is
empty: a disengaged axis is never lifted and never asks for one.

This is what a batch run has to be given to reproduce a streaming one, and it
is what `test/TestStreamingCore.jl` checks the equivalence with.
"""
function batchaxes(core::StreamingCore{Tc, N}) where {Tc, N}
    W = streamsketch(core)
    st = core.stream
    return _QDNAxis{Tc}[i == st ?
                        _QDNAxis{Tc}(size(W, 1), core.r[st], false, W,
                                     Matrix{Tc}(undef, size(W, 1), 0)) :
                        core.axs[i] for i in 1:N]
end

# ---------------------------------------------------------------------------
# The streaming Z-law residual
# ---------------------------------------------------------------------------

"""
    StreamingResidual(Ms, P; stream, block_bytes = 2^28)

Running `der_residual` for a candidate derivation over a stream.

The Z-law residual decomposes exactly frame by frame when the stream axis is
disengaged, because for every OTHER axis

    (Γ ×_a M_a)[…, k, …]  =  F_k ×_a M_a

and the stream axis's own term is skipped by `P[ρ, stream] == 0`.  So this is
not a new numerical kernel: each frame is handed to `der_residual_squares` with
the stream column of the chisel and the stream operator dropped, and the
per-row sums of squares accumulate.  The normaliser is `‖Γ‖` accumulated the
same way, times `max_a ‖M_a‖` over ALL axes -- the stream one included, so that
the number agrees with the batch `der_residual` and not merely with a
restriction of it.

`Ms` is one matrix per axis of the FULL tensor, frame index first, the
`embedITensors` convention that `der_residual` documents.
"""
mutable struct StreamingResidual{Tc}
    stream::Int
    Pr::Matrix{Tc}
    Ms::Vector{Matrix{Tc}}
    acc::Vector{Float64}
    nrm2::Float64
    nframes::Int
    opmax::Float64
    block_bytes::Int
end

function StreamingResidual(Ms::AbstractVector{<:AbstractMatrix}, P::AbstractMatrix;
                           stream::Int, block_bytes::Integer = 2^28)
    N = length(Ms)
    size(P, 2) == N ||
        error("StreamingResidual: the chisel has $(size(P,2)) columns but there are " *
              "$N operators.")
    1 <= stream <= N ||
        error("StreamingResidual: stream axis $stream is outside 1:$N.")
    all(iszero, view(P, :, stream)) ||
        error("StreamingResidual: column $stream of the chisel is not zero, so the " *
              "stream axis is engaged and the residual does not decompose frame by " *
              "frame. See docs/design/Streaming-Stratification.md section 1.")

    Tc = eltype(Ms[1])
    keep = Int[a for a in 1:N if a != stream]
    # `opmax` runs over EVERY axis, the disengaged stream one included, because
    # that is what the batch normaliser does; dropping it would make a streaming
    # residual and a batch residual disagree by a factor whenever the stream
    # operator happens to be the largest.
    opmax = Float64(maximum(a -> norm(Ms[a]), 1:N))
    return StreamingResidual{Tc}(stream, Matrix{Tc}(P[:, keep]),
                                 Matrix{Tc}[Matrix{Tc}(Ms[a]) for a in keep],
                                 zeros(Float64, size(P, 1)), 0.0, 0, opmax,
                                 Int(block_bytes))
end

"""
    push!(sr::StreamingResidual, frame) -> sr

Fold one frame's contribution into the running residual.
"""
function Base.push!(sr::StreamingResidual, frame::AbstractArray)
    ndims(frame) == length(sr.Ms) ||
        error("StreamingResidual: a frame has $(length(sr.Ms)) axes, got " *
              "$(ndims(frame)).")
    sq = der_residual_squares(frame, sr.Ms, sr.Pr; block_bytes = sr.block_bytes)
    sr.acc .+= Float64.(sq)
    sr.nrm2 += Float64(_qdn_safe_norm(frame))^2
    sr.nframes += 1
    return sr
end

"""
    der_residual(sr::StreamingResidual) -> Float64

`‖Σ_a P[ρ,a]·(Γ ×_a D_a)‖ / (‖Γ‖·max_a‖D_a‖)` over the frames seen so far --
the same quantity `der_residual(Γ, D, chisel)` returns for the whole tensor,
summed in a different order.

Because the derivation space is non-increasing along the stream, this number is
non-decreasing in expectation: it is the live measurement of whether a
candidate that survived warm-up is still a derivation.
"""
der_residual(sr::StreamingResidual) =
    sqrt(sum(sr.acc)) / max(sqrt(sr.nrm2) * sr.opmax, eps(Float64))

nframes(sr::StreamingResidual) = sr.nframes
