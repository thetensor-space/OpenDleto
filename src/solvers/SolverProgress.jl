#
# Progress reporting for the null solves.
#
# Chiseling spends its time applying *our own* maps -- `sylve`/`ester` in
# `sylvesterLM`, `forward`/`adjoint` in `denLM` -- so every unit of work is a
# call we control and can count.  That is what makes a meaningful progress
# report possible without instrumenting anybody's linear-algebra library: wrap
# the `LinearMap` in one that ticks a counter on each application, and compare
# the count against a complexity estimate.
#
# The estimate is exact for the case that actually makes people wait.
# `Matrix(L)` applies `L` once per column, so densifying a map with
# `prod(dims)` columns is exactly `prod(dims)` applications -- at n = 19 that
# is 6859 tensor contractions in a loop with no output at all.  For iterative
# solves the total is not known in advance, so the count and rate are reported
# without an ETA rather than with a fabricated one.
#
# OFF BY DEFAULT, selected by tag.  A library that writes to the terminal
# uninvited is a nuisance, and which stage you care about depends on what you
# are debugging.
#
#     der(Γ; progress = true)                  # every stage
#     den(Ω, P, Δ; progress = :densify)        # just the dense build
#     den(Ω, P, Δ; progress = [:solve])        # just the iterative applications
#

"""
    PROGRESS_TAGS

The stages that can be reported.

- `:densify` -- map applications spent building a dense matrix, one per column.
  Exact estimate, so this reports a percentage and an ETA.
- `:solve`   -- map applications inside an iterative solve.  Total unknown in
  advance, so this reports count and rate only.
"""
const PROGRESS_TAGS = (:densify, :solve)

"""
    ProgressSpec

Which stages to report, where, and how often.  Build with `progress_spec`.
"""
struct ProgressSpec
    tags::Set{Symbol}
    io::IO
    interval::Float64   # seconds between refreshes
    delay::Float64      # stay silent until the stage has run this long
end

"""
    progress_spec(p; io, interval, delay) -> ProgressSpec

Normalise the `progress` option.

Accepts `false`/`nothing` (silent, the default), `true`/`:all` (every stage),
or a tag or collection of tags from `PROGRESS_TAGS`.

`delay` keeps short solves silent: nothing is printed until a stage has been
running for that long, so the report appears exactly when a human starts
wondering whether anything is happening.
"""
progress_spec(p; io::IO = stderr, interval::Real = 0.25, delay::Real = 1.0) =
    ProgressSpec(__progress_tags(p), io, Float64(interval), Float64(delay))

__progress_tags(p::Bool) = p ? Set(PROGRESS_TAGS) : Set{Symbol}()
__progress_tags(::Nothing) = Set{Symbol}()
__progress_tags(p::ProgressSpec) = p.tags
__progress_tags(p::Symbol) =
    p === :all ? Set(PROGRESS_TAGS) :
    p === :none || p === :off ? Set{Symbol}() :
    (p in PROGRESS_TAGS ? Set((p,)) :
     error("Unknown progress tag :$p. Known: $(PROGRESS_TAGS), or true/false/:all."))
function __progress_tags(p)
    tags = Set{Symbol}()
    for t in p
        union!(tags, __progress_tags(t))
    end
    return tags
end

progress_spec(p::ProgressSpec; kwargs...) = p

"""Whether `tag` should be reported under this spec."""
wants_progress(spec::ProgressSpec, tag::Symbol) = tag in spec.tags

"""
    ProgressTracker

Counts map applications for one stage and prints a throttled single-line
report.  `expected == 0` means the total is unknown.
"""
mutable struct ProgressTracker
    label::String
    tag::Symbol
    expected::Int
    count::Int
    t0::Float64
    tlast::Float64
    spec::ProgressSpec
    shown::Bool
end

ProgressTracker(label::AbstractString, tag::Symbol, expected::Integer,
                spec::ProgressSpec) =
    ProgressTracker(String(label), tag, Int(expected), 0, time(), time(),
                    spec, false)

active(tr::ProgressTracker) = wants_progress(tr.spec, tr.tag)

function tick!(tr::ProgressTracker, k::Integer = 1)
    tr.count += k
    active(tr) || return tr
    now = time()
    elapsed = now - tr.t0
    (elapsed < tr.spec.delay || now - tr.tlast < tr.spec.interval) && return tr
    tr.tlast = now
    tr.shown = true
    __progress_print(tr, elapsed, false)
    return tr
end

function finish!(tr::ProgressTracker)
    (active(tr) && tr.shown) || return tr
    __progress_print(tr, time() - tr.t0, true)
    print(tr.spec.io, "\n")
    flush(tr.spec.io)
    return tr
end

function __progress_print(tr::ProgressTracker, elapsed::Float64, done::Bool)
    rate = tr.count / max(elapsed, eps())
    if tr.expected > 0
        frac = tr.count / tr.expected
        eta = frac > 0 ? elapsed * (1 - frac) / frac : NaN
        msg = string("\r  ", tr.label, " ", tr.count, "/", tr.expected, " (",
                     round(100 * frac; digits = 1), "%)  ",
                     round(elapsed; digits = 1), "s",
                     done ? "" : string("  eta ", round(eta; digits = 1), "s"))
    else
        msg = string("\r  ", tr.label, " ", tr.count, " applications  ",
                     round(elapsed; digits = 1), "s  ",
                     round(rate; digits = 1), "/s")
    end
    print(tr.spec.io, rpad(msg, 78))
    flush(tr.spec.io)
end

"""
    progress_wrap(L, tr::ProgressTracker) -> LinearMap

`L` with a counter on it.  Ticks once per forward or adjoint application, so a
`Matrix(L)` densification ticks once per column and an iterative solve ticks
once per matvec -- the same wrapper serves both stages.

Returns `L` unchanged when the tracker's tag is not being reported, so there is
no cost when progress is off.
"""
function progress_wrap(L, tr::ProgressTracker)
    active(tr) || return L
    fwd(v) = (tick!(tr); L * v)
    adj(v) = (tick!(tr); L' * v)
    return LinearMaps.LinearMap{eltype(L)}(fwd, adj, size(L, 1), size(L, 2);
                                ismutating = false,
                                issymmetric = LinearMaps.issymmetric(L))
end
